import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import dispatch_model

# ------------------ 基础模型加载 ------------------
def load_model(model_name, device="cuda"):
    model_kwargs = {"torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                    "use_cache": False,
                    "trust_remote_code": True}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    device_map = {}
    num_layers = base_model.config.num_hidden_layers
    for i in range(num_layers):
        device_map[f"model.decoder.layers.{i}"] = device
    device_map["model.decoder.embed_tokens"] = device
    device_map["model.decoder.embed_positions"] = device
    device_map["model.decoder.final_layer_norm"] = device
    device_map["lm_head"] = device

    model = dispatch_model(base_model, device_map=device_map)
    model.gradient_checkpointing = True
    return model, tokenizer

def get_lora_config(target_lora_layer):
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=target_lora_layer,
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config

# ------------------ 融合模块 ------------------
class SimpleFusion(nn.Module):
    """
    对prompt做token-level隐私embedding混合。
    """
    def __init__(self, prompt_len, emb_dim, init_alpha=0.0, noise_std=1e-3, device='cuda'):
        super().__init__()
        self.prompt_len = prompt_len
        self.emb_dim = emb_dim
        self.alpha_logit = nn.Parameter(torch.full((prompt_len,), float(init_alpha)))
        self.noise = nn.Parameter(torch.randn(prompt_len, emb_dim) * noise_std)
        self.device = device

    def forward(self, orig_embs, prompt_mask=None):
        # orig_embs: (batch, prompt_len, emb_dim)
        alpha = torch.sigmoid(self.alpha_logit).unsqueeze(0).unsqueeze(-1)
        noise = self.noise.unsqueeze(0)
        fused = alpha * orig_embs + (1.0 - alpha) * noise
        if prompt_mask is not None:
            fused = fused * prompt_mask.unsqueeze(-1).float()
        return fused

# ------------------ 数据集准备 ------------------
def prepare_dolly_dataset(tokenizer, max_len=256, batch_size=16, frac=0.9, seed=42):
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    if frac < 1.0:
        dataset = dataset.train_test_split(test_size=1-frac, seed=seed)["train"]

    def tokenize_fn(example):
        if example["context"]:
            prompt = f"Instruction: {example['instruction']}\nContext: {example['context']}\nAnswer:"
        else:
            prompt = f"Instruction: {example['instruction']}\nAnswer:"
        model_inputs = tokenizer(prompt, truncation=True, max_length=max_len, padding="max_length")
        labels = tokenizer(example["response"], truncation=True, max_length=max_len, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)
    collate_fn = DataCollatorForSeq2Seq(tokenizer, padding="longest", return_tensors="pt")
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

# ------------------ Policy Network & 采样 ------------------
class PolicyNetwork(nn.Module):
    """
    Token级mask策略网络，MLP结构。
    """
    def __init__(self, hidden_dim, hidden_policy=128):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_policy)
        self.lin2 = nn.Linear(hidden_policy, 1)

    def forward(self, src_embeds):
        x = F.relu(self.lin1(src_embeds))
        logits = self.lin2(x).squeeze(-1)
        return logits

def sample_concrete(logits, temperature=0.5, hard=False, eps=1e-9):
    """
    Gumbel-Sigmoid采样Bernoulli mask，同时返回mask和log_prob。
    """
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    y = (logits + g) / temperature
    y = torch.sigmoid(y)
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
    log_prob = y * torch.log(torch.sigmoid(logits) + eps) + (1 - y) * torch.log(1 - torch.sigmoid(logits) + eps)
    return y, log_prob

def policy_ppo_update(policy, policy_optimizer, old_log_probs, new_log_probs, advantages, mask=None, clip_epsilon=0.2):
    """
    PPO-like update，兼容padding mask。
    """
    if mask is not None:
        adv = advantages * mask
        ratio = ((new_log_probs - old_log_probs).exp()) * mask
    else:
        adv = advantages
        ratio = (new_log_probs - old_log_probs).exp()
    obj = ratio * adv
    obj_clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv
    loss = -torch.min(obj, obj_clipped)
    if mask is not None:
        loss = (loss.sum() / (mask.sum() + 1e-8))
    else:
        loss = loss.mean()
    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    return loss.item()

# ------------------ 训练迭代 ------------------
def training_step_with_rl(
    fusion_model, model_lora, base_model, tokenizer,
    policy, policy_optimizer,
    src_ids, attention_mask, labels,
    optimizer_lora, device="cuda",
    temp=0.6, clip_epsilon=0.2,
    lambda_privacy=1.0, lambda_kd=0.2
):
    # 准备数据
    model_lora.train()
    fusion_model.train()
    policy.train()
    src_ids = src_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    mask = (labels != tokenizer.pad_token_id).float()  # [B, T]
    # Embedding
    embed_layer = base_model.model.get_input_embeddings()
    src_embeds = embed_layer(src_ids).detach()  # [B, T, D]
    # Policy采样
    alpha_logits = policy(src_embeds)  # [B, T]
    alpha_mask, log_prob = sample_concrete(alpha_logits, temperature=temp, hard=True)  # [B, T], [B, T]
    alpha_bias = alpha_logits.unsqueeze(-1)  # [B, T, 1]
    # 融合
    alpha_logit = (fusion_model.module.alpha_logit if hasattr(fusion_model, "module") else fusion_model.alpha_logit)
    alpha = torch.sigmoid(alpha_logit.unsqueeze(0).unsqueeze(-1) + alpha_bias)
    noise = (fusion_model.module.noise if hasattr(fusion_model, "module") else fusion_model.noise).unsqueeze(0)
    fused_embs = alpha * src_embeds + (1 - alpha) * noise
    # LoRA forward
    lora_outputs = model_lora(inputs_embeds=fused_embs, attention_mask=attention_mask)
    lora_logits = lora_outputs.logits
    with torch.no_grad():
        base_outputs = base_model(inputs_embeds=src_embeds, attention_mask=attention_mask)
        base_logits = base_outputs.logits
    # LoRA损失
    loss_ce = F.cross_entropy(
        lora_logits.reshape(-1, lora_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=tokenizer.pad_token_id
    )
    loss_kd = F.kl_div(
        F.log_softmax(lora_logits.float(), dim=-1),
        F.softmax(base_logits.float(), dim=-1),
        reduction="batchmean"
    )
    loss_lora = (1 - lambda_kd) * loss_ce + lambda_kd * loss_kd
    # 隐私扰动loss
    loss_secure = -F.cosine_similarity(
        fused_embs.flatten(1),
        src_embeds.flatten(1),
        dim=-1
    ).mean()
    # LoRA参数更新
    optimizer_lora.zero_grad()
    loss_lora.backward(retain_graph=True)
    optimizer_lora.step()
    # reward计算（mask掉pad）
    with torch.no_grad():
        lora_logprobs = F.log_softmax(lora_logits, dim=-1)
        token_logps = lora_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        per_token_ce = -token_logps
    utility_reward = -per_token_ce * mask
    privacy_reward = (1 - F.cosine_similarity(fused_embs, src_embeds, dim=-1)) * mask
    reward = (utility_reward + lambda_privacy * privacy_reward)
    reward_mean = (reward.sum() / (mask.sum() + 1e-8))
    reward_std = ((reward * mask - reward_mean).pow(2).sum() / (mask.sum() + 1e-8)).sqrt()
    reward = (reward - reward_mean) / (reward_std + 1e-8) * mask
    # PPO policy更新
    old_log_probs = log_prob.detach()

    new_logits = policy(src_embeds)
    new_prob = torch.sigmoid(new_logits)
    new_log_prob = alpha_mask * (new_prob + 1e-9).log() + (1 - alpha_mask) * (1 - new_prob + 1e-9).log()
    advantages = reward.detach()
    policy_loss_val = policy_ppo_update(policy, policy_optimizer, old_log_probs, new_log_prob, advantages, mask, clip_epsilon)
    metrics = {
        "loss_lora": loss_lora.item(),
        "loss_secure": loss_secure.item(),
        "policy_loss": policy_loss_val,
        "reward_mean": reward_mean.item()
    }
    return metrics

# ------------------ 推理/测试 ------------------
@torch.no_grad()
def test_step(fusion_model, model_lora, base_model, tokenizer, policy, prompt, device="cuda", max_new_tokens=50):
    src_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    embed_layer = base_model.model.get_input_embeddings().to(device)
    src_embeds = embed_layer(src_ids).detach().clone()
    base_text_ids = base_model.generate(input_ids=src_ids, max_new_tokens=max_new_tokens)
    base_text = tokenizer.decode(base_text_ids[0], skip_special_tokens=True)
    # Policy -> mask
    logits = policy(src_embeds)
    mask = (torch.sigmoid(logits) > 0.5).float()
    perturbed_embeds = fusion_model(src_embeds * mask.unsqueeze(-1))
    # Greedy generate with LoRA
    lora_model = model_lora.module if isinstance(model_lora, nn.DataParallel) else model_lora
    generated_ids = src_ids
    for _ in range(max_new_tokens):
        logits = lora_model(inputs_embeds=perturbed_embeds, attention_mask=(generated_ids != tokenizer.pad_token_id)).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        perturbed_embeds = fusion_model(embed_layer(generated_ids))
    lora_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n====== TEST ======")
    print(f"[Base   ] {base_text}")
    print(f"[LoRA   ] {lora_text}")
    print("==================\n")

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = "cuda"
    base_model, tokenizer = load_model("facebook/opt-1.3b", device=device)
    base_model.eval()
    vocab_size = tokenizer.vocab_size
    prompt_len = 128
    emb_dim = base_model.config.hidden_size
    fusion_model = SimpleFusion(prompt_len=prompt_len, emb_dim=emb_dim).to(device)
    # LoRA target layers
    num_layers = base_model.config.num_hidden_layers
    last_layer_idx = num_layers - 1
    num_lora = 3
    target_lora_layer = []
    for i in range(num_lora):
        layer_idx = last_layer_idx - i
        target_lora_layer.extend([
            f"model.decoder.layers.{layer_idx}.self_attn.q_proj",
            f"model.decoder.layers.{layer_idx}.self_attn.k_proj",
            f"model.decoder.layers.{layer_idx}.self_attn.v_proj",
            f"model.decoder.layers.{layer_idx}.self_attn.o_proj",
            f"model.decoder.layers.{layer_idx}.mlp.down_proj",
            f"model.decoder.layers.{layer_idx}.mlp.up_proj",
        ])
    lora_config = get_lora_config(target_lora_layer)
    model_lora = get_peft_model(base_model, lora_config).to(device)
    policy = PolicyNetwork(hidden_dim=base_model.config.hidden_size).to(device)
    optimizer_lora = torch.optim.AdamW(model_lora.parameters(), lr=1e-5)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5)
    dataloader = prepare_dolly_dataset(tokenizer, max_len=128, batch_size=8)
    # DataParallel
    if torch.cuda.device_count() > 1:
        fusion_model = nn.DataParallel(fusion_model)
        model_lora = nn.DataParallel(model_lora)
    for epoch in range(30):
        for step, batch in enumerate(dataloader):
            src_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            metrics = training_step_with_rl(
                fusion_model, model_lora, base_model, tokenizer,
                policy, policy_optimizer,
                src_ids, attention_mask, labels,
                optimizer_lora, device=device,
                temp=0.6, clip_epsilon=0.2, lambda_privacy=0.5, lambda_kd=0.2
            )
            if step % 50 == 0:
                print(f"[epoch {epoch} step {step}] loss_lora={metrics['loss_lora']:.4f}, secure={metrics['loss_secure']:.4f}, policy_loss={metrics['policy_loss']:.4f}, reward={metrics['reward_mean']:.4f}")
        # 每个epoch测试一次
        test_step(fusion_model, model_lora, base_model, tokenizer, policy, prompt="Write a poetic summary of Reinforcement Learning.")
