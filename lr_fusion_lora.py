# -*- coding: utf-8 -*-
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import dispatch_model

# ---------- 你原来的 load_model / get_lora_config / SimpleFusion / dataset 等 ------------
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


class SimpleFusion(nn.Module):
    """
    For a prompt of length L, produce fused embeddings:
      e_fused[i] = sigmoid(alpha[i]) * e_orig[i] + (1 - sigmoid(alpha[i])) * n[i]
    - alpha: (L,) learnable pre-sigmoid scalars (float)
    - n: (L, D) learnable noise embeddings (float)
    Note: if prompt lengths vary, pad/truncate to max_len or use per-batch lengths.
    """
    def __init__(self, prompt_len, emb_dim, init_alpha=0.0, noise_std=1e-3, device='cuda'):
        super().__init__()
        self.prompt_len = prompt_len
        self.emb_dim = emb_dim
        # pre-sigmoid alpha parameters (so alpha = sigmoid(logit))
        self.alpha_logit = nn.Parameter(torch.full((prompt_len,), float(init_alpha)))
        # per-token noise embeddings
        self.noise = nn.Parameter(torch.randn(prompt_len, emb_dim) * noise_std)
        self.device = device

    def forward(self, orig_embs, prompt_mask=None):
        """
        orig_embs: (batch, prompt_len, emb_dim)  -- embeddings of original prompt tokens
                   if orig_embs shorter/padded, ensure shape consistent
        prompt_mask: (batch, prompt_len) boolean, True for valid token positions (optional)
        returns fused_embs with same shape
        """
        # alpha: (prompt_len,) -> (1, prompt_len, 1)
        alpha = torch.sigmoid(self.alpha_logit).unsqueeze(0).unsqueeze(-1)
        noise = self.noise.unsqueeze(0)  # (1, prompt_len, emb_dim)
        fused = alpha * orig_embs + (1.0 - alpha) * noise

        if prompt_mask is not None:
            fused = fused * prompt_mask.unsqueeze(-1).float()

        return fused


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

# ---------------- 新增：Policy Network for token-level mask ----------------
class PolicyNetwork(nn.Module):
    """
    给每个 token 输出一个 mask logit。采用简单的 per-token MLP。
    输入: src_embeds [B, T, D]
    输出: mask_logits [B, T]  (通过 Gumbel-Softmax / Concrete 采样得到 [0,1] mask)
    """
    def __init__(self, hidden_dim, hidden_policy=128):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_policy)
        self.lin2 = nn.Linear(hidden_policy, 1)  # per-token scalar logit

    def forward(self, src_embeds):
        # src_embeds: [B, T, D]
        B, T, D = src_embeds.size()
        x = F.relu(self.lin1(src_embeds))  # [B, T, H]
        logits = self.lin2(x).squeeze(-1)  # [B, T]
        return logits

def sample_concrete(logits, temperature=0.5, hard=False):
    """
    logits: [B, T]  ->  返回 mask [B, T] in (0,1)
    使用 Gumbel-Softmax/Concrete for Bernoulli (relaxed).
    """
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
    y = (logits + g) / temperature
    y = torch.sigmoid(y)  # relaxed Bernoulli
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
    return y
def policy_ppo_update(policy, policy_optimizer, old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    """
    PPO-like update for token-level actions.
    Args:
        old_log_probs: [B, T] tensor of old log probs from policy
        new_log_probs: [B, T] tensor of new log probs from policy
        advantages: [B] tensor (sentence-level) or [B, T] tensor (token-level)
        clip_epsilon: PPO clip parameter
    Returns:
        loss value (float)
    """
    # 1. Ensure advantages is token-level [B, T]
    if advantages.dim() == 1:  # sentence-level
        adv = advantages.detach().unsqueeze(-1).expand(-1, new_log_probs.size(1))  # [B, T]
    else:
        adv = advantages.detach()  # already [B, T]

    # 2. Compute ratio
    ratio = (new_log_probs.squeeze(-1) - old_log_probs.squeeze(-1)).exp()  # [B, T]

    # 3. PPO clipped objective
    obj = ratio * adv
    obj_clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv

    # 4. Loss (negate for gradient ascent)
    loss = -torch.min(obj, obj_clipped).mean()

    # 5. Update
    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()

    return loss.item()




# ---------------- Training step (融合 RL + LoRA) ----------------
def training_step_with_rl(fusion_model, model_lora, base_model, tokenizer,
                          policy, policy_optimizer,
                          src_ids, attention_mask, labels,
                          optimizer_lora, device="cuda",
                          temp=0.6, clip_epsilon=0.2,
                          lambda_privacy=1.0, lambda_kd=0.2):
    """
    与原逻辑类似，但 SimplePerTokenFusion 内部控制 alpha，
    PPO policy 控制 alpha 的分布偏置（或噪声强度）。
    """
    model_lora.train()
    fusion_model.train()
    policy.train()

    src_ids = src_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # 1. Embedding
    embed_layer = base_model.model.get_input_embeddings()
    src_embeds = embed_layer(src_ids).detach()  # [B,T,D]

    # 2. Policy 输出 per-token bias，影响 fusion alpha
    # policy 输出 (B,T)，代表调整 fusion alpha 的偏置
    alpha_bias = policy(src_embeds)  # [B,T]
    alpha_bias = alpha_bias.unsqueeze(-1)  # [B,T,1]
    # fusion 模块中的 alpha_logit 是 (L,)
    # alpha_base = torch.sigmoid(fusion_model.module.alpha_logit if isinstance(fusion_model, nn.DataParallel) else fusion_model.alpha_logit)
    # alpha = torch.sigmoid(alpha_base.unsqueeze(0).unsqueeze(-1) + alpha_bias)  # [B,T,1]
    alpha_logit = (fusion_model.module.alpha_logit if isinstance(fusion_model, nn.DataParallel) else fusion_model.alpha_logit)  # shape (L,)
    alpha = torch.sigmoid(alpha_logit.unsqueeze(0).unsqueeze(-1) + alpha_bias)  # -> [B, T, 1]


    # 3. 融合 embedding
    noise = (fusion_model.module.noise if isinstance(fusion_model, nn.DataParallel) else fusion_model.noise).unsqueeze(0)  # [1,T,D]
    fused_embs = alpha * src_embeds + (1 - alpha) * noise

    # 保存 PPO 的 old_log_probs（相当于 action 概率）
    eps = 1e-9
    old_log_probs = alpha_bias.sigmoid().log()  # 简化：policy 输出的激活概率

    # 4. LoRA forward
    lora_outputs = model_lora(inputs_embeds=fused_embs, attention_mask=attention_mask)
    lora_logits = lora_outputs.logits

    with torch.no_grad():
        base_outputs = base_model(inputs_embeds=src_embeds, attention_mask=attention_mask)
        base_logits = base_outputs.logits

    # 5. LoRA loss
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

    # 6. 隐私/扰动目标 (差异越大越好)
    loss_secure = -F.cosine_similarity(
        fused_embs.flatten(1),
        src_embeds.flatten(1),
        dim=-1
    ).mean()

    # 7. 更新 LoRA 参数
    optimizer_lora.zero_grad()
    loss_lora.backward(retain_graph=True)
    optimizer_lora.step()

    # 8. 计算 reward
    with torch.no_grad():
        lora_logprobs = F.log_softmax(lora_logits, dim=-1)  # [B, T, V]
        # gather token log probs
        token_logps = lora_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
        per_token_ce = -token_logps  # 每个 token 的 CE 作为 reward
                # [B]

    utility_reward = -per_token_ce
    # privacy_reward = loss_secure.detach() * torch.ones_like(utility_reward)
    # fused_embs: [B,T,D], src_embeds: [B,T,D]
    privacy_reward = (1 - F.cosine_similarity(fused_embs, src_embeds, dim=-1))  # [B, T]
  
    # reward = utility_reward + lambda_privacy * privacy_reward
    reward = (utility_reward + lambda_privacy * privacy_reward)
    reward = (reward - reward.mean()) / (reward.std() + 1e-8)  # 标准化


    # 9. PPO policy 更新
    new_log_probs = alpha_bias.sigmoid().log()  # 相同结构
    # baseline = reward.mean()
    # advantages = reward - baseline
    advantages = reward.detach()  # 已经中心化

    policy_loss_val = policy_ppo_update(policy, policy_optimizer, old_log_probs.detach(), new_log_probs, advantages, clip_epsilon)

    metrics = {
        "loss_lora": loss_lora.item(),
        "loss_secure": loss_secure.item(),
        "policy_loss": policy_loss_val,
        "reward_mean": reward.mean().item()
    }
    return metrics

# ----------------- Test step (保留/微调) -----------------
@torch.no_grad()
def test_step(fusion_model, model_lora, base_model, tokenizer, policy, prompt, device="cuda", max_new_tokens=50):
    src_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    embed_layer = base_model.model.get_input_embeddings().to(device)
    src_embeds = embed_layer(src_ids).detach().clone()

    # Base model generate
    base_text = base_model.generate(input_ids=src_ids, max_new_tokens=max_new_tokens)
    base_text = tokenizer.decode(base_text[0], skip_special_tokens=True)

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


# ----------------- Main -----------------
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = "cuda"

    # Load base
    base_model, tokenizer = load_model("facebook/opt-1.3b", device=device)
    base_model.eval()
    vocab_size = tokenizer.vocab_size

    # fusion_model = SimpleFusion(hidden_dim=base_model.config.hidden_size).to(device)
    prompt_len = 128  # 与 prepare_dolly_dataset 中的 max_len 保持一致
    emb_dim = base_model.config.hidden_size
    fusion_model = SimpleFusion(prompt_len=prompt_len, emb_dim=emb_dim).to(device)


    # LoRA target layers (same你原来的)
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

    # Policy init
    policy = PolicyNetwork(hidden_dim=base_model.config.hidden_size).to(device)

    # Optimizers
    optimizer_lora = torch.optim.AdamW(model_lora.parameters(), lr=1e-5)
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-5)

    # Dataloader
    dataloader = prepare_dolly_dataset(tokenizer, max_len=128, batch_size=8)

    # DataParallel (保持)
    if torch.cuda.device_count() > 1:
        fusion_model = nn.DataParallel(fusion_model)
        model_lora = nn.DataParallel(model_lora)
        # policy 可以在单卡上或改为 DistributedDataParallel（这里保持简单）

    # Training loop
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

        # 每个 epoch 后跑一次测试
        test_step(fusion_model, model_lora, base_model, tokenizer, policy, prompt="Write a poetic summary of Reinforcement Learning.")

