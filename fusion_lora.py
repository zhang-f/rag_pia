from fusion import PositionalEncoding, load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# ----------------- LoRA config -----------------
def get_lora_config(target_lora_layer):
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=target_lora_layer,
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config


# ----------------- Fusion Model (Lightweight, no decoder) -----------------
import torch
import torch.nn as nn
import torch.nn.init as init
import math



# class FusionModel(nn.Module):
#     def __init__(self, vocab_size, d_model=4096, nhead=4, num_encoder_layers=4, dim_ff=2048, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_enc = PositionalEncoding(d_model, dropout)

#         # Encoder only
#         encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

#     def forward(self, src_ids, noise_std=0.01):

#         # 1. Encoder embedding
#         src = self.embedding(src_ids) * math.sqrt(self.d_model)
#         src = self.pos_enc(src)
#         z = self.encoder(src)  # [B, T, d_model]

#         # # 2. Mask RAG latent positions
#         # if mask_rag is not None:
#         #     z[:, mask_rag] = 0

#         # 3. Add noise for privacy / obfuscation
#         z_tilde = z + torch.randn_like(z) * noise_std

#         return z, z_tilde


class FusionModel(nn.Module):
    def __init__(self, d_model=4096, r=8, alpha=16, scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.r = r
        self.alpha = alpha
        self.scale = scale

        # 低秩分解，模仿 LoRA
        self.A = nn.Linear(d_model, r, bias=False)
        self.B = nn.Linear(r, d_model, bias=False)

        # 缩放因子 (LoRA 里面常用 alpha/r)
        self.scaling = self.alpha / self.r

        # 初始化 (LoRA 初始化：A 全 0，B 正态)
        nn.init.zeros_(self.A.weight)
        nn.init.normal_(self.B.weight, std=1e-4)

    def forward(self, src_embeds, noise_std=0.01):
        # 低秩扰动
        delta = self.B(self.A(src_embeds)) * self.scaling

        # 融合：原 embedding + 扰动
        z = src_embeds + delta * self.scale

        # 添加噪声（可选）
        z_tilde = z + torch.randn_like(z) * noise_std
        return z, z_tilde




def prepare_dataset(tokenizer, max_len=128, batch_size=2):
    # 1. Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # 2. Tokenize
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )
    tokenized = dataset.map(tokenize_fn, batched=True)

    # 3. Remove unused columns
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c != "input_ids"])

    # 4. Collate_fn to return tensors
    collate_fn = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

    # 5. DataLoader
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader




# ----------------- Training step -----------------
def training_step(fusion_model, model_lora, base_model, tokenizer, src_ids,  
                  optimizer_fusion, optimizer_lora, device="cuda"):
    fusion_model.train()
    model_lora.train()

    src_ids = src_ids.to(device)

    # ---------------- Forward pass ----------------
    # 1. Base embeddings
    embed_layer = base_model.model.get_input_embeddings()
    src_embeds = embed_layer(src_ids)

    # 2. Fusion forward (add noise / obfuscation)
    z, z_tilde = fusion_model(src_ids)
    perturbed_embeds = src_embeds + z_tilde.half()


    # 3. LoRA forward
    lora_outputs = model_lora(inputs_embeds=perturbed_embeds,
                              attention_mask=(src_ids != tokenizer.pad_token_id))
    print(lora_outputs)
    exit()
    lora_logits = lora_outputs.logits

    # 4. Teacher signal from ORIGINAL base_model
    with torch.no_grad():
        base_outputs = base_model(input_ids=src_ids)
        base_logits = base_outputs.logits


    # ---------------- Loss design ----------------
    min_len = min(lora_logits.size(1), base_logits.size(1))

    # LoRA tries to recover base outputs
    loss_lora = F.kl_div(
        F.log_softmax(lora_logits[:, :min_len], dim=-1),
        F.softmax(base_logits[:, :min_len], dim=-1),
        reduction="batchmean"
    )

    # Fusion tries to obfuscate RAG region while keeping utility
    rag_embeds = src_embeds
    obf_embeds = perturbed_embeds

    loss_secure = -F.mse_loss(obf_embeds, rag_embeds)
    loss_utility = F.kl_div(
        F.log_softmax(lora_logits[:, :min_len], dim=-1),
        F.softmax(base_logits[:, :min_len], dim=-1),
        reduction="batchmean"
    )

    # ---------------- Update LoRA ----------------
    optimizer_lora.zero_grad()
    loss_lora.backward(retain_graph=True)
    optimizer_lora.step()

    # ---------------- Update Fusion ----------------
    optimizer_fusion.zero_grad()
    loss_fusion = loss_utility + 0.5 * loss_secure
    loss_fusion.backward()
    optimizer_fusion.step()

    return {
        "loss_lora": loss_lora.item(),
        "loss_fusion": loss_fusion.item(),
        "loss_utility": loss_utility.item(),
        "loss_secure": loss_secure.item()
    }


# ----------------- Main -----------------
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 多GPU
    device = "cuda"

    # 1. Load models
    base_model, tokenizer = load_model("huggyllama/llama-7b", device=device)
    base_model.eval()  # teacher, frozen
    vocab_size = tokenizer.vocab_size
    fusion_model = FusionModel(vocab_size=vocab_size).to(device)

    # 2. LoRA target layers
    num_layers = base_model.config.num_hidden_layers
    last_layer_idx = num_layers - 1
    target_lora_layer = [
        f"model.layers.{last_layer_idx}.self_attn.q_proj",
        f"model.layers.{last_layer_idx}.self_attn.k_proj",
        f"model.layers.{last_layer_idx}.self_attn.v_proj",
        f"model.layers.{last_layer_idx}.self_attn.o_proj",
        f"model.layers.{last_layer_idx}.mlp.down_proj",
        f"model.layers.{last_layer_idx}.mlp.up_proj",
        "lm_head"
    ]
    lora_config = get_lora_config(target_lora_layer)
    model_lora = get_peft_model(base_model, lora_config).to(device)

    # ---------------- Optimizers ----------------
    optimizer_fusion = torch.optim.AdamW(fusion_model.parameters(), lr=1e-3)
    optimizer_lora = torch.optim.AdamW(model_lora.parameters(), lr=1e-4)

    # ---------------- Example data ----------------
    # prompt_text = "What is the salary of Mary?"
    # rag_text = " Mary is a software engineer at a tech company. She earns $720,000 per year."
    # src_text = prompt_text + " " + rag_text
    # src_ids = tokenizer(src_text, return_tensors="pt").input_ids
    # tgt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    # prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    # rag_mask = list(range(prompt_len, src_ids.size(1)))

    # ---------------- Dataset & Dataloader ----------------
    dataloader = prepare_dataset(tokenizer, max_len=128)

    # ---------------- Multi-GPU (DDP) ----------------
    if torch.cuda.device_count() > 1:
        fusion_model = nn.DataParallel(fusion_model)
        model_lora = nn.DataParallel(model_lora)

    # ---------------- Training loop ----------------
    for step, batch in enumerate(dataloader):
        src_ids = batch["input_ids"]
        metrics = training_step(
            fusion_model, model_lora, base_model, tokenizer, src_ids,
            optimizer_fusion, optimizer_lora, device
        )
        if step % 5 == 0:
            print(f"[{step}] LoRA={metrics['loss_lora']:.4f}, "
                  f"Fusion={metrics['loss_fusion']:.4f}, "
                  f"Utility={metrics['loss_utility']:.4f}, "
                  f"Secure={metrics['loss_secure']:.4f}")
            

    ## test
    fusion_model.eval()
    model_lora.eval()
    base_model.eval()
    with torch.no_grad():
        # 1. Fusion forward
        logits_fusion, z, z_tilde = fusion_model(src_ids)

        # 2. 原始 embedding
        embed_layer = base_model.model.get_input_embeddings()
        orig_embeds = embed_layer(src_ids)  # [B, T, d_model]

        # 3. Fusion perturbation embedding
        perturbed_embeds = orig_embeds + z_tilde.half()  

        # 4. Fusion + LLaMA generate (只取新生成部分)
        outputs = model_lora.generate(
            inputs_embeds=perturbed_embeds,
            attention_mask=(src_ids != tokenizer.pad_token_id),
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        gen_text = tokenizer.decode(outputs[0][src_ids.size(1):], skip_special_tokens=True)
        print("Generated Text with LoRA:", gen_text)

        # 5. Base model generate (no LoRA, no Fusion)
        base_outputs = base_model.generate(
            input_ids=src_ids,
            max_new_tokens=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        base_gen_text = tokenizer.decode(base_outputs[0][src_ids.size(1):], skip_special_tokens=True)
        print("Generated Text without LoRA:", base_gen_text)
            
