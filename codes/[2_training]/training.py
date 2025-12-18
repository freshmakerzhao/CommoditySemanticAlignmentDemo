import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoTokenizer
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

# ===================== 配置区 =====================
# 数据
INPUT_CSV = "../../train_master.csv" # 含 id,image_path,text,split

# 模型
IMAGE_MODEL = "openai/clip-vit-large-patch14"       # 图像编码器
TEXT_MODEL  = "hfl/chinese-roberta-wwm-ext-large"   # 中文文本编码器
PROJ_DIM    = 512                                    # 投影维度
FREEZE_IMAGE_LAYERS = 22  # ViT-L/14 共 24 层，冻结前 22 层
FREEZE_TEXT_LAYERS  = 22  # RoBERTa-large 共 24 层，冻结前 22 层

# 训练
BATCH_SIZE  = 32          # 可调，24GB 显存可尝试 64
EPOCHS      = 20
LR          = 1e-4        # 若解冻层多，可降至 5e-5
WARMUP_EPOCHS = 2
MAX_TEXT_LEN  = 128
DEVICE = "cuda"

# 保存
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 数据集 =====================
class ImageTextDataset(Dataset):
    def __init__(self, csv_path, split, image_processor, tokenizer, max_len=128):
        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 图像
        image = Image.open(row["image_path"]).convert("RGB")
        image_input = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        # 文本
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        text_input = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return {
            "image": image_input,
            "input_ids": text_input["input_ids"].squeeze(0),
            "attention_mask": text_input["attention_mask"].squeeze(0),
        }

# ===================== 模型 =====================
class DualEncoder(nn.Module):
    def __init__(self, image_model_name, text_model_name, proj_dim=512):
        super().__init__()
        # 图像编码器（CLIP ViT）
        clip_model = CLIPModel.from_pretrained(image_model_name)
        self.image_encoder = clip_model.vision_model
        img_dim = self.image_encoder.config.hidden_size
        
        # 文本编码器（中文 RoBERTa）
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        txt_dim = self.text_encoder.config.hidden_size
        
        # 投影头
        self.image_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(txt_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # 可学习温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, input_ids, attention_mask):
        # 图像特征
        img_out = self.image_encoder(pixel_values=image)
        img_feat = img_out.pooler_output  # [B, img_dim]
        img_emb = F.normalize(self.image_proj(img_feat), dim=-1)
        
        # 文本特征
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.pooler_output  # [B, txt_dim]
        txt_emb = F.normalize(self.text_proj(txt_feat), dim=-1)
        
        return img_emb, txt_emb, self.logit_scale.exp()

# ===================== 损失函数 =====================
def clip_loss(img_emb, txt_emb, logit_scale):
    """双向 InfoNCE 损失"""
    logits_per_image = logit_scale * img_emb @ txt_emb.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(len(img_emb), device=img_emb.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2

# ===================== 评估函数 =====================
@torch.no_grad()
def evaluate(model, dataloader, device):
    """计算 Recall@1, @5, @10"""
    model.eval()
    all_img_emb, all_txt_emb = [], []
    
    for batch in dataloader:
        img_emb, txt_emb, _ = model(
            batch["image"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )
        all_img_emb.append(img_emb.cpu())
        all_txt_emb.append(txt_emb.cpu())
    
    all_img_emb = torch.cat(all_img_emb, dim=0)
    all_txt_emb = torch.cat(all_txt_emb, dim=0)
    
    # 相似度矩阵
    sims = all_img_emb @ all_txt_emb.t()  # [N, N]
    n = sims.size(0)
    
    # Text → Image Recall
    t2i_ranks = []
    for i in range(n):
        rank = (sims[: , i] >= sims[i, i]).sum().item()
        t2i_ranks.append(rank)
    
    # Image → Text Recall
    i2t_ranks = []
    for i in range(n):
        rank = (sims[i, : ] >= sims[i, i]).sum().item()
        i2t_ranks.append(rank)
    
    def recall_at_k(ranks, k):
        return 100 * sum([1 for r in ranks if r <= k]) / len(ranks)
    
    metrics = {
        "t2i_R@1": recall_at_k(t2i_ranks, 1),
        "t2i_R@5": recall_at_k(t2i_ranks, 5),
        "t2i_R@10": recall_at_k(t2i_ranks, 10),
        "i2t_R@1":  recall_at_k(i2t_ranks, 1),
        "i2t_R@5": recall_at_k(i2t_ranks, 5),
        "i2t_R@10": recall_at_k(i2t_ranks, 10),
    }
    return metrics

# ===================== 冻结层 =====================
def freeze_layers(model, freeze_image_layers, freeze_text_layers):
    # 冻结图像编码器
    for name, param in model.image_encoder.named_parameters():
        layer_num = None
        if "layers." in name:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
            except: 
                pass
        if layer_num is None or layer_num < freeze_image_layers:
            param.requires_grad = False
    
    # 冻结文本编码器
    for name, param in model.text_encoder.named_parameters():
        layer_num = None
        if "layer." in name:
            try:
                layer_num = int(name.split("layer.")[1].split(".")[0])
            except: 
                pass
        if layer_num is None or layer_num < freeze_text_layers:
            param.requires_grad = False
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数:  {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")

# ===================== 主函数 =====================
def main():
    print(f"Device: {DEVICE}")
    
    # 加载 processor 和 tokenizer
    image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    
    # 数据集
    train_ds = ImageTextDataset(INPUT_CSV, "train", image_processor, tokenizer, MAX_TEXT_LEN)
    val_ds   = ImageTextDataset(INPUT_CSV, "val", image_processor, tokenizer, MAX_TEXT_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train:  {len(train_ds)} | Val: {len(val_ds)}")
    
    # 模型
    model = DualEncoder(IMAGE_MODEL, TEXT_MODEL, PROJ_DIM).to(DEVICE)
    freeze_layers(model, FREEZE_IMAGE_LAYERS, FREEZE_TEXT_LAYERS)
    
    # 优化器 + 学习率调度
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    
    def lr_lambda(step):
        if step < warmup_steps: 
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度
    
    # 训练循环
    best_recall = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar: 
            with torch.cuda.amp.autocast():
                img_emb, txt_emb, logit_scale = model(
                    batch["image"].to(DEVICE),
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE)
                )
                loss = clip_loss(img_emb, txt_emb, logit_scale)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        
        avg_loss = total_loss / len(train_loader)
        
        # 验证
        metrics = evaluate(model, val_loader, DEVICE)
        avg_recall = (metrics["t2i_R@1"] + metrics["i2t_R@1"]) / 2
        
        print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f}")
        print(f"  T→I: R@1={metrics['t2i_R@1']:.1f}, R@5={metrics['t2i_R@5']:.1f}, R@10={metrics['t2i_R@10']:.1f}")
        print(f"  I→T: R@1={metrics['i2t_R@1']:.1f}, R@5={metrics['i2t_R@5']:.1f}, R@10={metrics['i2t_R@10']:.1f}")
        
        # 保存最佳模型
        if avg_recall > best_recall: 
            best_recall = avg_recall
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  ✓ Best model saved (Avg R@1: {avg_recall:.1f})")
    
    print(f"\n训练完成！最佳 Avg R@1: {best_recall:.1f}")
    print(f"模型保存于: {SAVE_DIR}/best_model.pt")

if __name__ == "__main__": 
    main()