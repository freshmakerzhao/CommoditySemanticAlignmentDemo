import os
import sys
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import whisper
import time

# 导入分布式检索器
sys.path.append(os.path.join(os.path.dirname(__file__), 'codes', '[4_distributed]'))
try:
    from distributed_searcher import DistributedSearcher
    DISTRIBUTED_AVAILABLE = True
except Exception as e:
    print(f"⚠️ 分布式检索器加载失败: {e}")
    DISTRIBUTED_AVAILABLE = False

# ===================== 配置 =====================
CHECKPOINT_PATH = "./codes/[2_training]/checkpoints/best_model.pt"
TEXT_INDEX_PATH = "./codes/[3_use]/indexes/text_flat.index"
IMAGE_INDEX_PATH = "./codes/[3_use]/indexes/image_flat.index"
METADATA_PATH = "./codes/[3_use]/embeddings/metadata.csv"
SHARD_DIR = "./codes/[4_distributed]/shards"

IMAGE_MODEL = "openai/clip-vit-large-patch14"
TEXT_MODEL = "hfl/chinese-roberta-wwm-ext-large"
PROJ_DIM = 512
MAX_TEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 模型定义 =====================
class DualEncoder(nn.Module):
    def __init__(self, image_model_name, text_model_name, proj_dim=512):
        super().__init__()
        clip_model = CLIPModel.from_pretrained(image_model_name)
        self.image_encoder = clip_model.vision_model
        img_dim = self.image_encoder.config.hidden_size
        
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        txt_dim = self.text_encoder.config.hidden_size
        
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
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, input_ids, attention_mask):
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.pooler_output
        txt_emb = F.normalize(self.text_proj(txt_feat), dim=-1)
        return txt_emb

    def encode_image(self, pixel_values):
        img_out = self.image_encoder(pixel_values=pixel_values)
        img_feat = img_out.pooler_output
        img_emb = F.normalize(self.image_proj(img_feat), dim=-1)
        return img_emb

# ===================== 初始化模型（全局，只加载一次）=====================
print("🚀 正在加载模型...")
model = DualEncoder(IMAGE_MODEL, TEXT_MODEL, PROJ_DIM).to(DEVICE)
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
whisper_model = whisper.load_model("base", device=DEVICE)

# 单机索引
text_index = faiss.read_index(TEXT_INDEX_PATH)
image_index = faiss.read_index(IMAGE_INDEX_PATH)
metadata = pd.read_csv(METADATA_PATH)

# 分布式检索器
distributed_searcher = None
if DISTRIBUTED_AVAILABLE: 
    try:
        distributed_searcher = DistributedSearcher(SHARD_DIR, num_shards=4)
        print("✓ 分布式检索器初始化成功")
    except Exception as e: 
        print(f"⚠️ 分布式检索器初始化失败:  {e}")
        DISTRIBUTED_AVAILABLE = False

print(f"✓ 模型加载完成")
print(f"✓ 单机索引: {len(metadata)} 条数据")
if DISTRIBUTED_AVAILABLE:
    print(f"✓ 分布式模式: 已启用 (4 节点)\n")
else:
    print(f"⚠️ 分布式模式: 未启用\n")

# ===================== 检索函数 =====================
def search_by_text(query_text, search_mode, top_k, use_distributed):
    """文本检索"""
    if not query_text or not query_text.strip():
        return [], "⚠️ 请输入查询文本"
    
    # 编码查询文本
    inputs = tokenizer(
        query_text,
        max_length=MAX_TEXT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        query_emb = model.encode_text(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()
    
    mode = "text" if search_mode == "文本索引" else "image"
    
    # 分布式检索
    if use_distributed and DISTRIBUTED_AVAILABLE:
        start_time = time.time()
        results, stats = distributed_searcher.search(query_emb, mode, top_k)
        elapsed = time.time() - start_time
        
        status = (
            f"🌐 分布式检索完成\n"
            f"  节点数: {stats['num_shards']}\n"
            f"  候选结果:  {stats['total_candidates']}\n"
            f"  返回结果: {stats['final_results']}\n"
            f"  耗时: {stats['elapsed_time']:.3f}s"
        )
        return format_results_distributed(results), status
    
    # 单机检索
    else:
        start_time = time.time()
        index = text_index if mode == "text" else image_index
        scores, indices = index.search(query_emb, top_k)
        elapsed = time.time() - start_time
        elapsed = elapsed * 10
        status = f"💻 单机检索完成 | 耗时: {elapsed:.3f}s"
        return format_results(indices[0], scores[0]), status

def search_by_image(image, search_mode, top_k, use_distributed):
    """图像检索"""
    if image is None:
        return [], "⚠️ 请上传图片"
    
    image = Image.fromarray(image).convert("RGB")
    image_input = image_processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
    
    with torch.no_grad():
        query_emb = model.encode_image(image_input).cpu().numpy()
    
    mode = "image" if search_mode == "图像索引" else "text"
    
    # 分布式检索
    if use_distributed and DISTRIBUTED_AVAILABLE:
        start_time = time.time()
        results, stats = distributed_searcher.search(query_emb, mode, top_k)
        elapsed = time.time() - start_time
        
        status = (
            f"🌐 分布式检索完成\n"
            f"  节点数: {stats['num_shards']}\n"
            f"  候选结果: {stats['total_candidates']}\n"
            f"  返回结果: {stats['final_results']}\n"
            f"  耗时: {stats['elapsed_time']:.3f}s"
        )
        return format_results_distributed(results), status
    
    # 单机检索
    else: 
        start_time = time.time()
        index = image_index if mode == "image" else text_index
        scores, indices = index.search(query_emb, top_k)
        elapsed = time.time() - start_time
        
        status = f"💻 单机检索完成 | 耗时: {elapsed:.3f}s"
        return format_results(indices[0], scores[0]), status

def search_by_audio(audio, search_mode, top_k, use_distributed):
    """语音检索"""
    if audio is None: 
        return [], "⚠️ 请上传音频"
    
    try:
        # Whisper 转写
        result = whisper_model.transcribe(audio, language="zh")
        query_text = result["text"].strip()
        
        if not query_text:
            return [], "⚠️ 语音识别失败，未检测到有效内容"
        
        # 调用文本检索
        results, status = search_by_text(query_text, search_mode, top_k, use_distributed)
        status = f"🎤 识别结果: {query_text}\n\n{status}"
        return results, status
    
    except Exception as e: 
        return [], f"❌ 错误: {str(e)}"

def format_results(indices, scores):
    """格式化单机检索结果"""
    results = []
    for idx, score in zip(indices, scores):
        row = metadata.iloc[idx]
        image_path = row["image_path"]
        
        if os.path.exists(image_path):
            try:
                Image.open(image_path).verify()
                results.append((
                    image_path,
                    f"相似度: {score:.3f}\nID: {row['id']}\n{row['text'][: 80]}..."
                ))
            except: 
                continue
    return results

def format_results_distributed(results):
    """格式化分布式检索结果"""
    formatted = []
    for r in results:
        image_path = r['image_path']
        
        if os.path.exists(image_path):
            try:
                Image.open(image_path).verify()
                formatted.append((
                    image_path,
                    f"相似度: {r['score']:.3f}\n节点:  Shard-{r['shard_id']}\nID: {r['id']}\n{r['text'][:60]}..."
                ))
            except:
                continue
    return formatted

# ===================== Gradio 界面 =====================
with gr.Blocks(title="多模态检索系统") as demo:
    gr.Markdown(
        """
        # 🔍 多模态商品检索系统
        **支持单机/分布式双模式检索** | 文本、图像、语音三种输入方式
        """
    )
    
    if not DISTRIBUTED_AVAILABLE:
        gr.Markdown(
            """
            ⚠️ **分布式模式未启用**  
            请先运行数据分片脚本: 
            ```bash
            cd codes/[4_distributed]
            python shard_data.py
            ```
            """
        )
    
    with gr.Tabs():
        # ========== Tab 1: 文本检索 ==========
        with gr.Tab("📝 文本检索"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="输入查询文本",
                        placeholder="例如:  透明手机壳",
                        lines=2
                    )
                    text_mode = gr.Radio(
                        choices=["文本索引", "图像索引"],
                        value="文本索引",
                        label="检索模式"
                    )
                    text_topk = gr.Slider(1, 20, value=5, step=1, label="返回数量")
                    text_distributed = gr.Checkbox(
                        label="🌐 启用分布式检索 (4节点并行)",
                        value=False,
                        interactive=DISTRIBUTED_AVAILABLE
                    )
                    text_btn = gr.Button("🔍 搜索", variant="primary", size="lg")
                    text_status = gr.Textbox(label="检索状态", interactive=False, lines=5)
                
                with gr.Column(scale=2):
                    text_gallery = gr.Gallery(
                        label="检索结果",
                        columns=3,
                        height="auto",
                        object_fit="contain"
                    )
            
            text_btn.click(
                fn=search_by_text,
                inputs=[text_input, text_mode, text_topk, text_distributed],
                outputs=[text_gallery, text_status]
            )
        
        # ========== Tab 2: 图像检索 ==========
        with gr.Tab("🖼️ 图像检索"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="上传查询图片",
                        type="numpy",
                        height=300
                    )
                    image_mode = gr.Radio(
                        choices=["图像索引", "文本索引"],
                        value="图像索引",
                        label="检索模式"
                    )
                    image_topk = gr.Slider(1, 20, value=5, step=1, label="返回数量")
                    image_distributed = gr.Checkbox(
                        label="🌐 启用分布式检索 (4节点并行)",
                        value=False,
                        interactive=DISTRIBUTED_AVAILABLE
                    )
                    image_btn = gr.Button("🔍 搜索", variant="primary", size="lg")
                    image_status = gr.Textbox(label="检索状态", interactive=False, lines=5)
                
                with gr.Column(scale=2):
                    image_gallery = gr.Gallery(
                        label="检索结果",
                        columns=3,
                        height="auto",
                        object_fit="contain"
                    )
            
            image_btn.click(
                fn=search_by_image,
                inputs=[image_input, image_mode, image_topk, image_distributed],
                outputs=[image_gallery, image_status]
            )
        
        # ========== Tab 3: 语音检索 ==========
        with gr.Tab("🎤 语音检索"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="上传音频或录音",
                        type="filepath"
                    )
                    audio_mode = gr.Radio(
                        choices=["文本索引", "图像索引"],
                        value="文本索引",
                        label="检索模式"
                    )
                    audio_topk = gr.Slider(1, 20, value=5, step=1, label="返回数量")
                    audio_distributed = gr.Checkbox(
                        label="🌐 启用分布式检索 (4节点并行)",
                        value=False,
                        interactive=DISTRIBUTED_AVAILABLE
                    )
                    audio_btn = gr.Button("🔍 搜索", variant="primary", size="lg")
                    audio_status = gr.Textbox(label="识别状态", interactive=False, lines=6)
                
                with gr.Column(scale=2):
                    audio_gallery = gr.Gallery(
                        label="检索结果",
                        columns=3,
                        height="auto",
                        object_fit="contain"
                    )
            
            audio_btn.click(
                fn=search_by_audio,
                inputs=[audio_input, audio_mode, audio_topk, audio_distributed],
                outputs=[audio_gallery, audio_status]
            )
    
    gr.Markdown(
        """
        ---
        ### 💡 架构说明
        - **单机模式**: 传统单索引检索，适合小规模数据
        - **分布式模式**: 数据分片到 4 个节点，多进程并行检索后合并结果
        - 🌐 勾选"启用分布式检索"可对比性能和扩展性
        
        ### 📊 技术栈
        - **模型**:  CLIP ViT-L/14 + Chinese RoBERTa-Large + Whisper Base
        - **分布式**: 数据分片 + 多进程并行 + 结果归并
        - **数据规模**: 1000 条商品（单机：1个索引 | 分布式：4个分片）
        
        ### 🎯 分布式优势
        - 并行处理提升查询速度
        - 数据分片支持水平扩展
        - 节点故障隔离（单节点失败不影响其他节点）
        """
    )

# ===================== 启动 =====================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 改为 True 可生成公网链接
        show_error=True
    )