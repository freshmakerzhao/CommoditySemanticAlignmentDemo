import os
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

# ===================== 配置 =====================
CHECKPOINT_PATH = "./codes/[2_training]/checkpoints/best_model.pt"
TEXT_INDEX_PATH = "./codes/[3_use]/indexes/text_flat.index"
IMAGE_INDEX_PATH = "./codes/[3_use]/indexes/image_flat.index"
METADATA_PATH = "./codes/[3_use]/embeddings/metadata.csv"

IMAGE_MODEL = "openai/clip-vit-large-patch14"
TEXT_MODEL = "hfl/chinese-roberta-wwm-ext-large"
PROJ_DIM = 512
MAX_TEXT_LEN = 128
DEVICE = "cuda"

# ===================== 模型定义 =====================
class DualEncoder(nn.Module):
    def __init__(self, image_model_name, text_model_name, proj_dim=512):
        super().__init__()
        clip_model = CLIPModel.from_pretrained(image_model_name)
        self.image_encoder = clip_model.vision_model
        img_dim = self.image_encoder.config.hidden_size
        
        self.text_encoder = AutoModel. from_pretrained(text_model_name)
        txt_dim = self.text_encoder.config.hidden_size
        
        self.image_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim),
            nn. GELU(),
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
print("Loading models...")
model = DualEncoder(IMAGE_MODEL, TEXT_MODEL, PROJ_DIM).to(DEVICE)
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
whisper_model = whisper.load_model("base", device=DEVICE)

text_index = faiss.read_index(TEXT_INDEX_PATH)
image_index = faiss. read_index(IMAGE_INDEX_PATH)
metadata = pd.read_csv(METADATA_PATH)

print("✓ Models loaded!\n")

# ===================== 检索函数 =====================
def search_by_text(query_text, search_mode, top_k):
    """文本检索"""
    inputs = tokenizer(
        query_text,
        max_length=MAX_TEXT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        query_emb = model. encode_text(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()
    
    index = text_index if search_mode == "文本索引" else image_index
    scores, indices = index.search(query_emb, top_k)
    
    return format_results(indices[0], scores[0])

def search_by_image(image, search_mode, top_k):
    """图像检索"""
    if image is None:
        return [], "⚠️ 请上传图片"
    
    image = Image.fromarray(image).convert("RGB")
    image_input = image_processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
    
    with torch.no_grad():
        query_emb = model. encode_image(image_input).cpu().numpy()
    
    index = image_index if search_mode == "图像索引" else text_index
    scores, indices = index.search(query_emb, top_k)
    
    return format_results(indices[0], scores[0])

def search_by_audio(audio, search_mode, top_k):
    """语音检索"""
    if audio is None: 
        return [], "⚠️ 请上传音频"
    
    # Whisper 转写
    result = whisper_model.transcribe(audio, language="zh")
    query_text = result["text"].strip()
    
    if not query_text:
        return [], "⚠️ 语音识别失败"
    
    # 文本检索
    inputs = tokenizer(
        query_text,
        max_length=MAX_TEXT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        query_emb = model.encode_text(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()
    
    index = text_index if search_mode == "文本索引" else image_index
    scores, indices = index. search(query_emb, top_k)
    
    return format_results(indices[0], scores[0]), f"🎤 识别结果: {query_text}"

def format_results(indices, scores):
    """格式化检索结果为图片列表"""
    results = []
    for idx, score in zip(indices, scores):
        row = metadata.iloc[idx]
        image_path = row["image_path"]
        
        # 检查图片是否存在
        if os.path.exists(image_path):
            results.append((
                image_path,
                f"相似度: {score:.3f}\nID: {row['id']}\n{row['text'][: 80]}..."
            ))
        else:
            # 图片不存在时用占位符
            results.append((
                None,
                f"⚠️ 图片缺失\n相似度: {score:.3f}\nID: {row['id']}"
            ))
    
    return results

# ===================== Gradio 界面 =====================
with gr.Blocks(title="多模态检索系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 多模态商品检索系统
        支持文本、图像、语音三种输入方式的跨模态检索
        """
    )
    
    with gr. Tabs():
        # ========== Tab 1: 文本检索 ==========
        with gr. Tab("文本检索"):
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
                    text_btn = gr.Button("🔍 搜索", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    text_gallery = gr.Gallery(
                        label="检索结果",
                        columns=3,
                        height="auto",
                        object_fit="contain"
                    )
            
            text_btn.click(
                fn=search_by_text,
                inputs=[text_input, text_mode, text_topk],
                outputs=text_gallery
            )
        
        # ========== Tab 2: 图像检索 ==========
        with gr.Tab("图像检索"):
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
                    image_btn = gr.Button("🔍 搜索", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    image_gallery = gr.Gallery(
                        label="检索结果",
                        columns=3,
                        height="auto",
                        object_fit="contain"
                    )
            
            image_btn.click(
                fn=search_by_image,
                inputs=[image_input, image_mode, image_topk],
                outputs=image_gallery
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
                    audio_btn = gr.Button("🔍 搜索", variant="primary", size="lg")
                    audio_status = gr.Textbox(label="识别状态", interactive=False)
                
                with gr.Column(scale=2):
                    audio_gallery = gr.Gallery(
                        label="检索结果",
                        columns=3,
                        height="auto",
                        object_fit="contain"
                    )
            
            audio_btn. click(
                fn=search_by_audio,
                inputs=[audio_input, audio_mode, audio_topk],
                outputs=[audio_gallery, audio_status]
            )
    
    gr.Markdown(
        """
        ---
        ### 使用说明
        - **文本索引**: 在商品文本描述中检索（语义匹配）
        - **图像索引**: 在商品图片中检索（视觉相似）
        - **跨模态**: 文本查图片 / 图片查文本（多模态对齐）
        
        ### 模型信息
        - 图像编码器: CLIP ViT-L/14
        - 文本编码器: Chinese RoBERTa-Large
        - 语音识别: Whisper Base
        - 训练数据: 1000 条商品 (图片 + 文本 + 音频)
        """
    )

# ===================== 启动 =====================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=False,  # 改为 True 可生成公网链接
        show_error=True
    )