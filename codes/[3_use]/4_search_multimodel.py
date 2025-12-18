import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import whisper
import argparse

# ===================== 配置 =====================
CHECKPOINT_PATH = "../[2_training]/checkpoints/best_model.pt"
TEXT_INDEX_PATH = "./indexes/text_flat.index"
IMAGE_INDEX_PATH = "./indexes/image_flat.index"
METADATA_PATH = "./embeddings/metadata.csv"

IMAGE_MODEL = "openai/clip-vit-large-patch14"
TEXT_MODEL = "hfl/chinese-roberta-wwm-ext-large"
PROJ_DIM = 512
MAX_TEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 模型定义（复用训练时的结构）=====================
class DualEncoder(nn.Module):
    def __init__(self, image_model_name, text_model_name, proj_dim=512):
        super().__init__()
        # 图像编码器
        clip_model = CLIPModel.from_pretrained(image_model_name)
        self.image_encoder = clip_model.vision_model
        img_dim = self.image_encoder.config.hidden_size
        
        # 文本编码器
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

# ===================== 多模态检索器 =====================
class MultiModalSearcher: 
    def __init__(self):
        print("Initializing Multi-Modal Searcher...")
        
        # 加载模型
        print("Loading DualEncoder model...")
        self.model = DualEncoder(IMAGE_MODEL, TEXT_MODEL, PROJ_DIM).to(DEVICE)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 加载处理器
        print("Loading processors...")
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.image_processor = CLIPProcessor.from_pretrained(IMAGE_MODEL)
        
        # 加载 Whisper（语音识别）
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base", device=DEVICE)
        
        # 加载索引
        print("Loading indexes...")
        self.text_index = faiss.read_index(TEXT_INDEX_PATH)
        self.image_index = faiss.read_index(IMAGE_INDEX_PATH)
        
        # 加载元数据
        self.metadata = pd.read_csv(METADATA_PATH)
        
        print(f"✓ Ready!  Text index:  {self.text_index.ntotal}, Image index: {self.image_index.ntotal}\n")

    def search_by_text(self, query_text, top_k=10, search_mode="text"):
        """
        文本搜索
        Args:
            query_text: 查询文本
            top_k: 返回数量
            search_mode: "text" (文本索引) 或 "image" (图像索引，跨模态)
        """
        # 编码查询文本
        inputs = self.tokenizer(
            query_text,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            query_emb = self.model.encode_text(
                inputs["input_ids"],
                inputs["attention_mask"]
            ).cpu().numpy()
        
        # 选择索引
        index = self.text_index if search_mode == "text" else self.image_index
        
        # 检索
        scores, indices = index.search(query_emb, top_k)
        
        return self._format_results(indices[0], scores[0], search_mode)

    def search_by_image(self, image_path, top_k=10, search_mode="image"):
        """
        图像搜索
        Args:
            image_path: 图片路径
            top_k:  返回数量
            search_mode: "image" (图像索引) 或 "text" (文本索引，跨模态)
        """
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        image_input = self.image_processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].to(DEVICE)
        
        # 编码图像
        with torch.no_grad():
            query_emb = self.model.encode_image(image_input).cpu().numpy()
        
        # 选择索引
        index = self.image_index if search_mode == "image" else self.text_index
        
        # 检索
        scores, indices = index.search(query_emb, top_k)
        
        return self._format_results(indices[0], scores[0], search_mode)

    def search_by_audio(self, audio_path, top_k=10, search_mode="text"):
        """
        语音搜索（先 ASR 转文本，再文本检索）
        Args:
            audio_path: 音频路径
            top_k: 返回数量
            search_mode: "text" 或 "image"
        """
        # 语音识别
        print(f"Transcribing audio: {audio_path}")
        result = self.whisper_model.transcribe(audio_path, language="zh")
        query_text = result["text"].strip()
        print(f"ASR Result: {query_text}")
        
        # 用文本检索
        return self.search_by_text(query_text, top_k, search_mode)

    def _format_results(self, indices, scores, search_mode):
        """格式化结果"""
        results = []
        for i, (idx, score) in enumerate(zip(indices, scores)):
            row = self.metadata.iloc[idx]
            results.append({
                "rank": i + 1,
                "score": float(score),
                "search_mode": search_mode,
                "id": str(row["id"]),
                "text": str(row["text"]),
                "image_path": str(row["image_path"]),
                "audio_path": str(row["audio_path"])
            })
        return results

    def print_results(self, results, query_info=""):
        """打印结果"""
        print(f"\n{'='*80}")
        if query_info:
            print(f"Query: {query_info}")
        print(f"Search Mode: {results[0]['search_mode'].upper()} Index")
        print(f"Found {len(results)} results:")
        print(f"{'='*80}\n")
        
        for r in results:
            print(f"[{r['rank']}] Score: {r['score']:.3f}")
            print(f"  ID: {r['id']}")
            print(f"  Text:  {r['text'][: 100]}...")
            print(f"  Image: {r['image_path']}")
            print(f"  Audio: {r['audio_path']}")
            print()

# ===================== 交互式界面 =====================
def interactive_mode():
    searcher = MultiModalSearcher()
    
    print("\n" + "="*80)
    print("          多模态检索 - Multi-Modal Search System")
    print("="*80)
    print("\n支持的搜索模式:")
    print("  1.文本搜索文本:  text <关键词>")
    print("  2.文本搜索图像:  text2image <关键词>")
    print("  3.图像搜索图像:  image <图片路径>")
    print("  4.图像搜索文本:  image2text <图片路径>")
    print("  5.语音搜索文本:  audio <音频路径>")
    print("  6.语音搜索图像:  audio2image <音频路径>")
    print("\n输入 'q' 或 'quit' 退出\n")
    
    while True:
        try:
            user_input = input("Search> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Goodbye!")
                break
            
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("[Error] 格式错误!  示例: text 透明手机壳")
                continue
            
            mode, query = parts[0].lower(), parts[1]
            
            # 执行搜索
            if mode == "text":
                results = searcher.search_by_text(query, top_k=5, search_mode="text")
                searcher.print_results(results, f"Text Query: {query}")
            
            elif mode == "text2image":
                results = searcher.search_by_text(query, top_k=5, search_mode="image")
                searcher.print_results(results, f"Text Query: {query} (cross-modal)")
            
            elif mode == "image":
                if not os.path.exists(query):
                    print(f"[Error] 图片不存在: {query}")
                    continue
                results = searcher.search_by_image(query, top_k=5, search_mode="image")
                searcher.print_results(results, f"Image Query:  {query}")
            
            elif mode == "image2text": 
                if not os.path.exists(query):
                    print(f"[Error] 图片不存在: {query}")
                    continue
                results = searcher.search_by_image(query, top_k=5, search_mode="text")
                searcher.print_results(results, f"Image Query: {query} (cross-modal)")
            
            elif mode == "audio": 
                if not os.path.exists(query):
                    print(f"[Error] 音频不存在: {query}")
                    continue
                results = searcher.search_by_audio(query, top_k=5, search_mode="text")
                searcher.print_results(results, f"Audio Query: {query}")
            
            elif mode == "audio2image":
                if not os.path.exists(query):
                    print(f"[Error] 音频不存在: {query}")
                    continue
                results = searcher.search_by_audio(query, top_k=5, search_mode="image")
                searcher.print_results(results, f"Audio Query: {query} (cross-modal)")
            
            else:
                print(f"[Error] 未知模式: {mode}")
        
        except KeyboardInterrupt: 
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[Error] 错误: {e}")

# ===================== 命令行模式 =====================
def cli_mode(args):
    searcher = MultiModalSearcher()
    
    if args.mode == "text":
        results = searcher.search_by_text(args.query, args.top_k, "text")
    elif args.mode == "text2image":
        results = searcher.search_by_text(args.query, args.top_k, "image")
    elif args.mode == "image":
        results = searcher.search_by_image(args.query, args.top_k, "image")
    elif args.mode == "image2text":
        results = searcher.search_by_image(args.query, args.top_k, "text")
    elif args.mode == "audio": 
        results = searcher.search_by_audio(args.query, args.top_k, "text")
    elif args.mode == "audio2image":
        results = searcher.search_by_audio(args.query, args.top_k, "image")
    else:
        print(f"[Error] 未知模式: {args.mode}")
        return
    
    searcher.print_results(results, args.query)

# ===================== 主函数 =====================
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Multi-Modal Search System")
    parser.add_argument("--mode", type=str, choices=[
        "text", "text2image", "image", "image2text", "audio", "audio2image"
    ], help="搜索模式")
    parser.add_argument("--query", type=str, help="查询内容（文本/图片路径/音频路径）")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    
    args = parser.parse_args()
    
    if args.mode and args.query:
        # 命令行模式
        cli_mode(args)
    else:
        # 交互式模式
        interactive_mode()