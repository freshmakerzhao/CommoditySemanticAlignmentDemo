"""
数据分片脚本 - 将embeddings和索引按区间切分成4片
"""
import os
import sys
import numpy as np
import pandas as pd
import faiss

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== 配置 =====================
EMBEDDING_DIR = "../[3_use]/embeddings"
OUTPUT_DIR = "./shards"
NUM_SHARDS = 4  # 分成4片，模拟4个节点

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 加载数据 =====================
print("="*60)
print("数据分片工具 - 分布式检索准备")
print("="*60)
print(f"\n正在加载数据...")

text_embs = np.load(os.path.join(EMBEDDING_DIR, "text_embeddings.npy")).astype("float32")
image_embs = np.load(os.path.join(EMBEDDING_DIR, "image_embeddings.npy")).astype("float32")
metadata = pd.read_csv(os.path.join(EMBEDDING_DIR, "metadata.csv"))

n = len(text_embs)
dim = text_embs.shape[1]
shard_size = n // NUM_SHARDS

print(f"✓ 数据加载完成")
print(f"  - 总样本数: {n}")
print(f"  - 向量维度: {dim}")
print(f"  - 分片数量: {NUM_SHARDS}")
print(f"  - 每片大小: ~{shard_size}")

# ===================== 分片 =====================
print(f"\n开始分片...")

for i in range(NUM_SHARDS):
    start = i * shard_size
    end = (i + 1) * shard_size if i < NUM_SHARDS - 1 else n
    
    print(f"\n[Shard {i}] 样本范围: {start}-{end-1} ({end-start} 条)")
    
    # 提取分片数据
    text_shard = text_embs[start:end]
    image_shard = image_embs[start:end]
    meta_shard = metadata.iloc[start:end]. copy()
    
    # 添加分片信息
    meta_shard['shard_id'] = i
    meta_shard['local_idx'] = range(len(meta_shard))
    meta_shard['global_idx'] = range(start, end)
    
    # 构建文本索引
    text_index = faiss.IndexFlatIP(dim)
    text_index.add(text_shard)
    text_index_path = os.path.join(OUTPUT_DIR, f"text_shard_{i}.index")
    faiss.write_index(text_index, text_index_path)
    print(f"  ✓ 文本索引:  {text_index_path}")
    
    # 构建图像索引
    image_index = faiss.IndexFlatIP(dim)
    image_index.add(image_shard)
    image_index_path = os.path.join(OUTPUT_DIR, f"image_shard_{i}.index")
    faiss.write_index(image_index, image_index_path)
    print(f"  ✓ 图像索引: {image_index_path}")
    
    # 保存元数据
    meta_path = os.path.join(OUTPUT_DIR, f"metadata_shard_{i}.csv")
    meta_shard.to_csv(meta_path, index=False)
    print(f"  ✓ 元数据: {meta_path}")

print(f"\n{'='*60}")
print(f"✓ 分片完成！")
print(f"✓ 输出目录: {os.path.abspath(OUTPUT_DIR)}")
print(f"✓ 生成文件: {NUM_SHARDS * 3} 个 (每片3个文件)")
print(f"{'='*60}\n")