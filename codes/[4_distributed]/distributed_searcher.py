"""
分布式检索器 - 并行查询多个分片
"""
import os
import faiss
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time

class DistributedSearcher: 
    def __init__(self, shard_dir, num_shards=4):
        self.shard_dir = shard_dir
        self.num_shards = num_shards
        
        # 验证分片文件是否存在
        for i in range(num_shards):
            text_path = os.path.join(shard_dir, f"text_shard_{i}.index")
            image_path = os.path.join(shard_dir, f"image_shard_{i}.index")
            meta_path = os.path.join(shard_dir, f"metadata_shard_{i}.csv")
            
            if not os.path.exists(text_path):
                raise FileNotFoundError(f"分片文件不存在: {text_path}")
            if not os.path. exists(image_path):
                raise FileNotFoundError(f"分片文件不存在:  {image_path}")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"分片文件不存在: {meta_path}")
        
        print(f"✓ 分布式检索器初始化完成 ({num_shards} 个节点)")
    
    def search(self, query_embedding, search_mode="text", top_k=10):
        """
        分布式检索
        Args:
            query_embedding: 查询向量 [1, dim]
            search_mode: "text" 或 "image"
            top_k: 返回结果数量
        Returns:
            results: 检索结果列表
            stats: 统计信息
        """
        start_time = time.time()
        
        # 准备任务参数
        tasks = [
            (i, query_embedding, search_mode, top_k, self.shard_dir)
            for i in range(self.num_shards)
        ]
        
        # 并行查询各分片
        with Pool(self.num_shards) as pool:
            shard_results = pool.map(self._search_single_shard, tasks)
        
        # 合并结果
        all_results = []
        for results in shard_results:
            all_results.extend(results)
        
        # 按分数排序，取全局 Top-K
        all_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = all_results[:top_k]
        
        elapsed = time.time() - start_time
        
        stats = {
            'num_shards': self.num_shards,
            'total_candidates': len(all_results),
            'final_results': len(final_results),
            'elapsed_time': elapsed
        }
        
        return final_results, stats
    
    @staticmethod
    def _search_single_shard(args):
        """单个分片检索（静态方法，供多进程调用）"""
        shard_id, query_embedding, search_mode, top_k, shard_dir = args
        
        # 加载分片索引
        if search_mode == "text":
            index_path = os.path.join(shard_dir, f"text_shard_{shard_id}.index")
        else: 
            index_path = os.path.join(shard_dir, f"image_shard_{shard_id}.index")
        
        index = faiss.read_index(index_path)
        
        # 加载分片元数据
        meta_path = os.path.join(shard_dir, f"metadata_shard_{shard_id}.csv")
        metadata = pd.read_csv(meta_path)
        
        # 检索
        scores, indices = index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            row = metadata.iloc[idx]
            results.append({
                'shard_id': shard_id,
                'global_idx': int(row['global_idx']),
                'local_idx': int(row['local_idx']),
                'score': float(score),
                'id': str(row['id']),
                'text': str(row['text']),
                'image_path': str(row['image_path']),
                'audio_path': str(row['audio_path'])
            })
        
        return results