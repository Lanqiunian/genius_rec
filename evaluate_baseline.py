import torch
import argparse
import pickle
import numpy as np
import pandas as pd
import math
import logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F # 必须导入 F

from src.config import get_config
from src.encoder.encoder import Hstu

# --- Part 1: 从训练脚本中完整复制、经过验证的评估函数 ---
def get_metrics_from_training_script(user_embeddings, all_item_embeddings, target_item_ids, k=10):
    """
    这是从 train_encoder.py 中完整复制的核心评估逻辑。
    它包含了正确的归一化和索引处理。
    """
    batch_size = user_embeddings.size(0)
    
    # 关键步骤 1: L2 归一化
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算余弦相似度
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 排序
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    hr_list, ndcg_list = [], []
    
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0: continue

        # 关键步骤 2: 正确的ID到索引转换
        # 因为输入的 all_item_embeddings 是从 item_id=1 开始的，所以它的索引和 (id-1) 完美对应。
        target_idx = target_id - 1
        
        target_rank_positions = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0]
        
        hr, ndcg = 0.0, 0.0
        if len(target_rank_positions) > 0:
            rank = target_rank_positions[0].item() + 1
            if rank <= k:
                hr = 1.0
                ndcg = 1.0 / np.log2(rank + 1)
        
        hr_list.append(hr)
        ndcg_list.append(ndcg)
    
    return hr_list, ndcg_list

# --- Part 2: 保持与训练时一致的数据集 ---
class LeaveOneOutDataset(Dataset):
    def __init__(self, data_path, max_len, pad_token_id=0):
        self.data = pd.read_parquet(data_path)
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data.iloc[idx]['history']
        ground_truth_item = full_seq[-1]
        input_seq = full_seq[:-1]
        padded_input_seq = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
        if len(input_seq) > self.max_len:
            input_seq = input_seq[-self.max_len:]
        padded_input_seq[-len(input_seq):] = input_seq
        return {
            'input_ids': torch.tensor(padded_input_seq, dtype=torch.long),
            'ground_truth': torch.tensor(ground_truth_item, dtype=torch.long)
        }

# --- Part 3: 最终的、决定性的 main 函数 ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate the PRE-TRAINED HSTU Encoder Baseline.")
    parser.add_argument('--encoder_weights_path', type=str, required=True, help='Path to the PRE-TRAINED HSTU encoder weights file.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for evaluation.')
    args = parser.parse_args()

    print("--- 1. Loading Config ---")
    config = get_config()
    device = torch.device(config['device'])
    pad_token_id = config['pad_token_id']
    top_k = config['evaluation']['top_k']
    max_len = config['encoder_model']['max_len']

    print("--- 2. Loading Test Dataset ---")
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
        item_num = id_maps['num_items']
    test_dataset = LeaveOneOutDataset(config['data']['test_file'], max_len, pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=config['finetune']['num_workers'])
    print(f"Test set loaded. Size: {len(test_dataset)}")

    print(f"--- 3. Loading PRE-TRAINED ENCODER ---")
    config['encoder_model']['item_num'] = item_num
    encoder = Hstu(**config['encoder_model']).to(device)

    try:
        checkpoint = torch.load(args.encoder_weights_path, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Pre-trained HSTU encoder weights loaded successfully!")
    except Exception as e:
        print(f"❌ FATAL: Error loading weights: {e}")
        return

    print("\n--- 4. Starting Final Evaluation (Replicating Training Logic) ---")
    encoder.eval()
    all_hr_scores, all_ndcg_scores = [], []
    
    # --- 【最终修正】: 100% 复刻训练时的候选物品集构建方式 ---
    with torch.no_grad():
        # 1. 创建只包含有效物品ID的列表
        all_item_ids = torch.arange(1, item_num + 1, device=device)
        # 2. 用这个列表，向模型请求一个【不含padding】的、干净的候选物品嵌入矩阵
        clean_item_embeddings = encoder.item_embedding(all_item_ids)

        for batch in tqdm(test_loader, desc="Evaluating with 100% Aligned Logic"):
            input_ids = batch['input_ids'].to(device)
            ground_truth_ids = batch['ground_truth'].to(device)

            encoder_output = encoder(input_ids)
            user_embeddings = encoder_output[:, -1, :]
            
            # 3. 将用户向量和这个【干净的】候选矩阵送入评估函数
            hr_list, ndcg_list = get_metrics_from_training_script(
                user_embeddings,
                clean_item_embeddings, # 使用干净的矩阵
                ground_truth_ids,
                k=top_k
            )
            all_hr_scores.extend(hr_list)
            all_ndcg_scores.extend(ndcg_list)
            
    avg_hr = np.mean(all_hr_scores)
    avg_ndcg = np.mean(all_ndcg_scores)

    print("\n" + "="*60)
    print(" " * 8 + "HSTU Encoder - Final Baseline Performance")
    print("="*60)
    print(f" ✅ -> HR@{top_k}: {avg_hr:.4f}")
    print(f" ✅ -> NDCG@{top_k}: {avg_ndcg:.4f}")
    print("="*60)
    print("\nℹ️  Methodology: Evaluation logic is now 100% aligned with the training script.")

if __name__ == '__main__':
    main()