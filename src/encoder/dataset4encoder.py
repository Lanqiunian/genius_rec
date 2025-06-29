# src/dataset.py (修正版)

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import pickle

class RecDataset(data.Dataset):
    def __init__(self, data_path, id_maps_path, max_seq_len, num_neg_samples=100, mode='train'):
        self.data = pd.read_parquet(data_path)
        self.max_seq_len = max_seq_len
        self.num_neg_samples = num_neg_samples
        self.mode = mode

        with open(id_maps_path, 'rb') as f:
            id_maps = pickle.load(f)
            
            # 【关键修正】直接使用'num_items'键来获取物品总数
            self.item_num = id_maps['num_items']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_seq = self.data.iloc[idx]['history']

        if self.mode == 'train':
            # 模式一：N-to-N 训练
            input_ids = np.zeros(self.max_seq_len, dtype=np.int32)
            target_ids = np.zeros(self.max_seq_len, dtype=np.int32)
            
            # 从序列末尾截取，保证数据最新
            end_idx = len(item_seq)
            start_idx = max(0, end_idx - self.max_seq_len)
            
            seq_slice = item_seq[start_idx:end_idx]
            
            # 填充到max_seq_len
            input_ids[-len(seq_slice)+1:] = seq_slice[:-1]
            target_ids[-len(seq_slice)+1:] = seq_slice[1:]
            
            return torch.LongTensor(input_ids), torch.LongTensor(target_ids)

        else:
            # 模式二：Leave-One-Out 验证/测试
            input_seq = item_seq[:-1]
            
            # 截断
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
            
            input_ids = np.zeros(self.max_seq_len, dtype=np.int32)
            input_ids[-len(input_seq):] = input_seq
            
            positive_item = item_seq[-1]
            
            negative_samples = []
            while len(negative_samples) < self.num_neg_samples:
                # 【修正】randint上界是开区间，所以要+1才能包含item_num
                neg_candidate = np.random.randint(1, self.item_num + 1)
                if neg_candidate != positive_item and neg_candidate not in item_seq:
                    negative_samples.append(neg_candidate)
            
            return (
                torch.LongTensor(input_ids),
                torch.LongTensor([positive_item]),
                torch.LongTensor(negative_samples)
            )