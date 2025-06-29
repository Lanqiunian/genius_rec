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
            self.item_num = id_maps['num_items']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 原始的完整序列
        item_seq = self.data.iloc[idx]['history']

        if self.mode == 'train':
            # --- 训练模式逻辑修正 ---
            # 为了进行next-item预测, 我们需要至少2个物品
            if len(item_seq) < 2:
                # 返回空的/全零的张量，可以在训练循环中过滤掉
                return torch.zeros(self.max_seq_len, dtype=torch.long), torch.zeros(self.max_seq_len, dtype=torch.long)

            # 截取最后的 max_len+1 个物品用于构建输入和目标
            seq = item_seq[-(self.max_seq_len + 1):]
            
            # input是序列到倒数第二个，target是序列从第二个到最后一个
            input_ids = seq[:-1]
            target_ids = seq[1:]
            
            # 获取实际长度
            seq_len = len(input_ids)
            
            # 创建全零的模板
            padded_input = np.zeros(self.max_seq_len, dtype=np.int32)
            padded_target = np.zeros(self.max_seq_len, dtype=np.int32) # padding target也是0
            
            # 从右侧填充
            padded_input[-seq_len:] = input_ids
            padded_target[-seq_len:] = target_ids
            
            return torch.LongTensor(padded_input), torch.LongTensor(padded_target)

        else: # 验证/测试模式逻辑保持不变，因为它是leave-one-out，没有这个问题
            input_seq = item_seq[:-1]
            
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
            
            padded_input = np.zeros(self.max_seq_len, dtype=np.int32)
            padded_input[-len(input_seq):] = input_seq
            
            positive_item = item_seq[-1]
            
            negative_samples = []
            while len(negative_samples) < self.num_neg_samples:
                neg_candidate = np.random.randint(1, self.item_num + 1)
                if neg_candidate != positive_item and neg_candidate not in item_seq:
                    negative_samples.append(neg_candidate)
            
            return (
                torch.LongTensor(padded_input),
                torch.LongTensor([positive_item]),
                torch.LongTensor(negative_samples)
            )