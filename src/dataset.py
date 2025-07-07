# src/dataset.py (最终修复版：随机分割的前缀预测后缀)

import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class Seq2SeqRecDataset(Dataset):
    """
    【最终修复版】为Encoder-Decoder架构准备训练和验证数据。
    采用“随机分割的前缀预测后缀”模式，优雅地解决数据泄露并最大化数据利用率。
    """
    def __init__(self, config, data_path, is_validation=False):
        self.data = pd.read_parquet(data_path)
        self.max_len = config['encoder_model']['max_len']
        self.pad_token_id = config['pad_token_id']
        self.sos_token_id = config['sos_token_id']
        self.eos_token_id = config['eos_token_id']
        self.min_source_len = 5 # 保证源序列至少有5个物品
        self.is_validation = is_validation # 区分训练和验证

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data.iloc[idx]['history']

        # --- 【核心逻辑】分割序列 ---
        if self.is_validation:
            # 在验证时，使用固定的Leave-One-Out，保证评估标准统一
            split_point = len(full_seq) - 1
        else:
            # 在训练时，随机选择分割点
            if len(full_seq) <= self.min_source_len + 1:
                split_point = len(full_seq) - 1
            else:
                split_point = random.randint(self.min_source_len, len(full_seq) - 1)

        source_seq = full_seq[:split_point]
        target_seq = full_seq[split_point:]

        # --- 截断与填充 (与之前逻辑相同) ---
        if len(source_seq) > self.max_len:
            source_seq = source_seq[-self.max_len:]
        
        source_ids = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
        source_ids[:len(source_seq)] = source_seq

        decoder_input_seq = [self.sos_token_id] + target_seq
        if len(decoder_input_seq) > self.max_len:
            decoder_input_seq = decoder_input_seq[:self.max_len]
        decoder_input_ids = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
        decoder_input_ids[:len(decoder_input_seq)] = decoder_input_seq

        labels_seq = target_seq + [self.eos_token_id]
        if len(labels_seq) > self.max_len:
            labels_seq = labels_seq[:self.max_len]
        labels = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
        labels[:len(labels_seq)] = labels_seq
        
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }