# src/dataset.py (最终修复版：随机分割的前缀预测后缀)

import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset

# src/dataset.py (修改版)

import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class Seq2SeqRecDataset(Dataset):
    """
    【Sampled Softmax 修改版】为Encoder-Decoder架构准备训练和验证数据。
    """
    def __init__(self, config, data_path, is_validation=False, item_maps=None):
        self.data = pd.read_parquet(data_path)
        self.max_len = config['encoder_model']['max_len']
        self.pad_token_id = config['pad_token_id']
        self.sos_token_id = config['sos_token_id']
        self.eos_token_id = config['eos_token_id']
        self.min_source_len = 5
        self.is_validation = is_validation

        # --- ↓↓↓ 新增/修改的代码 ↓↓↓ ---
        self.use_sampled_softmax = config['finetune'].get('use_sampled_softmax', False)
        # 只在训练模式下且开启了sampled softmax时，才初始化负采样器
        if self.use_sampled_softmax and not self.is_validation:
            self.num_neg_samples = config['finetune']['num_neg_samples']
            self.num_special_tokens = config['num_special_tokens']
            
            if item_maps is None:
                raise ValueError("item_maps must be provided for negative sampling.")
            
            num_items = item_maps['num_items']
            # 创建一个包含所有有效物品ID的列表，用于高效采样
            self.valid_item_ids = list(range(self.num_special_tokens, num_items + self.num_special_tokens))
        # --- ↑↑↑ 新增/修改的代码 ↑↑↑ ---

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data.iloc[idx]['history']

        # --- 序列分割逻辑 (保持不变) ---
        if self.is_validation:
            split_point = len(full_seq) - 1
        else:
            if len(full_seq) <= self.min_source_len + 1:
                split_point = len(full_seq) - 1
            else:
                split_point = random.randint(self.min_source_len, len(full_seq) - 1)

        source_seq = full_seq[:split_point]
        target_seq = full_seq[split_point:]

        # --- 截断与填充 (保持不变) ---
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
        
        # --- ↓↓↓ 新增/修改的代码: 生成负样本 ↓↓↓ ---
        negative_samples = np.zeros((self.max_len, 1), dtype=np.int64) # 默认占位符

        if self.use_sampled_softmax and not self.is_validation:
            negative_samples = np.zeros((self.max_len, self.num_neg_samples), dtype=np.int64)
            for i, target_item in enumerate(labels):
                if i >= self.max_len: break
                # 只为有效的目标物品（非padding, 非eos）进行采样
                if target_item != self.pad_token_id and target_item != self.eos_token_id:
                    # 从有效物品池中采样
                    sampled_negs = random.sample(self.valid_item_ids, self.num_neg_samples + 1)
                    # 如果不巧抽到了正样本，就移除它
                    if target_item in sampled_negs:
                        sampled_negs.remove(target_item)
                    # 截取所需数量的负样本
                    negative_samples[i] = sampled_negs[:self.num_neg_samples]
        # --- ↑↑↑ 新增/修改的代码: 生成负样本 ↑↑↑ ---
        
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'negative_samples': torch.tensor(negative_samples, dtype=torch.long) # 新增返回项
        }