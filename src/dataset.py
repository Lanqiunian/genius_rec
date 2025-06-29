import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
# --- 1. 数据集准备 (适配Seq2Seq任务) ---
# 这个Dataset类用于将序列数据分割为编码器输入和解码器目标
class Seq2SeqRecDataset(Dataset):
    def __init__(self, data_path, max_seq_len, pad_token_id=0, split_ratio=0.5):
        self.data = pd.read_parquet(data_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        # 分割点，例如0.5表示一半历史用于编码，一半用于解码
        self.split_point = int(max_seq_len * split_ratio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取完整的历史序列
        full_seq = self.data.iloc[idx]['history']
        
        # 截断到最大长度
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[-self.max_seq_len:]
        
        # 分割为源序列(编码器输入)和目标序列(解码器输入/目标)
        source_seq = full_seq[:self.split_point]
        target_seq = full_seq[self.split_point:]

        # 如果目标序列为空，则跳过此样本 (或者使用整个序列作为源)
        if len(target_seq) == 0:
            source_seq = full_seq[:-1]
            target_seq = full_seq[1:]

        # 创建带padding的输入张量
        source_ids = np.full(self.split_point, self.pad_token_id, dtype=np.int64)
        source_ids[-len(source_seq):] = source_seq

        # 解码器的输入是目标序列的前n-1项, 并在开头加上<SOS> (这里用pad_token_id=0模拟)
        decoder_input_len = self.max_seq_len - self.split_point
        decoder_input_ids = np.full(decoder_input_len, self.pad_token_id, dtype=np.int64)
        # 解码器的输入通常以一个特殊的“开始”符开头，我们这里用pad_id=0
        # 并且只到target_seq的倒数第二个元素
        if len(target_seq) > 1:
            copy_len = min(len(target_seq) - 1, decoder_input_len - 1)
            decoder_input_ids[1:1+copy_len] = target_seq[:copy_len]


        # 解码器的目标是完整的序列
        labels = np.full(decoder_input_len, self.pad_token_id, dtype=np.int64)
        copy_len = min(len(target_seq), decoder_input_len)
        labels[:copy_len] = target_seq[:copy_len]
        
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }