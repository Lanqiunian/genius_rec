import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# --- 1. 数据集准备 (适配Seq2Seq任务) ---
# 🔧 修复版本：避免数据泄露，正确构造序列分割
class Seq2SeqRecDataset(Dataset):
    def __init__(self, data_path, max_seq_len, pad_token_id=0, split_ratio=0.5, sos_token_id=1):
        self.data = pd.read_parquet(data_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id  # 明确的开始标记
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
        
        # 🔧 修复：使用时间分割避免数据泄露
        # 对于推荐任务，应该是历史预测未来，而不是任意分割
        if len(full_seq) <= 2:
            # 序列太短，跳过或使用最小配置
            source_seq = [full_seq[0]] if len(full_seq) > 0 else [self.pad_token_id]
            target_seq = full_seq[1:] if len(full_seq) > 1 else [self.pad_token_id]
        else:
            # 使用前80%作为编码器输入，后20%作为解码器目标
            split_idx = max(1, int(len(full_seq) * 0.8))
            source_seq = full_seq[:split_idx]
            target_seq = full_seq[split_idx:]

        # 创建带padding的编码器输入
        source_ids = np.full(self.split_point, self.pad_token_id, dtype=np.int64)
        if len(source_seq) > 0:
            copy_len = min(len(source_seq), self.split_point)
            source_ids[-copy_len:] = source_seq[-copy_len:]  # 右对齐

        # 🔧 修复：正确构造解码器输入和标签
        decoder_input_len = self.max_seq_len - self.split_point
        
        # 解码器输入：[SOS, target_seq[:-1]]
        decoder_input_ids = np.full(decoder_input_len, self.pad_token_id, dtype=np.int64)
        decoder_input_ids[0] = self.sos_token_id  # 开始标记
        if len(target_seq) > 0:
            copy_len = min(len(target_seq), decoder_input_len - 1)
            decoder_input_ids[1:1+copy_len] = target_seq[:copy_len]

        # 解码器标签：[target_seq, EOS/PAD]
        labels = np.full(decoder_input_len, self.pad_token_id, dtype=np.int64)
        if len(target_seq) > 0:
            copy_len = min(len(target_seq), decoder_input_len)
            labels[:copy_len] = target_seq[:copy_len]
        
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }