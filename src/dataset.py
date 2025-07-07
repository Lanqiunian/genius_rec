# ==============================================================================
# 替换 src/dataset.py 中的 Seq2SeqRecDataset 类
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import random 
from torch.utils.data import Dataset

class Seq2SeqRecDataset(Dataset):
    """
    【最终重构版 - 全程Next-Token Prediction模式】
    为GENIUS-Rec模型准备训练和验证数据的数据集。

    核心逻辑：
    1. 不再分割序列，将完整的用户历史作为一个整体进行学习。
    2. 对一个完整的用户序列 S = [i1, i2, ..., iN]，生成：
       - decoder_input_ids: [<SOS>, i1, i2, ..., i_{N-1}]  (作为模型的输入)
       - labels:            [i1, i2, ..., iN, <EOS>]      (作为预测的目标)
    3. 序列会被截断并用PAD填充到最大长度，以支持批处理。
    """
    def __init__(self, config, data_path):
        """
        初始化数据集。

        Args:
            config (dict): 全局配置字典。
            data_path (str or Path): 数据文件路径 (.parquet格式)。
        """
        self.data = pd.read_parquet(data_path)
        
        # --- 从配置中读取所有必要的参数 ---
        # 在此模式下，编码器和解码器处理的序列长度是相同的
        self.max_len = config['encoder_model']['max_len']
        self.pad_token_id = config['pad_token_id']
        self.sos_token_id = config['sos_token_id']
        self.eos_token_id = config['eos_token_id']

        finetune_config = config.get('finetune', {}) # 安全地获取finetune配置
        self.use_stochastic_length = finetune_config.get('use_stochastic_length', True)
        self.stochastic_threshold = finetune_config.get('stochastic_threshold', 20)
        self.stochastic_prob = finetune_config.get('stochastic_prob', 0.5)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 获取一个用户的完整历史序列
        full_seq = self.data.iloc[idx]['history']

        # --- 【新增】随机序列长度逻辑 ---
        if self.use_stochastic_length and len(full_seq) > self.stochastic_threshold:
            if random.random() < self.stochastic_prob:
                # 以一定概率截取最近的一部分
                # 截取的长度可以在阈值和最大长度之间随机，增加多样性
                sample_len = random.randint(self.stochastic_threshold, self.max_len - 1)
                full_seq = full_seq[-sample_len:]
        # --- 新增逻辑结束 ---
        
        # 2. 截断过长的序列 (这段逻辑依然保留，作为最后的保险)
        if len(full_seq) >= self.max_len:
            full_seq = full_seq[-(self.max_len - 1):]

        # 3. 创建解码器输入 (decoder_input_ids)，以 <SOS> 开头
        input_seq = [self.sos_token_id] + full_seq
        decoder_input_ids = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
        decoder_input_ids[:len(input_seq)] = input_seq

        # 4. 创建标签 (labels)，以 <EOS> 结尾
        label_seq = full_seq + [self.eos_token_id]
        labels = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
        labels[:len(label_seq)] = label_seq
        
        return {
            'source_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }