import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Seq2SeqRecDataset(Dataset):
    """
    为GENIUS-Rec模型准备训练和验证数据的数据集。
    核心逻辑：
    1. 动态分割：根据真实序列长度和配置中的比例，将用户历史分割为源(source)和目标(target)。
    2. 固定长度填充：将分割后的序列填充到配置中指定的固定长度，以便进行批处理。
    3. 注入特殊Token：为解码器的输入添加[SOS]，为标签添加[EOS]。
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
        self.max_seq_len = config['encoder_model']['max_len']
        self.pad_token_id = config['pad_token_id']
        self.sos_token_id = config['sos_token_id']
        self.eos_token_id = config['eos_token_id']
        
        # 【最终优化】读取真正用于序列分割的比例
        # 这个比例决定了用多少历史(e.g., 80%)去预测多少未来(e.g., 20%)
        self.sequence_split_ratio = config['finetune'].get('split_ratio', 0.8) # 提供默认值以增加健壮性
        
        # 【最终优化】计算编码器和解码器输入最终需要被填充到的固定长度
        # 这确保了DataLoader输出的每个Tensor形状都一致
        self.encoder_target_len = int(self.max_seq_len * self.sequence_split_ratio)
        self.decoder_target_len = self.max_seq_len - self.encoder_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 获取一个用户的完整历史序列
        full_seq = self.data.iloc[idx]['history']
        
        # 2. 截断过长的序列，只保留最近的行为
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[-self.max_seq_len:]
        
        # 3. 动态分割序列，生成源(source)和目标(target)
        if len(full_seq) <= 2:
            # 处理极短序列的边缘情况
            source_seq = [full_seq[0]] if len(full_seq) > 0 else []
            target_seq = full_seq[1:] if len(full_seq) > 1 else []
        else:
            # 使用配置驱动的比例，基于当前序列的真实长度进行分割
            split_idx = max(1, int(len(full_seq) * self.sequence_split_ratio))
            source_seq = full_seq[:split_idx]
            target_seq = full_seq[split_idx:]

        # 4. 创建编码器输入，并进行左对齐填充 (关键修正)
        source_ids = np.full(self.encoder_target_len, self.pad_token_id, dtype=np.int64)
        if len(source_seq) > 0:
            # 从左侧开始填充真实序列
            copy_len = min(len(source_seq), self.encoder_target_len)
            source_ids[:copy_len] = source_seq[:copy_len]

        # 5. 创建解码器输入 (decoder_input_ids)
        # 格式: [SOS, item1, item2, ..., PAD, PAD]
        decoder_input_ids = np.full(self.decoder_target_len, self.pad_token_id, dtype=np.int64)
        decoder_input_ids[0] = self.sos_token_id  # 序列以[SOS]开始
        if len(target_seq) > 0: # 检查非空
            copy_len = min(len(target_seq), self.decoder_target_len - 1)
            decoder_input_ids[1:1+copy_len] = target_seq[:copy_len]

        # 6. 创建解码器标签 (labels)
        # 格式: [item1, item2, item3, ..., EOS, PAD]
        labels = np.full(self.decoder_target_len, self.pad_token_id, dtype=np.int64)
        if len(target_seq) > 0: # 检查非空
            copy_len = min(len(target_seq), self.decoder_target_len - 1)
            labels[:copy_len] = target_seq[:copy_len]
            # 🔧 修复：确保EOS token位置安全，并且紧跟在实际内容后
            eos_position = copy_len
            if eos_position < self.decoder_target_len:
                labels[eos_position] = self.eos_token_id
        
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }