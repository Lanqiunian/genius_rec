import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import logging

class HSTUDataset(Dataset):
    """
    为HSTU模型准备训练、验证和测试数据的Dataset类。(修正版)
    
    核心思想:
    1. 直接使用由preprocess.py生成的、每个用户的单一完整交互序列。
    2. 使用滑动窗口在长序列上生成多个 (输入序列, 目标物品) 的训练样本。
    3. 在 __getitem__ 中对每个样本进行截断和左侧填充，使其长度统一。
    """
    def __init__(self, data: list, config: dict):
        """
        Args:
            data (list): 预处理好的数据，格式为: 
                         [{'user_id': int, 'sequence': [item_id, ...]}, ...]
            config (dict): 配置字典，需要包含 'max_seq_len'。
        """
        super().__init__()
        self.max_seq_len = config['max_seq_len']
        
        # self.samples 存储所有可以用来训练的 (输入序列, 目标) 对
        self.samples = []

        logging.info("正在准备训练样本...")
        # --- 修正点：直接使用'sequence'键 ---
        progress_bar = tqdm(data, desc="准备样本")
        for user_data in progress_bar:
            # 直接获取用户的完整交互序列
            long_sequence = user_data['sequence']

            # 使用滑动窗口生成样本
            # 我们至少需要2个物品才能构成一个样本 (1个输入, 1个目标)
            if len(long_sequence) < 2:
                continue
            
            # 例如，对于序列 [A, B, C]，我们会生成：
            # sample 1: input=[A], target=B
            # sample 2: input=[A, B], target=C
            for i in range(1, len(long_sequence)):
                input_seq = long_sequence[:i]
                target_item = long_sequence[i]
                
                self.samples.append({
                    "input": input_seq,
                    "target": target_item
                })
        
        logging.info(f"样本准备完成，共计 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        input_seq = sample['input']
        target_item = sample['target']

        # ---  padding 和 truncation ---
        padded_input = np.zeros(self.max_seq_len, dtype=np.int64)
        
        # 截取最后的 max_seq_len 个元素
        truncated_input = input_seq[-self.max_seq_len:]
        
        # 将截取后的序列填充到padded_input的右侧（左侧填充）
        start_index = self.max_seq_len - len(truncated_input)
        padded_input[start_index:] = truncated_input
        
        return {
            "input_seq": torch.tensor(padded_input, dtype=torch.long),
            "target": torch.tensor(target_item, dtype=torch.long)
        }