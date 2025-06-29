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
        # 【BUG FIX】Ensure item_seq is a Python list, not a NumPy array.
        item_seq = self.data.iloc[idx]['history'].tolist()

        if self.mode == 'train':
            # 1. Get the last max_len + 1 items.
            seq = item_seq[-(self.max_seq_len + 1):]
            
            # 2. Pad to the left if the sequence is short.
            padding_len = (self.max_seq_len + 1) - len(seq)
            if padding_len > 0:
                # Now this is a safe list concatenation.
                seq = [0] * padding_len + seq
            
            # 3. Create input and target.
            input_ids = seq[:-1]
            target_ids = seq[1:]
            
            return torch.LongTensor(input_ids), torch.LongTensor(target_ids)

        else: # Validation/Test mode
            input_seq = item_seq[:-1]
            
            # Truncate or pad to a fixed length of max_len.
            if len(input_seq) > self.max_seq_len:
                input_seq = input_seq[-self.max_seq_len:]
            else:
                padding_len = self.max_seq_len - len(input_seq)
                # Safe list concatenation.
                input_seq = [0] * padding_len + input_seq

            positive_item = item_seq[-1]
            
            negative_samples = []
            while len(negative_samples) < self.num_neg_samples:
                neg_candidate = np.random.randint(1, self.item_num + 1)
                if neg_candidate != positive_item and neg_candidate not in item_seq:
                    negative_samples.append(neg_candidate)
            
            return (
                torch.LongTensor(input_seq),
                torch.LongTensor([positive_item]),
                torch.LongTensor(negative_samples)
            )