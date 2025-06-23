# src/model.py

import torch
import torch.nn as nn
import math

# 辅助模块: 位置编码器 (这部分无需修改)
class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2) 
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2)

# --- 核心修正点：简化HSTU_Encoder，移除sep_token_id ---
class HSTU_Encoder(nn.Module):
    def __init__(self, num_items: int, config: dict):
        super().__init__()
        self.config = config
        self.embedding_dim = config['embedding_dim']
        
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim, padding_idx=0)
        self.positional_encoder = PositionalEncoder(self.embedding_dim, config['dropout'], config['max_seq_len'])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_encoder_layers']
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.item_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        key_padding_mask = (seq == 0)
        item_embed = self.item_embedding(seq) * math.sqrt(self.embedding_dim)
        pos_embed = self.positional_encoder(item_embed)
        output = self.transformer_encoder(pos_embed, src_key_padding_mask=key_padding_mask)
        return output

# --- 核心修正点：简化Standalone_HSTU_Model的初始化 ---
class Standalone_HSTU_Model(nn.Module):
    def __init__(self, num_items: int, config: dict):
        super().__init__()
        # 现在这里的调用是匹配的
        self.hstu_encoder = HSTU_Encoder(num_items, config)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        sequence_output = self.hstu_encoder(seq)
        
        # 为了预测，我们只取序列中最后一个有效（非padding）物品的表征
        # 一个简单但有效的近似是直接取最后一个时间步的输出
        # 注意: DataLoader送出的batch中，每个序列的有效长度可能不同
        # 在更精细的实现中，需要根据每个序列的真实长度来提取对应的表征
        last_item_representation = sequence_output[:, -1, :]
        
        item_embeddings = self.hstu_encoder.item_embedding.weight
        scores = torch.matmul(last_item_representation, item_embeddings.T)
        
        return scores