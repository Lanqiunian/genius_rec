# src/baseline.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    标准的正弦位置编码，为Transformer提供位置信息。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BaselineTransformer(nn.Module):
    """
    一个标准的基于 nn.TransformerEncoder 的序列推荐模型。
    接口与 Hstu 模型完全对齐，以用于公平比较。
    """
    def __init__(self, item_num, embedding_dim=128, linear_hidden_dim=128, attention_dim=64,
                 num_heads=8, num_layers=4, max_len=50, dropout=0.1, pad_token_id=0):
        super(BaselineTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len)

        # Transformer Encoder 层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4, # 经典设置为4倍
            dropout=dropout,
            activation='gelu', # 使用GELU激活
            batch_first=True # 输入格式为 [B, L, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        Args:
            x: [B, L] item ids
        Returns:
            [B, L, D] sequence representations
        """
        # 创建 padding mask
        # attention mask 中, True/1 表示该位置被忽略, False/0 表示不被忽略
        padding_mask = (x == self.pad_token_id) # [B, L]

        # 创建 causal mask
        causal_mask = self._generate_square_subsequent_mask(x.size(1), x.device) # [L, L]

        # 物品嵌入
        item_emb = self.item_embedding(x) * math.sqrt(self.embedding_dim) # [B, L, D]

        # 添加位置编码
        # TransformerEncoderLayer 的输入需要是 [L, B, D] 如果 batch_first=False
        # 但我们用了 batch_first=True, 所以输入是 [B, L, D]
        pos_encoded_emb = self.pos_encoder(item_emb.transpose(0, 1)).transpose(0, 1)

        # 通过Transformer Encoder层
        encoded_seq = self.transformer_encoder(
            pos_encoded_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # 最终归一化
        encoded_seq = self.final_norm(encoded_seq)
        return encoded_seq

    def predict(self, item_seq, item_candidates):
        """用于评估的预测函数，与Hstu保持一致"""
        log_feats = self.forward(item_seq)
        final_feat = log_feats[:, -1, :]  # [B, D]
        item_embs = self.item_embedding(item_candidates)  # [B, num_candidates, D]
        logits = torch.bmm(final_feat.unsqueeze(1), item_embs.transpose(1, 2)).squeeze(1)
        return logits