# baseline/model_corrected.py

import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoder(nn.Module):
    """
    更标准的正弦/余弦位置编码器.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # -> [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        """
        # x的尺寸是 [B, L, D], pe的尺寸是 [1, max_len, D]
        # 直接截取所需长度并相加
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BaselineTransformer(nn.Module):
    """
    经过关键修正的Transformer baseline模型
    1. 使用Pre-LN (norm_first=True)
    2. 使用Sinusoidal (正弦) 位置编码
    3. 使用GELU激活函数
    """
    def __init__(self, item_num, embedding_dim, max_len, num_layers, num_heads, dropout, pad_token_id):
        super().__init__()
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # 物品嵌入层
        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=pad_token_id)
        
        # --- 修正 #2: 使用正弦位置编码 ---
        self.positional_encoder = SinusoidalPositionalEncoder(embedding_dim, dropout, max_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4, # 保持与常见实现一致
            dropout=dropout,
            activation='gelu',  # --- 修正 #3: 使用GELU, 效果通常优于ReLU ---
            batch_first=True,
            norm_first=True  # --- 修正 #1: 切换到Pre-LN, 关键修正！ ---
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim) # 最终输出前再加一个LN
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重，与HSTU实践对齐"""
        initrange = 0.02 # 与HSTU对齐
        self.item_embedding.weight.data.normal_(mean=0.0, std=initrange)

    def _generate_causal_mask(self, sz, device):
        """生成一个上三角矩阵的mask，用于阻止看到未来的token"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def forward(self, input_ids):
        """
        前向传播
        Args:
            input_ids: [B, L] 输入序列
        Returns:
            sequence_output: [B, L, D] 序列输出
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # 创建 padding mask (padding位置为True, 非padding为False)
        padding_mask = (input_ids == self.pad_token_id)

        # 创建 causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # 物品嵌入
        item_emb = self.item_embedding(input_ids) # [B, L, D]
        
        # 应用位置编码
        pos_emb = self.positional_encoder(item_emb)
        
        # 通过Transformer编码器
        sequence_output = self.transformer_encoder(
            pos_emb, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask
        )
        
        return sequence_output