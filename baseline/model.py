# baseline/model_corrected.py
# 创建一个更强的baseline以进行公平比较

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # 如果序列长度超过预设的max_len，需要扩展位置编码
            print(f"WARNING: sequence length {seq_len} > max_len {self.pe.size(1)}")
            pe_slice = self.pe[:, :seq_len, :]
        else:
            pe_slice = self.pe[:, :seq_len, :]
        
        # 检查位置编码是否正常
        if torch.isnan(pe_slice).any():
            print("ERROR: NaN in positional encoding!")
            
        x = x + pe_slice
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """自定义的多头自注意力，添加一些改进"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            
        # Apply key padding mask - 修复mask应用顺序
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L]
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(expanded_mask, float('-inf'))
        
        # 防止全为-inf导致NaN
        # 检查是否所有位置都被mask掉了
        max_scores = scores.max(dim=-1, keepdim=True)[0]
        if torch.isinf(max_scores).any():
            # 如果有全为-inf的行，给第一个位置一个很小但不是-inf的值
            inf_mask = torch.isinf(max_scores).expand_as(scores)
            scores = torch.where(inf_mask, torch.full_like(scores, -1e9), scores)
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # 检查注意力权重是否包含NaN
        if torch.isnan(attn_weights).any():
            print(f"WARNING: NaN in attention weights! scores range: [{scores.min():.4f}, {scores.max():.4f}]")
            attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
            
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        return output

class FeedForward(nn.Module):
    """改进的前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            self.activation = F.relu
            
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """改进的Transformer块"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation='silu')  # 使用SiLU激活
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        # Pre-LN: 先归一化再计算
        normed_x = self.norm1(x)
        attn_output = self.self_attn(normed_x, mask, key_padding_mask)
        x = x + self.dropout(attn_output)
        
        # Pre-LN for FFN
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x

class BaselineTransformer(nn.Module):
    """
    更强的Baseline模型 - 使用与HSTU相同的超参数和一些改进技巧
    目标是创建一个合理但不过于复杂的基线对比
    """
    def __init__(self, item_num, embedding_dim, max_len, num_layers, num_heads, dropout, pad_token_id):
        super().__init__()
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # 物品嵌入层 - 使用与HSTU相同的初始化
        # 修复：嵌入层大小应为item_num (已经包含了4个特殊标记)，不需要再+1
        self.item_embedding = nn.Embedding(item_num, embedding_dim, padding_idx=pad_token_id)
        
        # 位置编码 - 使用正弦位置编码而不是学习位置编码
        self.positional_encoder = SinusoidalPositionalEncoder(embedding_dim, dropout, max_len)
        
        # 使用自定义的Transformer块而不是PyTorch内置的
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=embedding_dim,
                num_heads=num_heads,
                d_ff=embedding_dim * 4,  # 标准的4倍扩展
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 最终归一化层
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """使用更稳定的初始化策略"""
        # 使用更小的初始化范围，防止梯度爆炸
        initrange = 0.01  # 减小初始化范围
        self.item_embedding.weight.data.normal_(mean=0.0, std=initrange)
        
        # 使用更保守的初始化方法
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更稳定的初始化
                nn.init.xavier_normal_(module.weight, gain=0.1)  # 减小gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

    def _generate_causal_mask(self, sz, device):
        """生成因果掩码"""
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

        # 创建padding mask和causal mask
        padding_mask = (input_ids == self.pad_token_id)
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # 物品嵌入，去掉缩放因子避免梯度爆炸
        item_emb = self.item_embedding(input_ids)  # 移除 * math.sqrt(self.embedding_dim)
        
        # 应用位置编码
        x = self.positional_encoder(item_emb)
        
        # 通过Transformer块
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=causal_mask, key_padding_mask=padding_mask)
        
        # 最终归一化
        sequence_output = self.final_norm(x)
        
        return sequence_output