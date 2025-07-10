import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

class RelativePositionalBias(nn.Module):
    def __init__(self, max_seq_len: int, num_heads: int) -> None:
        super().__init__()
        self._max_seq_len = max_seq_len
        self._w = nn.Parameter(
            torch.empty(2 * max_seq_len - 1, num_heads).normal_(mean=0, std=0.02)
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """返回 [num_heads, seq_len, seq_len] 的相对位置偏置"""
        n = min(seq_len, self._max_seq_len)
        
        # 简化的相对位置偏置实现，避免复杂的Toeplitz矩阵
        # 直接根据相对位置索引获取偏置
        device = self._w.device
        
        # 创建相对位置索引矩阵
        q_indices = torch.arange(n, device=device)[:, None]  # [n, 1]
        k_indices = torch.arange(n, device=device)[None, :]  # [1, n]
        relative_indices = k_indices - q_indices + (self._max_seq_len - 1)  # [n, n]
        
        # 确保索引在有效范围内
        relative_indices = torch.clamp(relative_indices, 0, 2 * self._max_seq_len - 2)
        
        # 根据相对位置索引获取偏置值
        bias = self._w[relative_indices]  # [n, n, num_heads]
        bias = bias.permute(2, 0, 1)  # [num_heads, n, n]
        
        # 如果序列长度超过max_seq_len，需要填充
        if seq_len > n:
            padding_row = torch.zeros(bias.size(0), seq_len - n, n, 
                                    device=device, dtype=bias.dtype)
            bias = torch.cat([bias, padding_row], dim=1)
            
            padding_col = torch.zeros(bias.size(0), seq_len, seq_len - n,
                                    device=device, dtype=bias.dtype)
            bias = torch.cat([bias, padding_col], dim=2)
        
        return bias

class HstuBlock(nn.Module):
    def __init__(self, embedding_dim: int, linear_hidden_dim: int, attention_dim: int, 
                 num_heads: int, dropout_ratio: float = 0.1, normalization: str = "rel_bias"):
        super(HstuBlock, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_hidden_dim  
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._dropout_ratio = dropout_ratio
        self._normalization = normalization
        
        # 官方的uvqk参数矩阵
        self._uvqk = nn.Parameter(
            torch.empty(
                embedding_dim,
                linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2,
            ).normal_(mean=0, std=0.02)
        )
        
        # 输出投影
        self._o = nn.Linear(linear_hidden_dim * num_heads, embedding_dim)
        nn.init.xavier_uniform_(self._o.weight)
        
        self._eps = 1e-6

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps)

    def forward(self, x, rel_pos_bias, invalid_attn_mask):
        """
        Args:
            x: [B, N, D]
            rel_pos_bias: [num_heads, N, N] 
            invalid_attn_mask: [B, N, N]
        """
        batch_size, seq_len, _ = x.size()
        residual = x
        
        # 输入归一化
        normed_x = self._norm_input(x.view(-1, self._embedding_dim))  # [B*N, D]
        
        # 线性变换 uvqk
        batched_mm_output = torch.mm(normed_x, self._uvqk)
        batched_mm_output = F.silu(batched_mm_output)  # 官方使用silu激活
        
        # 分割为 u, v, q, k
        u, v, q, k = torch.split(
            batched_mm_output,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads, 
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=1,
        )
        
        # reshape回batch形式
        u = u.view(batch_size, seq_len, self._linear_dim * self._num_heads)
        v = v.view(batch_size, seq_len, self._linear_dim * self._num_heads)
        q = q.view(batch_size, seq_len, self._num_heads, self._attention_dim)
        k = k.view(batch_size, seq_len, self._num_heads, self._attention_dim)
        v_reshaped = v.view(batch_size, seq_len, self._num_heads, self._linear_dim)
        
        # 注意力计算
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k)  # [B, H, N, N]
        
        # 添加相对位置偏置
        qk_attn = qk_attn + rel_pos_bias.unsqueeze(0)  # [B, H, N, N]
        
        # 官方的关键步骤：SiLU激活
        qk_attn = F.silu(qk_attn) 
        
        # 应用attention mask
        qk_attn = qk_attn * invalid_attn_mask.unsqueeze(1)
        
        # 计算attention输出
        attn_output = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v_reshaped)
        attn_output = attn_output.reshape(batch_size, seq_len, self._linear_dim * self._num_heads)
        
        # 归一化attention输出
        norm_attn_output = self._norm_attn_output(attn_output.view(-1, self._linear_dim * self._num_heads))
        norm_attn_output = norm_attn_output.view(batch_size, seq_len, -1)
        
        # 门控机制：u * normalized_attention_output
        gated_output = u * norm_attn_output
        
        # 输出投影，最后的线性层
        output = self._o(F.dropout(gated_output.view(-1, self._linear_dim * self._num_heads), 
                                  p=self._dropout_ratio, training=self.training))
        output = output.view(batch_size, seq_len, self._embedding_dim)
        
        # 残差连接
        return residual + output

class Hstu(nn.Module):
    """完全对齐官方的HSTU模型"""
    def __init__(self, item_num=None, embedding_dim=128, linear_hidden_dim=128, attention_dim=64,
                 num_heads=8, num_layers=4, max_len=50, dropout=0.1, pad_token_id=0, **kwargs):
        """
        初始化HSTU编码器
        
        Args:
            item_num: 物品总数（包括特殊标记）
            embedding_dim: 嵌入维度
            linear_hidden_dim: 线性层隐藏维度
            attention_dim: 注意力维度
            num_heads: 多头注意力头数
            num_layers: 编码器层数
            max_len: 最大序列长度
            dropout: dropout比例
            pad_token_id: padding标记ID
            **kwargs: 其他参数（向前兼容）
        """
        super(Hstu, self).__init__()
        
        if item_num is None:
            raise ValueError("item_num is required for HSTU encoder")
        
        # 修复：嵌入层大小应为item_num (已经包含了4个特殊标记)，保持与传入参数一致
        self.item_embedding = nn.Embedding(item_num, embedding_dim, padding_idx=pad_token_id)
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # 相对位置偏置
        self.relative_pos_bias = RelativePositionalBias(max_len, num_heads)
        
        # HSTU层
        self.encoder_layers = nn.ModuleList([
            HstuBlock(
                embedding_dim=embedding_dim,
                linear_hidden_dim=linear_hidden_dim,
                attention_dim=attention_dim,
                num_heads=num_heads,
                dropout_ratio=dropout
            ) for _ in range(num_layers)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, L] item ids
        Returns:
            [B, L, D] sequence representations
        """
        batch_size, seq_len = x.size()
        
        # 创建causal mask
        invalid_attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        invalid_attn_mask = invalid_attn_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
        
        # 物品嵌入
        item_emb = self.item_embedding(x)  # [B, L, D]
        input_emb = self.dropout(item_emb)
        
        # 相对位置偏置
        rel_pos_bias = self.relative_pos_bias(seq_len)  # [H, L, L]
        
        # 通过HSTU层
        encoded_seq = input_emb
        for layer in self.encoder_layers:
            encoded_seq = layer(encoded_seq, rel_pos_bias, invalid_attn_mask)
        
        # 最终归一化
        encoded_seq = self.final_norm(encoded_seq)
        return encoded_seq

    def predict(self, item_seq, item_candidates):
        """用于评估的预测函数"""
        log_feats = self.forward(item_seq)
        final_feat = log_feats[:, -1, :]  # [B, D]
        item_embs = self.item_embedding(item_candidates)  # [B, num_candidates, D]
        logits = torch.bmm(final_feat.unsqueeze(1), item_embs.transpose(1, 2)).squeeze(1)
        return logits