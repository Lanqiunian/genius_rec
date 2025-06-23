# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSTU_EncoderLayer(nn.Module):
    """
    单个HSTU编码器层。
    此版本经过优化，解决了数值稳定性问题，并为最高速度而设计。
    """
    def __init__(self, embedding_dim, nhead, dim_feedforward, dropout=0.1, activation=F.silu):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.d_head = embedding_dim // nhead
        if self.d_head * nhead != self.embedding_dim:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) 必须是 nhead ({self.nhead}) 的整数倍")

        self.in_proj = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.activation = activation

    def forward(self, src, src_mask=None):
        S, B, E = src.shape
        x = src
        x_norm = self.norm1(x)
        
        proj_output = self.in_proj(x_norm)
        proj_output_activated = self.activation(proj_output)
        qkv, u = torch.chunk(proj_output_activated, 2, dim=-1)
        
        q = qkv.view(S, B, self.nhead, self.d_head).permute(1, 2, 0, 3)
        k = qkv.view(S, B, self.nhead, self.d_head).permute(1, 2, 0, 3)
        v = qkv.view(S, B, self.nhead, self.d_head).permute(1, 2, 0, 3)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if src_mask is not None:
            # 关键修复: 使用很大的负数而非-inf，防止SiLU激活函数计算出NaN
            attn_scores = attn_scores.masked_fill(src_mask == 0, -1e9)

        attn_weights = self.activation(attn_scores)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(S, B, E)
        
        gated_output = attn_output * u
        gated_output_proj = self.out_proj(gated_output)
        
        x = x + self.out_dropout(gated_output_proj)
        
        return x

class HSTU_Encoder(nn.Module):
    """
    完整的HSTU编码器，此版本不使用梯度检查点，以实现最高训练速度。
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src, mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)
        return output

class Standalone_HSTU_Model(nn.Module):
    """
    最终版：与Baseline对齐，优先保证训练速度，并修复了所有已知问题。
    """
    def __init__(self, num_items, config):
        super().__init__()
        self.config = config
        
        embedding_dim = config['embedding_dim']
        nhead = config['nhead']
        num_encoder_layers = config['num_encoder_layers']
        dim_feedforward = config['dim_feedforward']
        dropout = config['dropout']
        max_seq_len = config['max_seq_len']
        self.embedding_dim = embedding_dim 
        
        # 关键修复：直接使用传递进来的num_items作为词汇表大小
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=config['pad_token_id'])
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, 1, embedding_dim))
        
        hstu_layer = HSTU_EncoderLayer(
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.hstu_encoder = HSTU_Encoder(
            encoder_layer=hstu_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.item_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seq):
        embedded = self.item_embedding(input_seq)
        embedded = embedded * math.sqrt(self.embedding_dim)
        
        embedded = embedded.permute(1, 0, 2)
        seq_len = embedded.size(0)
        src = embedded + self.positional_encoding[:seq_len, :]
        
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.config['device'])
        
        sequence_output = self.hstu_encoder(src, mask=attn_mask)
        final_representation = sequence_output[-1, :, :]
        
        scores = torch.matmul(final_representation, self.item_embedding.weight.t())

        return scores