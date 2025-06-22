import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HSTU_EncoderLayer(nn.Module):
    """
    单个HSTU编码器层，其参数名已与您的config.py对齐。
    """
    def __init__(self, embedding_dim, nhead, dim_feedforward, dropout=0.1, activation=F.silu):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.d_head = embedding_dim // nhead
        
        if self.d_head * nhead != self.embedding_dim:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) 必须是 nhead ({self.nhead}) 的整数倍")

        # 将Q,K,V和门控U的投影合并在一个大的线性层中
        self.in_proj = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # 使用Pre-Norm结构，在操作之前进行LayerNorm
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.activation = activation

    def forward(self, src, src_mask=None):
        """
        前向传播
        :param src: (S, B, E) - S=序列长度, B=批次大小, E=embedding_dim
        :param src_mask: (S, S) - 注意力掩码
        """
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
            # 使用-inf进行掩码，以在激活后得到接近于0的值
            attn_scores = attn_scores.masked_fill(src_mask == 0, float('-inf'))

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
    完整的HSTU编码器，由多个HSTU_EncoderLayer堆叠而成。
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
    用于第一阶段独立训练的完整模型，读取您的config文件。
    """
    def __init__(self, num_items, config):
        super().__init__()
        self.config = config
        self.num_items = num_items

        # --- 从您的config字典中读取参数 ---
        embedding_dim = config['embedding_dim']
        nhead = config['nhead']
        num_encoder_layers = config['num_encoder_layers']
        dim_feedforward = config['dim_feedforward']
        dropout = config['dropout']
        max_seq_len = config['max_seq_len']

        # --- 模型层定义 ---
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=config['pad_token_id'])
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_len, 1, embedding_dim))
        
        # 实例化HSTU层
        hstu_layer = HSTU_EncoderLayer(
            embedding_dim=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 实例化HSTU编码器，并在最后添加一个LayerNorm
        self.hstu_encoder = HSTU_Encoder(
            encoder_layer=hstu_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(embedding_dim)
        )

        self.output_layer = nn.Linear(embedding_dim, num_items + 1)

    def forward(self, input_seq):
        # input_seq: (B, S)
        
        embedded = self.item_embedding(input_seq)
        embedded = embedded.permute(1, 0, 2)  # (S, B, E)
        
        seq_len = embedded.size(0)
        # 注意: 截取与当前输入序列长度匹配的位置编码
        src = embedded + self.positional_encoding[:seq_len, :]
        
        # 创建causal mask (后续掩码)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.config['device'])
        
        encoded = self.hstu_encoder(src, mask=attn_mask) # (S, B, E)
        encoded = encoded.permute(1, 0, 2) # (B, S, E)
        
        logits = self.output_layer(encoded) # (B, S, num_items+1)
        
        return logits