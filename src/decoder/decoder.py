import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
    """
    单个解码器模块，遵循标准Transformer设计.
    它包含三个核心部分：
    1. 带掩码的多头自注意力 (Masked Multi-Head Self-Attention)
    2. 编码器-解码器交叉注意力 (Encoder-Decoder Cross-Attention)
    3. 前馈神经网络 (Feed-Forward Network)
    """
    def __init__(self, embedding_dim: int, num_heads: int, ffn_hidden_dim: int, dropout_ratio: float = 0.1):
        super(DecoderBlock, self).__init__()
        
        # 1. 带掩码的自注意力层 (用于解码器自身)
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_ratio, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_ratio)
        
        # 2. 交叉注意力层 (连接编码器和解码器)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_ratio, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_ratio)
        
        # 3. 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(ffn_hidden_dim, embedding_dim)
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout_ratio)

    def forward(self, target, encoder_output, target_mask, memory_mask):
        """
        Args:
            target: 解码器的输入序列, e.g., (B, target_len, D)
            encoder_output: 编码器(HSTU)的最终输出, (B, source_len, D)
            target_mask: 目标序列的注意力掩码 (防止看到未来), (target_len, target_len)
            memory_mask: 源序列(编码器输出)的填充掩码, (B, source_len)
        """
        # --- 第一部分: 带掩码的自注意力 ---
        # 解码器关注自身已经生成的部分
        self_attn_output, _ = self.self_attention(
            query=target, key=target, value=target, 
            attn_mask=target_mask,
            key_padding_mask=None  # 通常解码器输入在训练时不带padding
        )
        # 残差连接和归一化
        target = self.norm1(target + self.dropout1(self_attn_output))
        
        # --- 第二部分: 编码器-解码器交叉注意力 ---
        # 这是连接编码器和解码器的关键！
        # Query 来自解码器，Key 和 Value 来自编码器的输出
        cross_attn_output, _ = self.cross_attention(
            query=target, key=encoder_output, value=encoder_output,
            key_padding_mask=memory_mask # 屏蔽掉编码器输入中的填充部分
        )
        # 残差连接和归一化
        target = self.norm2(target + self.dropout2(cross_attn_output))
        
        # --- 第三部分: 前馈神经网络 ---
        ffn_output = self.ffn(target)
        # 残差连接和归一化
        target = self.norm3(target + self.dropout3(ffn_output))
        
        return target


class GenerativeDecoder(nn.Module):
    """
    完整的生成式解码器.
    它由物品嵌入层、位置编码、多个解码器模块和一个最终的线性输出层组成.
    """
    def __init__(self, num_items: int, embedding_dim: int, num_layers: int, num_heads: int, 
                 ffn_hidden_dim: int, max_seq_len: int, dropout_ratio: float = 0.1):
        super(GenerativeDecoder, self).__init__()
        
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout_ratio) for _ in range(num_layers)]
        )
        
        # 最终的线性层，将输出映射到词汇表大小，用于预测下一个物品
        self.output_layer = nn.Linear(embedding_dim, num_items)
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding_dim = embedding_dim

    @staticmethod
    def _generate_square_subsequent_mask(sz: int):
        """生成一个上三角矩阵的掩码，用于防止自注意力看到未来的token"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, target_ids, encoder_output, memory_padding_mask):
        """
        Args:
            target_ids: 目标物品ID序列, (B, target_len)
            encoder_output: 来自HSTU编码器的输出, (B, source_len, D)
            memory_padding_mask: HSTU输入序列的填充掩码, (B, source_len)
        """
        batch_size, target_len = target_ids.size()
        
        # 1. 准备解码器输入
        # 生成位置ID
        positions = torch.arange(0, target_len, device=target_ids.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 物品嵌入 + 位置嵌入
        target_emb = self.item_embedding(target_ids) * math.sqrt(self.embedding_dim)
        pos_emb = self.pos_embedding(positions)
        decoder_input = self.dropout(target_emb + pos_emb)
        
        # 2. 生成自注意力掩码
        target_mask = self._generate_square_subsequent_mask(target_len).to(target_ids.device)
        
        # 3. 逐层通过DecoderBlock
        output = decoder_input
        for layer in self.decoder_layers:
            output = layer(output, encoder_output, target_mask, memory_padding_mask)
            
        # 4. 最终输出预测
        # 在这里可以集成MoE系统，根据不同的专家意见来调整output
        # Future work: output = self.moe_layer(output, encoder_output)
        
        logits = self.output_layer(output) # (B, target_len, num_items)
        
        return logits