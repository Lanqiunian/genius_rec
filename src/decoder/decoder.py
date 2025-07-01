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
    def __init__(self, num_items: int, embedding_dim: int, num_layers: int, num_heads: int, 
                 ffn_hidden_dim: int, max_seq_len: int, dropout_ratio: float = 0.1, 
                 pad_token_id: int = 0, text_embedding_dim: int = 768): # 假设您的Gemini嵌入是768维
        super(GenerativeDecoder, self).__init__()
        
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout_ratio) for _ in range(num_layers)]
        )
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding_dim = embedding_dim
        self.num_items = num_items

        # ==================== MoE 核心组件定义 ====================
        
        # 1. 文本嵌入层 (用于存储所有物品的文本嵌入，设为不可训练)
        self.text_embedding = nn.Embedding(num_items, text_embedding_dim, padding_idx=pad_token_id)
        self.text_embedding.weight.requires_grad = False
        
        # 2. 行为专家网络 (就是原先的输出层)
        self.behavior_expert_fc = nn.Linear(embedding_dim, num_items)
        
        # 3. 内容专家的查询投影层
        # 将解码器隐藏状态投影到与文本嵌入相同的维度，以便进行点积
        self.content_expert_projection = nn.Linear(embedding_dim, text_embedding_dim)

        # 4. 门控网络 (Router)
        # 输入是解码器的隐藏状态，输出是2个专家的权重
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim, 2),
            nn.Softmax(dim=-1)
        )
        # ==========================================================

    def load_text_embeddings(self, embedding_matrix: torch.Tensor):
        """
        一个辅助函数，用于从外部加载预训练好的文本嵌入矩阵。
        """
        if self.text_embedding.weight.shape != embedding_matrix.shape:
            raise ValueError(f"Shape mismatch! Model expects {self.text_embedding.weight.shape}, but got {embedding_matrix.shape}")
        
        print("Loading pretrained text embeddings into the decoder...")
        self.text_embedding.weight.data.copy_(embedding_matrix)
        print("Text embeddings loaded successfully.")

    @staticmethod
    def _generate_square_subsequent_mask(sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, target_ids: torch.Tensor, encoder_output: torch.Tensor, memory_padding_mask: torch.Tensor, return_weights: bool = False):
        batch_size, target_len = target_ids.size()
        positions = torch.arange(0, target_len, device=target_ids.device).unsqueeze(0)
        target_emb = self.item_embedding(target_ids) * math.sqrt(self.embedding_dim)
        pos_emb = self.pos_embedding(positions)
        decoder_input = self.dropout(target_emb + pos_emb)
        target_mask = self._generate_square_subsequent_mask(target_len).to(target_ids.device)
        
        hidden_state = decoder_input
        for layer in self.decoder_layers:
            hidden_state = layer(hidden_state, encoder_output, target_mask, memory_padding_mask)
        
        behavior_logits = self.behavior_expert_fc(hidden_state)
        content_query = self.content_expert_projection(hidden_state)
        all_text_embeddings = self.text_embedding.weight.transpose(0, 1)
        content_logits = torch.matmul(content_query, all_text_embeddings)
        
        gate_weights = self.gate_network(hidden_state)
        w_behavior = gate_weights[:, :, 0].unsqueeze(-1)
        w_content = gate_weights[:, :, 1].unsqueeze(-1)
        
        final_logits = w_behavior * behavior_logits + w_content * content_logits
        
        if return_weights:
            return final_logits, gate_weights
        else:
            return final_logits
