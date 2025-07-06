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
    【最终版】生成式解码器
    - 增加了对 trainable_embeddings 的配置支持
    - 实现了负载均衡损失以防止专家极化
    """
    def __init__(self, num_items: int, embedding_dim: int, num_layers: int, num_heads: int,
                 ffn_hidden_dim: int, max_seq_len: int, dropout_ratio: float = 0.1,
                 pad_token_id: int = 0, text_embedding_dim: int = 768,
                 expert_config: dict = None, **kwargs):
        super(GenerativeDecoder, self).__init__()

        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout_ratio) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding_dim = embedding_dim
        self.num_items = num_items
        
        self.expert_config = expert_config or {"experts": {}}
        self.enabled_experts = [k for k, v in self.expert_config["experts"].items() if v]
        num_experts = len(self.enabled_experts)
        if num_experts == 0: raise ValueError("至少需要启用一个专家！")
        print(f"🧠 启用的专家: {self.enabled_experts} (共{num_experts}个)")
        
        if self.expert_config["experts"].get("behavior_expert", False):
            self.behavior_expert_fc = nn.Linear(embedding_dim, num_items)

        if self.expert_config["experts"].get("content_expert", False):
            content_config = self.expert_config["content_expert"]
            if content_config.get("trainable_embeddings", False):
                print(" thawed Content Expert embeddings (trainable).")
                self.text_embedding = nn.Embedding(num_items, content_config["text_embedding_dim"], padding_idx=pad_token_id)
                self.text_embedding_matrix = None
            else:
                print("🧊 Frozen Content Expert embeddings (buffer).")
                self.register_buffer('text_embedding_matrix', torch.zeros(1, 1))
                self.text_embedding = None
            if content_config.get("use_cross_attention", True):
                self.content_expert_attention = nn.MultiheadAttention(embedding_dim, content_config["attention_heads"], dropout=dropout_ratio, batch_first=True)
                self.content_attention_projection = nn.Linear(embedding_dim, content_config["text_embedding_dim"])
            else:
                self.content_expert_fc = nn.Linear(embedding_dim, content_config["text_embedding_dim"])

        if self.expert_config["experts"].get("image_expert", False):
            image_config = self.expert_config["image_expert"]
            if image_config.get("trainable_embeddings", False):
                print(" thawed Image Expert embeddings (trainable).")
                self.image_embedding = nn.Embedding(num_items, image_config["image_embedding_dim"], padding_idx=pad_token_id)
                self.image_embedding_matrix = None
            else:
                print("🧊 Frozen Image Expert embeddings (buffer).")
                self.register_buffer('image_embedding_matrix', torch.zeros(1, 1))
                self.image_embedding = None
            if image_config.get("use_cross_attention", True):
                self.image_expert_attention = nn.MultiheadAttention(embedding_dim, image_config["attention_heads"], dropout=dropout_ratio, batch_first=True)
                self.image_attention_projection = nn.Linear(embedding_dim, image_config["image_embedding_dim"])
            else:
                self.image_expert_fc = nn.Linear(embedding_dim, image_config["image_embedding_dim"])

        gate_config = self.expert_config.get("gate_config", {})
        gate_hidden_dim = gate_config.get("gate_hidden_dim", 64)
        if gate_config.get("gate_type") == "mlp":
            self.gate_network = nn.Sequential(nn.Linear(embedding_dim, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, num_experts))
        else:
            self.gate_network = nn.Linear(embedding_dim, num_experts)

    def load_text_embeddings(self, embedding_matrix: torch.Tensor, verbose: bool = True):
        if not self.expert_config["experts"].get("content_expert", False): return
        if verbose: print("📄 正在加载预训练文本嵌入...")
        if self.text_embedding is not None:
            self.text_embedding.weight.data.copy_(embedding_matrix)
        elif 'text_embedding_matrix' in self._buffers:
            self.text_embedding_matrix = embedding_matrix.clone()
        if verbose: print("✅ 文本嵌入加载成功")

    def load_image_embeddings(self, embedding_matrix: torch.Tensor, verbose: bool = True):
        if not self.expert_config["experts"].get("image_expert", False): return
        if verbose: print("🖼️ 正在加载预训练图像嵌入...")
        if self.image_embedding is not None:
            self.image_embedding.weight.data.copy_(embedding_matrix)
        elif 'image_embedding_matrix' in self._buffers:
            self.image_embedding_matrix = embedding_matrix.clone()
        if verbose: print("✅ 图像嵌入加载成功")

    @staticmethod
    def _generate_square_subsequent_mask(sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, target_ids: torch.Tensor, encoder_output: torch.Tensor, memory_padding_mask: torch.Tensor,
                return_weights: bool = False):
        
        batch_size, target_len = target_ids.size()
        positions = torch.arange(0, target_len, device=target_ids.device).unsqueeze(0)
        target_emb = self.item_embedding(target_ids) * math.sqrt(self.embedding_dim)
        pos_emb = self.pos_embedding(positions)
        decoder_input = self.dropout(target_emb + pos_emb)
        target_mask = self._generate_square_subsequent_mask(target_len).to(target_ids.device)

        hidden_state = decoder_input
        for layer in self.decoder_layers:
            hidden_state = layer(hidden_state, encoder_output, target_mask, memory_padding_mask)

        # 1. 动态计算专家权重
        gate_input = hidden_state
        gate_logits = self.gate_network(gate_input.view(-1, self.embedding_dim))
        gate_logits = gate_logits.view(batch_size, target_len, -1)
        expert_weights = F.softmax(gate_logits, dim=-1) # Shape: [B, T, num_experts]

        # 2. 计算平衡损失 (balancing_loss)
        balancing_loss = torch.tensor(0.0, device=target_ids.device)
        if self.training:
            # 使用更标准的 .mean(dim=(0, 1))
            avg_probs_per_expert = expert_weights.mean(dim=(0, 1))
            balancing_loss = len(self.enabled_experts) * torch.sum(avg_probs_per_expert.pow(2))

        # 3. 💡 **核心修正**: 初始化 final_logits 并使用新的 `expert_weights` 变量
        final_logits = torch.zeros_like(hidden_state @ self.item_embedding.weight.t()) # 确保形状正确
        expert_idx = 0

        if self.expert_config["experts"].get("behavior_expert", False):
            behavior_logits = self.behavior_expert_fc(hidden_state)
            # 直接使用 expert_weights，不再有 'expanded'
            weight = expert_weights[:, :, expert_idx].unsqueeze(-1)
            final_logits = weight * behavior_logits
            expert_idx += 1

        if self.expert_config["experts"].get("content_expert", False):
            content_config = self.expert_config["content_expert"]
            if content_config.get("use_cross_attention", True):
                content_context_vector, _ = self.content_expert_attention(query=hidden_state, key=encoder_output, value=encoder_output, key_padding_mask=memory_padding_mask)
                content_query = self.content_attention_projection(content_context_vector)
            else:
                content_query = self.content_expert_fc(hidden_state)
            
            text_weights_t = self.text_embedding.weight.t() if self.text_embedding is not None else self.text_embedding_matrix.t()
            content_logits = torch.matmul(content_query, text_weights_t)
            
            # 直接使用 expert_weights
            weight = expert_weights[:, :, expert_idx].unsqueeze(-1)
            # 注意：这里的逻辑是 +=，因为 behavior 专家已经初始化了 final_logits
            final_logits += weight * content_logits
            expert_idx += 1

        if self.expert_config["experts"].get("image_expert", False):
            image_config = self.expert_config["image_expert"]
            if image_config.get("use_cross_attention", True):
                visual_context_vector, _ = self.image_expert_attention(query=hidden_state, key=encoder_output, value=encoder_output, key_padding_mask=memory_padding_mask)
                visual_query = self.image_attention_projection(visual_context_vector)
            else:
                visual_query = self.image_expert_fc(hidden_state)
            
            image_weights_t = self.image_embedding.weight.t() if self.image_embedding is not None else self.image_embedding_matrix.t()
            image_logits = torch.matmul(visual_query, image_weights_t)
            
            # 直接使用 expert_weights
            weight = expert_weights[:, :, expert_idx].unsqueeze(-1)
            final_logits += weight * image_logits
            expert_idx += 1
                
        if expert_idx == 0: 
            raise RuntimeError("至少需要启用一个专家！")

        # 4. 💡 **返回正确的权重变量**
        weights_to_return = expert_weights if return_weights else None
        return final_logits, weights_to_return, balancing_loss
