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
    【最终修复与优化版】生成式解码器
    - 修正了交叉注意力的共享逻辑，确保所有专家正确参与训练。
    - 保留了“后期投影”MoE架构以保证训练速度。
    - 增加了门控噪声选项，以解决专家极化问题。
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
            if num_experts == 0: raise ValueError("At least one expert must be enabled!")
            print(f"🧠 [Final Version] Enabled Experts: {self.enabled_experts} (Total: {num_experts})")

            # --- 【核心修改】创建一个辅助函数来动态构建投影层 ---
            def create_projection_layer(expert_name, config):
                projection_type = config.get(f"{expert_name}_projection_type", "simple")
                if projection_type == "mlp":
                    print(f"  - {expert_name.replace('_', ' ').title()} Projection: MLP (with Dropout)")
                    return nn.Sequential(
                        nn.Linear(embedding_dim, embedding_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(dropout_ratio),
                        nn.Linear(embedding_dim * 2, embedding_dim)
                    )
                else:
                    print(f"  - {expert_name.replace('_', ' ').title()} Projection: Simple (Linear Layer)")
                    return nn.Linear(embedding_dim, embedding_dim)
            
            # --- 共享交叉注意力模块 (逻辑不变) ---
            self.shared_cross_attention = None
            if "content_expert" in self.enabled_experts or "image_expert" in self.enabled_experts:
                attention_heads = self.expert_config.get("content_expert", {}).get("attention_heads", num_heads)
                self.shared_cross_attention = nn.MultiheadAttention(embedding_dim, attention_heads, dropout=dropout_ratio, batch_first=True)
                print(f"✅ Shared Cross-Attention module created for multimodal experts.")

            # --- 专家网络定义 (使用新的辅助函数) ---
            if "behavior_expert" in self.enabled_experts:
                self.behavior_expert = nn.Linear(embedding_dim, embedding_dim)

            if "content_expert" in self.enabled_experts:
                content_config = self.expert_config["content_expert"]
                self.content_expert_projection = create_projection_layer("text", content_config)
                self.register_buffer('text_embedding_matrix', torch.zeros(1, 1))

            if "image_expert" in self.enabled_experts:
                image_config = self.expert_config["image_expert"]
                self.image_expert_projection = create_projection_layer("image", image_config)
                self.register_buffer('image_embedding_matrix', torch.zeros(1, 1))

            # --- 门控网络和最终输出层 (逻辑不变) ---
            gate_config = self.expert_config.get("gate_config", {})
            gate_type = gate_config.get("gate_type", "simple")
            self.gate_noise_epsilon = gate_config.get("noise_epsilon", 0.1)

            if gate_type == "mlp":
                gate_hidden_dim = gate_config.get("gate_hidden_dim", 64)
                self.gate_network = nn.Sequential(
                    nn.Linear(embedding_dim, gate_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_ratio),
                    nn.Linear(gate_hidden_dim, num_experts)
                )
                print("🚪 Gating Network Type: MLP (with Dropout for regularization)")
            else:
                self.gate_network = nn.Linear(embedding_dim, num_experts)
                print("🚪 Gating Network Type: Simple (Single Linear Layer)")
            
            self.final_projection = nn.Linear(embedding_dim, num_items)

    def load_text_embeddings(self, embedding_matrix: torch.Tensor, verbose: bool = True):
        if verbose: print("📄 Loading pre-trained text embeddings...")
        self.text_embedding_matrix = embedding_matrix.clone()
        if verbose: print("✅ Text embeddings loaded successfully.")

    def load_image_embeddings(self, embedding_matrix: torch.Tensor, verbose: bool = True):
        if verbose: print("🖼️ Loading pre-trained image embeddings...")
        self.image_embedding_matrix = embedding_matrix.clone()
        if verbose: print("✅ Image embeddings loaded successfully.")

    @staticmethod
    def _generate_square_subsequent_mask(sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # --- 核心修复：确保`forward`方法签名正确 ---
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

        # 1. 门控网络计算
        gate_logits = self.gate_network(hidden_state)
        # --- 【核心修复 3/3】仅在训练时加入门控噪声，对抗专家极化 ---
        if self.training and self.gate_noise_epsilon > 0:
            noise = torch.randn_like(gate_logits) * self.gate_noise_epsilon
            gate_logits += noise
        expert_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)

        # 2. 平衡损失计算
        balancing_loss = torch.tensor(0.0, device=target_ids.device)
        if self.training:
            avg_probs_per_expert = expert_weights.squeeze(-1).mean(dim=(0, 1))
            balancing_loss = len(self.enabled_experts) * torch.sum(avg_probs_per_expert.pow(2))

        # 3. 共享的交叉注意力计算
        cross_attention_context = None
        if self.shared_cross_attention is not None:
            cross_attention_context, _ = self.shared_cross_attention(
                query=hidden_state, key=encoder_output, value=encoder_output, key_padding_mask=memory_padding_mask
            )

        # 4. 在隐藏空间中计算专家输出
        expert_outputs = []
        if "behavior_expert" in self.enabled_experts:
            expert_outputs.append(self.behavior_expert(hidden_state))
        if "content_expert" in self.enabled_experts and cross_attention_context is not None:
            expert_outputs.append(self.content_expert_projection(cross_attention_context))
        if "image_expert" in self.enabled_experts and cross_attention_context is not None:
            expert_outputs.append(self.image_expert_projection(cross_attention_context))
        
        # 5. 加权融合
        stacked_expert_outputs = torch.stack(expert_outputs, dim=2)
        final_hidden_state = (expert_weights * stacked_expert_outputs).sum(dim=2)
        
        # 6. 最终投影
        final_logits = self.final_projection(final_hidden_state)

        weights_to_return = expert_weights.squeeze(-1) if return_weights else None
        return final_logits, weights_to_return, balancing_loss, final_hidden_state
