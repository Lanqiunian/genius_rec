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
                 pad_token_id: int = 0, text_embedding_dim: int = 768, 
                 expert_config: dict = None):  # 【新增】专家配置参数
        super(GenerativeDecoder, self).__init__()
        
        # 基础组件
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout_ratio) for _ in range(num_layers)]
        )
        
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding_dim = embedding_dim
        self.num_items = num_items
        
        # ==================== 配置驱动的专家系统 ====================
        # 专家配置
        self.expert_config = expert_config or {
            "experts": {"behavior_expert": True, "content_expert": True, "image_expert": False},
            "gate_config": {"gate_type": "mlp", "temperature": 1.0},
            "content_expert": {"text_embedding_dim": text_embedding_dim, "attention_heads": num_heads},
            "image_expert": {"image_embedding_dim": 512, "attention_heads": num_heads}
        }
        
        # 启用的专家列表
        self.enabled_experts = [k for k, v in self.expert_config["experts"].items() if v]
        num_experts = len(self.enabled_experts)
        
        print(f"🧠 启用的专家: {self.enabled_experts} (共{num_experts}个)")
        
        # 1. 行为专家 (Behavior Expert)
        if self.expert_config["experts"]["behavior_expert"]:
            self.behavior_expert_fc = nn.Linear(embedding_dim, num_items)
        
        # 2. 内容专家 (Content Expert) - 基于文本嵌入
        if self.expert_config["experts"]["content_expert"]:
            content_config = self.expert_config["content_expert"]
            self.text_embedding = nn.Embedding(num_items, content_config["text_embedding_dim"], padding_idx=pad_token_id)
            self.text_embedding.weight.requires_grad = False
            
            if content_config.get("use_cross_attention", True):
                self.content_expert_attention = nn.MultiheadAttention(
                    embed_dim=embedding_dim, 
                    num_heads=content_config["attention_heads"], 
                    dropout=dropout_ratio, 
                    batch_first=True
                )
                self.content_attention_projection = nn.Linear(embedding_dim, content_config["text_embedding_dim"])
            else:
                # 简单线性投影方案
                self.content_expert_fc = nn.Linear(embedding_dim, content_config["text_embedding_dim"])
        
        # 3. 图像专家 (Image Expert) - 基于书封面嵌入 【预留】
        if self.expert_config["experts"]["image_expert"]:
            image_config = self.expert_config["image_expert"]
            self.image_embedding = nn.Embedding(num_items, image_config["image_embedding_dim"], padding_idx=pad_token_id)
            self.image_embedding.weight.requires_grad = False
            
            if image_config.get("use_cross_attention", True):
                self.image_expert_attention = nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=image_config["attention_heads"],
                    dropout=dropout_ratio,
                    batch_first=True
                )
                self.image_attention_projection = nn.Linear(embedding_dim, image_config["image_embedding_dim"])
            else:
                self.image_expert_fc = nn.Linear(embedding_dim, image_config["image_embedding_dim"])

        # 4. 动态门控网络
        gate_config = self.expert_config["gate_config"]
        if gate_config["gate_type"] == "mlp":
            # 新增的MLP门控（多层）
            self.gate_network = nn.Sequential(
                nn.Linear(embedding_dim, gate_config.get("gate_hidden_dim", 64)),
                nn.ReLU(),
                nn.Linear(gate_config.get("gate_hidden_dim", 64), num_experts),
                nn.Softmax(dim=-1)
            )
        else:
            # 原始的简单线性门控（与原代码一致）
            self.gate_network = nn.Sequential(
                nn.Linear(embedding_dim, num_experts),
                nn.Softmax(dim=-1)
            )
        # =================================================================

    def load_text_embeddings(self, embedding_matrix: torch.Tensor):
        """
        加载预训练的文本嵌入矩阵。
        """
        if not self.expert_config["experts"]["content_expert"]:
            print("⚠️  内容专家未启用，跳过文本嵌入加载")
            return
            
        if self.text_embedding.weight.shape != embedding_matrix.shape:
            raise ValueError(f"文本嵌入形状不匹配! 模型期望 {self.text_embedding.weight.shape}, 但得到 {embedding_matrix.shape}")
        
        print("📄 正在加载预训练文本嵌入...")
        self.text_embedding.weight.data.copy_(embedding_matrix)
        print("✅ 文本嵌入加载成功")

    def load_image_embeddings(self, embedding_matrix: torch.Tensor):
        """
        加载预训练的图像嵌入矩阵（书封面嵌入）。【新增】
        """
        if not self.expert_config["experts"]["image_expert"]:
            print("⚠️  图像专家未启用，跳过图像嵌入加载")
            return
            
        if self.image_embedding.weight.shape != embedding_matrix.shape:
            raise ValueError(f"图像嵌入形状不匹配! 模型期望 {self.image_embedding.weight.shape}, 但得到 {embedding_matrix.shape}")
        
        print("🖼️  正在加载预训练图像嵌入...")
        self.image_embedding.weight.data.copy_(embedding_matrix)
        print("✅ 图像嵌入加载成功")

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
        
        # 经过N层解码器模块
        hidden_state = decoder_input
        for layer in self.decoder_layers:
            hidden_state = layer(hidden_state, encoder_output, target_mask, memory_padding_mask)
        
        # ==================== 动态专家系统推理（显存优化版）====================
        # 计算门控权重
        gate_weights = self.gate_network(hidden_state)  # (B, T, num_experts)
        
        # 【显存优化】初始化结果tensor，逐个专家计算并累加
        final_logits = torch.zeros(batch_size, target_len, self.num_items, device=target_ids.device)
        expert_idx = 0
        
        # 1. 行为专家
        if self.expert_config["experts"]["behavior_expert"]:
            behavior_logits = self.behavior_expert_fc(hidden_state)
            weight = gate_weights[:, :, expert_idx].unsqueeze(-1)  # (B, T, 1)
            final_logits += weight * behavior_logits
            expert_idx += 1
        
        # 2. 内容专家（基于文本嵌入）
        if self.expert_config["experts"]["content_expert"]:
            content_config = self.expert_config["content_expert"]
            
            if content_config.get("use_cross_attention", True):
                # 使用交叉注意力机制
                content_context_vector, _ = self.content_expert_attention(
                    query=hidden_state,
                    key=encoder_output,
                    value=encoder_output,
                    key_padding_mask=memory_padding_mask
                )
                content_query = self.content_attention_projection(content_context_vector)
            else:
                # 使用简单线性投影
                content_query = self.content_expert_fc(hidden_state)
            
            # 计算与所有文本嵌入的相似度
            all_text_embeddings = self.text_embedding.weight.transpose(0, 1)
            content_logits = torch.matmul(content_query, all_text_embeddings)
            
            weight = gate_weights[:, :, expert_idx].unsqueeze(-1)  # (B, T, 1)
            final_logits += weight * content_logits
            expert_idx += 1
        
        # 3. 图像专家（基于书封面嵌入）【预留实现】
        if self.expert_config["experts"]["image_expert"]:
            image_config = self.expert_config["image_expert"]
            
            if image_config.get("use_cross_attention", True):
                # 使用交叉注意力机制
                image_context_vector, _ = self.image_expert_attention(
                    query=hidden_state,
                    key=encoder_output,
                    value=encoder_output,
                    key_padding_mask=memory_padding_mask
                )
                image_query = self.image_attention_projection(image_context_vector)
            else:
                # 使用简单线性投影
                image_query = self.image_expert_fc(hidden_state)
            
            # 计算与所有图像嵌入的相似度
            all_image_embeddings = self.image_embedding.weight.transpose(0, 1)
            image_logits = torch.matmul(image_query, all_image_embeddings)
            
            weight = gate_weights[:, :, expert_idx].unsqueeze(-1)  # (B, T, 1)
            final_logits += weight * image_logits
            expert_idx += 1
        
        if return_weights:
            return final_logits, gate_weights
        else:
            return final_logits