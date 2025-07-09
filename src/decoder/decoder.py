import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
    """
    单个解码器模块，遵循标准Transformer设计.
    (此部分代码保持不变)
    """
    def __init__(self, embedding_dim: int, num_heads: int, ffn_hidden_dim: int, dropout_ratio: float = 0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_ratio, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_ratio, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(ffn_hidden_dim, embedding_dim)
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout3 = nn.Dropout(dropout_ratio)

    def forward(self, target, encoder_output, target_mask, memory_mask):
        self_attn_output, _ = self.self_attention(
            query=target, key=target, value=target, 
            attn_mask=target_mask,
            key_padding_mask=None
        )
        target = self.norm1(target + self.dropout1(self_attn_output))
        cross_attn_output, _ = self.cross_attention(
            query=target, key=encoder_output, value=encoder_output,
            key_padding_mask=memory_mask
        )
        target = self.norm2(target + self.dropout2(cross_attn_output))
        ffn_output = self.ffn(target)
        target = self.norm3(target + self.dropout3(ffn_output))
        return target


class GenerativeDecoder(nn.Module):
    """
    【Sampled Softmax 修改版】生成式解码器
    - 在训练时，根据配置动态切换到Sampled Softmax损失计算。
    - 在评估时，保持原有的全词汇表输出，确保评估逻辑不变。
    """
    def __init__(self, config: dict): # 接收总配置
        super(GenerativeDecoder, self).__init__()
        
        # --- ↓↓↓ 新增/修改的代码 ↓↓↓ ---
        self.config = config
        self.decoder_config = config['decoder_model']
        self.expert_config = config.get('expert_system', {"experts": {}})
        self.finetune_config = config['finetune']
        
        # 从配置中获取参数
        num_items = self.decoder_config['num_items']
        embedding_dim = self.decoder_config['embedding_dim']
        num_layers = self.decoder_config['num_layers']
        num_heads = self.decoder_config['num_heads']
        ffn_hidden_dim = self.decoder_config['ffn_hidden_dim']
        max_seq_len = self.decoder_config['max_seq_len']
        dropout_ratio = self.decoder_config['dropout_ratio']
        pad_token_id = config['pad_token_id']
        # --- ↑↑↑ 新增/修改的代码 ↑↑↑ ---

        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout_ratio) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding_dim = embedding_dim
        self.num_items = num_items
        
        self.enabled_experts = [k for k, v in self.expert_config["experts"].items() if v]
        num_experts = len(self.enabled_experts)
        if num_experts == 0: raise ValueError("At least one expert must be enabled!")
        print(f"🧠 [Sampled Softmax Ready] Enabled Experts: {self.enabled_experts}")

        # --- 专家和门控网络定义 (保持不变) ---
        if "behavior_expert" in self.enabled_experts:
            self.behavior_expert_projection = nn.Linear(embedding_dim, embedding_dim)
        if "content_expert" in self.enabled_experts:
            content_config = self.expert_config["content_expert"]
            text_dim = content_config["text_embedding_dim"]
            self.text_embedding_projection = nn.Linear(text_dim, embedding_dim)
            self.content_attention = nn.MultiheadAttention(embedding_dim, content_config["attention_heads"], dropout=dropout_ratio, batch_first=True)
            self.content_expert_projection = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 2), nn.ReLU(), nn.Dropout(dropout_ratio), nn.Linear(embedding_dim * 2, embedding_dim))
            self.register_buffer('text_embedding_matrix', torch.zeros(1, 1))
        if "image_expert" in self.enabled_experts:
            image_config = self.expert_config["image_expert"]
            image_dim = image_config["image_embedding_dim"]
            self.image_embedding_projection = nn.Linear(image_dim, embedding_dim)
            self.image_attention = nn.MultiheadAttention(embedding_dim, image_config["attention_heads"], dropout=dropout_ratio, batch_first=True)
            self.image_expert_projection = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 2), nn.ReLU(), nn.Dropout(dropout_ratio), nn.Linear(embedding_dim * 2, embedding_dim))
            self.register_buffer('image_embedding_matrix', torch.zeros(1, 1))
        gate_config = self.expert_config.get("gate_config", {})
        gate_type = gate_config.get("gate_type", "mlp")
        self.gate_noise_epsilon = gate_config.get("noise_epsilon", 0.1)
        if gate_type == "mlp":
            gate_hidden_dim = gate_config.get("gate_hidden_dim", 64)
            self.gate_network = nn.Sequential(nn.Linear(embedding_dim, gate_hidden_dim), nn.ReLU(), nn.Dropout(dropout_ratio), nn.Linear(gate_hidden_dim, num_experts))
        else:
            self.gate_network = nn.Linear(embedding_dim, num_experts)
        self.final_projection = nn.Linear(embedding_dim, num_items)

    def load_text_embeddings(self, embedding_matrix: torch.Tensor, verbose: bool = True):
        self.text_embedding_matrix = embedding_matrix.clone()

    def load_image_embeddings(self, embedding_matrix: torch.Tensor, verbose: bool = True):
        self.image_embedding_matrix = embedding_matrix.clone()

    @staticmethod
    def _generate_square_subsequent_mask(sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, target_ids: torch.Tensor, encoder_output: torch.Tensor, memory_padding_mask: torch.Tensor, return_weights: bool = False, **kwargs):
        # --- 步骤 1: 主解码器流程 (保持不变) ---
        batch_size, target_len = target_ids.size()
        positions = torch.arange(0, target_len, device=target_ids.device).unsqueeze(0)
        target_emb = self.item_embedding(target_ids) * math.sqrt(self.embedding_dim)
        pos_emb = self.pos_embedding(positions)
        decoder_input = self.dropout(target_emb + pos_emb)
        target_mask = self._generate_square_subsequent_mask(target_len).to(target_ids.device)
        hidden_state = decoder_input
        for layer in self.decoder_layers:
            hidden_state = layer(hidden_state, encoder_output, target_mask, memory_padding_mask)

        # --- 步骤 2: 专家系统计算 (保持不变) ---
        gate_logits = self.gate_network(hidden_state)
        if self.training and self.gate_noise_epsilon > 0:
            gate_logits += torch.randn_like(gate_logits) * self.gate_noise_epsilon
        expert_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)
        balancing_loss = torch.tensor(0.0, device=target_ids.device)
        if self.training:
            avg_probs_per_expert = expert_weights.squeeze(-1).mean(dim=(0, 1))
            balancing_loss = len(self.enabled_experts) * torch.sum(avg_probs_per_expert.pow(2))
        expert_outputs = []
        if "behavior_expert" in self.enabled_experts:
            expert_outputs.append(self.behavior_expert_projection(hidden_state))
        if "content_expert" in self.enabled_experts:
            text_history_emb = F.embedding(target_ids, self.text_embedding_matrix)
            projected_text_history_emb = self.text_embedding_projection(text_history_emb)
            content_context, _ = self.content_attention(query=hidden_state, key=projected_text_history_emb, value=projected_text_history_emb)
            expert_outputs.append(self.content_expert_projection(content_context))
        if "image_expert" in self.enabled_experts:
            image_history_emb = F.embedding(target_ids, self.image_embedding_matrix)
            projected_image_history_emb = self.image_embedding_projection(image_history_emb)
            image_context, _ = self.image_attention(query=hidden_state, key=projected_image_history_emb, value=projected_image_history_emb)
            expert_outputs.append(self.image_expert_projection(image_context))
        stacked_expert_outputs = torch.stack(expert_outputs, dim=2)
        final_hidden_state = (expert_weights * stacked_expert_outputs).sum(dim=2)

        # --- ↓↓↓ 步骤 3: 【核心修改】根据模式选择输出方式 ↓↓↓ ---
        use_sampled = self.finetune_config.get('use_sampled_softmax', False) and self.training

        if use_sampled:
            # --- 训练模式: Sampled Softmax ---
            labels = kwargs.get('labels')
            negative_samples = kwargs.get('negative_samples')
            if labels is None or negative_samples is None:
                raise ValueError("Missing 'labels' or 'negative_samples' for sampled softmax training.")
            
            # 获取正样本嵌入: [B, T, D]
            positive_embed = self.item_embedding(labels)
            
            # 获取负样本嵌入: [B, T, num_neg, D]
            negative_embed = self.item_embedding(negative_samples)

            # 拼接正负样本 -> [B, T, 1 + num_neg, D]
            candidate_embeds = torch.cat([positive_embed.unsqueeze(2), negative_embed], dim=2)

            # 计算点积得分: [B, T, 1, D] @ [B, T, D, 1+num_neg] -> [B, T, 1, 1+num_neg] -> [B, T, 1+num_neg]
            final_logits = torch.matmul(final_hidden_state.unsqueeze(2), candidate_embeds.transpose(-2, -1)).squeeze(2)

            # 应用温度系数
            final_logits = final_logits / self.finetune_config.get('temperature', 0.05)
        else:
            # --- 评估/推理模式: 全量词汇表 ---
            final_logits = self.final_projection(final_hidden_state)
        # --- ↑↑↑ 步骤 3: 【核心修改】根据模式选择输出方式 ↑↑↑ ---

        weights_to_return = expert_weights.squeeze(-1) if return_weights else None
        return final_logits, weights_to_return, balancing_loss, final_hidden_state