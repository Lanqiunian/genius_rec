# src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SampledSoftmaxLoss(nn.Module):
    """完全对齐官方的Sampled Softmax损失，处理索引偏移问题"""
    def __init__(self, num_negatives=100, temperature=0.05):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature
    
    def forward(self, output_embeddings, target_ids, all_item_embeddings, supervision_weights):
        batch_size, seq_len, embed_dim = output_embeddings.shape
        num_items = all_item_embeddings.size(0) - 1  # 不包括padding(0)的物品数量
        
        # 展平所有维度
        flat_output = output_embeddings.reshape(-1, embed_dim)  # (batch_size * seq_len, embed_dim)
        flat_targets = target_ids.reshape(-1)                   # (batch_size * seq_len,)
        flat_weights = supervision_weights.reshape(-1)          # (batch_size * seq_len,)
        
        # 关键修正：只过滤掉权重为0的位置（这已经排除了target_id=0的padding位置）
        valid_mask = flat_weights > 0
        num_valid = valid_mask.sum().item()
        
        # 调试信息
        print(f"DEBUG: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
        print(f"DEBUG: num_items={num_items}, all_item_embeddings.shape={all_item_embeddings.shape}")
        print(f"DEBUG: flat_weights.shape={flat_weights.shape}, num_valid={num_valid}")
        print(f"DEBUG: flat_weights.sum()={flat_weights.sum()}, flat_weights.mean()={flat_weights.mean()}")
        
        if num_valid == 0:
            return torch.tensor(0.0, device=output_embeddings.device, requires_grad=True)
        
        # 提取有效的样本
        valid_output = flat_output[valid_mask]      # (num_valid, embed_dim)
        valid_targets = flat_targets[valid_mask]    # (num_valid,)
        valid_weights = flat_weights[valid_mask]    # (num_valid,)
        
        # 关键修正：将target_ids转换为embedding索引（减1处理索引偏移）
        valid_target_indices = valid_targets - 1   # target_id从1开始，embedding索引从0开始
        
        # 边界检查：确保索引在有效范围内
        max_item_idx = all_item_embeddings.size(0) - 1
        valid_target_indices = torch.clamp(valid_target_indices, 0, max_item_idx)
        
        # L2归一化
        valid_output = F.normalize(valid_output, p=2, dim=1)
        norm_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
        
        # 正样本logits
        pos_embeddings = norm_item_embeddings[valid_target_indices]  # (num_valid, embed_dim)
        pos_logits = (valid_output * pos_embeddings).sum(dim=1, keepdim=True)  # (num_valid, 1)
        
        # 负采样
        neg_item_ids = torch.randint(1, num_items + 1,
                                    (num_valid, self.num_negatives),
                                    device=output_embeddings.device)
        
        # 避免负样本与正样本重复
        neg_mask = neg_item_ids != valid_targets.unsqueeze(1)
        neg_item_ids = torch.where(neg_mask, neg_item_ids, 
                                  torch.randint(1, num_items + 1, neg_item_ids.shape, 
                                               device=neg_item_ids.device))
        
        # 转换为embedding索引
        neg_indices = torch.clamp(neg_item_ids - 1, 0, max_item_idx)
        
        # 负样本logits
        neg_embeddings = norm_item_embeddings[neg_indices]  # (num_valid, num_negatives, embed_dim)
        neg_logits = torch.bmm(valid_output.unsqueeze(1), 
                              neg_embeddings.transpose(1, 2)).squeeze(1)  # (num_valid, num_negatives)
        
        # 组合正负样本logits并应用温度
        all_logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
        
        # 标签：正样本在第0位
        labels = torch.zeros(num_valid, dtype=torch.long, device=output_embeddings.device)
        
        # 计算交叉熵损失
        loss_per_sample = F.cross_entropy(all_logits, labels, reduction='none')
        weighted_loss = (loss_per_sample * valid_weights).sum() / valid_weights.sum()
        
        return weighted_loss