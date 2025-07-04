import torch
import math
import numpy as np
import torch.nn.functional as F

def get_metrics(batch_logits, batch_labels, k, pad_token_id=0):
    """
    计算一个批次的 HR@K 和 NDCG@K。

    Args:
        batch_logits (torch.Tensor): 模型的输出 logits，形状为 (B, T, V)。
        batch_labels (torch.Tensor): 真实的标签，形状为 (B, T)。
        k (int): 评估指标的 top-K 值。
        pad_token_id (int): 用于填充的token ID。

    Returns:
        tuple: (list_of_hr_scores, list_of_ndcg_scores)
    """
    # 获取 Top-K 预测结果
    # topk会返回 (values, indices)，我们只需要 indices
    # 形状: (B, T, k)
    _, top_k_preds = torch.topk(batch_logits, k=k, dim=-1)

    batch_size, seq_len = batch_labels.shape
    
    hr_scores = []
    ndcg_scores = []

    # 遍历批次中的每一个序列
    for i in range(batch_size):
        # 遍历序列中的每一个时间步 (每一个需要预测的物品)
        for t in range(seq_len):
            true_item = batch_labels[i, t].item()

            # 如果真实标签是填充符，则跳过这个时间步的评估
            if true_item == pad_token_id:
                continue

            # 获取当前时间步的 Top-K 预测列表
            pred_items = top_k_preds[i, t].tolist()

            # --- 计算 HR@K ---
            # 只要真实物品在Top-K列表中，就算命中
            hit = 1.0 if true_item in pred_items else 0.0
            hr_scores.append(hit)
            
            # --- 计算 NDCG@K ---
            # 只有命中了，NDCG才有可能不为0
            if hit > 0:
                # 找到命中物品在预测列表中的位置 (索引从0开始)
                position = pred_items.index(true_item)
                # DCG = 1 / log2(position + 2)  (position从0开始，所以+2)
                # IDCG 永远是 1.0，因为只有一个正确答案，理想情况是排在第一位
                dcg = 1.0 / math.log2(position + 2)
                idcg = 1.0
                ndcg = dcg / idcg
                ndcg_scores.append(ndcg)
            else:
                # 如果没命中，NDCG就是0
                ndcg_scores.append(0.0)

    return hr_scores, ndcg_scores


def compute_hr_ndcg_full(user_embeddings, all_item_embeddings, target_item_ids, k=10):
    """
    全量评估：计算HR@K和NDCG@K（与HSTU和baseline完全一致的实现）
    
    此函数的实现逻辑与HSTU和baseline完全一致，确保评估结果的准确性和可比性。
    
    Args:
        user_embeddings: 用户嵌入 [batch_size, embed_dim]
        all_item_embeddings: 所有物品的嵌入 [num_items-1, embed_dim]
        target_item_ids: 目标物品ID [batch_size]
        k: 推荐列表长度
        
    Returns:
        hr: HR@k
        ndcg: NDCG@k
    """
    batch_size = user_embeddings.size(0)
    
    # L2归一化（对齐HSTU原始实现）
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算用户与所有物品的相似度 [batch_size, num_items-1]
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 完全排序获取排名（降序）- 与HSTU完全一致
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    hr_count = 0
    ndcg_sum = 0.0
    valid_samples = 0
    
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0:  # 跳过无效样本
            continue
        valid_samples += 1
            
        # 修复：由于物品ID从4开始，需要减去4，使下标从0开始
        target_idx = target_id - 4  # ID到索引的转换
        target_rank_positions = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0]
        
        if len(target_rank_positions) > 0:
            rank = target_rank_positions[0].item() + 1  # 排名从1开始
            
            if rank <= k:
                hr_count += 1
                ndcg_sum += 1.0 / np.log2(rank + 1)
    
    # 与HSTU保持一致：使用总批次大小而非有效样本数作为分母
    # 这确保了与原始HSTU实现的完全一致性
    hr = hr_count / batch_size  # 注意这里使用总批次大小
    ndcg = ndcg_sum / batch_size  # 注意这里使用总批次大小
    
    return hr, ndcg


def compute_hr_ndcg_batch(user_embeddings, all_item_embeddings, target_item_ids, k=10):
    """
    全量评估：计算HR@K和NDCG@K（返回每个样本的结果）
    
    Args:
        user_embeddings: 用户嵌入 [batch_size, embed_dim]
        all_item_embeddings: 所有物品的嵌入 [num_items-1, embed_dim]
        target_item_ids: 目标物品ID [batch_size]
        k: 推荐列表长度
        
    Returns:
        hr_list: 每个样本的HR@k
        ndcg_list: 每个样本的NDCG@k
    """
    batch_size = user_embeddings.size(0)
    
    # L2归一化
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算余弦相似度
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 排序
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    hr_list, ndcg_list = [], []
    
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0: 
            continue

        target_idx = target_id - 4  # ID到索引的转换
        target_rank_positions = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0]
        
        hr, ndcg = 0.0, 0.0
        if len(target_rank_positions) > 0:
            rank = target_rank_positions[0].item() + 1
            if rank <= k:
                hr = 1.0
                ndcg = 1.0 / np.log2(rank + 1)
        
        hr_list.append(hr)
        ndcg_list.append(ndcg)
    
    return hr_list, ndcg_list