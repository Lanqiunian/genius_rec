import torch
import math
import numpy as np
import torch.nn.functional as F

def get_metrics(batch_logits, batch_labels, k, pad_token_id=0):
    """
    【向量优化版】计算一个批次的 HR@K 和 NDCG@K。
    通过并行化操作替代for循环，极大提升计算效率。

    Args:
        batch_logits (torch.Tensor): 模型的输出 logits，形状为 (B, T, V)。
        batch_labels (torch.Tensor): 真实的标签，形状为 (B, T)。
        k (int): 评估指标的 top-K 值。
        pad_token_id (int): 用于填充的token ID。

    Returns:
        tuple: (list_of_hr_scores, list_of_ndcg_scores) 
               返回Python列表以保持与原函数输出类型一致。
    """
    # 1. 找出所有需要评估的有效位置 (非padding)
    valid_mask = (batch_labels != pad_token_id)
    if not valid_mask.any():
        return [], []

    # 2. 获取 Top-K 预测结果
    _, top_k_preds = torch.topk(batch_logits, k=k, dim=-1)

    # 3. 将真实标签的维度扩展为 (B, T, 1) 以便进行广播比较
    batch_labels_expanded = batch_labels.unsqueeze(-1)

    # 4. 一次性比较所有位置，判断是否命中
    #    (top_k_preds == batch_labels_expanded) -> [B, T, k]
    #    .any(dim=-1) 检查 top-k 列表中是否有命中的 -> [B, T]
    hits_matrix = (top_k_preds == batch_labels_expanded).any(dim=-1)
    
    # 筛选出所有有效位置的命中结果
    hr_scores_tensor = hits_matrix[valid_mask].float()

    # 5. 计算NDCG
    ndcg_scores_tensor = torch.zeros_like(hr_scores_tensor)
    
    # 找出所有命中（hit）的位置
    hit_indices_mask = (hr_scores_tensor == 1.0)
    
    if hit_indices_mask.any():
        # 仅对命中的情况进行处理
        # 筛选出命中了的 top_k 预测结果和对应的真实标签
        hit_top_k_preds = top_k_preds[valid_mask][hit_indices_mask]
        hit_labels = batch_labels_expanded[valid_mask][hit_indices_mask]
        
        # 找出命中物品在 Top-K 列表中的位置 (索引从0开始)
        # position_matrix 形状: [num_hits, k]
        position_matrix = (hit_top_k_preds == hit_labels)
        
        # .nonzero() 返回命中位置的索引 (row, col)
        # 我们只需要列索引 (col)，即在k个物品中的排名
        positions = position_matrix.nonzero(as_tuple=True)[1]
        
        # DCG = 1 / log2(position + 2)
        # 将计算出的ndcg值放回原ndcg_scores张量的对应位置
        ndcg_scores_tensor[hit_indices_mask] = 1.0 / torch.log2(positions.float() + 2)
        
    # 转换为list以匹配原始函数签名
    return hr_scores_tensor.tolist(), ndcg_scores_tensor.tolist()


def compute_hr_ndcg_full(user_embeddings, all_item_embeddings, target_item_ids, k=10, special_token_offset=4):
    """
    【向量优化版】全量评估：计算HR@K和NDCG@K（与HSTU和baseline完全一致的实现）
    通过并行化操作一次性计算所有样本的排名，并保持与原版相同的平均逻辑。

    Args:
        user_embeddings: 用户嵌入 [B, D]
        all_item_embeddings: 所有物品的嵌入 [N, D] (N = num_items - offset)
        target_item_ids: 目标物品ID [B]
        k: 推荐列表长度
        special_token_offset (int): ID到索引的偏移量。原代码为4。

    Returns:
        tuple: (hr_float, ndcg_float)
    """
    batch_size = user_embeddings.size(0)

    # 1. 筛选出有效的样本进行计算
    valid_mask = (target_item_ids > 0)
    if not valid_mask.any():
        return 0.0, 0.0

    active_user_embeddings = user_embeddings[valid_mask]
    active_target_ids = target_item_ids[valid_mask]
    
    # 2. L2归一化
    active_user_embeddings = F.normalize(active_user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)

    # 3. 计算用户与所有物品的相似度
    # [num_valid, D] x [D, N] -> [num_valid, N]
    scores = torch.matmul(active_user_embeddings, all_item_embeddings.t())

    # 4. 【核心优化】向量化计算排名
    # 4.1. 获取每个有效目标物品的分数
    target_indices = active_target_ids - special_token_offset
    target_scores = scores.gather(1, target_indices.unsqueeze(1))

    # 4.2. 计算排名：计算有多少物品的分数 >= 目标物品的分数
    rank = (scores >= target_scores).sum(dim=1)

    # 5. 计算指标
    in_top_k = (rank <= k)
    
    # 命中总数
    hr_count = in_top_k.float().sum().item()
    
    # NDCG总和
    ndcg_sum = (1.0 / torch.log2(rank[in_top_k].float() + 1)).sum().item()

    # 【保持一致】与原函数逻辑对齐，使用总批次大小作为分母
    hr = hr_count / batch_size
    ndcg = ndcg_sum / batch_size
    
    return hr, ndcg


def compute_hr_ndcg_batch(user_embeddings, all_item_embeddings, target_item_ids, k=10, special_token_offset=4):
    """
    【向量优化版】全量评估：计算HR@K和NDCG@K（返回每个样本的结果）

    Args:
        user_embeddings: 用户嵌入 [B, D]
        all_item_embeddings: 所有物品的嵌入 [N, D]
        target_item_ids: 目标物品ID [B]
        k: 推荐列表长度
        special_token_offset (int): ID到索引的偏移量。原代码为4。

    Returns:
        tuple: (hr_list, ndcg_list)
    """
    batch_size = user_embeddings.size(0)

    # 1. 筛选出有效的样本进行计算
    valid_mask = (target_item_ids > 0)
    
    # 初始化返回列表
    hr_list = [0.0] * batch_size
    ndcg_list = [0.0] * batch_size
    
    if not valid_mask.any():
        # 如果没有有效样本，根据原逻辑，应该返回对应长度的0值列表
        # (取决于原代码中对于无效样本是跳过还是记为0)
        # 这里假设跳过，即不填充列表，返回空列表
        return [], []

    active_user_embeddings = user_embeddings[valid_mask]
    active_target_ids = target_item_ids[valid_mask]
    
    # 2. L2归一化
    active_user_embeddings = F.normalize(active_user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)

    # 3. 计算相似度分数
    scores = torch.matmul(active_user_embeddings, all_item_embeddings.t())

    # 4. 向量化计算排名
    target_indices = active_target_ids - special_token_offset
    target_scores = scores.gather(1, target_indices.unsqueeze(1))
    rank = (scores >= target_scores).sum(dim=1)

    # 5. 计算每个有效样本的指标
    in_top_k = (rank <= k)
    
    # HR结果
    hr_results = in_top_k.float()
    
    # NDCG结果
    ndcg_results = torch.zeros_like(hr_results)
    ndcg_results[in_top_k] = 1.0 / torch.log2(rank[in_top_k].float() + 1)

    # 6. 将计算结果填充回原始batch对应的位置
    #    这样可以保证即使有无效样本，输出列表的长度也与batch_size一致
    #    但这里为了匹配原函数的行为（只返回有效样本的结果），我们直接转换成list
    #    如果需要对齐batch_size，需要更复杂的索引操作
    
    # 原函数是跳过无效样本，所以我们只返回有效样本的结果
    return hr_results.tolist(), ndcg_results.tolist()