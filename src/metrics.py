import torch
import math

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