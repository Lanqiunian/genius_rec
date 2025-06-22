import torch
import numpy as np
from tqdm import tqdm

def evaluate_model(model, data_loader, device, top_k=10):
    """
    在给定的数据集上评估模型性能。

    Args:
        model (nn.Module): 待评估的模型。
        data_loader (DataLoader): 评估数据的DataLoader。
        device: 运行设备 (e.g., 'cuda' or 'cpu')。
        top_k (int): 计算指标时使用的K值。

    Returns:
        dict: 包含评估指标的字典, e.g., {'Recall@K': ..., 'NDCG@K': ...}。
    """
    model.eval()
    recalls = []
    ndcgs = []

    progress_bar = tqdm(data_loader, desc="评估中", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            input_seq = batch['input_seq'].to(device)
            targets = batch['target'].to(device) # shape: [batch_size]

            # 得到模型对所有物品的打分
            scores = model(input_seq) # shape: [batch_size, num_items]

            # --- 为每个样本计算指标 ---
            # scores.topk() 会返回 (values, indices)
            _, top_k_indices = torch.topk(scores, k=top_k, dim=1) # shape: [batch_size, k]

            # 将Tensor转到CPU并转为Numpy进行计算
            targets_np = targets.cpu().numpy()
            top_k_indices_np = top_k_indices.cpu().numpy()

            for i in range(len(targets_np)):
                target_item = targets_np[i]
                predicted_top_k = top_k_indices_np[i]

                # --- 计算 Recall@K ---
                if target_item in predicted_top_k:
                    recalls.append(1)
                else:
                    recalls.append(0)

                # --- 计算 NDCG@K ---
                # 找到目标物品在推荐列表中的位置 (rank)
                # 如果不在列表中, rank是无穷大, ndcg为0
                try:
                    rank = np.where(predicted_top_k == target_item)[0][0] + 1
                    ndcg = 1.0 / np.log2(rank + 1)
                    ndcgs.append(ndcg)
                except IndexError:
                    ndcgs.append(0)
    
    # --- 计算平均指标 ---
    recall_at_k = np.mean(recalls)
    ndcg_at_k = np.mean(ndcgs)

    return {f'Recall@{top_k}': recall_at_k, f'NDCG@{top_k}': ndcg_at_k}