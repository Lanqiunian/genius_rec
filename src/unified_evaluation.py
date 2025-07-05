"""
统一评估模块 (unified_evaluation.py)

该模块整合了所有模型(HSTU, Baseline, GENIUS-Rec)的评估函数，
确保三个模型使用完全一致的评估逻辑和指标计算方法。
"""

import math
import torch
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src.metrics import compute_hr_ndcg_full, compute_hr_ndcg_batch


class ValidationDataset(Dataset):
    """
    用于排序指标评估的数据集：
    - 只从验证/测试集中取数据
    - 使用Leave-One-Out方式评估
    - 确保训练时没有见过完整序列
    """
    def __init__(self, data_path, max_len, pad_token_id=0):
        self.data = pd.read_parquet(data_path)
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data.iloc[idx]['history']
        # Leave-One-Out: 最后一个作为目标，其余作为输入
        ground_truth_item = full_seq[-1]
        input_seq = full_seq[:-1]
        
        # 截断和填充
        if len(input_seq) > self.max_len:
            input_seq = input_seq[-self.max_len:]
        
        padded_input_seq = np.full(self.max_len, self.pad_token_id, dtype=np.int64)
        padded_input_seq[-len(input_seq):] = input_seq
        
        return {
            'input_ids': torch.tensor(padded_input_seq, dtype=torch.long),
            'ground_truth': torch.tensor(ground_truth_item, dtype=torch.long)
        }


def evaluate_model_validation(model, val_loader, criterion, device, pad_token_id):
    """
    验证集评估：计算loss和perplexity（不计算排序指标）
    适用于基础评估需求
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)
            
            # 前向传播
            logits = model(source_ids, decoder_input_ids, source_padding_mask)
            
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 统计非padding的token数量
            non_padding_mask = (labels != pad_token_id).float()
            num_tokens = non_padding_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return {
        'val_loss': avg_loss,
        'val_ppl': perplexity
    }


def evaluate_model_validation_with_ranking(model, val_loader, criterion, device, epoch, num_epochs, pad_token_id, num_candidates=None, top_k=10):
    """
    【优化版】验证集评估：计算loss、ppl和排序指标
    
    对采样评估逻辑进行了完全的向量化重构，解决了性能瓶颈。
    """
    model.eval()
    
    total_loss_tokens = 0.0
    total_tokens = 0
    total_gate_weights = None
    total_valid_batches = 0
    
    # 排序指标计算
    hr_total = 0.0
    ndcg_total = 0.0
    total_samples = 0
    
    # 预先计算所有物品嵌入 (仅用于全量评估模式)
    with torch.no_grad():
        item_num = model.encoder.item_embedding.num_embeddings
        all_item_embeddings = None
        if num_candidates is None or num_candidates <= 0:
            all_item_ids = torch.arange(1, item_num, device=device) # 排除 padding token 0
            all_item_embeddings = model.encoder.item_embedding(all_item_ids)

    progress_bar_desc = f"Epoch {epoch+1}/{num_epochs} [Validation"
    if num_candidates is not None and num_candidates > 0:
        progress_bar_desc += " - Sampled Eval]"
    else:
        progress_bar_desc += " - Full Eval]"
    progress_bar = tqdm(val_loader, desc=progress_bar_desc)

    with torch.no_grad():
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)
            batch_size = source_ids.size(0)

            # 计算模型输出（包括损失计算所需的logits）
            logits, gate_weights = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=True)
            
            # 计算损失 (这部分逻辑不变)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            non_padding_mask = (labels != pad_token_id)
            num_tokens = non_padding_mask.sum().item()
            if num_tokens > 0:
                total_loss_tokens += loss.item() * num_tokens
                total_tokens += num_tokens
            
            # 记录专家权重 (这部分逻辑不变)
            if gate_weights is not None:
                label_mask = non_padding_mask.float().unsqueeze(-1)
                valid_gate_weights = gate_weights * label_mask
                batch_sum = valid_gate_weights.sum(dim=1)
                batch_count = label_mask.sum(dim=1)
                batch_mean = batch_sum / (batch_count + 1e-8)
                masked_gate_weights = batch_mean.mean(dim=0)
                
                if total_gate_weights is None:
                    total_gate_weights = masked_gate_weights
                else:
                    total_gate_weights += masked_gate_weights
                total_valid_batches += 1
            
            # ======================== 排序指标计算 ========================
            # 对整个批次获取编码器输出
            encoder_outputs = model.encoder(source_ids)  # [B, L, D]
            user_embeddings = encoder_outputs[:, -1, :]  # [B, D] - 取每个序列的最后一个位置作为用户表示
            
            # 【优化】向量化提取真实目标物品ID
            # 创建一个巨大的列索引，这样找到的第一个非零值就是第一个非pad标签
            col_indices = torch.arange(labels.size(1), device=device)
            # 将pad位置的索引设为一个巨大值，这样它们就不会被argmin选中
            masked_labels_indices = col_indices.expand_as(labels).masked_fill(non_padding_mask == 0, labels.size(1) + 1)
            # 找到第一个非pad标签的列索引
            first_label_indices = torch.argmin(masked_labels_indices, dim=1)
            # 使用这些索引提取目标ID
            target_item_ids = labels[torch.arange(batch_size), first_label_indices]
            # 对于完全是pad的行，argmin会返回0，其ID也为0，我们以此作为无效样本的标记
            valid_samples_mask = (target_item_ids != pad_token_id)
            
            # 如果整个批次都没有有效样本，则跳过排序指标计算
            if not valid_samples_mask.any():
                hr, ndcg = 0.0, 0.0
            
            # --- 【核心优化】向量化的采样评估逻辑 ---
            elif num_candidates is not None and num_candidates > 0:
                # 仅对有效样本进行操作
                active_user_embeddings = user_embeddings[valid_samples_mask]
                active_target_ids = target_item_ids[valid_samples_mask]
                num_valid_samples = active_user_embeddings.size(0)

                # [num_valid, 1]
                positive_ids = active_target_ids.unsqueeze(1)

                # 1. 在GPU上一次性为整个batch生成负样本
                # [num_valid, num_candidates - 1]
                negative_ids = torch.randint(
                    1, item_num,
                    (num_valid_samples, num_candidates - 1),
                    device=device
                )

                # 2. 检查并替换与正样本冲突的负样本 (防止采样到正样本)
                # [num_valid, num_candidates - 1]
                collisions = (negative_ids == positive_ids)
                while torch.any(collisions):
                    # 只为冲突的位置重新采样
                    new_neg_samples = torch.randint(
                        1, item_num,
                        (collisions.sum().item(),), # 只生成需要替换的数量
                        device=device
                    )
                    negative_ids[collisions] = new_neg_samples
                    collisions = (negative_ids == positive_ids)

                # 3. 组合正负样本
                # [num_valid, num_candidates]
                all_candidate_ids = torch.cat([positive_ids, negative_ids], dim=1)

                # 4. 一次性获取所有候选物品的嵌入
                # candidate_embeddings: [num_valid, num_candidates, D]
                candidate_embeddings = model.encoder.item_embedding(all_candidate_ids)
                
                # 5. 一次性计算相似度 (使用批处理矩阵乘法)
                # user_embeddings: [num_valid, D] -> [num_valid, 1, D]
                # scores: [num_valid, num_candidates]
                scores = torch.bmm(active_user_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)
                
                # 6. 高效计算指标
                # 正样本的分数在第一列
                positive_scores = scores[:, 0].unsqueeze(1)
                
                # 计算每个正样本在候选列表中的排名 (有多少个负样本分数比它高)
                rank = (scores[:, 1:] >= positive_scores).sum(dim=1) + 1
                
                # 计算 HR@K
                hr = (rank <= top_k).float().mean().item()
                
                # 计算 NDCG@K
                in_top_k = (rank <= top_k)
                # 只对在top_k内的样本计算NDCG
                ndcg_values = 1.0 / torch.log2(rank[in_top_k] + 1)
                ndcg = ndcg_values.mean().item() if len(ndcg_values) > 0 else 0.0

            # --- 全量评估分支 (保持不变，因为其本身是高效的) ---
            else:
                hr, ndcg = compute_hr_ndcg_full(
                    user_embeddings,
                    all_item_embeddings,
                    target_item_ids,
                    k=top_k
                )

            # 乘以 batch_size 是为了与后面的 / len(val_loader) 对应
            hr_total += hr * batch_size 
            ndcg_total += ndcg * batch_size
            total_samples += batch_size
            
            progress_bar.set_postfix(loss=(loss.item() if num_tokens > 0 else 0.0), hr=hr, ndcg=ndcg)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算最终指标
    avg_loss = total_loss_tokens / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss > 0 and avg_loss < 20 else float('inf')
    avg_gate_weights = total_gate_weights / total_valid_batches if total_valid_batches > 0 else None
    
    # 排序指标 - 完全对齐HSTU计算方式
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = ndcg_total / total_samples if total_samples > 0 else 0.0
    
    result = {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
        'val_hr': avg_hr,
        'val_ndcg': avg_ndcg,
        'avg_gate_weights': avg_gate_weights,
        'evaluated_samples': total_samples
    }
    
    # 动态添加专家权重信息
    if avg_gate_weights is not None:
        enabled_experts = [k for k, v in model.decoder.expert_config["experts"].items() if v]
        for i, expert_name in enumerate(enabled_experts):
            if i < len(avg_gate_weights):
                result[f'avg_{expert_name}_weight'] = avg_gate_weights[i].item()
    
    return result

def evaluate_model_test(model, test_loader, device, item_num, num_candidates=None, top_k=10):
    """
    测试集评估：只计算排序指标，训练结束后调用一次
    
    使用与HSTU完全相同的评估逻辑
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 设备
        item_num: 物品总数
        num_candidates: 评估方式控制参数
                        - None: 使用全量评估（与所有物品计算相似度）- 与HSTU完全一致
                        - 整数值: 使用采样评估（每个用户随机抽取n-1个负样本+1个正样本）
        top_k: 推荐列表长度K
    """
    model.eval()
    hr_total = 0.0
    ndcg_total = 0.0
    total_samples = 0
    
    with torch.no_grad():
        # 预先计算所有物品嵌入
        all_item_ids = torch.arange(1, item_num, device=device)
        all_item_embeddings = model.encoder.item_embedding(all_item_ids)
        
        progress_bar = tqdm(test_loader, desc="Test Set Evaluation - Full Eval")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            ground_truth_ids = batch['ground_truth'].to(device)

            # 获取用户嵌入
            encoder_output = model.encoder(input_ids)
            user_embeddings = encoder_output[:, -1, :]  # 取最后一个位置
            
            # 使用统一的评估函数
            batch_size = input_ids.size(0)
            
            if num_candidates is not None and num_candidates > 0:
                # 使用采样评估（随机抽取n个负样本+1个正样本）
                hr_batch_sum, ndcg_batch_sum = 0.0, 0.0
                valid_batch_samples = 0
                
                for i in range(batch_size):
                    target_id = ground_truth_ids[i].item()
                    if target_id == 0:
                        continue  # 跳过无效样本
                    
                    valid_batch_samples += 1
                    
                    # 随机选择num_candidates-1个负样本ID (排除0和目标ID)
                    candidate_ids = set(range(1, item_num))
                    candidate_ids.discard(target_id)  # 排除正样本
                    neg_ids = random.sample(list(candidate_ids), min(num_candidates-1, len(candidate_ids)))
                    
                    # 合并正负样本
                    all_candidate_ids = [target_id] + neg_ids
                    random.shuffle(all_candidate_ids)  # 打乱顺序
                    
                    # 转换为张量
                    all_candidate_ids = torch.tensor(all_candidate_ids, device=device)
                    candidate_embeddings = model.encoder.item_embedding(all_candidate_ids)
                    
                    # 计算单个用户的指标
                    user_emb = user_embeddings[i].unsqueeze(0)  # [1, D]
                    hr, ndcg = compute_hr_ndcg_full(
                        user_emb,
                        F.normalize(candidate_embeddings, p=2, dim=1),
                        torch.tensor([target_id], device=device),
                        k=top_k
                    )
                    
                    hr_batch_sum += hr
                    ndcg_batch_sum += ndcg
                
                # 计算批次平均值
                if valid_batch_samples > 0:
                    hr = hr_batch_sum / valid_batch_samples
                    ndcg = ndcg_batch_sum / valid_batch_samples
                else:
                    hr, ndcg = 0.0, 0.0
            else:
                # 使用全量评估（与所有物品计算相似度）
                hr, ndcg = compute_hr_ndcg_full(
                    user_embeddings,
                    all_item_embeddings,
                    ground_truth_ids,
                    k=top_k
                )
            
            hr_total += hr * batch_size
            ndcg_total += ndcg * batch_size
            total_samples += batch_size
    
    # 与HSTU保持一致：使用批次数量作为分母
    avg_hr = hr_total / len(test_loader) if len(test_loader) > 0 else 0.0
    avg_ndcg = ndcg_total / len(test_loader) if len(test_loader) > 0 else 0.0
    
    return {
        'test_hr': avg_hr,
        'test_ndcg': avg_ndcg,
        'evaluated_samples': total_samples
    }


def evaluate_encoder(encoder, val_loader, device, top_k=10):
    """
    编码器评估函数：用于HSTU编码器的独立评估
    """
    encoder.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_samples = 0
    
    with torch.no_grad():
        # 获取所有物品的嵌入
        item_num = encoder.item_embedding.num_embeddings
        all_item_ids = torch.arange(1, item_num, device=device)
        all_item_embeddings = encoder.item_embedding(all_item_ids)
        
        for batch in tqdm(val_loader, desc="Evaluating Encoder"):
            seq = batch[0].to(device)
            target_item_ids = batch[1].to(device).squeeze(1)
            
            # 获取用户表示（序列的最后一个位置）
            sequence_output = encoder.forward(seq)
            user_embeddings = sequence_output[:, -1, :]
            
            # 使用统一评估函数
            batch_size = seq.size(0)
            hr, ndcg = compute_hr_ndcg_full(
                user_embeddings,
                all_item_embeddings,
                target_item_ids,
                k=top_k
            )
            
            total_hr += hr * batch_size
            total_ndcg += ndcg * batch_size
            total_samples += batch_size
    
    avg_hr = total_hr / total_samples if total_samples > 0 else 0.0
    avg_ndcg = total_ndcg / total_samples if total_samples > 0 else 0.0
    
    return avg_hr, avg_ndcg
