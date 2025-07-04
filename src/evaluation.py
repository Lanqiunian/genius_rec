# src/evaluation.py
"""
评估相关的函数模块
包含验证集和测试集的评估函数
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd


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


def compute_ranking_metrics(user_embeddings, all_item_embeddings, target_item_ids, k=10):
    """
    计算HR@K和NDCG@K指标
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

        target_idx = target_id - 1  # ID到索引的转换
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


def evaluate_model_validation(model, val_loader, criterion, device, epoch, num_epochs, pad_token_id):
    """
    验证集评估：只计算loss和ppl，用于早停和模型选择
    """
    model.eval()
    
    total_loss_tokens = 0.0
    total_tokens = 0
    total_gate_weights = None
    total_valid_batches = 0

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")

    with torch.no_grad():
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)

            logits, gate_weights = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=True)
            
            # 修正：使用传统的CrossEntropyLoss调用方式
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            valid_tokens = (labels.view(-1) != pad_token_id).sum().item()
            total_loss_tokens += loss.item() * valid_tokens
            total_tokens += valid_tokens

            # 累计门控权重（动态支持多个专家）
            non_padding_mask = (decoder_input_ids != pad_token_id)
            if gate_weights.size(-1) > 0:  # 确保有专家
                masked_gate_weights = gate_weights[non_padding_mask]  # (N, num_experts)
                if masked_gate_weights.numel() > 0:
                    if total_gate_weights is None:
                        total_gate_weights = masked_gate_weights.mean(dim=0)  # (num_experts,)
                        total_valid_batches = 1
                    else:
                        total_gate_weights += masked_gate_weights.mean(dim=0)
                        total_valid_batches += 1

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss_tokens / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    # 计算平均门控权重
    avg_gate_weights = total_gate_weights / total_valid_batches if total_valid_batches > 0 else None
    
    result = {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
    }
    
    # 动态添加专家权重信息
    if avg_gate_weights is not None:
        enabled_experts = [k for k, v in model.decoder.expert_config["experts"].items() if v]
        for i, expert_name in enumerate(enabled_experts):
            if i < len(avg_gate_weights):
                result[f'avg_{expert_name}_weight'] = avg_gate_weights[i].item()
    
    return result


def evaluate_model_test(model, test_loader, device, item_num, top_k=10):
    """
    测试集评估：只计算排序指标，训练结束后调用一次
    
    🔧 修复：使用与HSTU完全相同的评估逻辑
    """
    model.eval()
    hr_total = 0.0
    ndcg_total = 0.0
    total_samples = 0
    
    with torch.no_grad():
        # 预先计算所有物品嵌入，避免重复计算
        all_item_ids = torch.arange(1, item_num, device=device)
        all_item_embeddings = model.encoder.item_embedding(all_item_ids)
        
        progress_bar = tqdm(test_loader, desc="Test Set Evaluation - HSTU Style")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            ground_truth_ids = batch['ground_truth'].to(device)

            # 获取用户嵌入
            encoder_output = model.encoder(input_ids)
            user_embeddings = encoder_output[:, -1, :]  # 取最后一个位置
            
            # 🔧 修复：使用与HSTU完全相同的评估逻辑
            batch_size = input_ids.size(0)
            hr, ndcg = get_metrics_hstu_style(
                user_embeddings,
                all_item_embeddings,
                ground_truth_ids,
                k=top_k
            )
            
            hr_total += hr * batch_size
            ndcg_total += ndcg * batch_size
            total_samples += batch_size
    
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = ndcg_total / total_samples if total_samples > 0 else 0.0
    
    return {
        'test_hr': avg_hr,
        'test_ndcg': avg_ndcg,
        'evaluated_samples': total_samples
    }


def evaluate_model_validation_with_ranking(model, val_loader, criterion, device, epoch, num_epochs, pad_token_id, num_candidates=1000, top_k=10):
    """
    验证集评估：计算loss、ppl和排序指标
    
    ⚠️ 完全重构：使用与HSTU/baseline完全相同的评估逻辑
    
    Args:
        num_candidates: 保留参数以保持接口兼容性，但实际使用全量评估
    """
    model.eval()
    
    total_loss_tokens = 0.0
    total_tokens = 0
    total_gate_weights = None
    total_valid_batches = 0
    
    # 🚀 排序指标计算（使用完全相同的HSTU计算方法）
    hr_total = 0.0
    ndcg_total = 0.0
    total_samples = 0
    
    # 🔧 预先计算所有物品嵌入（不包含padding token 0）- 与HSTU完全一致
    with torch.no_grad():
        # 获取所有有效物品的嵌入（ID从1开始，跳过padding token 0）
        item_num = model.encoder.item_embedding.num_embeddings
        all_item_ids = torch.arange(1, item_num, device=device)
        all_item_embeddings = model.encoder.item_embedding(all_item_ids)  # [num_items-1, embed_dim]
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation - Full Eval]")

    with torch.no_grad():
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)

            logits, gate_weights = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=True)
            
            # 计算loss和ppl
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            valid_tokens = (labels.view(-1) != pad_token_id).sum().item()
            total_loss_tokens += loss.item() * valid_tokens
            total_tokens += valid_tokens

            # 累计门控权重
            non_padding_mask = (decoder_input_ids != pad_token_id)
            if gate_weights.size(-1) > 0:
                masked_gate_weights = gate_weights[non_padding_mask]
                if masked_gate_weights.numel() > 0:
                    if total_gate_weights is None:
                        total_gate_weights = masked_gate_weights.mean(dim=0)
                        total_valid_batches = 1
                    else:
                        total_gate_weights += masked_gate_weights.mean(dim=0)
                        total_valid_batches += 1
            
            # 🚀 全面重构：排序指标计算 - 完全采用HSTU的评估方法
            # 获取用户表示
            batch_size = source_ids.size(0)
            # 对整个批次获取编码器输出
            encoder_outputs = model.encoder(source_ids)  # [B, L, D]
            user_embeddings = encoder_outputs[:, -1, :]  # [B, D] - 取每个序列的最后一个位置作为用户表示
            
            # 从标签中提取真实目标物品ID
            target_item_ids = []
            for i in range(batch_size):
                valid_positions = (labels[i] != pad_token_id).nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    # 如果没有有效标签，用0填充（后面会跳过）
                    target_item_ids.append(0)
                else:
                    # 取第一个非padding的标签作为目标
                    target_item_ids.append(labels[i, valid_positions[0]].item())
            
            target_item_ids = torch.tensor(target_item_ids, device=device)
            
            # 完全采用HSTU的计算方式
            hr, ndcg = get_metrics_hstu_style(
                user_embeddings,
                all_item_embeddings,
                target_item_ids,
                k=top_k
            )
            
            hr_total += hr * batch_size
            ndcg_total += ndcg * batch_size
            total_samples += batch_size

            progress_bar.set_postfix(loss=loss.item())
        
        # 🔧 新增：清理GPU显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算最终指标
    avg_loss = total_loss_tokens / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    avg_gate_weights = total_gate_weights / total_valid_batches if total_valid_batches > 0 else None
    
    # 🚀 修复：排序指标 - 完全对齐HSTU计算方式
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = ndcg_total / total_samples if total_samples > 0 else 0.0
    
    result = {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
        'val_hr': avg_hr,          # 与HSTU完全对齐的HR@K
        'val_ndcg': avg_ndcg,      # 与HSTU完全对齐的NDCG@K
        'evaluated_samples': total_samples
    }
    
    # 动态添加专家权重信息
    if avg_gate_weights is not None:
        enabled_experts = [k for k, v in model.decoder.expert_config["experts"].items() if v]
        for i, expert_name in enumerate(enabled_experts):
            if i < len(avg_gate_weights):
                result[f'avg_{expert_name}_weight'] = avg_gate_weights[i].item()
    
    return result


def get_metrics_hstu_style(user_embeddings, all_item_embeddings, target_item_ids, k=10):
    """
    全量评估：计算用户嵌入与所有物品嵌入的相似度（与HSTU完全一致）
    
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
    
    # L2归一化（对齐HSTU实现）
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算用户与所有物品的相似度 [batch_size, num_items-1]
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 排序获取排名（降序）
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    hr_count = 0
    ndcg_sum = 0.0
    valid_samples = 0
    
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0:
            continue
        valid_samples += 1
            
        target_idx = target_id - 1  # ID到索引的转换
        target_rank_positions = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0]
        
        if len(target_rank_positions) > 0:
            rank = target_rank_positions[0].item() + 1
            
            if rank <= k:
                hr_count += 1
                ndcg_sum += 1.0 / np.log2(rank + 1)
    
    # 注意：这里使用有效样本数而非总批次大小
    hr = hr_count / valid_samples if valid_samples > 0 else 0
    ndcg = ndcg_sum / valid_samples if valid_samples > 0 else 0
    
    return hr, ndcg
