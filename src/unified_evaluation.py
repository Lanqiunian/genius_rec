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
    验证集评估：计算loss、ppl和排序指标
    
    完全使用与HSTU/baseline一致的评估逻辑，确保指标的可比性
    
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前轮次
        num_epochs: 总轮次
        pad_token_id: 填充标记ID
        num_candidates: 评估方式控制参数
                        - None: 使用全量评估（与所有物品计算相似度）- 与HSTU和baseline完全一致
                        - 整数值(如500): 使用采样评估（每个用户随机抽取n-1个负样本+1个正样本）- 速度更快
        top_k: 推荐列表长度K
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
    
    # 预先计算所有物品嵌入
    with torch.no_grad():
        item_num = model.encoder.item_embedding.num_embeddings
        all_item_ids = torch.arange(1, item_num, device=device)
        all_item_embeddings = model.encoder.item_embedding(all_item_ids)  # [num_items-1, embed_dim]
    
    # 根据评估模式设置不同的进度条描述
    if num_candidates is not None and num_candidates > 0:
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation - Sampled Eval ({num_candidates} candidates)]")
    else:
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation - Full Eval]")

    with torch.no_grad():
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)

            # 计算模型输出（包括损失计算所需的logits）
            logits, gate_weights = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=True)
            
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 统计非padding的token数量
            non_padding_mask = (labels != pad_token_id).float()
            num_tokens = non_padding_mask.sum().item()
            
            total_loss_tokens += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # 记录专家权重
            if gate_weights is not None:
                # 🔧 修复：从原理上理解和处理门控权重
                # gate_weights形状为 [batch_size, target_len, num_experts]
                # labels形状为 [batch_size, target_len]
                
                # 1. 创建一个掩码，标识哪些位置是有效标签（非pad）
                label_mask = (labels != pad_token_id).float().unsqueeze(-1)  # [B, T, 1]
                
                # 2. 应用掩码，只考虑有效位置的权重
                valid_gate_weights = gate_weights * label_mask
                
                # 3. 对每个batch的有效位置取平均（避免切片操作）
                batch_sum = valid_gate_weights.sum(dim=1)  # [B, num_experts]
                batch_count = label_mask.sum(dim=1)  # [B, 1]
                # 防止除零
                batch_mean = batch_sum / (batch_count + 1e-8)  # [B, num_experts]
                
                # 4. 再对整个batch取平均得到最终权重
                masked_gate_weights = batch_mean.mean(dim=0)  # [num_experts]
                
                if total_gate_weights is None:
                    total_gate_weights = masked_gate_weights
                    total_valid_batches = 1
                else:
                    total_gate_weights += masked_gate_weights
                    total_valid_batches += 1
            
            # 全量评估排序指标计算
            # 对整个批次获取编码器输出
            encoder_outputs = model.encoder(source_ids)  # [B, L, D]
            user_embeddings = encoder_outputs[:, -1, :]  # [B, D] - 取每个序列的最后一个位置作为用户表示
            
            # 从标签中提取真实目标物品ID
            target_item_ids = []
            for i in range(source_ids.size(0)):
                valid_positions = (labels[i] != pad_token_id).nonzero(as_tuple=True)[0]
                if len(valid_positions) == 0:
                    # 如果没有有效标签，用0填充（后面会跳过）
                    target_item_ids.append(0)
                else:
                    # 取第一个非padding的标签作为目标
                    target_item_ids.append(labels[i, valid_positions[0]].item())
            
            target_item_ids = torch.tensor(target_item_ids, device=device)
            
            # 使用统一的评估指标计算
            if num_candidates is not None and num_candidates > 0:
                # 使用采样评估（随机抽取n个负样本+1个正样本）
                batch_size = user_embeddings.size(0)
                
                # 为每个用户选择随机负样本
                hr_batch_sum, ndcg_batch_sum = 0.0, 0.0
                valid_batch_samples = 0
                
                for i in range(batch_size):
                    target_id = target_item_ids[i].item()
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
                    # 注意：这里我们知道目标ID就是all_candidate_ids[0]
                    target_position = (all_candidate_ids == target_id).nonzero(as_tuple=True)[0].item()
                    target_id_tensor = torch.tensor([target_id], device=device)
                    
                    # 使用单个用户的计算函数
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
                    target_item_ids,
                    k=top_k
                )
            
            hr_total += hr * source_ids.size(0)
            ndcg_total += ndcg * source_ids.size(0)
            total_samples += source_ids.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        
        # 清理GPU显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算最终指标
    avg_loss = total_loss_tokens / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    avg_gate_weights = total_gate_weights / total_valid_batches if total_valid_batches > 0 else None
    
    # 排序指标 - 完全对齐HSTU计算方式
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = ndcg_total / total_samples if total_samples > 0 else 0.0
    
    result = {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
        'val_hr': avg_hr,          # 与HSTU完全对齐的HR@K
        'val_ndcg': avg_ndcg,      # 与HSTU完全对齐的NDCG@K
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
        
        # 根据评估模式设置不同的进度条描述
        if num_candidates is not None and num_candidates > 0:
            progress_bar = tqdm(test_loader, desc=f"Test Set Evaluation - Sampled Eval ({num_candidates} candidates)")
        else:
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
    
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = ndcg_total / total_samples if total_samples > 0 else 0.0
    
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
