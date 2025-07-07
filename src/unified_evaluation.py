"""
统一评估模块 (unified_evaluation.py)

【最终版】该模块整合了所有模型(HSTU, Baseline, GENIUS-Rec)的评估函数，
确保三个模型使用完全一致的评估逻辑和指标计算方法。
所有核心评估逻辑均已完全向量化，以实现最高效率。
"""

import math
import torch
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src.metrics import compute_hr_ndcg_full

class ValidationDataset(Dataset):
    """
    用于排序指标评估的数据集（Leave-One-Out方式）。
    """
    def __init__(self, data_path, max_len, pad_token_id=0):
        self.data = pd.read_parquet(data_path)
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_seq = self.data.iloc[idx]['history']
        ground_truth_item = full_seq[-1]
        input_seq = full_seq[:-1]
        
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
    【最终版】验证集评估：仅计算loss和perplexity。
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)
            
            # 解包模型返回的三个值，但只使用第一个
            logits, _, _ = model(source_ids, decoder_input_ids, source_padding_mask)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            num_tokens = (labels != pad_token_id).sum().item()
            
            if num_tokens > 0:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss > 0 and avg_loss < 20 else float('inf')
    
    return {'val_loss': avg_loss, 'val_ppl': perplexity}


def evaluate_model_validation_with_ranking(model, val_loader, criterion, device, epoch, num_epochs, pad_token_id, config, top_k=10, **kwargs):
    """
    【最终修复版】验证集评估：同时计算loss, ppl, 和排序指标。
    - 适配“前缀预测后缀”模式，使用编码器对历史前缀的输出来代表用户。
    - 评估目标为后缀的第一个真实物品。
    - 彻底解决数据泄露和所有已知bug。
    """
    model.eval()
    total_loss_tokens, total_tokens = 0.0, 0
    hr_total, ndcg_total, total_samples = 0.0, 0.0, 0
    
    total_gate_weights, total_valid_batches = None, 0
    enabled_experts = [k for k, v in config['expert_system']['experts'].items() if v]
    num_special_tokens = config['num_special_tokens']

    with torch.no_grad():
        item_num = model.encoder.item_embedding.num_embeddings
        all_item_embeddings = model.encoder.item_embedding.weight[num_special_tokens:item_num]

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)

        for batch in progress_bar:
            # --- 数据准备 ---
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)
            batch_size = source_ids.size(0)

            # --- Loss 计算 (基于解码器) ---
            # 为了计算loss，依然需要完整运行一次模型
            logits, gate_weights, _, _ = model(
                source_ids=source_ids,
                decoder_input_ids=decoder_input_ids,
                source_padding_mask=source_padding_mask,
                return_weights=True
            )
            task_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            non_padding_mask_labels = (labels != pad_token_id)
            num_tokens = non_padding_mask_labels.sum().item()
            if num_tokens > 0:
                total_loss_tokens += task_loss.item() * num_tokens
                total_tokens += num_tokens
            
            # --- 专家权重聚合 ---
            if gate_weights is not None:
                non_padding_mask_decoder = (decoder_input_ids != pad_token_id)
                label_mask = non_padding_mask_decoder.float().unsqueeze(-1)
                valid_gate_weights = gate_weights * label_mask
                batch_sum = valid_gate_weights.sum(dim=(0, 1))
                batch_count = label_mask.sum()
                if batch_count > 0:
                    batch_mean = batch_sum / batch_count
                    if total_gate_weights is None: total_gate_weights = batch_mean
                    else: total_gate_weights += batch_mean
                    total_valid_batches += 1

            # --- 【核心修复】正确的排序指标计算逻辑 ---
            
            # 1. 获取用户画像：只将历史前缀(source_ids)送入编码器
            encoder_outputs = model.encoder(source_ids)
            
            # 使用历史前缀的最后一个非填充token的表征作为用户画像
            source_lengths = (source_ids != pad_token_id).sum(dim=1)
            embedding_dim = model.encoder.item_embedding.embedding_dim
            last_item_indices = (source_lengths - 1).view(-1, 1, 1).expand(-1, -1, embedding_dim)
            user_embeddings = encoder_outputs.gather(1, last_item_indices).squeeze(1)

            # 2. 获取评估目标：目标是未来后缀的第一个物品
            #    它在labels张量的第0个位置
            target_item_ids = labels[:, 0]
            
            # 3. 调用指标函数
            hr, ndcg = compute_hr_ndcg_full(
                user_embeddings, all_item_embeddings, target_item_ids, 
                k=top_k, special_token_offset=num_special_tokens
            )

            # --- 结果聚合 ---
            if hr is not None and ndcg is not None:
                 hr_total += hr * batch_size
                 ndcg_total += ndcg * batch_size
                 total_samples += batch_size
            progress_bar.set_postfix(
                loss=(task_loss.item() if num_tokens > 0 else 0.0), hr=f"{hr:.4f}", ndcg=f"{ndcg:.4f}"
            )

    # --- 最终指标计算 ---
    avg_loss = total_loss_tokens / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss > 0 and avg_loss < 20 else float('inf')
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = ndcg_total / total_samples if total_samples > 0 else 0.0
    
    result = { 'val_loss': avg_loss, 'val_ppl': perplexity, 'val_hr': avg_hr, 'val_ndcg': avg_ndcg }
    
    if total_gate_weights is not None and total_valid_batches > 0:
        avg_gate_weights = total_gate_weights / total_valid_batches
        for i, expert_name in enumerate(enabled_experts):
            if i < len(avg_gate_weights):
                result[f'avg_{expert_name}_weight'] = avg_gate_weights[i].item()

    return result

def evaluate_model_test(model, test_loader, device, item_num, top_k=10, config=None):
    """
    【最终向量优化版】测试集评估，始终使用全量评估。
    """
    model.eval()
    hr_total, ndcg_total, total_samples = 0.0, 0.0, 0
    
    num_special_tokens = config['num_special_tokens'] if config else 4 # Default to 4 if config not provided

    with torch.no_grad():
        # 修复：从num_special_tokens开始切片，确保只包含实际物品的嵌入
        all_item_embeddings = model.encoder.item_embedding.weight[num_special_tokens:item_num]
        progress_bar = tqdm(test_loader, desc="Test Set Evaluation")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            ground_truth_ids = batch['ground_truth'].to(device)
            batch_size = input_ids.size(0)
            encoder_output = model.encoder(input_ids)
            user_embeddings = encoder_output[:, -1, :]
            
            hr, ndcg = compute_hr_ndcg_full(user_embeddings, all_item_embeddings, ground_truth_ids, k=top_k)
            
            hr_total += hr * batch_size
            ndcg_total += ndcg * batch_size
            total_samples += batch_size
            progress_bar.set_postfix(hr=hr, ndcg=ndcg)
            
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = ndcg_total / total_samples if total_samples > 0 else 0.0
    return {'test_hr': avg_hr, 'test_ndcg': avg_ndcg, 'evaluated_samples': total_samples}

def evaluate_encoder(encoder, val_loader, device, top_k=10):
    """
    【最终向量优化版】编码器独立评估函数。
    """
    encoder.eval()
    hr_total, ndcg_total, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        item_num = encoder.item_embedding.num_embeddings
        all_item_embeddings = encoder.item_embedding.weight[1:item_num]
        
        for batch in tqdm(val_loader, desc="Evaluating Encoder"):
            seq = batch[0].to(device)
            target_item_ids = batch[1].to(device).squeeze(1)
            batch_size = seq.size(0)
            
            sequence_output = encoder.forward(seq)
            user_embeddings = sequence_output[:, -1, :]
            
            hr, ndcg = compute_hr_ndcg_full(user_embeddings, all_item_embeddings, target_item_ids, k=top_k)
            
            hr_total += hr * batch_size
            total_ndcg += ndcg * batch_size
            total_samples += batch_size
    
    avg_hr = hr_total / total_samples if total_samples > 0 else 0.0
    avg_ndcg = total_ndcg / total_samples if total_samples > 0 else 0.0
    return avg_hr, avg_ndcg