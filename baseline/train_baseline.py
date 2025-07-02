# baseline/train_baseline.py (完全对齐HSTU的Baseline实现)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import os
from tqdm import tqdm
import pickle
import logging
import numpy as np

# 使用主项目的配置和数据集
from src.config import get_config
from src.encoder.dataset4encoder import RecDataset 
from baseline.model import BaselineTransformer 

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_metrics_full_eval(user_embeddings, all_item_embeddings, target_item_ids, k=10):
    """
    全量评估：计算用户嵌入与所有物品嵌入的相似度（完全对齐HSTU评估）
    Args:
        user_embeddings: [B, D] 用户/序列的嵌入表示
        all_item_embeddings: [num_items, D] 所有物品的嵌入（不包含padding）
        target_item_ids: [B] 目标物品ID
        k: Top-K
    Returns:
        hr: Hit Rate@k
        ndcg: NDCG@k
    """
    batch_size = user_embeddings.size(0)
    
    # L2归一化（对齐HSTU实现）
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算用户与所有物品的相似度 [B, num_items]
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 排序获取排名（降序）
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    # 为每个样本找到目标物品的排名
    hr_count = 0
    ndcg_sum = 0.0
    
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0:  # 跳过padding
            continue
            
        # 找到目标物品在排序中的位置
        target_idx = target_id - 1
        sorted_items = sorted_indices[i]
        target_rank_positions = (sorted_items == target_idx).nonzero(as_tuple=True)[0]
        
        if len(target_rank_positions) > 0:
            rank = target_rank_positions[0].item() + 1  # 排名从1开始
            
            # 计算HR@k
            if rank <= k:
                hr_count += 1
                # 计算NDCG@k
                ndcg_sum += 1.0 / np.log2(rank + 1)
    
    hr = hr_count / batch_size
    ndcg = ndcg_sum / batch_size
    
    return hr, ndcg

class SampledSoftmaxLoss(nn.Module):
    """完全对齐HSTU的Sampled Softmax损失"""
    def __init__(self, num_negatives=100, temperature=0.05):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature
    
    def forward(self, output_embeddings, target_ids, all_item_embeddings, supervision_weights):
        """
        Args:
            output_embeddings: [B, L-1, D] 模型输出(去掉最后一个位置)
            target_ids: [B, L-1] 目标物品ID(去掉第一个位置)
            all_item_embeddings: [num_items+1, D] 所有物品的嵌入
            supervision_weights: [B, L-1] 有效位置的权重
        """
        batch_size, seq_len_minus_1, embed_dim = output_embeddings.shape
        num_items = all_item_embeddings.size(0) - 1  # 去掉pad_token
        
        # 展平处理，只保留有效位置
        flat_output = output_embeddings.reshape(-1, embed_dim)  # [B*(L-1), D]
        flat_targets = target_ids.reshape(-1)  # [B*(L-1)]
        flat_weights = supervision_weights.reshape(-1)  # [B*(L-1)]
        
        # 过滤掉padding位置
        valid_mask = flat_weights > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=output_embeddings.device)
            
        valid_output = flat_output[valid_mask]  # [N_valid, D]
        valid_targets = flat_targets[valid_mask]  # [N_valid]
        valid_weights = flat_weights[valid_mask]  # [N_valid]
        
        # L2归一化
        valid_output = F.normalize(valid_output, p=2, dim=1)
        norm_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
        
        # 正样本嵌入
        pos_embeddings = norm_item_embeddings[valid_targets]  # [N_valid, D]
        pos_logits = (valid_output * pos_embeddings).sum(dim=1, keepdim=True)  # [N_valid, 1]
        
        # 确保logits在合理范围内
        pos_logits = torch.clamp(pos_logits, min=-10.0, max=10.0)
        
        # 负采样
        neg_indices = torch.randint(1, num_items + 1,  # 避免采样到pad_token(0)
                                   (valid_output.size(0), self.num_negatives),
                                   device=output_embeddings.device)
        
        # 去除与正样本相同的负样本
        neg_mask = neg_indices != valid_targets.unsqueeze(1)
        neg_indices = torch.where(neg_mask, neg_indices, 
                                 torch.randint(1, num_items + 1, neg_indices.shape, 
                                             device=neg_indices.device))
        
        neg_embeddings = norm_item_embeddings[neg_indices]  # [N_valid, num_neg, D]
        neg_logits = torch.bmm(valid_output.unsqueeze(1), 
                              neg_embeddings.transpose(1, 2)).squeeze(1)  # [N_valid, num_neg]
        
        # 确保logits在合理范围内
        neg_logits = torch.clamp(neg_logits, min=-10.0, max=10.0)
        
        # 合并正负样本logits，应用温度
        all_logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature  # [N_valid, 1+num_neg]
        
        # 再次确保logits稳定
        all_logits = torch.clamp(all_logits, min=-20.0, max=20.0)
        
        # 标签（正样本在位置0）
        labels = torch.zeros(valid_output.size(0), dtype=torch.long, 
                           device=output_embeddings.device)
        
        # 计算交叉熵损失，加权平均
        loss_per_sample = F.cross_entropy(all_logits, labels, reduction='none')
        weighted_loss = (loss_per_sample * valid_weights).sum() / valid_weights.sum()
        
        return weighted_loss



def train():
    config = get_config()
    
    config['data']['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    config['data']['log_dir'].mkdir(parents=True, exist_ok=True)
    
    log_file = config['data']['log_dir'] / 'baseline_training.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    
    set_seed(config['seed'])
    device = torch.device(config['device'])
    logging.info(f"配置加载完成. 使用设备: {device}")

    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
        item_num = id_maps['num_items']

    # 数据加载 - 完全使用HSTU的数据管道
    train_dataset = RecDataset(config['data']['train_file'], config['data']['id_maps_file'], config['encoder_model']['max_len'], mode='train')
    val_dataset = RecDataset(config['data']['validation_file'], config['data']['id_maps_file'], config['encoder_model']['max_len'], mode='validation')
    test_dataset = RecDataset(config['data']['test_file'], config['data']['id_maps_file'], config['encoder_model']['max_len'], mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config['pretrain']['batch_size'], shuffle=True, num_workers=config['pretrain']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['pretrain']['batch_size'], shuffle=False, num_workers=config['pretrain']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['pretrain']['batch_size'], shuffle=False, num_workers=config['pretrain']['num_workers'])
    logging.info(f"数据集加载完成. 物品总数: {item_num}")

    # 模型实例化 - Baseline Transformer，使用与HSTU完全相同的超参数
    model = BaselineTransformer(
        item_num=item_num,
        embedding_dim=config['encoder_model']['embedding_dim'],  # 64
        max_len=config['encoder_model']['max_len'],              # 50
        num_layers=config['encoder_model']['num_layers'],        # 4
        num_heads=config['encoder_model']['num_heads'],          # 4
        dropout=config['encoder_model']['dropout'],              # 0.1
        pad_token_id=config['pad_token_id']                      # 0
    ).to(device)

    logging.info(f"Baseline模型参数: embedding_dim={config['encoder_model']['embedding_dim']}, "
                f"num_layers={config['encoder_model']['num_layers']}, "
                f"num_heads={config['encoder_model']['num_heads']}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Baseline模型总参数量: {total_params:,}, 可训练参数: {trainable_params:,}")

    # 使用与HSTU相同但更保守的优化器设置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['pretrain']['learning_rate'] * 0.1,  # 减小学习率10倍防止梯度爆炸
        weight_decay=config['pretrain']['weight_decay'],
        betas=(0.9, 0.98),  # 与HSTU保持一致
        eps=1e-8  # 增加数值稳定性
    )
    
    logging.info(f"优化器设置: lr={config['pretrain']['learning_rate'] * 0.1}, weight_decay={config['pretrain']['weight_decay']}")
    
    # 使用与HSTU相同的Sampled Softmax损失
    criterion = SampledSoftmaxLoss(
        num_negatives=config['pretrain']['num_neg_samples'], 
        temperature=config['pretrain']['temperature']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    best_ndcg = 0.0
    patience_counter = 0
    logging.info("--- 开始Baseline训练 (对齐HSTU训练流程) ---")
    
    # 添加一个简单的梯度检查
    logging.info("执行初始梯度检查...")
    model.train()
    sample_batch = next(iter(train_loader))
    input_ids, target_ids = sample_batch[0][:2].to(device), sample_batch[1][:2].to(device)
    
    logging.info(f"样本输入形状: {input_ids.shape}, 目标形状: {target_ids.shape}")
    logging.info(f"输入ID范围: [{input_ids.min().item()}, {input_ids.max().item()}]")
    logging.info(f"目标ID范围: [{target_ids.min().item()}, {target_ids.max().item()}]")
    
    # 检查模型输出
    sequence_output = model(input_ids)
    logging.info(f"模型输出形状: {sequence_output.shape}")
    logging.info(f"输出范围: [{sequence_output.min().item():.4f}, {sequence_output.max().item():.4f}]")
    logging.info(f"输出是否包含NaN: {torch.isnan(sequence_output).any().item()}")
    
    # 检查嵌入权重
    embed_weights = model.item_embedding.weight
    logging.info(f"嵌入权重形状: {embed_weights.shape}")
    logging.info(f"嵌入权重范围: [{embed_weights.min().item():.4f}, {embed_weights.max().item():.4f}]")
    logging.info(f"嵌入权重是否包含NaN: {torch.isnan(embed_weights).any().item()}")
    
    supervision_weights = (target_ids != config['pad_token_id']).float()
    logging.info(f"监督权重形状: {supervision_weights.shape}")
    logging.info(f"有效位置数量: {supervision_weights.sum().item()}")
    
    try:
        test_loss = criterion(sequence_output, target_ids, model.item_embedding.weight, supervision_weights)
        logging.info(f"初始测试损失: {test_loss.item():.4f}")
    except Exception as e:
        logging.error(f"损失计算出错: {e}")
        logging.error(f"sequence_output形状: {sequence_output.shape}")
        logging.error(f"target_ids形状: {target_ids.shape}")
        logging.error(f"embed_weights形状: {model.item_embedding.weight.shape}")
        logging.error(f"supervision_weights形状: {supervision_weights.shape}")
        return
    if torch.isnan(test_loss) or torch.isinf(test_loss):
        logging.error("初始损失就是NaN/Inf! 请检查模型和数据！")
        return
    
    # 检查初始梯度
    test_loss.backward()
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
    
    logging.info(f"初始最大梯度范数: {max_grad:.4f}")
    if max_grad > 100:
        logging.warning(f"初始梯度过大: {max_grad:.4f}, 建议进一步调整学习率或初始化")
    
    # 清空梯度开始正式训练
    model.zero_grad()

    for epoch in range(config['pretrain']['num_epochs']):
        logging.info(f"--- Epoch {epoch+1}/{config['pretrain']['num_epochs']} ---")
        
        # 训练循环 - N-to-N方式，每个位置预测下一个（与HSTU完全一致）
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        
        for input_ids, target_ids in train_bar:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            
            # 1. 直接将dataset产出的input_ids喂给模型
            sequence_output = model(input_ids)
            
            # 2. 【关键】不再对输出和目标进行任何切片操作
            output_embeddings = sequence_output
            supervision_ids = target_ids
            
            # 3. 权重掩码会自然地处理padding部分
            supervision_weights = (supervision_ids != config['pad_token_id']).float()
            
            # 4. 将对齐好的数据送入损失函数
            loss = criterion(output_embeddings, supervision_ids, 
                        model.item_embedding.weight, supervision_weights)
            
            # 检查损失是否为NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"训练过程中发现NaN/Inf损失! Batch {len(train_bar)}, Loss: {loss.item()}")
                # 跳过这个batch
                continue
            
            loss.backward()
            
            # 检查梯度是否过大
            max_grad = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_grad = max(max_grad, grad_norm)
            
            if max_grad > 10.0:  # 如果梯度过大，记录警告
                logging.warning(f"梯度过大: {max_grad:.4f}")
            
            # 更强的梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 从1.0减小到0.5
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")

        # 验证循环 - 全量评估（完全对齐HSTU）
        model.eval()
        total_hr, total_ndcg = 0.0, 0.0
        with torch.no_grad():
            # 获取所有物品的嵌入（用于全量评估）
            all_item_ids = torch.arange(1, item_num + 1, device=device)  # 排除padding物品(0)
            all_item_embeddings = model.item_embedding(all_item_ids)  # [num_items, D]
            
            eval_bar = tqdm(val_loader, desc=f"全量评估 Epoch {epoch+1}")
            for seq, pos, neg in eval_bar:
                seq = seq.to(device)
                target_item_ids = pos.to(device).squeeze(1)  # [B] 目标物品ID
                
                # 获取序列的最终表示（用于预测下一个物品）
                sequence_output = model.forward(seq)  # [B, L, D]
                user_embeddings = sequence_output[:, -1, :]  # [B, D] 取最后一个位置
                
                # 进行全量评估
                hr, ndcg = get_metrics_full_eval(user_embeddings, all_item_embeddings, target_item_ids, k=config['evaluation']['top_k'])
                total_hr += hr
                total_ndcg += ndcg

        avg_hr = total_hr / len(val_loader)
        avg_ndcg = total_ndcg / len(val_loader)
        logging.info(f"验证集（全量评估） ==> HR@{config['evaluation']['top_k']}: {avg_hr:.4f}, NDCG@{config['evaluation']['top_k']}: {avg_ndcg:.4f}")
        
        scheduler.step(avg_ndcg)

        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            patience_counter = 0
            model_path = config['data']['checkpoint_dir'] / 'baseline_transformer_best.pth'
            logging.info(f"发现新的最佳Baseline模型! NDCG: {best_ndcg:.4f}. 保存至 {model_path}")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= config['pretrain']['early_stopping_patience']:
                logging.info("触发 EarlyStopping. 停止训练.")
                break

    logging.info("--- Baseline训练完成 ---")
    
    # 测试部分
    logging.info("--- 使用最佳Baseline模型进行测试（全量评估） ---")
    best_model_path = config['data']['checkpoint_dir'] / 'baseline_transformer_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        total_hr, total_ndcg = 0.0, 0.0
        with torch.no_grad():
            # 获取所有物品的嵌入
            all_item_ids = torch.arange(1, item_num + 1, device=device)
            all_item_embeddings = model.item_embedding(all_item_ids)
            
            test_bar = tqdm(test_loader, desc="测试中（全量评估）")
            for seq, pos, neg in test_bar:
                seq = seq.to(device)
                target_item_ids = pos.to(device).squeeze(1)
                
                sequence_output = model.forward(seq)  # [B, L, D]
                user_embeddings = sequence_output[:, -1, :]  # [B, D] 取最后一个位置
                hr, ndcg = get_metrics_full_eval(user_embeddings, all_item_embeddings, target_item_ids, k=config['evaluation']['top_k'])
                total_hr += hr
                total_ndcg += ndcg

        avg_hr = total_hr / len(test_loader)
        avg_ndcg = total_ndcg / len(test_loader)
        logging.info(f"Baseline测试集结果（全量评估） ==> HR@{config['evaluation']['top_k']}: {avg_hr:.4f}, NDCG@{config['evaluation']['top_k']}: {avg_ndcg:.4f}")
        logging.info(f"*** Baseline最终结果 ==> HR@{config['evaluation']['top_k']}: {avg_hr:.4f}, NDCG@{config['evaluation']['top_k']}: {avg_ndcg:.4f} ***")
    else:
        logging.warning("未找到最佳Baseline模型文件，跳过测试。")

if __name__ == '__main__':
    train()
    print("Baseline训练和测试完成。请查看日志文件以获取详细信息。")