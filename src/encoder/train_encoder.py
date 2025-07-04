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


from src.config import get_config
from src.encoder.encoder import Hstu  
from src.encoder.dataset4encoder import RecDataset 

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
    全量评估：计算用户嵌入与所有物品嵌入的相似度（与HSTU原始实现完全对齐）
    
    注意：此函数使用完整排序以确保与原始HSTU的评估结果一致，
    同时优化了实现，提高了运行效率。
    """
    batch_size = user_embeddings.size(0)
    
    # L2归一化（对齐官方实现）
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算用户与所有物品的相似度 [B, num_items]
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 完全排序获取排名（降序）- 与HSTU原始实现对齐
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    hr_list, ndcg_list = [], []
    valid_samples = 0
    
    # 批量处理但保持与原始HSTU相同的逻辑
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0:  # 跳过无效样本
            continue
            
        valid_samples += 1
        # 修复：由于物品ID从4开始，需要减去4，使下标从0开始
        target_idx = target_id - 4  # 将ID转换为索引
        
        # 找出目标物品的排名位置
        rank_positions = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0]
        
        if len(rank_positions) > 0:
            rank = rank_positions[0].item() + 1  # 排名从1开始
            
            if rank <= k:
                hr_list.append(1.0)
                ndcg_list.append(1.0 / np.log2(rank + 1))
            else:
                hr_list.append(0.0)
                ndcg_list.append(0.0)
        else:
            hr_list.append(0.0)
            ndcg_list.append(0.0)
    
    # 计算平均值（与HSTU原始实现相同）
    hr = sum(hr_list) / batch_size if batch_size > 0 else 0.0
    ndcg = sum(ndcg_list) / batch_size if batch_size > 0 else 0.0
    
    return hr, ndcg

class SampledSoftmaxLoss(nn.Module):
    """完全对齐官方的Sampled Softmax损失 (从baseline/train_baseline.py同步)"""
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
        
        # 确保logits在合理范围内 - 增加数值稳定性
        pos_logits = torch.clamp(pos_logits, min=-10.0, max=10.0)
        
        # 负采样
        # 修复：避开所有特殊标记(0,1,2,3)，从4开始采样
        neg_indices = torch.randint(4, num_items + 4,
                                   (valid_output.size(0), self.num_negatives),
                                   device=output_embeddings.device)
        
        # 去除与正样本相同的负样本
        neg_mask = neg_indices != valid_targets.unsqueeze(1)
        neg_indices = torch.where(neg_mask, neg_indices, 
                                 torch.randint(4, num_items + 4, neg_indices.shape, 
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

def get_optimal_eval_batch_size(item_num, embedding_dim, device_vram_gb=None):
    """
    根据模型大小和GPU显存动态确定最优评估批次大小
    
    Args:
        item_num: 物品数量
        embedding_dim: 嵌入维度
        device_vram_gb: GPU显存大小(GB)，如果不提供则自动检测
        
    Returns:
        推荐的评估批次大小
    """
    # 估算单个物品嵌入所需字节
    bytes_per_item = 4  # float32占用4字节
    
    # 估算物品嵌入矩阵大小(GB)
    item_embeddings_gb = (item_num * embedding_dim * bytes_per_item) / (1024**3)
    
    # 如果未提供显存大小，尝试自动检测
    if device_vram_gb is None and torch.cuda.is_available():
        device_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        # 默认值(如果无法检测)
        device_vram_gb = 8
    
    # 计算可用显存(减去基本开销和物品嵌入)
    available_vram_gb = max(1, device_vram_gb * 0.6 - item_embeddings_gb)
    
    # 估算每个样本的内存占用
    sample_vram_gb = embedding_dim * bytes_per_item * 3 / (1024**3)  # 用户嵌入+序列嵌入+临时计算
    
    # 计算可容纳的最大批次大小(受制于显存)
    max_batch_size = int(available_vram_gb / sample_vram_gb)
    
    # 设置合理的上下限
    min_batch_size = 64
    max_reasonable_batch_size = 1024
    
    optimal_batch_size = max(min_batch_size, min(max_batch_size, max_reasonable_batch_size))
    
    # 确保是8的倍数(提高GPU利用率)
    optimal_batch_size = (optimal_batch_size // 8) * 8
    
    return optimal_batch_size

def train():
    config = get_config()
    
    config['data']['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
    config['data']['log_dir'].mkdir(parents=True, exist_ok=True)
    
    log_file = config['data']['log_dir'] / config['pretrain']['log_file']
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    
    set_seed(config['seed'])
    device = torch.device(config['device'])
    logging.info(f"配置加载完成. 使用设备: {device}")

    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
        item_num = id_maps['num_items']

    train_dataset = RecDataset(config['data']['train_file'], config['data']['id_maps_file'], config['encoder_model']['max_len'], mode='train')
    val_dataset = RecDataset(config['data']['validation_file'], config['data']['id_maps_file'], config['encoder_model']['max_len'], mode='validation')
    test_dataset = RecDataset(config['data']['test_file'], config['data']['id_maps_file'], config['encoder_model']['max_len'], mode='test')

    # 设置训练批次大小
    train_batch_size = config['pretrain']['batch_size']
    
    # 计算并设置最优评估批次大小
    embedding_dim = config['encoder_model']['embedding_dim']
    eval_batch_size = get_optimal_eval_batch_size(item_num, embedding_dim)
    config['pretrain']['eval_batch_size'] = eval_batch_size
    
    logging.info(f"自动配置评估批次大小: {eval_batch_size} (训练批次: {train_batch_size})")

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=config['pretrain']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, num_workers=config['pretrain']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False, num_workers=config['pretrain']['num_workers'])
    logging.info(f"数据集加载完成. 物品总数: {item_num}")

    model = Hstu(
        item_num=item_num + 4 ,
        embedding_dim=config['encoder_model']['embedding_dim'],
        linear_hidden_dim=config['encoder_model']['linear_hidden_dim'],
        attention_dim=config['encoder_model']['attention_dim'],
        num_heads=config['encoder_model']['num_heads'],
        num_layers=config['encoder_model']['num_layers'],
        max_len=config['encoder_model']['max_len'],
        dropout=config['encoder_model']['dropout'],
        pad_token_id=config['pad_token_id']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['pretrain']['learning_rate'], 
        weight_decay=config['pretrain']['weight_decay'],
        betas=(0.9, 0.98)
    )
    
    criterion = SampledSoftmaxLoss(
        num_negatives=config['pretrain']['num_neg_samples'], 
        temperature=config['pretrain']['temperature']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)

    # 【新增】断点续传的初始化逻辑
    model_path = config['data']['checkpoint_dir'] / 'hstu_official_aligned_best.pth'
    start_epoch = 0
    best_ndcg = 0.0
    patience_counter = 0

    if os.path.exists(model_path):
        logging.info(f"发现检查点: {model_path}，正在尝试加载...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查是否为新格式的checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态（如果存在且你需要继续使用它）
            if checkpoint.get('optimizer_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("成功加载模型和优化器状态。")
            else:
                logging.warning("检查点中未找到优化器状态，优化器将从头开始。")

            start_epoch = checkpoint.get('epoch', -1) + 1
            best_ndcg = checkpoint.get('best_ndcg', 0.0)
            logging.info(f"将从 Epoch {start_epoch} 继续训练。已记录的最佳NDCG为 {best_ndcg:.4f}")
        else:
            # 兼容只存了模型参数的旧格式ckpt
            model.load_state_dict(checkpoint)
            logging.info("成功加载旧格式的模型参数。将从 Epoch 0 开始训练。")

    logging.info("--- 开始官方风格的训练 (支持断点续传) ---")

    # 【修改】训练循环的起始点
    for epoch in range(start_epoch, config['pretrain']['num_epochs']):
        logging.info(f"--- Epoch {epoch}/{config['pretrain']['num_epochs']-1} ---")
        
        model.train()
        total_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
        # train_encoder.py (正确的训练循环部分)

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
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch} 平均训练损失: {avg_train_loss:.4f}")

        model.eval()
        total_hr, total_ndcg = 0.0, 0.0
        total_samples = 0  # 记录有效样本数量
        
        with torch.no_grad():
            # 预计算所有物品的嵌入，提高效率
            logging.info("预计算物品嵌入向量...")
            # 修复：使用从4开始的物品ID（跳过所有特殊标记0,1,2,3）
            all_item_ids = torch.arange(4, item_num, device=device)
            all_item_embeddings = model.item_embedding(all_item_ids)
            # 预先L2归一化，避免重复计算
            all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
            
            # 使用更大的批次进行评估，减少循环次数
            eval_batch_size = config['pretrain'].get('eval_batch_size', config['pretrain']['batch_size'] * 4)
            
            # 使用原始验证数据加载器来保持与HSTU和baseline一致的评估逻辑
            # 但增加批次大小以提高效率
            eval_batch_size = config['pretrain'].get('eval_batch_size', config['pretrain']['batch_size'] * 2)
            
            # 重新创建数据加载器，保持原始数据顺序
            eval_dataloader = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,  # 不打乱顺序，与原始实现一致
                num_workers=config['pretrain']['num_workers'],
                pin_memory=True  # 启用内存固定，加速数据传输
            )
            
            eval_bar = tqdm(eval_dataloader, desc=f"全量评估 Epoch {epoch} (与HSTU原始实现对齐)")
            for seq, pos, neg in eval_bar:
                # 与原始HSTU一致的数据加载方式
                seq = seq.to(device)
                target_item_ids = pos.to(device).squeeze(1)
                
                batch_size = seq.size(0)
                
                # 获取用户嵌入
                sequence_output = model.forward(seq)
                user_embeddings = sequence_output[:, -1, :]  # 取最后一个位置的嵌入作为用户表示
                
                # 完全对齐HSTU的评估函数
                hr, ndcg = get_metrics_full_eval(user_embeddings, all_item_embeddings, target_item_ids, k=config['evaluation']['top_k'])
                
                # 累积指标
                total_hr += hr * batch_size
                total_ndcg += ndcg * batch_size
                total_samples += batch_size
                
                # 实时显示当前批次的指标
                eval_bar.set_postfix(HR=f"{hr:.4f}", NDCG=f"{ndcg:.4f}")

        # 计算总体平均值
        avg_hr = total_hr / total_samples if total_samples > 0 else 0
        avg_ndcg = total_ndcg / total_samples if total_samples > 0 else 0
        logging.info(f"验证集（全量评估） ==> HR@{config['evaluation']['top_k']}: {avg_hr:.4f}, NDCG@{config['evaluation']['top_k']}: {avg_ndcg:.4f}, 样本数: {total_samples}")
        
        scheduler.step(avg_ndcg)

        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            patience_counter = 0
            
            logging.info(f"发现新的最佳模型! NDCG: {best_ndcg:.4f}. 保存至 {model_path}")
            # 【修改】使用新的格式保存检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ndcg': best_ndcg,
                'config': config # (可选) 保存当时的配置
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= config['pretrain']['early_stopping_patience']:
                logging.info(f"触发 EarlyStopping (patience={patience_counter}). 停止训练.")
                break

    logging.info("--- 训练完成 ---")
    
    logging.info("--- 使用最佳模型进行测试（全量评估） ---")
    # 【修改】测试时也需要加载新格式的checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        total_hr, total_ndcg = 0.0, 0.0
        total_samples = 0
        
        with torch.no_grad():
            # 预计算所有物品的嵌入，提高效率
            logging.info("预计算测试集评估所需的物品嵌入向量...")
            # 修复：使用从4开始的物品ID（跳过所有特殊标记0,1,2,3）
            all_item_ids = torch.arange(4, item_num, device=device)
            all_item_embeddings = model.item_embedding(all_item_ids)
            # 预先L2归一化，避免重复计算
            all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
            
            # 使用原始测试集加载器来保持与HSTU和baseline一致的评估逻辑
            # 但允许增加批次大小以提高效率
            test_batch_size = config['pretrain'].get('eval_batch_size', config['pretrain']['batch_size'] * 2)
            
            # 确保与原始实现一致的数据加载器配置
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,  # 不打乱顺序，与原始实现一致
                num_workers=config['pretrain']['num_workers'],
                pin_memory=True
            )
            
            test_bar = tqdm(test_dataloader, desc="测试中（全量评估 - 与HSTU原始实现对齐）")
            for seq, pos, neg in test_bar:
                # 与原始HSTU一致的数据处理方式
                seq = seq.to(device)
                target_item_ids = pos.to(device).squeeze(1)
                
                batch_size = seq.size(0)
                
                # 获取用户嵌入
                sequence_output = model.forward(seq)
                user_embeddings = sequence_output[:, -1, :]
                
                # 使用完全对齐HSTU的评估函数
                hr, ndcg = get_metrics_full_eval(user_embeddings, all_item_embeddings, target_item_ids, k=config['evaluation']['top_k'])
                
                # 累积指标
                total_hr += hr * batch_size
                total_ndcg += ndcg * batch_size
                total_samples += batch_size
                
                # 实时显示当前批次的指标
                test_bar.set_postfix(HR=f"{hr:.4f}", NDCG=f"{ndcg:.4f}")
            
        # 计算总体平均值
        avg_hr = total_hr / total_samples if total_samples > 0 else 0
        avg_ndcg = total_ndcg / total_samples if total_samples > 0 else 0
        logging.info(f"测试集结果 ==> HR@{config['evaluation']['top_k']}: {avg_hr:.4f}, NDCG@{config['evaluation']['top_k']}: {avg_ndcg:.4f}, 样本数: {total_samples}")
    else:
        logging.warning("未找到最佳模型文件，跳过测试。")

if __name__ == '__main__':
    train()