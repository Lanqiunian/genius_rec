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
    全量评估：计算用户嵌入与所有物品嵌入的相似度（对齐官方HSTU）
    (此函数无需修改)
    """
    batch_size = user_embeddings.size(0)
    
    # L2归一化（对齐官方实现）
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算用户与所有物品的相似度 [B, num_items]
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 排序获取排名（降序）
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    hr_count = 0
    ndcg_sum = 0.0
    
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0:
            continue
            
        target_idx = target_id - 1
        sorted_items = sorted_indices[i]
        target_rank_positions = (sorted_items == target_idx).nonzero(as_tuple=True)[0]
        
        if len(target_rank_positions) > 0:
            rank = target_rank_positions[0].item() + 1
            
            if rank <= k:
                hr_count += 1
                ndcg_sum += 1.0 / np.log2(rank + 1)
    
    hr = hr_count / batch_size
    ndcg = ndcg_sum / batch_size
    
    return hr, ndcg

class SampledSoftmaxLoss(nn.Module):
    """完全对齐官方的Sampled Softmax损失 (此部分无需修改)"""
    def __init__(self, num_negatives=100, temperature=0.05):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature
    
    def forward(self, output_embeddings, target_ids, all_item_embeddings, supervision_weights):
        batch_size, seq_len_minus_1, embed_dim = output_embeddings.shape
        num_items = all_item_embeddings.size(0) - 1
        
        flat_output = output_embeddings.reshape(-1, embed_dim)
        flat_targets = target_ids.reshape(-1)
        flat_weights = supervision_weights.reshape(-1)
        
        valid_mask = flat_weights > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=output_embeddings.device)
            
        valid_output = flat_output[valid_mask]
        valid_targets = flat_targets[valid_mask]
        valid_weights = flat_weights[valid_mask]
        
        valid_output = F.normalize(valid_output, p=2, dim=1)
        norm_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
        
        pos_embeddings = norm_item_embeddings[valid_targets]
        pos_logits = (valid_output * pos_embeddings).sum(dim=1, keepdim=True)
        
        neg_indices = torch.randint(1, num_items + 1,
                                   (valid_output.size(0), self.num_negatives),
                                   device=output_embeddings.device)
        
        neg_mask = neg_indices != valid_targets.unsqueeze(1)
        neg_indices = torch.where(neg_mask, neg_indices, 
                                 torch.randint(1, num_items + 1, neg_indices.shape, 
                                             device=neg_indices.device))
        
        neg_embeddings = norm_item_embeddings[neg_indices]
        neg_logits = torch.bmm(valid_output.unsqueeze(1), 
                              neg_embeddings.transpose(1, 2)).squeeze(1)
        
        all_logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
        
        labels = torch.zeros(valid_output.size(0), dtype=torch.long, 
                           device=output_embeddings.device)
        
        loss_per_sample = F.cross_entropy(all_logits, labels, reduction='none')
        weighted_loss = (loss_per_sample * valid_weights).sum() / valid_weights.sum()
        
        return weighted_loss

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

    train_loader = DataLoader(train_dataset, batch_size=config['pretrain']['batch_size'], shuffle=True, num_workers=config['pretrain']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['pretrain']['batch_size'], shuffle=False, num_workers=config['pretrain']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['pretrain']['batch_size'], shuffle=False, num_workers=config['pretrain']['num_workers'])
    logging.info(f"数据集加载完成. 物品总数: {item_num}")

    model = Hstu(
        item_num=item_num,
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
        with torch.no_grad():
            all_item_ids = torch.arange(1, item_num + 1, device=device)
            all_item_embeddings = model.item_embedding(all_item_ids)
            
            eval_bar = tqdm(val_loader, desc=f"全量评估 Epoch {epoch}")
            for seq, pos, neg in eval_bar:
                seq = seq.to(device)
                target_item_ids = pos.to(device).squeeze(1)
                
                sequence_output = model.forward(seq)
                user_embeddings = sequence_output[:, -1, :]
                
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
        with torch.no_grad():
            all_item_ids = torch.arange(1, item_num + 1, device=device)
            all_item_embeddings = model.item_embedding(all_item_ids)
            
            test_bar = tqdm(test_loader, desc="测试中（全量评估）")
            for seq, pos, neg in test_bar:
                seq = seq.to(device)
                target_item_ids = pos.to(device).squeeze(1)
                
                sequence_output = model.forward(seq)
                user_embeddings = sequence_output[:, -1, :]
                hr, ndcg = get_metrics_full_eval(user_embeddings, all_item_embeddings, target_item_ids, k=config['evaluation']['top_k'])
                total_hr += hr
                total_ndcg += ndcg

        avg_hr = total_hr / len(test_loader)
        avg_ndcg = total_ndcg / len(test_loader)
        logging.info(f"测试集结果 ==> HR@{config['evaluation']['top_k']}: {avg_hr:.4f}, NDCG@{config['evaluation']['top_k']}: {avg_ndcg:.4f}")
    else:
        logging.warning("未找到最佳模型文件，跳过测试。")

if __name__ == '__main__':
    train()