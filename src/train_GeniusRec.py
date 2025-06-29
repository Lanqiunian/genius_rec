import argparse
import logging
import os
import pickle
import random
import math
from tqdm import tqdm
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 训练参数
# 冻结编码器参数，防止其在微调过程中被更新
# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --freeze_encoder
# 不使用冻结编码器参数
# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth

from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.dataset import Seq2SeqRecDataset
import platform


# --- 1. 训练和评估函数 ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for batch in progress_bar:
        source_ids = batch['source_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)
        source_padding_mask = (source_ids == 0).to(device)
        optimizer.zero_grad()
        
        # 模型前向传播
        logits = model(source_ids, decoder_input_ids, source_padding_mask)

        # 计算损失
        # logits: (B, target_len, num_items) -> (B * target_len, num_items)
        # labels: (B, target_len) -> (B * target_len)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/len(progress_bar))
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")

    with torch.no_grad():
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == 0).to(device)

            logits = model(source_ids, decoder_input_ids, source_padding_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/len(progress_bar))

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) # 困惑度是评估生成模型常用指标
    return avg_loss, perplexity


# --- 2. 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script for GENIUS-Rec model.")
    parser.add_argument('--config_path', type=str, help='Path to your project config file (optional).')
    parser.add_argument('--encoder_weights_path', type=str, required=True, help='Path to pre-trained HSTU encoder weights.')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights during fine-tuning.')
    parser.add_argument('--save_dir', type=str, default='checkpoints/checkpoints_genius_rec', help='Directory to save fine-tuned models.')
    args = parser.parse_args()

    # --- 初始化 ---
    config = get_config() # 使用您提供的config.py
    device = torch.device(config['device'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    os.makedirs(args.save_dir, exist_ok=True)
    # 确保 config['data']['log_dir'] 是 pathlib.Path 对象
    log_dir_path = config['data']['log_dir']
    if not isinstance(log_dir_path, pathlib.Path):
        from pathlib import Path
        log_dir_path = Path(log_dir_path)

    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / config['finetune']['log_file']
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info("--- Starting GENIUS-Rec Fine-tuning ---")
    logging.info(f"Device: {device}")
    logging.info(f"Arguments: {args}")

    # --- 准备数据 ---
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
        num_items = id_maps['num_items'] + 1 # +1 for padding token
    
    train_dataset = Seq2SeqRecDataset(config['data']['train_file'], config['decoder_model']['max_seq_len'], split_ratio=config['finetune']['split_ratio'])
    val_dataset = Seq2SeqRecDataset(config['data']['validation_file'], config['decoder_model']['max_seq_len'], split_ratio=config['finetune']['split_ratio'])
    train_loader = DataLoader(train_dataset, batch_size=config['finetune']['batch_size'], shuffle=True, num_workers=config['finetune']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['finetune']['batch_size'], shuffle=False, num_workers=config['finetune']['num_workers'])
    logging.info(f"Data loaded. Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # --- 构建模型 ---
    # 构建encoder配置
    encoder_config = {
        'item_num': num_items,
        'embedding_dim': config['encoder_model']['embedding_dim'],
        'linear_hidden_dim': config['encoder_model']['linear_hidden_dim'],
        'attention_dim': config['encoder_model']['attention_dim'],
        'num_heads': config['encoder_model']['num_heads'],
        'num_layers': config['encoder_model']['num_layers'],
        'max_len': config['encoder_model']['max_len'],
        'dropout': config['encoder_model']['dropout'],
        'pad_token_id': config['pad_token_id']
    }
    # 构建decoder配置
    decoder_config = {
        'num_items': num_items,
        'embedding_dim': config['decoder_model']['embedding_dim'],
        'num_layers': config['decoder_model']['num_layers'],
        'num_heads': config['decoder_model']['num_heads'],
        'ffn_hidden_dim': config['decoder_model']['ffn_hidden_dim'],
        'max_seq_len': config['decoder_model']['max_seq_len'],
        'dropout_ratio': config['decoder_model']['dropout_ratio']
    }
    model = GENIUSRecModel(encoder_config=encoder_config, decoder_config=decoder_config).to(device)
    logging.info("GENIUS-Rec model created.")

    # --- 加载预训练权重 ---
    logging.info(f"Loading pre-trained encoder weights from: {args.encoder_weights_path}")
    
  
    # 如果当前系统是Linux，并且要加载的文件存在
    if platform.system() == "Linux" and os.path.exists(args.encoder_weights_path):
        # 暂时将 WindowsPath 指向 PosixPath，欺骗 unpickler
        temp_windows_path = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
    # +++ 结束修改 +++

    # 加载状态字典 (这行代码本身不变)
    encoder_state_dict = torch.load(args.encoder_weights_path, map_location=device)
    
    # +++ 开始修改: 恢复pathlib设置 +++
    # 加载完成后，恢复原始的 pathlib.WindowsPath，避免影响后续操作
    if platform.system() == "Linux":
        pathlib.WindowsPath = temp_windows_path
    # +++ 结束修改 +++
    
    # 为了安全，只加载键匹配的权重
    model.encoder.load_state_dict(encoder_state_dict, strict=False) 
    logging.info("Encoder weights loaded successfully.")
    
    # --- 冻结与优化器设置 ---
    if args.freeze_encoder:
        logging.info("Freezing encoder parameters.")
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=config['finetune']['learning_rate']['decoder_lr'], weight_decay=config['finetune']['weight_decay'])
    else:
        logging.info("Fine-tuning both encoder and decoder with different learning rates.")
        optimizer = torch.optim.AdamW([
            {'params': model.decoder.parameters(), 'lr': config['finetune']['learning_rate']['decoder_lr']},
            {'params': model.encoder.parameters(), 'lr': config['finetune']['learning_rate']['encoder_lr']}
        ], weight_decay=config['finetune']['weight_decay'])
        
    criterion = nn.CrossEntropyLoss(ignore_index=config['pad_token_id'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    # --- 断点续传逻辑 ---
    start_epoch = 0
    best_perplexity = float('inf')
    patience_counter = 0
    
    # 检查点文件路径
    latest_ckpt_path = os.path.join(args.save_dir, 'genius_rec_latest.pth')
    best_ckpt_path = os.path.join(args.save_dir, 'genius_rec_best.pth')
    
    # 尝试加载最新检查点进行断点续传
    if os.path.exists(latest_ckpt_path):
        logging.info(f"发现最新检查点: {latest_ckpt_path}，正在加载...")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练状态
            start_epoch = checkpoint['epoch'] + 1
            best_perplexity = checkpoint.get('best_perplexity', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            
            logging.info(f"成功加载检查点! 从 Epoch {start_epoch} 继续训练")
            logging.info(f"当前最佳困惑度: {best_perplexity:.4f}")
            logging.info(f"当前耐心计数: {patience_counter}")
            
        except Exception as e:
            logging.warning(f"加载检查点失败: {e}")
            logging.info("将从头开始训练")
            start_epoch = 0
            best_perplexity = float('inf')
            patience_counter = 0
    else:
        logging.info("未发现检查点，从头开始训练")

    # --- 训练循环 ---
    for epoch in range(start_epoch, config['finetune']['num_epochs']):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config['finetune']['num_epochs'])
        avg_val_loss, perplexity = evaluate(model, val_loader, criterion, device, epoch, config['finetune']['num_epochs'])
        
        logging.info(f"Epoch {epoch+1}/{config['finetune']['num_epochs']} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # 保存最新检查点（每个epoch都保存）
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_perplexity': best_perplexity,
            'current_perplexity': perplexity,
            'patience_counter': patience_counter,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'config': config,
            'args': vars(args)
        }
        torch.save(latest_checkpoint, latest_ckpt_path)
        logging.info(f"最新检查点已保存: {latest_ckpt_path}")
        
        # 保存最佳检查点（仅在性能提升时保存）
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            patience_counter = 0
            
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_perplexity': best_perplexity,
                'current_perplexity': perplexity,
                'patience_counter': patience_counter,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
                'args': vars(args)
            }
            torch.save(best_checkpoint, best_ckpt_path)
            logging.info(f"🎉 发现新的最佳模型! 困惑度: {best_perplexity:.4f}")
            logging.info(f"最佳检查点已保存: {best_ckpt_path}")
        else:
            patience_counter += 1
            logging.info(f"性能未提升，耐心计数: {patience_counter}/{config['finetune']['early_stopping_patience']}")
            if patience_counter >= config['finetune']['early_stopping_patience']:
                logging.info(f"触发早停! 连续 {patience_counter} 个epoch性能未提升")
                break
                
    logging.info("--- Fine-tuning finished ---")
    # 如果循环因为早停而中断，epoch可能没有达到最大值，所以用 epoch+1
    completed_epochs = epoch + 1 if 'epoch' in locals() else start_epoch
    logging.info(f"训练总轮次: {completed_epochs}/{config['finetune']['num_epochs']}")
    logging.info(f"最佳验证困惑度: {best_perplexity:.4f}")
    
    # 显示检查点信息
    if os.path.exists(best_ckpt_path):
        logging.info(f"✅ 最佳模型检查点: {best_ckpt_path}")
    if os.path.exists(latest_ckpt_path):
        logging.info(f"✅ 最新模型检查点: {latest_ckpt_path}")

def load_checkpoint_info(checkpoint_path):
    """加载并显示检查点信息的辅助函数"""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        info = {
            'epoch': checkpoint.get('epoch', -1),
            'best_perplexity': checkpoint.get('best_perplexity', float('inf')),
            'current_perplexity': checkpoint.get('current_perplexity', float('inf')),
            'train_loss': checkpoint.get('train_loss', 0.0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'patience_counter': checkpoint.get('patience_counter', 0)
        }
        return info
    except Exception as e:
        print(f"加载检查点信息失败: {e}")
        return None

if __name__ == '__main__':
    main()