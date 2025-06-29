import argparse
import logging
import os
import pickle
import random
import math
from tqdm import tqdm
import pathlib
from transformers import get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.dataset import Seq2SeqRecDataset

import platform

# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --freeze_encoder
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, num_epochs, pad_token_id):
    """
    【已修改】: 新增 scheduler 参数和 pad_token_id 参数, 并在每个batch后调用 scheduler.step()
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

    for batch in progress_bar:
        source_ids = batch['source_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)
        source_padding_mask = (source_ids == pad_token_id).to(device)
        optimizer.zero_grad()
        
        # 模型前向传播
        logits = model(source_ids, decoder_input_ids, source_padding_mask)

        # 计算损失
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
        optimizer.step()
        
        # 在每个batch后更新学习率
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/len(progress_bar))
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, epoch, num_epochs, pad_token_id):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")

    with torch.no_grad():
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id).to(device)

            logits = model(source_ids, decoder_input_ids, source_padding_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss/len(progress_bar))

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# --- 2. 主函数 (已修改调度器逻辑) ---
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script for GENIUS-Rec model.")
    parser.add_argument('--config_path', type=str, help='Path to your project config file (optional).')
    parser.add_argument('--encoder_weights_path', type=str, required=True, help='Path to pre-trained HSTU encoder weights.')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights during fine-tuning.')
    parser.add_argument('--save_dir', type=str, default='checkpoints/checkpoints_genius_rec', help='Directory to save fine-tuned models.')
    args = parser.parse_args()

    # --- 初始化 ---
    config = get_config()
    device = torch.device(config['device'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir_path = pathlib.Path(config['data']['log_dir'])
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / config['finetune']['log_file']
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info("--- Starting GENIUS-Rec Fine-tuning with Warmup Scheduler ---")
    logging.info(f"Device: {device}")
    logging.info(f"Arguments: {args}")

    # --- 准备数据 ---
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
        num_items = id_maps['num_items'] + 1

    # 修改数据集初始化
    train_dataset = Seq2SeqRecDataset(
        config['data']['train_file'], 
        config['decoder_model']['max_seq_len'], 
        pad_token_id=config['pad_token_id'],
        split_ratio=config['finetune']['split_ratio']
    )
    val_dataset = Seq2SeqRecDataset(
        config['data']['validation_file'], 
        config['decoder_model']['max_seq_len'], 
        pad_token_id=config['pad_token_id'],
        split_ratio=config['finetune']['split_ratio']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['finetune']['batch_size'], shuffle=True, num_workers=config['finetune']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['finetune']['batch_size'], shuffle=False, num_workers=config['finetune']['num_workers'])
    logging.info(f"Data loaded. Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # --- 构建模型 ---
    encoder_config = {**config['encoder_model'], 'item_num': num_items, 'pad_token_id': config['pad_token_id']}
    decoder_config = {
        **config['decoder_model'], 
        'num_items': num_items,
        'pad_token_id': config['pad_token_id']
    }
    model = GENIUSRecModel(encoder_config=encoder_config, decoder_config=decoder_config).to(device)
    logging.info("GENIUS-Rec model created.")

    # --- 加载预训练权重 (包含跨平台补丁) ---
    logging.info(f"Loading pre-trained encoder weights from: {args.encoder_weights_path}")
    if platform.system() == "Linux" and os.path.exists(args.encoder_weights_path):
        temp_windows_path = getattr(pathlib, 'WindowsPath', None)
        pathlib.WindowsPath = pathlib.PosixPath
    
    encoder_state_dict = torch.load(args.encoder_weights_path, map_location=device)
    
    if platform.system() == "Linux" and 'temp_windows_path' in locals():
        pathlib.WindowsPath = temp_windows_path
    
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
    
    # --- 【核心修改】: 设置新的学习率调度器 ---
    num_training_steps = len(train_loader) * config['finetune']['num_epochs']
    num_warmup_steps = config['finetune'].get('warmup_steps', 500) # 从配置获取，提供默认值
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    logging.info(f"Using learning rate scheduler with {num_warmup_steps} warmup steps and {num_training_steps} total training steps.")

    # --- 断点续传逻辑 ---
    start_epoch = 0
    best_perplexity = float('inf')
    patience_counter = 0
    latest_ckpt_path = os.path.join(args.save_dir, 'genius_rec_latest.pth')
    best_ckpt_path = os.path.join(args.save_dir, 'genius_rec_best.pth')
    
    if os.path.exists(latest_ckpt_path):
        logging.info(f"发现最新检查点: {latest_ckpt_path}，正在加载...")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 【重要】: 同样需要加载调度器的状态
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_perplexity = checkpoint.get('best_perplexity', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            logging.info(f"成功加载检查点! 从 Epoch {start_epoch} 继续训练")
        except Exception as e:
            logging.warning(f"加载检查点失败: {e}. 将从头开始训练")
            start_epoch = 0

    # --- 训练循环 ---
    for epoch in range(start_epoch, config['finetune']['num_epochs']):
        # 【修改】: 传入scheduler和pad_token_id
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, config['finetune']['num_epochs'], config['pad_token_id'])
        avg_val_loss, perplexity = evaluate(model, val_loader, criterion, device, epoch, config['finetune']['num_epochs'], config['pad_token_id'])
        
        logging.info(f"Epoch {epoch+1}/{config['finetune']['num_epochs']} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")
        
        # 保存检查点
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_perplexity': best_perplexity,
            'current_perplexity': perplexity,
            'patience_counter': patience_counter,
        }
        torch.save(checkpoint_data, latest_ckpt_path)
        
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            patience_counter = 0
            torch.save(checkpoint_data, best_ckpt_path)
            logging.info(f"🎉 发现新的最佳模型! 困惑度: {best_perplexity:.4f}. 已保存至 {best_ckpt_path}")
        else:
            patience_counter += 1
            logging.info(f"性能未提升，耐心计数: {patience_counter}/{config['finetune']['early_stopping_patience']}")
            if patience_counter >= config['finetune']['early_stopping_patience']:
                logging.info(f"触发早停! 连续 {patience_counter} 个epoch性能未提升")
                break
                
    logging.info("--- Fine-tuning finished ---")
    completed_epochs = epoch + 1 if 'epoch' in locals() else start_epoch
    logging.info(f"训练总轮次: {completed_epochs}/{config['finetune']['num_epochs']}")
    logging.info(f"最佳验证困惑度: {best_perplexity:.4f}")

if __name__ == '__main__':
    main()
