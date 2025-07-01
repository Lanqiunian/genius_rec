import argparse
import logging
import os
import pickle
import random
import math
import pathlib
import numpy as np

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入所有需要的模块
from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.dataset import Seq2SeqRecDataset
from src.encoder.encoder import Hstu
from src.decoder.decoder import GenerativeDecoder
# 【新】导入我们的指标计算模块
from src.metrics import get_metrics 

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, num_epochs, pad_token_id):
    # 这个函数保持不变
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
    for batch in progress_bar:
        source_ids, decoder_input_ids, labels = batch['source_ids'].to(device), batch['decoder_input_ids'].to(device), batch['labels'].to(device)
        source_padding_mask = (source_ids == pad_token_id)
        optimizer.zero_grad()
        logits = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=False)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_loss:.4f}")
    return avg_loss

# 【已修改】evaluate 函数现在会计算并返回所有指标
def evaluate(model, dataloader, criterion, device, epoch, num_epochs, pad_token_id, top_k):
    model.eval()
    total_loss = 0.0
    
    # 初始化用于累加指标的变量
    total_behavior_weight, total_content_weight = 0.0, 0.0
    total_valid_batches = 0
    all_hr_scores, all_ndcg_scores = [], []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")

    with torch.no_grad():
        for batch in progress_bar:
            source_ids, decoder_input_ids, labels = batch['source_ids'].to(device), batch['decoder_input_ids'].to(device), batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)

            logits, gate_weights = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=True)
            
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            
            # 【新】调用指标计算函数
            hr_k, ndcg_k = get_metrics(logits, labels, k=top_k, pad_token_id=pad_token_id)
            all_hr_scores.extend(hr_k)
            all_ndcg_scores.extend(ndcg_k)

            # 监控门控权重 (逻辑不变)
            non_padding_mask = (decoder_input_ids != pad_token_id)
            behavior_weights = gate_weights[:, :, 0][non_padding_mask]
            content_weights = gate_weights[:, :, 1][non_padding_mask]
            if behavior_weights.numel() > 0:
                total_behavior_weight += behavior_weights.mean().item()
                total_content_weight += content_weights.mean().item()
                total_valid_batches += 1

            progress_bar.set_postfix(loss=loss.item())

    # 计算平均指标
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    avg_hr = np.mean(all_hr_scores) if all_hr_scores else 0
    avg_ndcg = np.mean(all_ndcg_scores) if all_ndcg_scores else 0
    avg_behavior_w = total_behavior_weight / total_valid_batches if total_valid_batches > 0 else 0
    avg_content_w = total_content_weight / total_valid_batches if total_valid_batches > 0 else 0
    
    # 打印所有结果
    print(f"\nValidation Results for Epoch {epoch+1}:")
    print(f"  -> Val Loss: {avg_loss:.4f}, Val PPL: {perplexity:.4f}")
    print(f"  -> HR@{top_k}: {avg_hr:.4f}, NDCG@{top_k}: {avg_ndcg:.4f}")
    print(f"  -> Avg Gate Weight [Behavior Expert]: {avg_behavior_w:.4f}")
    print(f"  -> Avg Gate Weight [Content Expert]:  {avg_content_w:.4f}\n")

    return avg_loss, perplexity, avg_hr, avg_ndcg


def main():
    config = get_config()
    parser = argparse.ArgumentParser(description="Train GENIUS-Rec Model with MoE")
    parser.add_argument('--encoder_weights_path', type=str, default=None, help='Path to pre-trained HSTU encoder weights.')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights.')
    args = parser.parse_args()
    
    # 2. Setup environment
    device = torch.device(config['device'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # 3. Load data and ID maps
    print("Loading data from processed directory specified in config...")
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)

    train_dataset = Seq2SeqRecDataset(config['data']['train_file'], config['decoder_model']['max_seq_len'])
    val_dataset = Seq2SeqRecDataset(config['data']['validation_file'], config['decoder_model']['max_seq_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['finetune']['batch_size'], shuffle=True, num_workers=config['finetune']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['finetune']['batch_size'], shuffle=False, num_workers=config['finetune']['num_workers'])
    
    num_items = id_maps['num_items'] + 1
    pad_token_id = config['pad_token_id']
    
    # 4. Load text embeddings
    print("Loading pre-computed and FILTERED text embeddings...")
    text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered.npy'
    try:
        text_embeddings_dict = np.load(text_embedding_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Filtered embedding file not found at '{text_embedding_file}'! Please run filter_embeddings.py first.")
        return

    text_embedding_dim = next(iter(text_embeddings_dict.values())).shape[0]
    text_embedding_matrix = torch.zeros(num_items, text_embedding_dim, dtype=torch.float)
    
    item_asin_map = id_maps['item_map']
    loaded_count = 0
    for asin, embedding in text_embeddings_dict.items():
        if asin in item_asin_map:
            item_id = item_asin_map[asin]
            text_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
            loaded_count += 1
    
    print(f"Successfully loaded and mapped {loaded_count} text embeddings.")
    if loaded_count == 0:
        print("\nWARNING: No embeddings were mapped. The content expert will not function. Please check your data files.\n")


    # 5. Initialize model
    config['encoder_model']['item_num'] = num_items
    config['decoder_model']['num_items'] = num_items
    config['decoder_model']['text_embedding_dim'] = text_embedding_dim
    model = GENIUSRecModel(config['encoder_model'], config['decoder_model']).to(device)

    # 6. Load pre-trained weights
    if args.encoder_weights_path:
        try:
            encoder_weights = torch.load(args.encoder_weights_path, map_location=device)
            model.encoder.load_state_dict(encoder_weights, strict=False) 
            print(f"Pre-trained HSTU encoder weights loaded from '{args.encoder_weights_path}'.")
        except Exception as e:
            print(f"Warning: Could not load encoder weights: {e}. Training from scratch.")
            
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen.")

    # 7. Load text embeddings into model
    model.decoder.load_text_embeddings(text_embedding_matrix.to(device))
    
    # 8. Define optimizer and loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['finetune']['learning_rate']['decoder_lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
    
    # 9. Define learning rate scheduler
    num_training_steps = len(train_loader) * config['finetune']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['finetune']['warmup_steps'], num_training_steps=num_training_steps)

    # 【已修改】训练循环现在会处理和保存所有指标
    print("Starting training...")
    
    best_val_metric = -1 # 使用一个以越高越好的指标 (如NDCG) 作为标准
    output_dir = config['data']['checkpoint_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / 'genius_rec_moe_best.pth'
    latest_model_path = output_dir / 'genius_rec_moe_latest.pth'
    top_k = config['evaluation']['top_k']

    for epoch in range(config['finetune']['num_epochs']):
        train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, config['finetune']['num_epochs'], pad_token_id)
        
        val_loss, val_ppl, val_hr, val_ndcg = evaluate(model, val_loader, criterion, device, epoch, config['finetune']['num_epochs'], pad_token_id, top_k)

        torch.save(model.state_dict(), latest_model_path)
        print(f"Saved latest model checkpoint to '{latest_model_path}'")
        
        # 使用NDCG@K作为判断最佳模型的标准
        if val_ndcg > best_val_metric:
            best_val_metric = val_ndcg
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 New best model found! Saved to '{best_model_path}' with NDCG@{top_k}: {best_val_metric:.4f}")

    print("Training finished.")

if __name__ == '__main__':
    main()