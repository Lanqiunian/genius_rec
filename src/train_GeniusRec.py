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
    # 1. 参数解析 (此部分逻辑保持不变)
    parser = argparse.ArgumentParser(description="Train GENIUS-Rec Model")
    # ... (您原有的所有 argparse 参数定义)
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of the model embedding')
    parser.add_argument('--decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--ffn_hidden_dim', type=int, default=512, help='Hidden dimension of FFN')
    parser.add_argument('--max_seq_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='Dropout ratio')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()

    # 2. 设置设备 (此部分逻辑保持不变)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 加载数据集和ID映射 (此部分逻辑保持不变)
    data_path = 'data'
    train_loader, val_loader, test_loader, id_maps = load_data_and_create_loaders(data_path, args.batch_size, args.max_seq_len)
    num_users = id_maps['num_users']
    num_items = id_maps['num_items'] + 1  # +1 for padding token 0
    pad_token_id = 0
    
    # ==================== 新增模块: 加载并准备文本嵌入 ====================
    print("Loading pre-computed text embeddings from data/book_gemini_embeddings_final.npy...")
    try:
        text_embeddings_dict = np.load('data/book_gemini_embeddings_final.npy', allow_pickle=True).item()
    except FileNotFoundError:
        print("Error: `data/book_gemini_embeddings_final.npy` not found!")
        print("Please ensure the text embedding file is in the correct location.")
        return # 提前退出

    # 从字典中任意取一个嵌入来确定维度
    try:
        text_embedding_dim = next(iter(text_embeddings_dict.values())).shape[0]
        print(f"Detected text embedding dimension: {text_embedding_dim}")
    except StopIteration:
        print("Error: The text embedding dictionary is empty!")
        return # 提前退出

    # 创建一个空的嵌入矩阵
    text_embedding_matrix = torch.zeros(num_items, text_embedding_dim, dtype=torch.float)
    
    # 创建 asin -> item_id 的映射
    asin_to_id_map = {v: k for k, v in id_maps['item_map'].items()}
    
    # 填充嵌入矩阵
    loaded_count = 0
    for asin, embedding in text_embeddings_dict.items():
        if asin in asin_to_id_map:
            item_id = asin_to_id_map[asin]
            text_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
            loaded_count += 1
    
    print(f"Successfully loaded and mapped {loaded_count} text embeddings into the matrix.")
    # =======================================================================

    # 4. 初始化模型
    # 4.1 初始化编码器 (此部分逻辑保持不变)
    print("Initializing HSTU Encoder...")
    encoder = HSTUEncoder(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len,
        num_layers=4, # 根据您的HSTU配置
        num_heads=args.num_heads,
        dropout_ratio=args.dropout_ratio,
        pad_token_id=pad_token_id
    )
    # 加载预训练的HSTU权重
    try:
        encoder_weights = torch.load('models/hstu_encoder_weights.pth', map_location=device)
        encoder.load_state_dict(encoder_weights)
        print("Pre-trained HSTU encoder weights loaded successfully.")
    except FileNotFoundError:
        print("Warning: Pre-trained HSTU encoder weights not found. Training encoder from scratch.")
        
    # 4.2 初始化修改后的解码器
    print("Initializing Generative Decoder with MoE...")
    decoder = GenerativeDecoder(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.decoder_layers,
        num_heads=args.num_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        max_seq_len=args.max_seq_len,
        dropout_ratio=args.dropout_ratio,
        pad_token_id=pad_token_id,
        text_embedding_dim=text_embedding_dim # <<-- 传入新的参数
    )

    # 4.3 初始化完整的GeniusRec模型 (此部分逻辑保持不变)
    model = GeniusRec(encoder, decoder).to(device)

    # ==================== 新增模块: 将嵌入矩阵加载到模型中 ====================
    # 将包含所有文本嵌入的矩阵加载到解码器的对应层中
    model.decoder.load_text_embeddings(text_embedding_matrix.to(device))
    # =======================================================================

    # 5. 定义优化器和损失函数 (此部分逻辑保持不变)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 忽略padding token的损失计算
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # 6. 训练和验证循环 (此部分逻辑保持不变)
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        for user_id, sequence, target in train_loader:
            user_id, sequence, target = user_id.to(device), sequence.to(device), target.to(device)

            optimizer.zero_grad()
            
            logits = model(sequence, target[:, :-1]) # 输入目标序列除了最后一个token
            
            # 调整维度以匹配CrossEntropyLoss的期望输入
            # Logits: (B, T, V) -> (B*T, V)
            # Target: (B, T) -> (B*T)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for user_id, sequence, target in val_loader:
                user_id, sequence, target = user_id.to(device), sequence.to(device), target.to(device)
                
                logits = model(sequence, target[:, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PPL: {math.exp(avg_val_loss):.4f}")

    print("Training finished.")

    # 7. 保存最终模型 (可选)
    torch.save(model.state_dict(), 'models/genius_rec_moe_final.pth')
    print("Final model saved to models/genius_rec_moe_final.pth")


if __name__ == '__main__':
    main()
