# src/train_GeniusRec.py
import argparse
import logging
import os
import pickle
import random
import math
import pathlib
import platform
from pathlib import Path

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.dataset import Seq2SeqRecDataset
from src.unified_evaluation import (
    ValidationDataset, 
    evaluate_model_validation, 
    evaluate_model_test, 
    evaluate_model_validation_with_ranking
)

from src.encoder.encoder import Hstu
from src.decoder.decoder import GenerativeDecoder


# # 端到端微调训练（推荐方式）
# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --full_evaluation

# # 冻结编码器训练（对比实验）
# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --freeze_encoder

# # 使用与HSTU完全一致的全量评估模式（评估所有物品）
# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --full_evaluation

# # 或者使用采样评估（速度更快，指定候选物品数量）
# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --sample_eval_size 1000

# # 从检查点恢复训练
# python -m src.train_GeniusRec --resume_from checkpoints/genius_rec_moe_latest.pth --encoder_weights_path checkpoints/hstu_encoder.pth

# # 自定义保存目录
# python -m src.train_GeniusRec --save_dir my_checkpoints


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, num_epochs, pad_token_id, config, scaler=None):
    """
    【最终版】训练一个epoch。
    """
    model.train()
    total_loss_value = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=True)

    balancing_loss_alpha = config['finetune'].get('balancing_loss_alpha', 0.01)

    for batch_idx, batch in enumerate(progress_bar):
        source_ids = batch['source_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)
        source_padding_mask = (source_ids == pad_token_id)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits, gate_weights, balancing_loss = model(
                source_ids,
                decoder_input_ids,
                source_padding_mask,
                return_weights=True,
                
            )
            
            task_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss = task_loss + balancing_loss_alpha * balancing_loss
            bal_loss_item = balancing_loss.item()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if scheduler is not None:
             scheduler.step()
        
        total_loss_value += loss.item()
        
        if gate_weights is not None:
            enabled_experts = [k for k, v in model.decoder.expert_config["experts"].items() if v]
            weights_postfix = {}
            for i, expert_name in enumerate(enabled_experts):
                if i < gate_weights.shape[-1]:
                    display_name = f'{expert_name.split("_")[0][:3].title()}W'
                    weights_postfix[display_name] = f"{gate_weights[:, :, i].mean().item():.3f}"
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 'task_l': f"{task_loss.item():.4f}",
                'bal_l': f"{bal_loss_item:.4f}", **weights_postfix
            })

    avg_loss = total_loss_value / len(dataloader)
    logging.info(f"Epoch {epoch+1} training finished. Average Loss: {avg_loss:.4f}")
    return avg_loss


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    检查点加载函数，支持完整检查点和仅权重的.pth文件
    
    Returns:
        dict: 包含恢复信息的字典
    """
    logging.info(f"尝试加载检查点: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 检查是否为完整检查点格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整检查点
            logging.info("检测到完整检查点格式")
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 尝试加载optimizer状态
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("成功恢复optimizer状态")
                except Exception as e:
                    logging.warning(f"无法恢复optimizer状态: {e}")
            
            # 尝试加载scheduler状态
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logging.info("成功恢复scheduler状态")
                except Exception as e:
                    logging.warning(f"无法恢复scheduler状态: {e}")
            
            return {
                'epoch': checkpoint.get('epoch', 0),
                'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'val_ppl': checkpoint.get('val_ppl', float('inf')),
                'patience_counter': checkpoint.get('patience_counter', 0)
            }
            
        else:
            # 仅权重的检查点
            logging.info("检测到权重格式，仅恢复模型权重")
            
            # 处理可能的键名不匹配
            if hasattr(checkpoint, 'keys'):
                # 如果是state_dict格式
                model.load_state_dict(checkpoint, strict=False)
            else:
                # 如果是直接的权重
                logging.warning("无法识别的检查点格式，跳过加载")
                return None
            
            return {
                'epoch': 0,
                'best_val_loss': float('inf'),
                'val_loss': float('inf'),
                'val_ppl': float('inf'),
                'patience_counter': 0
            }
            
    except Exception as e:
        logging.error(f"加载检查点失败: {e}")
        return None

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, metrics_dict, config, num_items):
    """
    检查点保存函数，保存完整的训练状态
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config,
        'num_items': num_items,
        **metrics_dict  # 展开所有指标
    }
    torch.save(checkpoint_data, checkpoint_path)


def main():
    # 1. 参数解析和配置加载
    parser = argparse.ArgumentParser(description="Train GENIUS-Rec Model")
    parser.add_argument('--encoder_weights_path', type=str, default=None, help='Path to pre-trained HSTU encoder weights.')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights.')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints.')
    parser.add_argument('--disable_behavior_expert', action='store_true', help='Disable behavior expert.')
    parser.add_argument('--disable_content_expert', action='store_true', help='Disable content expert.')
    parser.add_argument('--disable_image_expert', action='store_true', help='Disable image expert.')
    parser.add_argument('--enable_image_expert', action='store_true', help='Enable image expert (requires image embeddings).')
    parser.add_argument('--image_embeddings_path', type=str, default=None, help='Path to image embeddings file.')
    parser.add_argument('--full_evaluation', action='store_true', help='使用全量评估(与所有物品计算相似度)，与HSTU和baseline完全一致，但速度较慢。')
    parser.add_argument('--sample_eval_size', type=int, default=500, help='采样评估的候选物品数量，默认为500，设为0使用全量评估。')
    parser.add_argument('--decoder_layers', type=int, default=None, help='Override decoder layers count for architecture experiments.')
    args = parser.parse_args()

    config = get_config()

    if args.disable_behavior_expert: config['expert_system']['experts']['behavior_expert'] = False
    if args.disable_content_expert: config['expert_system']['experts']['content_expert'] = False
    if args.disable_image_expert: config['expert_system']['experts']['image_expert'] = False
    if args.enable_image_expert: config['expert_system']['experts']['image_expert'] = True
    
    # Override decoder layers if specified
    if args.decoder_layers is not None:
        config['decoder_model']['num_layers'] = args.decoder_layers
        logging.info(f"🏗️ 覆盖解码器层数设置为: {args.decoder_layers}")

    # 2. 环境设置
    device = torch.device(config['device'])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config['seed'])

    # 3. 日志设置
    log_dir_path = pathlib.Path(config['data']['log_dir'])
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / 'train_genius_rec.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.info("=== Starting GENIUS-Rec Training ===")
    logging.info(f"Device: {device}")
    logging.info(f"Arguments: {args}")

    # 4. 数据加载和ID映射处理
    logging.info("Loading data from processed directory...")
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
    num_special_tokens = id_maps.get('num_special_tokens', 4)
    total_vocab_size = id_maps['num_items'] + num_special_tokens
    config['encoder_model']['item_num'] = total_vocab_size
    config['decoder_model']['num_items'] = total_vocab_size
    num_items = total_vocab_size
    logging.info(f"📊 Vocabulary Info: Total vocabulary size: {total_vocab_size}")
    enabled_experts = [k for k, v in config['expert_system']['experts'].items() if v]
    if not enabled_experts: raise ValueError("❌ 至少需要启用一个专家！")
    logging.info(f"🧠 已启用专家: {enabled_experts}")

    train_dataset = Seq2SeqRecDataset(config, config['data']['train_file'])
    val_dataset = Seq2SeqRecDataset(config, config['data']['validation_file'])
    test_dataset = ValidationDataset(config['data']['test_file'], config['encoder_model']['max_len'], config['pad_token_id'])
    train_loader = DataLoader(train_dataset, batch_size=config['finetune']['batch_size'], shuffle=True, num_workers=config['finetune']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['finetune']['batch_size'], shuffle=False, num_workers=config['finetune']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['finetune']['batch_size'], shuffle=False, num_workers=config['finetune']['num_workers'])
    pad_token_id = config['pad_token_id']
    top_k = config['evaluation']['top_k']
    logging.info(f"📊 Dataset Info: Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")


    # 5. 文本嵌入加载
    logging.info("Loading pre-computed and filtered text embeddings...")
    text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered_migrated.npy'
    try:
        text_embeddings_dict = np.load(text_embedding_file, allow_pickle=True).item()
    except FileNotFoundError:
        logging.error(f"Filtered embedding file not found at '{text_embedding_file}'! Please run filter_embeddings.py first.")
        return

    text_embedding_dim = next(iter(text_embeddings_dict.values())).shape[0]
    text_embedding_matrix = torch.zeros(total_vocab_size, text_embedding_dim, dtype=torch.float)
    
    item_asin_map = id_maps['item_map']
    loaded_count = 0
    for asin, embedding in text_embeddings_dict.items():
        if asin in item_asin_map:
            item_id = item_asin_map[asin]
            text_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
            loaded_count += 1
    
    logging.info(f"Successfully loaded and mapped {loaded_count} text embeddings.")
    if loaded_count == 0:
        logging.warning("No embeddings were mapped. The content expert will not function.")

    # 6. 模型初始化（使用配置字典）
    config['decoder_model']['text_embedding_dim'] = text_embedding_dim
    
    # 【新增】传递专家配置到模型
    model = GENIUSRecModel(config['encoder_model'], config['decoder_model'], config['expert_system']).to(device)
    logging.info("GENIUS-Rec model created with expert configuration.")
    
    # 打印启用的专家信息
    enabled_experts = [k for k, v in config['expert_system']['experts'].items() if v]
    logging.info(f"🧠 启用的专家: {enabled_experts}")
    
    # 显存使用报告
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"💾 GPU显存状态:")
        logging.info(f"   - 已分配: {memory_allocated:.2f}GB")
        logging.info(f"   - 已预留: {memory_reserved:.2f}GB") 
        logging.info(f"   - 总容量: {memory_total:.2f}GB")
        logging.info(f"   - 剩余可用: {memory_total - memory_reserved:.2f}GB")
    
    # 7. 预训练权重加载（优先使用迁移后的权重）
    if args.encoder_weights_path:
        try:
            # 检查是否为迁移后的权重文件
            weights_path = Path(args.encoder_weights_path)
            
            if "_migrated" in weights_path.stem:
                logging.info(f"Loading migrated encoder weights from: {weights_path}")
                load_path = weights_path
            else:
                # 检查是否存在迁移后的权重文件
                migrated_weights_path = weights_path.parent / f"{weights_path.stem}_migrated.pth"
                
                if migrated_weights_path.exists():
                    logging.info(f"Found migrated weights, loading from: {migrated_weights_path}")
                    load_path = migrated_weights_path
                else:
                    logging.info(f"Loading original encoder weights from: {weights_path}")
                    logging.warning("⚠️  使用原始权重可能导致维度不匹配，建议先运行权重迁移脚本")
                    load_path = weights_path
            
            checkpoint = torch.load(load_path, map_location=device, weights_only=False)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                encoder_state_dict = checkpoint['model_state_dict']
                logging.info("Found 'model_state_dict' in checkpoint")
            else:
                encoder_state_dict = checkpoint
                logging.info("Using checkpoint directly as state_dict")
            
            # 处理item_num不匹配问题
            current_item_embedding_size = model.encoder.item_embedding.weight.shape
            checkpoint_item_embedding_size = encoder_state_dict.get('item_embedding.weight', torch.empty(0)).shape
            
            if checkpoint_item_embedding_size != current_item_embedding_size:
                logging.warning(f"Item embedding size mismatch:")
                logging.warning(f"   Current model: {current_item_embedding_size}")
                logging.warning(f"   Checkpoint: {checkpoint_item_embedding_size}")
                logging.info("   Adjusting item embedding size...")
                
                if len(checkpoint_item_embedding_size) > 0:
                    old_embedding = encoder_state_dict['item_embedding.weight']
                    new_embedding = model.encoder.item_embedding.weight.data.clone()
                    min_items = min(old_embedding.shape[0], new_embedding.shape[0])
                    new_embedding[:min_items] = old_embedding[:min_items]
                    encoder_state_dict['item_embedding.weight'] = new_embedding
                    logging.info(f"   ✅ Copied {min_items} item embeddings")
            
            missing_keys, unexpected_keys = model.encoder.load_state_dict(encoder_state_dict, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys: {unexpected_keys}")
                
            logging.info("✅ Pre-trained HSTU encoder weights loaded successfully")
            
        except Exception as e:
            logging.error(f"Could not load encoder weights: {e}")
            logging.info("Training from scratch...")

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        logging.info("🔒 Encoder weights frozen.")

    # 8. 文本嵌入加载到模型
    model.decoder.load_text_embeddings(text_embedding_matrix.to(device))
    
    # 【新增】智能图像嵌入加载系统 🎨
    if config['expert_system']['experts']['image_expert']:
        image_embeddings_path = args.image_embeddings_path or "data/book_image_embeddings_migrated.npy"
        
        if os.path.exists(image_embeddings_path):
            logging.info(f"🎨 Loading visual expert embeddings from: {image_embeddings_path}")
            try:
                # 加载图像嵌入字典
                image_embeddings_dict = np.load(image_embeddings_path, allow_pickle=True).item()
                
                if isinstance(image_embeddings_dict, dict) and len(image_embeddings_dict) > 0:
                    # 获取嵌入维度
                    sample_embedding = next(iter(image_embeddings_dict.values()))
                    image_embedding_dim = sample_embedding.shape[0]
                    logging.info(f"📐 Image embedding dimension: {image_embedding_dim}")
                    
                    # 🔧 修复：更新配置但不重新初始化整个模型
                    config['expert_system']['image_expert']['image_embedding_dim'] = image_embedding_dim
                    
                    # 🔧 修复：从配置中获取当前维度，而不是从占位符矩阵
                    current_image_dim = config['expert_system']['image_expert'].get('image_embedding_dim', None)
                    
                    # 只有当配置中的维度与实际文件维度不匹配时才重新初始化
                    if current_image_dim != image_embedding_dim:
                        logging.warning(f"图像嵌入维度不匹配: 配置={current_image_dim}, 文件={image_embedding_dim}")
                        logging.info("需要重新初始化模型以适配图像嵌入维度...")
                        
                        # 重新初始化模型
                        model = GENIUSRecModel(
                            config['encoder_model'], 
                            config['decoder_model'],
                            config['expert_system']
                        ).to(device)
                        
                        # 重新加载编码器权重
                        if args.encoder_weights_path:
                            try:
                                logging.info("🔄 重新加载编码器权重...")
                                checkpoint = torch.load(args.encoder_weights_path, map_location=device, weights_only=False)
                                encoder_state_dict = checkpoint.get('model_state_dict', checkpoint)
                                
                                # 处理item_num不匹配
                                current_item_embedding_size = model.encoder.item_embedding.weight.shape
                                checkpoint_item_embedding_size = encoder_state_dict.get('item_embedding.weight', torch.empty(0)).shape
                                
                                if checkpoint_item_embedding_size != current_item_embedding_size:
                                    if len(checkpoint_item_embedding_size) > 0:
                                        old_embedding = encoder_state_dict['item_embedding.weight']
                                        new_embedding = model.encoder.item_embedding.weight.data.clone()
                                        min_items = min(old_embedding.shape[0], new_embedding.shape[0])
                                        new_embedding[:min_items] = old_embedding[:min_items]
                                        encoder_state_dict['item_embedding.weight'] = new_embedding
                                
                                model.encoder.load_state_dict(encoder_state_dict, strict=False)
                                
                                if args.freeze_encoder:
                                    for param in model.encoder.parameters():
                                        param.requires_grad = False
                                        
                                # 重新加载文本嵌入（静默模式，避免重复日志）
                                model.decoder.load_text_embeddings(text_embedding_matrix.to(device), verbose=False)
                                
                            except Exception as e:
                                logging.error(f"重新加载编码器权重失败: {e}")
                    
                    # 初始化图像嵌入矩阵
                    num_items = total_vocab_size  # 使用总词汇表大小
                    image_embedding_matrix = torch.randn(num_items, image_embedding_dim, dtype=torch.float) * 0.01
                    
                    # 映射item_id并加载嵌入 - 现在统一使用item_id作为键
                    loaded_image_count = 0
                    
                    for item_id, embedding in image_embeddings_dict.items():
                        # 所有键现在都是item_id (整数)
                        if isinstance(item_id, (int, np.int32, np.int64)) and 0 <= item_id < num_items:
                            image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                            loaded_image_count += 1
                    
                    # 加载到模型
                    model.decoder.load_image_embeddings(image_embedding_matrix.to(device))
                    
                    # 统计信息
                    coverage_rate = (loaded_image_count / num_items) * 100
                    logging.info(f"✅ Visual Expert Integration Complete!")
                    logging.info(f"   📊 Loaded {loaded_image_count:,} image embeddings (item_id keys)")
                    logging.info(f"   📈 Coverage: {coverage_rate:.1f}% of {num_items:,} items")
                    
                    if coverage_rate < 50:
                        logging.warning(f"⚠️  Low image coverage ({coverage_rate:.1f}%). Consider generating more image embeddings.")
                    
                else:
                    raise ValueError("Empty or invalid image embeddings dictionary")
                    
            except Exception as e:
                logging.error(f"❌ Failed to load image embeddings: {e}")
                logging.info("🔄 Gracefully disabling visual expert...")
                config['expert_system']['experts']['image_expert'] = False
        else:
            logging.warning(f"📁 Image embeddings file not found: {image_embeddings_path}")
            logging.info("💡 To enable visual expert, generate image embeddings first:")
            logging.info(f"   python generate_image_embeddings.py --input_dir data/book_covers_enhanced --output_file {image_embeddings_path}")
            logging.info("🔄 Disabling visual expert for this run...")
            config['expert_system']['experts']['image_expert'] = False

    # 🚀 显存优化：启用混合精度训练
    try:
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    except AttributeError:
        # 兼容旧版本PyTorch
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # 9. 优化器和损失函数 (最终版差异化优化器)
    logging.info("为模型不同部分设置差异化学习率...")
    gate_params = [p for n, p in model.named_parameters() if 'gate_network' in n and p.requires_grad]
    encoder_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    
    embedding_params = []
    if getattr(model.decoder, 'text_embedding', None) is not None and model.decoder.expert_config['content_expert'].get('trainable_embeddings', False):
        embedding_params.extend(model.decoder.text_embedding.parameters())
    if getattr(model.decoder, 'image_embedding', None) is not None and model.decoder.expert_config['image_expert'].get('trainable_embeddings', False):
        embedding_params.extend(model.decoder.image_embedding.parameters())
        
    main_param_ids = {id(p) for p_group in [gate_params, encoder_params, embedding_params] for p in p_group}
    main_model_params = [p for p in model.parameters() if id(p) not in main_param_ids and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': main_model_params, 'lr': config['finetune']['learning_rate']['decoder_lr']},
        {'params': gate_params, 'lr': config['finetune']['learning_rate']['gate_lr']},
        {'params': encoder_params, 'lr': config['finetune']['learning_rate']['encoder_lr']},
        {'params': embedding_params, 'lr': config['finetune']['learning_rate'].get('embedding_lr', 1e-6)}
    ], weight_decay=config['finetune']['weight_decay'])

    logging.info(f"  - 主干网络学习率: {config['finetune']['learning_rate']['decoder_lr']}")
    logging.info(f"  - 门控网络学习率: {config['finetune']['learning_rate']['gate_lr']}")
    logging.info(f"  - 编码器学习率: {config['finetune']['learning_rate']['encoder_lr']}")
    if embedding_params:
        logging.info(f"  - 解冻的嵌入层学习率: {config['finetune']['learning_rate'].get('embedding_lr', 1e-6)}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=config['finetune'].get('label_smoothing', 0))

    # 10. 学习率调度器
    num_training_steps = len(train_loader) * config['finetune']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['finetune']['warmup_steps'], num_training_steps=num_training_steps)
    
    # 11. 检查点目录和混合精度
    output_dir = pathlib.Path(args.save_dir or config['data']['checkpoint_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / 'genius_rec_best.pth'
    latest_model_path = output_dir / 'genius_rec_latest.pth'
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 12. 断点续传
    start_epoch, best_val_loss, patience_counter = 0, float('inf'), 0
    resume_path = args.resume_from or latest_model_path
    if os.path.exists(resume_path):
        resume_info = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        if resume_info:
            start_epoch = resume_info['epoch'] + 1
            best_val_loss = resume_info['best_val_loss']
            patience_counter = resume_info['patience_counter']
            logging.info(f"✅ 成功恢复训练状态! 从 Epoch {start_epoch} 继续 (Best Val Loss: {best_val_loss:.4f})")

    # 13. 训练主循环 (最终版)
    logging.info("=== Starting Training Loop ===")
    for epoch in range(start_epoch, config['finetune']['num_epochs']):
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch,
            config['finetune']['num_epochs'], pad_token_id, config, scaler
        )
        
        num_candidates = None if (args.full_evaluation or args.sample_eval_size == 0) else args.sample_eval_size
        eval_results = evaluate_model_validation_with_ranking(
            model, val_loader, criterion, device,
            epoch, config['finetune']['num_epochs'], pad_token_id,
            config=config, num_candidates=num_candidates, top_k=top_k
        )
        
        current_val_loss = eval_results['val_loss']
        is_best = current_val_loss < best_val_loss

        if is_best:
            best_val_loss = current_val_loss
            patience_counter = 0
            logging.info(f"🎉 发现新的最佳模型! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        checkpoint_metrics = {'best_val_loss': best_val_loss, 'patience_counter': patience_counter, **eval_results}
        
        if is_best:
            save_checkpoint(best_model_path, model, optimizer, scheduler, epoch, checkpoint_metrics, config, num_items)
            logging.info(f"已保存最佳模型到: {best_model_path}")
        
        save_checkpoint(latest_model_path, model, optimizer, scheduler, epoch, checkpoint_metrics, config, num_items)
        logging.info(f"已保存最新检查点到: {latest_model_path}")

        logging.info(f"Epoch {epoch+1}/{config['finetune']['num_epochs']} Results:")
        logging.info(f"  📈 Train Loss: {avg_train_loss:.4f}")
        logging.info(f"  📉 Val Loss: {current_val_loss:.4f} (Best: {best_val_loss:.4f})")
        logging.info(f"  📊 Val PPL: {eval_results['val_ppl']:.4f}")
        logging.info(f"  🎯 Val HR@{top_k}: {eval_results['val_hr']:.4f}")
        logging.info(f"  🎯 Val NDCG@{top_k}: {eval_results['val_ndcg']:.4f}")
        
        for expert_name in enabled_experts:
            weight_key = f'avg_{expert_name}_weight'
            if weight_key in eval_results and eval_results[weight_key] is not None:
                logging.info(f"  ⚖️  {expert_name.replace('_', ' ').title()} Weight: {eval_results[weight_key]:.4f}")

        early_stopping_patience = config['finetune'].get('early_stopping_patience', 10)
        logging.info(f"耐心计数: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            logging.info(f"触发早停! 连续 {patience_counter} 个epoch性能未提升")
            break
            
        logging.info("-" * 80)

    # 14. 最终测试评估
    logging.info("=== Final Test Evaluation ===")
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型 {best_model_path} 进行测试...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_results = evaluate_model_test(model, test_loader, device, num_items, top_k=top_k)
        
        logging.info(f"📈 Final Test Results:")
        logging.info(f"  🎯 Test HR@{top_k}: {test_results['test_hr']:.4f}")
        logging.info(f"  🎯 Test NDCG@{top_k}: {test_results['test_ndcg']:.4f}")
    else:
        logging.warning("未找到最佳模型文件，跳过测试集评估。")

if __name__ == '__main__':
    main()