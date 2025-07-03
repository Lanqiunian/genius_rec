# src/train_GeniusRec.py
import argparse
import logging
import os
import pickle
import random
import math
import pathlib
import platform

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

from src.encoder.encoder import Hstu
from src.decoder.decoder import GenerativeDecoder


# # 从头开始训练
# python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --freeze_encoder

# # 从检查点恢复训练
# python -m src.train_GeniusRec --resume_from checkpoints/genius_rec_moe_latest.pth --encoder_weights_path checkpoints/hstu_encoder.pth

# # 自定义保存目录
# python -m src.train_GeniusRec --save_dir my_checkpoints

# --- 无数据泄露的评估数据集 ---
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

def compute_ranking_metrics(user_embeddings, all_item_embeddings, target_item_ids, k=10):
    """
    计算HR@K和NDCG@K指标
    """
    batch_size = user_embeddings.size(0)
    
    # L2归一化
    user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    all_item_embeddings = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 计算余弦相似度
    scores = torch.matmul(user_embeddings, all_item_embeddings.t())
    
    # 排序
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
    
    hr_list, ndcg_list = [], []
    
    for i in range(batch_size):
        target_id = target_item_ids[i].item()
        if target_id == 0: 
            continue

        target_idx = target_id - 1  # ID到索引的转换
        target_rank_positions = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0]
        
        hr, ndcg = 0.0, 0.0
        if len(target_rank_positions) > 0:
            rank = target_rank_positions[0].item() + 1
            if rank <= k:
                hr = 1.0
                ndcg = 1.0 / np.log2(rank + 1)
        
        hr_list.append(hr)
        ndcg_list.append(ndcg)
    
    return hr_list, ndcg_list

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, num_epochs, pad_token_id, force_equal_weights=False):
    """
    训练一个epoch，并实时监控损失和专家权重。

    Args:
        model: 待训练的模型。
        dataloader: 训练数据加载器。
        criterion: 损失函数。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        device: 'cuda' 或 'cpu'。
        epoch (int): 当前的epoch数。
        num_epochs (int): 总的epoch数。
        pad_token_id (int): 用于padding的token ID。
        force_equal_weights (bool): 是否强制专家使用均等权重（用于预热）。

    Returns:
        float: 该epoch的平均训练损失。
    """
    model.train()
    total_loss = 0.0
    
    # 初始化用于在进度条上显示的权重信息
    weights_postfix = {'Bhv W': 0.0, 'Cnt W': 0.0}

    # 使用tqdm来实时显示loss和权重
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=True)

    for batch_idx, batch in enumerate(progress_bar):
        source_ids = batch['source_ids'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)
        source_padding_mask = (source_ids == pad_token_id)

        optimizer.zero_grad()
        
        # ==================== ✨ 核心修改 ✨ ====================
        # 调用模型时，传入force_equal_weights并要求返回权重
        logits, gate_weights = model(
            source_ids, 
            decoder_input_ids, 
            source_padding_mask,
            force_equal_weights=force_equal_weights, # 传入控制标志
            return_weights=True                      # 要求返回权重以供监控
        )
        # =======================================================

        # 计算损失 (保持不变)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 仅在需要时更新学习率 (取决于您的scheduler类型)
        if scheduler is not None:
             scheduler.step()
        
        total_loss += loss.item()
        
        # --- 实时更新进度条后缀 ---
        if gate_weights is not None:
            # 取一个batch的平均权重来观察
            # 假设第一个专家是Behavior, 第二个是Content
            weights_postfix['Bhv W'] = gate_weights[:, :, 0].mean().item()
            if gate_weights.shape[-1] > 1:
                 weights_postfix['Cnt W'] = gate_weights[:, :, 1].mean().item()
            else:
                 weights_postfix['Cnt W'] = 0.0 # 如果只有一个专家

        # 更新进度条的显示信息，合并loss和权重
        current_postfix = {'loss': f"{loss.item():.4f}", **weights_postfix}
        progress_bar.set_postfix(current_postfix)
        
    avg_loss = total_loss / len(dataloader)
    
    # 在每个epoch结束时，通过logging记录最终的平均loss
    # (权重信息已经在进度条中实时显示了)
    logging.info(f"Epoch {epoch+1} training finished. Average Loss: {avg_loss:.4f}")
    
    return avg_loss # <-- 保持返回值不变

def evaluate_model_validation(model, val_loader, criterion, device, epoch, num_epochs, pad_token_id):
    """
    验证集评估：只计算loss和ppl，用于早停和模型选择
    """
    model.eval()
    
    total_loss_tokens = 0.0
    total_tokens = 0
    total_gate_weights = None
    total_valid_batches = 0

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")

    with torch.no_grad():
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)

            logits, gate_weights = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=True)
            
            # 修正：使用传统的CrossEntropyLoss调用方式
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            valid_tokens = (labels.view(-1) != pad_token_id).sum().item()
            total_loss_tokens += loss.item() * valid_tokens
            total_tokens += valid_tokens

            # 累计门控权重（动态支持多个专家）
            non_padding_mask = (decoder_input_ids != pad_token_id)
            if gate_weights.size(-1) > 0:  # 确保有专家
                masked_gate_weights = gate_weights[non_padding_mask]  # (N, num_experts)
                if masked_gate_weights.numel() > 0:
                    if total_gate_weights is None:
                        total_gate_weights = masked_gate_weights.mean(dim=0)  # (num_experts,)
                        total_valid_batches = 1
                    else:
                        total_gate_weights += masked_gate_weights.mean(dim=0)
                        total_valid_batches += 1

            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss_tokens / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    # 计算平均门控权重
    avg_gate_weights = total_gate_weights / total_valid_batches if total_valid_batches > 0 else None
    
    result = {
        'val_loss': avg_loss,
        'val_ppl': perplexity,
    }
    
    # 动态添加专家权重信息
    if avg_gate_weights is not None:
        enabled_experts = [k for k, v in model.decoder.expert_config["experts"].items() if v]
        for i, expert_name in enumerate(enabled_experts):
            if i < len(avg_gate_weights):
                result[f'avg_{expert_name}_weight'] = avg_gate_weights[i].item()
    
    return result

def evaluate_model_test(model, test_loader, device, item_num, top_k=10):
    """
    测试集评估：只计算排序指标，训练结束后调用一次
    """
    model.eval()
    all_hr_scores, all_ndcg_scores = [], []
    
    with torch.no_grad():
        # 预先计算所有物品嵌入，避免重复计算
        all_item_ids = torch.arange(1, item_num, device=device)
        all_item_embeddings = model.encoder.item_embedding(all_item_ids)
        
        progress_bar = tqdm(test_loader, desc="Test Set Evaluation")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            ground_truth_ids = batch['ground_truth'].to(device)

            # 获取用户嵌入
            encoder_output = model.encoder(input_ids)
            user_embeddings = encoder_output[:, -1, :]  # 取最后一个位置
            
            # 计算HR和NDCG
            hr_list, ndcg_list = compute_ranking_metrics(
                user_embeddings,
                all_item_embeddings,
                ground_truth_ids,
                k=top_k
            )
            all_hr_scores.extend(hr_list)
            all_ndcg_scores.extend(ndcg_list)
    
    avg_hr = np.mean(all_hr_scores) if all_hr_scores else 0.0
    avg_ndcg = np.mean(all_ndcg_scores) if all_ndcg_scores else 0.0
    
    return {
        'test_hr': avg_hr,
        'test_ndcg': avg_ndcg,
        'evaluated_samples': len(all_hr_scores)
    }

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
    
    # 【新增】专家系统控制参数
    parser.add_argument('--disable_behavior_expert', action='store_true', help='Disable behavior expert.')
    parser.add_argument('--disable_content_expert', action='store_true', help='Disable content expert.')
    parser.add_argument('--enable_image_expert', action='store_true', help='Enable image expert (requires image embeddings).')
    parser.add_argument('--image_embeddings_path', type=str, default=None, help='Path to image embeddings file.')
    
    args = parser.parse_args()

    config = get_config()
    
    # 【新增】根据命令行参数动态调整专家配置
    if args.disable_behavior_expert:
        config['expert_system']['experts']['behavior_expert'] = False
    if args.disable_content_expert:
        config['expert_system']['experts']['content_expert'] = False
    if args.enable_image_expert:
        config['expert_system']['experts']['image_expert'] = True
    
    # 2. 环境设置
    device = torch.device(config['device'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # 3. 日志设置
    log_dir_path = pathlib.Path(config['data']['log_dir'])
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / 'train_genius_rec.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("=== Starting GENIUS-Rec Training ===")
    logging.info(f"Device: {device}")
    logging.info(f"Arguments: {args}")

    # 4. 数据加载 - 修正：使用独立的测试集
    logging.info("Loading data from processed directory...")
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)

    train_dataset = Seq2SeqRecDataset(config['data']['train_file'], config['decoder_model']['max_seq_len'])
    val_dataset = Seq2SeqRecDataset(config['data']['validation_file'], config['decoder_model']['max_seq_len'])
    test_dataset = ValidationDataset(
        config['data']['test_file'],  # 修正：使用独立的测试集
        config['encoder_model']['max_len']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['finetune']['batch_size'], shuffle=True, num_workers=config['finetune']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['finetune']['batch_size'], shuffle=False, num_workers=config['finetune']['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['finetune']['batch_size'], shuffle=False, num_workers=config['finetune']['num_workers'])
    
    num_items = id_maps['num_items'] + 1
    pad_token_id = config['pad_token_id']
    top_k = config['evaluation']['top_k']
    
    logging.info(f"📊 Dataset Info:")
    logging.info(f"  - Training samples: {len(train_dataset)}")
    logging.info(f"  - Validation samples: {len(val_dataset)}")
    logging.info(f"  - Test samples: {len(test_dataset)}")  # 修正：显示测试集样本数
    logging.info(f"  - Total items: {num_items}")

    # 5. 文本嵌入加载
    logging.info("Loading pre-computed and filtered text embeddings...")
    text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered.npy'
    try:
        text_embeddings_dict = np.load(text_embedding_file, allow_pickle=True).item()
    except FileNotFoundError:
        logging.error(f"Filtered embedding file not found at '{text_embedding_file}'! Please run filter_embeddings.py first.")
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
    
    logging.info(f"Successfully loaded and mapped {loaded_count} text embeddings.")
    if loaded_count == 0:
        logging.warning("No embeddings were mapped. The content expert will not function.")

    # 6. 模型初始化
    config['encoder_model']['item_num'] = num_items
    config['decoder_model']['num_items'] = num_items
    config['decoder_model']['text_embedding_dim'] = text_embedding_dim
    
    # 【新增】传递专家配置到模型
    model = GENIUSRecModel(
        config['encoder_model'], 
        config['decoder_model'],
        config['expert_system']  # 专家系统配置
    ).to(device)
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
    
    # 7. 预训练权重加载（如果指定）- 修正：移除跨平台hack
    if args.encoder_weights_path:
        try:
            logging.info(f"Loading encoder weights from: {args.encoder_weights_path}")
            
            checkpoint = torch.load(args.encoder_weights_path, map_location=device, weights_only=False)
            
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
        image_embeddings_path = args.image_embeddings_path or "data/book_image_embeddings.npy"
        
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
                    
                    # 更新配置中的图像嵌入维度
                    config['expert_system']['image_expert']['image_embedding_dim'] = image_embedding_dim
                    
                    # 初始化图像嵌入矩阵 (使用小的随机值初始化未匹配的项目)
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

    # 9. 优化器和损失函数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['finetune']['learning_rate']['decoder_lr'],
        weight_decay=config['finetune'].get('weight_decay', 0.01)
    )


    # 标签平滑
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=config['finetune'].get('label_smoothing', 0))

    # 10. 学习率调度器
    num_training_steps = len(train_loader) * config['finetune']['num_epochs']
    num_warmup_steps = config['finetune'].get('warmup_steps', 500)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )
    
    logging.info(f"Using learning rate scheduler with {num_warmup_steps} warmup steps and {num_training_steps} total training steps.")

    # 11. 检查点目录设置
    if args.save_dir:
        output_dir = pathlib.Path(args.save_dir)
    else:
        output_dir = config['data']['checkpoint_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = output_dir / 'genius_rec_best.pth'  # 修正：只保存一个最佳模型
    latest_model_path = output_dir / 'genius_rec_latest.pth'

    # 12. 断点续传
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    resume_path = args.resume_from or latest_model_path
    if os.path.exists(resume_path):
        resume_info = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        if resume_info:
            start_epoch = resume_info['epoch'] + 1
            best_val_loss = resume_info['best_val_loss']
            patience_counter = resume_info['patience_counter']
            logging.info(f"✅ 成功恢复训练状态! 从 Epoch {start_epoch} 继续")
            logging.info(f"   - Best Val Loss: {best_val_loss:.4f}")

    # 13. 训练主循环
    logging.info("=== Starting Training Loop ===")
    warmup_epochs = config['finetune'].get('warmup_epochs', 2)
    for epoch in range(start_epoch, config['finetune']['num_epochs']):
        # 训练一个epoch
        is_warmup_phase = (epoch < warmup_epochs)
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            device, epoch, config['finetune']['num_epochs'], pad_token_id,
            force_equal_weights=is_warmup_phase
        )
        
        # 验证集评估（只计算loss和ppl）
        eval_results = evaluate_model_validation(
            model, val_loader, criterion, device, 
            epoch, config['finetune']['num_epochs'], pad_token_id
        )
        
        # 日志输出
        logging.info(f"Epoch {epoch+1}/{config['finetune']['num_epochs']} Results:")
        logging.info(f"  📈 Train Loss: {avg_train_loss:.4f}")
        logging.info(f"  📉 Val Loss: {eval_results['val_loss']:.4f}")
        logging.info(f"  📊 Val PPL: {eval_results['val_ppl']:.4f}")
        
        # 动态显示专家权重
        enabled_experts = [k for k, v in config['expert_system']['experts'].items() if v]
        for expert_name in enabled_experts:
            weight_key = f'avg_{expert_name}_weight'
            if weight_key in eval_results:
                logging.info(f"  ⚖️  {expert_name.replace('_', ' ').title()} Weight: {eval_results[weight_key]:.4f}")

        # 准备保存的指标
        save_metrics = {
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            **eval_results
        }

        # 保存最新检查点
        save_checkpoint(
            latest_model_path, model, optimizer, scheduler, 
            epoch, save_metrics, config, num_items
        )
        logging.info(f"保存最新检查点到: {latest_model_path}")

        # 修正：只基于验证loss保存最佳模型
        if eval_results['val_loss'] < best_val_loss:
            best_val_loss = eval_results['val_loss']
            patience_counter = 0
            save_metrics['best_val_loss'] = best_val_loss
            
            save_checkpoint(
                best_model_path, model, optimizer, scheduler, 
                epoch, save_metrics, config, num_items
            )
            logging.info(f"🎉 发现新的最佳模型! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # 早停检查
        early_stopping_patience = config['finetune'].get('early_stopping_patience', 10)
        if patience_counter >= early_stopping_patience:
            logging.info(f"触发早停! 连续 {patience_counter} 个epoch性能未提升")
            break
            
        logging.info(f"耐心计数: {patience_counter}/{early_stopping_patience}")
        logging.info("-" * 80)

    # 14. 训练完成，在测试集上进行最终评估
    completed_epochs = epoch + 1 if 'epoch' in locals() else start_epoch
    logging.info("=== Training Finished ===")
    logging.info(f"训练总轮次: {completed_epochs}/{config['finetune']['num_epochs']}")
    logging.info(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 15. 加载最佳模型并在测试集上评估
    logging.info("=== Final Test Evaluation ===")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("已加载最佳模型进行测试集评估")
        
        test_results = evaluate_model_test(model, test_loader, device, num_items, top_k)
        
        logging.info(f"📈 Final Test Results:")
        logging.info(f"  🎯 Test HR@{top_k}: {test_results['test_hr']:.4f}")
        logging.info(f"  🎯 Test NDCG@{top_k}: {test_results['test_ndcg']:.4f}")
        logging.info(f"  📊 Test samples: {test_results['evaluated_samples']}")
    else:
        logging.warning("未找到最佳模型文件，跳过测试集评估")
    
    logging.info(f"检查点保存位置:")
    logging.info(f"  - Latest: {latest_model_path}")
    logging.info(f"  - Best: {best_model_path}")

if __name__ == '__main__':
    main()