# evaluate_checkpoint.py (最终修复版)

import argparse
import logging
import os
import pickle
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- 导入您项目中的必要组件 ---
from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.dataset import Seq2SeqRecDataset
from src.unified_evaluation import evaluate_model_validation_with_ranking

def fix_state_dict_keys(state_dict):
    """
    智能修复state_dict中的键名，以适配模型架构的变更。
    将旧的 content_expert_attention 权重重命名为新的 shared_cross_attention。
    """
    new_state_dict = OrderedDict()
    for old_key, value in state_dict.items():
        new_key = old_key
        # 如果键名包含旧的注意力模块名称，将其替换为新的共享模块名称
        if "content_expert_attention" in old_key:
            new_key = old_key.replace("content_expert_attention", "shared_cross_attention")
            logging.info(f"键名转换: '{old_key}' -> '{new_key}'")
        # 忽略掉已经被废弃的 image_expert_attention 的权重
        elif "image_expert_attention" in old_key:
            logging.warning(f"发现已废弃的键: '{old_key}'，将予以忽略。")
            continue
            
        new_state_dict[new_key] = value
        
    return new_state_dict


def run_evaluation():
    """
    一个健壮的临时脚本，用于加载检查点并使用修正后的逻辑进行一次性评估。
    """
    # 1. 设置: 参数、配置、设备、日志
    # =================================================================
    parser = argparse.ArgumentParser(description="Evaluate a GENIUS-Rec Checkpoint")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file (.pth).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = get_config()
    device = torch.device(config['device'])
    logging.info(f"使用设备: {device}")

    # 2. 加载数据和元数据
    # =================================================================
    logging.info("正在加载数据和ID映射...")
    try:
        with open(config['data']['id_maps_file'], 'rb') as f:
            id_maps = pickle.load(f)
        val_dataset = Seq2SeqRecDataset(config, config['data']['validation_file'])
    except FileNotFoundError as e:
        logging.error(f"数据文件未找到: {e}。请先运行 preprocess.py。")
        return

    num_special_tokens = id_maps.get('num_special_tokens', 4)
    total_vocab_size = id_maps['num_items'] + num_special_tokens
    config['encoder_model']['item_num'] = total_vocab_size
    config['decoder_model']['num_items'] = total_vocab_size
    pad_token_id = config['pad_token_id']
    top_k = config['evaluation']['top_k']

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['finetune']['batch_size'] * 4,
        shuffle=False,
        num_workers=config['finetune'].get('num_workers', 0)
    )
    logging.info(f"已加载 {len(val_dataset)} 个验证样本。")

    # 3. 初始化模型
    # =================================================================
    logging.info("正在初始化 GENIUS-Rec 模型...")
    model = GENIUSRecModel(config['encoder_model'], config['decoder_model'], config['expert_system']).to(device)

    # 4. 【核心修复】在加载检查点 *之前*，完成所有准备工作
    # =================================================================
    
    # 步骤 A: 加载外部嵌入以确保尺寸正确
    logging.info("正在加载外部嵌入矩阵以同步模型尺寸...")
    # 加载文本嵌入
    text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered_migrated.npy'
    text_embeddings_dict = np.load(text_embedding_file, allow_pickle=True).item()
    text_embedding_dim = next(iter(text_embeddings_dict.values())).shape[0]
    text_embedding_matrix = torch.zeros(total_vocab_size, text_embedding_dim, dtype=torch.float)
    item_asin_map = id_maps['item_map']
    for asin, embedding in text_embeddings_dict.items():
        if asin in item_asin_map:
            text_embedding_matrix[item_asin_map[asin]] = torch.tensor(embedding, dtype=torch.float)
    model.decoder.load_text_embeddings(text_embedding_matrix.to(device), verbose=False)
    
    # 加载图像嵌入
    image_embeddings_path = "data/book_image_embeddings_migrated.npy"
    image_embeddings_dict = np.load(image_embeddings_path, allow_pickle=True).item()
    image_embedding_dim = next(iter(image_embeddings_dict.values())).shape[0]
    image_embedding_matrix = torch.randn(total_vocab_size, image_embedding_dim, dtype=torch.float) * 0.01
    for item_id, embedding in image_embeddings_dict.items():
        if isinstance(item_id, (int, np.int32, np.int64)) and 0 <= item_id < total_vocab_size:
            image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
    model.decoder.load_image_embeddings(image_embedding_matrix.to(device), verbose=False)
    logging.info("✅ 外部嵌入加载完成，模型尺寸已同步。")

    # 步骤 B: 应用权重绑定
    if model.decoder.final_projection is not None:
        model.decoder.final_projection.weight = model.encoder.item_embedding.weight
        logging.info("✅ 权重绑定已应用。")

    # 5. 加载检查点 (包含智能修复逻辑)
    # =================================================================
    if not os.path.exists(args.checkpoint_path):
        logging.error(f"检查点文件未找到: {args.checkpoint_path}")
        return
        
    logging.info(f"正在从 '{args.checkpoint_path}' 加载模型状态...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # 步骤 C: 智能修复键名
    fixed_state_dict = fix_state_dict_keys(checkpoint['model_state_dict'])
    
    # 使用修复后的 state_dict 加载，设置 strict=False 以忽略不匹配的键
    model.load_state_dict(fixed_state_dict, strict=False)
    logging.info("✅ 模型状态加载成功 (已智能修复架构不匹配问题)。")

    # 6. 运行评估
    # =================================================================
    logging.info("正在使用修正后的逻辑开始评估...")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    eval_results = evaluate_model_validation_with_ranking(
        model=model, val_loader=val_loader, criterion=criterion, device=device,
        epoch=0, num_epochs=1, pad_token_id=pad_token_id, config=config, top_k=top_k
    )

    # 7. 打印结果
    # =================================================================
    logging.info("🎉 --- 评估完成 --- 🎉")
    logging.info(f"  📈 验证集 Loss: {eval_results['val_loss']:.4f}")
    logging.info(f"  📊 验证集 PPL: {eval_results['val_ppl']:.4f}")
    logging.info(f"  🎯 验证集 HR@{top_k}: {eval_results['val_hr']:.4f}")
    logging.info(f"  🎯 验证集 NDCG@{top_k}: {eval_results['val_ndcg']:.4f}")
    
    enabled_experts = [k for k, v in config['expert_system']['experts'].items() if v]
    for expert_name in enabled_experts:
        weight_key = f'avg_{expert_name}_weight'
        if weight_key in eval_results:
            logging.info(f"  ⚖️  {expert_name.replace('_', ' ').title()} 权重: {eval_results[weight_key]:.4f}")

if __name__ == '__main__':
    run_evaluation()