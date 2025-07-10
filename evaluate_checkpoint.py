"""
独立评估脚本 (evaluate_checkpoint.py) - 修正版 v2.0

该脚本用于加载一个已经训练好的 GENIUS-Rec 模型检查点，
并在指定的数据集（验证集或测试集）上进行评估，以复现或验证模型性能。

【v2.0 修正内容】
- 修复了因未加载专家系统（文本/图像）的外部嵌入矩阵，导致的 "size mismatch" 运行时错误。
- 确保评估时模型的结构与训练并保存时完全一致。

【使用示例】
# 评估默认的最新检查点在测试集上的表现
python evaluate_checkpoint.py

# 指定一个特定的检查点文件 (例如最好的模型)
python evaluate_checkpoint.py --checkpoint_path checkpoints/genius_rec_best.pth
"""
import argparse
import logging
import pickle
import pathlib

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# --- 从您的项目中导入必要的模块 ---
from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.dataset import Seq2SeqRecDataset
from src.evaluation import (
    ValidationDataset,
    evaluate_model_test,
    evaluate_model_validation_with_ranking
)

def main():
    # 1. --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Evaluate a GENIUS-Rec Model Checkpoint")
    parser.add_argument(
        '--checkpoint_path',
        type=pathlib.Path,
        default='checkpoints/genius_rec_latest.pth',
        help='要评估的检查点文件路径'
    )
    # (其余参数解析保持不变)
    parser.add_argument(
        '--dataset', type=str, default='test', choices=['test', 'validation'],
        help='在其上进行评估的数据集 ("test" 或 "validation")'
    )
    parser.add_argument('--batch_size', type=int, default=128, help='评估时使用的批处理大小')
    parser.add_argument('--top_k', type=int, default=10, help='评估指标的K值 (例如, HR@K, NDCG@K)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行评估的设备')
    args = parser.parse_args()

    # 2. --- 环境与日志设置 ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(args.device)

    # 3. --- 加载检查点和配置 ---
    if not args.checkpoint_path.exists():
        logging.error(f"检查点文件未找到: {args.checkpoint_path}")
        return
    logging.info(f"正在加载检查点: {args.checkpoint_path}")
    try:
        # 在新版 PyTorch 中, 推荐使用 weights_only=True 以保证安全，但这里我们信任自己的检查点
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        logging.error(f"加载检查点失败: {e}")
        return

    config = checkpoint.get('config')
    if config is None:
        logging.warning("检查点中未找到配置信息，将使用默认配置 `get_config()`")
        config = get_config()
    config['device'] = args.device
    logging.info(f"模型配置加载成功。将在 '{args.dataset}' 数据集上进行评估。")

    # 4. --- 数据加载 (包括ID映射) ---
    logging.info("正在加载ID映射和数据集...")
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
    
    num_items = id_maps['num_items'] + id_maps.get('num_special_tokens', 4)
    pad_token_id = config['pad_token_id']

    if args.dataset == 'test':
        dataset = ValidationDataset(config['data']['test_file'], config['encoder_model']['max_len'], pad_token_id)
    else:
        dataset = Seq2SeqRecDataset(config, config['data']['validation_file'], is_validation=True, item_maps=id_maps)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=config['finetune'].get('num_workers', 2))
    logging.info(f"数据集加载完毕，共 {len(dataset)} 条样本。")

    # 5. --- 模型初始化与权重加载 ---
    logging.info("正在初始化模型...")
    model = GENIUSRecModel(config).to(device)
    
    # --- ↓↓↓ 新增的核心修复代码 ↓↓↓ ---
    # 在加载 state_dict 之前，必须先加载外部嵌入，以确保模型结构完全匹配
    
    # 5.1 加载文本嵌入 (如果内容专家被启用)
    if config['expert_system']['experts'].get('content_expert', False):
        logging.info("内容专家已启用，正在加载文本嵌入...")
        text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered_migrated.npy'
        text_embeddings_dict = np.load(text_embedding_file, allow_pickle=True).item()
        text_embedding_dim = next(iter(text_embeddings_dict.values())).shape[0]
        text_embedding_matrix = torch.zeros(num_items, text_embedding_dim, dtype=torch.float)
        
        item_asin_map = id_maps['item_map']
        for asin, embedding in text_embeddings_dict.items():
            if asin in item_asin_map:
                item_id = item_asin_map[asin]
                text_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
        
        model.decoder.load_text_embeddings(text_embedding_matrix.to(device))
        logging.info("文本嵌入加载并设置成功。")

    # 5.2 加载图像嵌入 (如果图像专家被启用)
    if config['expert_system']['experts'].get('image_expert', False):
        logging.info("图像专家已启用，正在加载图像嵌入...")
        # 假设图像嵌入路径在训练脚本中是固定的
        image_embeddings_path = "data/book_image_embeddings_migrated.npy"
        image_embeddings_dict = np.load(image_embeddings_path, allow_pickle=True).item()
        
        sample_embedding = next(iter(image_embeddings_dict.values()))
        image_embedding_dim = sample_embedding.shape[0]
        image_embedding_matrix = torch.randn(num_items, image_embedding_dim, dtype=torch.float) * 0.01

        for item_id, embedding in image_embeddings_dict.items():
            if isinstance(item_id, (int, np.int32, np.int64)) and 0 <= item_id < num_items:
                image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                
        model.decoder.load_image_embeddings(image_embedding_matrix.to(device))
        logging.info("图像嵌入加载并设置成功。")

    # --- ↑↑↑ 新增的核心修复代码结束 ↑↑↑ ---
    
    # 5.3 现在可以安全地加载模型权重了
    if 'model_state_dict' not in checkpoint:
        logging.error("检查点中缺少 'model_state_dict'。")
        return
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 切换到评估模式
    logging.info("模型权重加载成功。")

    # 6. --- 执行评估 ---
    # (此部分无需修改)
    logging.info(f"--- 开始在 {args.dataset.upper()} 集上评估 (Top-K={args.top_k}) ---")
    results = {}
    if args.dataset == 'test':
        results = evaluate_model_test(model, dataloader, device, num_items, top_k=args.top_k, config=config)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        results = evaluate_model_validation_with_ranking(
            model, dataloader, criterion, device, 0, 1, pad_token_id, config, top_k=args.top_k)

    # 7. --- 打印评估结果 ---
    # (此部分无需修改)
    logging.info("--- 评估完成 ---")
    logging.info(f"评估检查点: {args.checkpoint_path.name}")
    logging.info(f"评估数据集: {args.dataset.upper()}")
    for metric, value in results.items():
        logging.info(f"  - {metric}: {value:.4f}")

if __name__ == '__main__':
    main()