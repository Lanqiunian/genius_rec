#!/usr/bin/env python3
"""
临时测试脚本：验证模型验证阶段是否正常工作
直接基于训练脚本的逻辑来构建和测试模型
"""

import os
import sys
import torch
import logging
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.dataset import Seq2SeqRecDataset
from src.unified_evaluation import ValidationDataset, evaluate_model_validation_with_ranking

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_validation():
    """测试验证功能 - 直接复制训练脚本的逻辑"""
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 检查模型文件是否存在
        model_path = "checkpoints/genius_rec_best_0.pth"
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型文件不存在: {model_path}")
            return False
            
        logger.info(f"✅ 找到模型文件: {model_path}")
        
        # === 完全复制训练脚本的数据加载逻辑 ===
        config = get_config()
        
        # 数据加载和ID映射处理 (来自train_GeniusRec.py line 249-262)
        logger.info("Loading data from processed directory...")
        with open(config['data']['id_maps_file'], 'rb') as f:
            id_maps = pickle.load(f)
        num_special_tokens = id_maps.get('num_special_tokens', 4)
        total_vocab_size = id_maps['num_items'] + num_special_tokens
        config['encoder_model']['item_num'] = total_vocab_size
        config['decoder_model']['num_items'] = total_vocab_size
        num_items = total_vocab_size
        logger.info(f"📊 Vocabulary Info: Total vocabulary size: {total_vocab_size}")
        
        # 创建数据集 (来自train_GeniusRec.py line 263-270)
        val_dataset = Seq2SeqRecDataset(config, config['data']['validation_file'])
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)  # 减小batch_size
        pad_token_id = config['pad_token_id']
        top_k = config['evaluation']['top_k']
        logger.info(f"📊 Dataset Info: Validation samples: {len(val_dataset)}")

        # 文本嵌入加载 (来自train_GeniusRec.py line 273-289)
        logger.info("Loading pre-computed and filtered text embeddings...")
        text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered_migrated.npy'
        try:
            text_embeddings_dict = np.load(text_embedding_file, allow_pickle=True).item()
        except FileNotFoundError:
            logger.error(f"Filtered embedding file not found at '{text_embedding_file}'!")
            return False

        text_embedding_dim = next(iter(text_embeddings_dict.values())).shape[0]
        text_embedding_matrix = torch.zeros(total_vocab_size, text_embedding_dim, dtype=torch.float)
        
        item_asin_map = id_maps['item_map']
        loaded_count = 0
        for asin, embedding in text_embeddings_dict.items():
            if asin in item_asin_map:
                item_id = item_asin_map[asin]
                text_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                loaded_count += 1
        
        logger.info(f"Successfully loaded and mapped {loaded_count} text embeddings.")

        # 模型初始化 (来自train_GeniusRec.py line 291-299)
        config['decoder_model']['text_embedding_dim'] = text_embedding_dim
        model = GENIUSRecModel(config['encoder_model'], config['decoder_model'], config['expert_system']).to(device)
        logger.info("GENIUS-Rec model created with expert configuration.")
        
        enabled_experts = [k for k, v in config['expert_system']['experts'].items() if v]
        logger.info(f"🧠 启用的专家: {enabled_experts}")
        
        # 加载模型权重
        logger.info("💾 加载模型权重...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # 使用strict=False
            logger.info("✅ 从 'model_state_dict' 加载权重")
        else:
            model.load_state_dict(checkpoint, strict=False)
            logger.info("✅ 直接加载权重")
            
        # 加载文本嵌入 (来自train_GeniusRec.py line 387)
        model.decoder.load_text_embeddings(text_embedding_matrix.to(device))
        
        # 加载图像嵌入 (来自train_GeniusRec.py line 393-480)
        if config['expert_system']['experts']['image_expert']:
            image_embeddings_path = "data/book_image_embeddings_migrated.npy"
            
            if os.path.exists(image_embeddings_path):
                logger.info(f"🎨 Loading visual expert embeddings from: {image_embeddings_path}")
                try:
                    image_embeddings_dict = np.load(image_embeddings_path, allow_pickle=True).item()
                    
                    if isinstance(image_embeddings_dict, dict) and len(image_embeddings_dict) > 0:
                        sample_embedding = next(iter(image_embeddings_dict.values()))
                        image_embedding_dim = sample_embedding.shape[0]
                        logger.info(f"📐 Image embedding dimension: {image_embedding_dim}")
                        
                        # 更新配置
                        config['expert_system']['image_expert']['image_embedding_dim'] = image_embedding_dim
                        
                        # 创建图像嵌入矩阵
                        image_embedding_matrix = torch.zeros(num_items, image_embedding_dim, dtype=torch.float)
                        loaded_image_count = 0
                        
                        for item_id, embedding in image_embeddings_dict.items():
                            if isinstance(item_id, (int, np.int32, np.int64)) and 0 <= item_id < num_items:
                                image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                                loaded_image_count += 1
                        
                        model.decoder.load_image_embeddings(image_embedding_matrix.to(device))
                        
                        coverage_rate = (loaded_image_count / num_items) * 100
                        logger.info(f"✅ Visual Expert Integration Complete!")
                        logger.info(f"   📊 Loaded {loaded_image_count:,} image embeddings")
                        logger.info(f"   📈 Coverage: {coverage_rate:.1f}% of {num_items:,} items")
                        
                except Exception as e:
                    logger.error(f"Failed to load image embeddings: {e}")
                    return False
            else:
                logger.warning(f"⚠️ 图像嵌入文件不存在: {image_embeddings_path}")
        
        # 设置模型为评估模式
        model.eval()
        logger.info("🔍 模型设置为评估模式")
        
        # 测试验证功能 - 直接复制训练脚本的全量评估逻辑
        logger.info("🧪 开始测试验证功能...")
        
        # 创建损失函数 (来自train_GeniusRec.py line 530)
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=config['finetune'].get('label_smoothing', 0))
        
        # 使用全量评估，直接对齐训练脚本 (来自train_GeniusRec.py line 560-567)
        num_candidates = None  # 全量评估，不限制候选数量
        
        try:
            eval_results = evaluate_model_validation_with_ranking(
                model, val_loader, criterion, device,
                0, 1, pad_token_id,  # epoch=0, num_epochs=1
                config=config, num_candidates=num_candidates, top_k=top_k
            )
            
            logger.info("🎉 验证功能测试成功！")
            logger.info("📊 验证结果:")
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   - {metric}: {value:.4f}")
                else:
                    logger.info(f"   - {metric}: {value}")
                
            return True
            
        except Exception as eval_error:
            logger.error(f"❌ 验证功能测试失败: {eval_error}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🧪 开始验证功能测试")
    logger.info("=" * 60)
    
    # 检查GPU内存
    if torch.cuda.is_available():
        logger.info(f"💾 GPU显存状态:")
        logger.info(f"   - 已分配: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        logger.info(f"   - 已预留: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        logger.info(f"   - 总容量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        logger.info(f"   - 剩余可用: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1024**3:.2f}GB")
    
    success = test_validation()
    
    logger.info("=" * 60)
    if success:
        logger.info("✅ 验证功能测试通过！可以安全进行完整训练。")
    else:
        logger.info("❌ 验证功能测试失败！需要修复问题。")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
