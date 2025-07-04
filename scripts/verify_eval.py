#!/usr/bin/env python3
# 验证脚本 - 检查修改后的评估方法

import sys
import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
import pickle
import argparse

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.unified_evaluation import ValidationDataset
from src.metrics import compute_hr_ndcg_full
from torch.utils.data import DataLoader

# 解析命令行参数
parser = argparse.ArgumentParser(description='验证修改后的评估方法')
parser.add_argument('--image-expert-weight', type=float, default=1.0, 
                    help='图像专家的权重因子(0.0-1.0)，1.0表示使用门控网络原始权重，0.0表示完全忽略图像专家')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# 获取图像专家权重因子
IMAGE_EXPERT_WEIGHT = args.image_expert_weight

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("使用CPU")
    return device

def load_model(config, device, checkpoint_path):
    """加载模型"""
    # 动态计算总词汇表大小（物品数量 + 特殊标记数量）
    with open(config['data']['id_maps_file'], 'rb') as f:
        import pickle
        id_maps = pickle.load(f)
    
    num_special_tokens = id_maps.get('num_special_tokens', 4)  # 默认4个特殊标记
    total_vocab_size = id_maps['num_items'] + num_special_tokens
    
    # 将动态计算的参数添加到配置中
    config['encoder_model']['item_num'] = total_vocab_size
    config['decoder_model']['num_items'] = total_vocab_size
    
    logging.info(f"词汇表大小: {total_vocab_size} (物品: {id_maps['num_items']}, 特殊标记: {num_special_tokens})")
    
    # 1. 加载文本嵌入 - 必须在模型初始化前准备好
    logging.info("加载文本嵌入...")
    text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered_migrated.npy'
    try:
        text_embeddings_dict = np.load(text_embedding_file, allow_pickle=True).item()
        text_embedding_dim = next(iter(text_embeddings_dict.values())).shape[0]
        text_embedding_matrix = torch.zeros(total_vocab_size, text_embedding_dim, dtype=torch.float)
        
        item_asin_map = id_maps['item_map']
        loaded_count = 0
        for asin, embedding in text_embeddings_dict.items():
            if asin in item_asin_map:
                item_id = item_asin_map[asin]
                text_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                loaded_count += 1
        
        logging.info(f"成功加载并映射 {loaded_count} 个文本嵌入")
        
        # 更新解码器配置
        config['decoder_model']['text_embedding_dim'] = text_embedding_dim
        
    except FileNotFoundError:
        logging.error(f"找不到文本嵌入文件: {text_embedding_file}")
        sys.exit(1)
    
    # 2. 准备图像嵌入（如果可用）
    image_embedding_file = config['data']['data_dir'] / 'book_image_embeddings_migrated.npy'
    image_embedding_matrix = None
    
    # 保持原始专家配置不变，我们将在评估时调整权重
    image_expert_enabled = config['expert_system']['experts'].get('image_expert', False)
    logging.info(f"图像专家状态: {'启用' if image_expert_enabled else '禁用'}")
    logging.info(f"图像专家权重因子: {IMAGE_EXPERT_WEIGHT} (1.0表示原始权重，0.0表示忽略该专家)")
    
    # 如果图像专家启用，尝试加载图像嵌入；否则创建随机嵌入
    if image_expert_enabled:
        try:
            logging.info(f"加载图像嵌入文件: {image_embedding_file}")
            image_embeddings_dict = np.load(image_embedding_file, allow_pickle=True).item()
            
            # 诊断信息
            total_embeddings = len(image_embeddings_dict)
            logging.info(f"图像嵌入文件中包含 {total_embeddings} 个嵌入")
            
            if total_embeddings > 0:
                sample_keys = list(image_embeddings_dict.keys())[:3]
                logging.info(f"图像嵌入样本键: {sample_keys}")
                logging.info(f"图像嵌入样本键类型: {[type(k).__name__ for k in sample_keys]}")
                
                # 检查键格式
                sample_item_keys = list(item_asin_map.keys())[:3]
                logging.info(f"物品映射样本键: {sample_item_keys}")
                logging.info(f"物品映射样本键类型: {[type(k).__name__ for k in sample_item_keys]}")
                
                # 检测图像嵌入键的类型并确定映射策略
                key_type = "unknown"
                if isinstance(sample_keys[0], (int, np.integer)):
                    key_type = "id"
                    logging.info("检测到图像嵌入使用ID作为键")
                elif isinstance(sample_keys[0], str):
                    if sample_keys[0].isdigit():
                        key_type = "id_str"
                        logging.info("检测到图像嵌入使用字符串形式的ID作为键")
                    else:
                        # 检查是否为ASIN
                        if any(k in item_asin_map for k in sample_keys):
                            key_type = "asin"
                            logging.info("检测到图像嵌入使用ASIN作为键")
                
                # 根据键类型计算重叠度
                if key_type == "asin":
                    overlap = set(image_embeddings_dict.keys()) & set(item_asin_map.keys())
                    logging.info(f"ASIN键重叠数: {len(overlap)}/{total_embeddings} ({len(overlap)/total_embeddings*100:.2f}%)")
                elif key_type in ["id", "id_str"]:
                    # 对于ID键，检查有多少在有效范围内
                    valid_ids = 0
                    for k in image_embeddings_dict.keys():
                        try:
                            item_id = int(k) if isinstance(k, str) else k
                            if 0 <= item_id < total_vocab_size:
                                valid_ids += 1
                        except:
                            pass
                    logging.info(f"有效ID数量: {valid_ids}/{total_embeddings} ({valid_ids/total_embeddings*100:.2f}%)")
                
                image_embedding_dim = next(iter(image_embeddings_dict.values())).shape[0]
                image_embedding_matrix = torch.zeros(total_vocab_size, image_embedding_dim, dtype=torch.float)
                
                # 添加随机初始化，避免空嵌入
                image_embedding_matrix.normal_(mean=0, std=0.01)
                
                loaded_count = 0
                # 根据检测到的键类型使用不同的加载策略
                if key_type in ["id", "id_str"]:
                    # 直接使用ID作为索引
                    for key, embedding in image_embeddings_dict.items():
                        try:
                            item_id = int(key) if isinstance(key, str) else key
                            if 0 <= item_id < total_vocab_size:
                                image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                                loaded_count += 1
                        except Exception as e:
                            # 忽略无效的键
                            continue
                elif key_type == "asin":
                    # 使用ASIN到ID的映射
                    for key, embedding in image_embeddings_dict.items():
                        if key in item_asin_map:
                            item_id = item_asin_map[key]
                            image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                            loaded_count += 1
                else:
                    # 尝试多种方法
                    for key, embedding in image_embeddings_dict.items():
                        try:
                            if isinstance(key, (int, np.integer)):
                                item_id = int(key)
                                if 0 <= item_id < total_vocab_size:
                                    image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                                    loaded_count += 1
                            elif isinstance(key, str) and key.isdigit():
                                item_id = int(key)
                                if 0 <= item_id < total_vocab_size:
                                    image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                                    loaded_count += 1
                            # 如果键是ASIN，尝试通过item_asin_map转换
                            elif isinstance(key, str) and key in item_asin_map:
                                item_id = item_asin_map[key]
                                image_embedding_matrix[item_id] = torch.tensor(embedding, dtype=torch.float)
                                loaded_count += 1
                        except Exception as e:
                            # 忽略无效的键
                            continue
                
                logging.info(f"成功加载并映射 {loaded_count}/{total_embeddings} 个图像嵌入")
            else:
                logging.warning("图像嵌入字典为空！创建随机嵌入...")
                # 创建一个随机初始化的矩阵，以便模型能继续运行
                image_embedding_dim = 512  # CLIP的标准维度
                image_embedding_matrix = torch.randn(total_vocab_size, image_embedding_dim) * 0.01
        except Exception as e:
            logging.warning(f"加载图像嵌入失败: {e}")
            # 创建一个随机初始化的矩阵，以便模型能继续运行
            logging.info("创建随机图像嵌入以继续执行...")
            image_embedding_dim = 512  # CLIP的标准维度
            image_embedding_matrix = torch.randn(total_vocab_size, image_embedding_dim) * 0.01
    
    # 3. 初始化模型
    model = GENIUSRecModel(
        config['encoder_model'], 
        config['decoder_model'],
        config['expert_system']
    ).to(device)
    
    # 4. 将嵌入矩阵加载到模型中
    model.decoder.load_text_embeddings(text_embedding_matrix.to(device))
    
    # 图像嵌入处理
    if image_expert_enabled:
        # 确保无论如何都加载图像嵌入（即使是随机初始化的）
        if image_embedding_matrix is not None:
            model.decoder.load_image_embeddings(image_embedding_matrix.to(device))
        else:
            logging.warning("没有可用的图像嵌入矩阵，图像专家可能无法正常工作")
    else:
        logging.info("图像专家已禁用，不需要加载图像嵌入")
    
    # 5. 加载模型检查点        
    try:
        # 使用weights_only=True避免安全警告
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # 仅加载模型状态（不加载优化器等）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logging.info(f"成功加载模型检查点: {checkpoint_path}")
        
        # 检查图像专家是否被启用且图像专家权重是否被修改
        if model.decoder.expert_config["experts"].get("image_expert", False) and IMAGE_EXPERT_WEIGHT != 1.0:
            logging.info(f"图像专家已启用，将应用图像专家权重因子: {IMAGE_EXPERT_WEIGHT}")
        
    except Exception as e:
        logging.error(f"加载检查点失败: {e}")
        sys.exit(1)

    return model, id_maps

def run_evaluation(model, config, device):
    """运行评估"""
    # 加载测试数据集
    test_dataset = ValidationDataset(
        config['data']['test_file'],
        config['encoder_model']['max_len'],
        config['pad_token_id']
    )
    
    batch_size = 128  # 使用较大批次加速评估
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logging.info(f"加载测试数据集: {len(test_dataset)} 样本, 批次大小: {batch_size}")
    
    # 打印专家情况
    experts_config = config['expert_system']['experts']
    enabled_experts = [name for name, enabled in experts_config.items() if enabled]
    logging.info(f"启用的专家: {enabled_experts} (共{len(enabled_experts)}个)")
    
    # 添加一个诊断函数来查看专家权重
    def check_expert_weights(sample_ids):
        """检查并打印专家权重，并应用图像专家权重因子"""
        try:
            with torch.no_grad():
                # 创建一个简单的目标序列
                target_ids = torch.ones((1, 1), dtype=torch.long, device=device) * 5  # 使用ID=5作为目标物品
                
                # 运行模型，并要求返回专家权重
                source_ids = sample_ids.unsqueeze(0)  # 添加批次维度
                encoder_outputs = model.encoder(source_ids)
                
                # 直接打印模型中启用的专家
                enabled_experts = model.decoder.enabled_experts
                logging.info(f"模型中启用的专家: {enabled_experts}")
                
                # 获取解码器输入
                target_emb = model.decoder.item_embedding(target_ids) * math.sqrt(model.decoder.embedding_dim)
                pos_emb = model.decoder.pos_embedding(torch.zeros(1, 1, dtype=torch.long, device=device))
                decoder_input = model.decoder.dropout(target_emb + pos_emb)
                
                # 获取门控权重
                gate_input = decoder_input.mean(dim=1)
                gate_weights = model.decoder.gate_network(gate_input)
                original_weights = F.softmax(gate_weights, dim=-1)
                
                # 找到图像专家的索引
                image_expert_idx = -1
                for i, name in enumerate(enabled_experts):
                    if name == 'image_expert':
                        image_expert_idx = i
                        break
                
                # 应用图像专家权重因子
                modified_weights = original_weights.clone()
                if image_expert_idx >= 0:
                    # 计算原始图像专家权重
                    original_image_weight = original_weights[0, image_expert_idx].item()
                    # 应用权重因子
                    new_image_weight = original_image_weight * IMAGE_EXPERT_WEIGHT
                    weight_diff = original_image_weight - new_image_weight
                    
                    # 修改权重
                    modified_weights[0, image_expert_idx] = new_image_weight
                    
                    # 将差额均匀分配给其他专家
                    other_experts_count = len(enabled_experts) - 1
                    if other_experts_count > 0:
                        weight_addition = weight_diff / other_experts_count
                        for i in range(len(enabled_experts)):
                            if i != image_expert_idx:
                                modified_weights[0, i] += weight_addition
                
                # 打印原始和修改后的权重信息
                original_weight_info = [f"{name}: {w.item():.4f}" for name, w in zip(enabled_experts, original_weights[0])]
                modified_weight_info = [f"{name}: {w.item():.4f}" for name, w in zip(enabled_experts, modified_weights[0])]
                logging.info(f"原始专家权重: {original_weight_info}")
                logging.info(f"调整后专家权重: {modified_weight_info}")
                
                # 返回修改后的权重以供评估使用
                return modified_weights
        except Exception as e:
            logging.warning(f"获取专家权重时出错: {e}")
            return None
    
    # 执行评估
    model.eval()
    total_hr = 0.0
    total_ndcg = 0.0
    total_samples = 0
    
    with torch.no_grad():
        # 预计算所有物品的嵌入，提高效率
        logging.info("预计算物品嵌入向量...")
        # 注意：这里使用从4开始的物品ID，跳过所有特殊标记
        all_item_ids = torch.arange(4, config['encoder_model']['item_num'], device=device)
        all_item_embeddings = model.encoder.item_embedding(all_item_ids)
        # L2归一化
        all_item_embeddings = torch.nn.functional.normalize(all_item_embeddings, p=2, dim=1)
        
        # 遍历测试集
        for idx, batch in enumerate(test_loader):
            # ValidationDataset返回字典格式
            source_ids = batch['input_ids'].to(device)
            target_item_ids = batch['ground_truth'].to(device)
            
            # 获取用户表示
            encoder_outputs = model.encoder(source_ids)  # [B, L, D]
            user_embeddings_encoder = encoder_outputs[:, -1, :]  # [B, D] - 取序列最后一个位置
            
            if idx == 0:
                logging.info("分析评估流程...")
                logging.info("使用两种评估方式进行对比：1) 专家融合评分 2) 编码器嵌入相似度")
            
            # ================ 方法1：专家融合评分方法 ================
            # 创建与batch_size相等的空序列作为目标序列输入
            dummy_targets = torch.ones(source_ids.size(0), 1, dtype=torch.long, device=device) * 4
            
            # 通过模型的解码器获取专家融合后的物品评分
            source_padding_mask = (source_ids == config['pad_token_id'])
            
            try:
                # 尝试通过解码器和专家系统获取物品评分
                with torch.no_grad():
                    # 检查第一个batch是否需要调整专家权重
                    if idx == 0 and IMAGE_EXPERT_WEIGHT != 1.0:
                        logging.info(f"正在应用图像专家权重因子: {IMAGE_EXPERT_WEIGHT}")
                        # 修改GenerativeDecoder的forward方法，临时注入专家权重调整
                        original_forward = model.decoder.forward
                        
                        def modified_forward(*args, **kwargs):
                            # 获取原始门控权重
                            kwargs['return_weights'] = True
                            logits, orig_weights = original_forward(*args, **kwargs)
                            
                            # 找到图像专家的索引
                            image_expert_idx = -1
                            for i, name in enumerate(model.decoder.enabled_experts):
                                if name == 'image_expert':
                                    image_expert_idx = i
                                    break
                            
                            if image_expert_idx >= 0:
                                # 复制原始权重
                                batch_size, seq_len, num_experts = orig_weights.shape
                                mod_weights = orig_weights.clone()
                                
                                # 应用权重因子
                                for b in range(batch_size):
                                    for t in range(seq_len):
                                        # 原始图像权重
                                        orig_image_weight = mod_weights[b, t, image_expert_idx].item()
                                        # 调整后的图像权重
                                        new_image_weight = orig_image_weight * IMAGE_EXPERT_WEIGHT
                                        # 计算差额
                                        weight_diff = orig_image_weight - new_image_weight
                                        
                                        # 修改图像权重
                                        mod_weights[b, t, image_expert_idx] = new_image_weight
                                        
                                        # 将差额均匀分配给其他专家
                                        other_experts_count = num_experts - 1
                                        if other_experts_count > 0:
                                            weight_addition = weight_diff / other_experts_count
                                            for i in range(num_experts):
                                                if i != image_expert_idx:
                                                    mod_weights[b, t, i] += weight_addition
                                
                                # 如果是第一个批次，打印专家权重信息
                                if idx == 0:
                                    # 取一个样本示例
                                    sample_b, sample_t = 0, 0
                                    original_weight_info = [f"{name}: {orig_weights[sample_b, sample_t, i].item():.4f}" 
                                                          for i, name in enumerate(model.decoder.enabled_experts)]
                                    modified_weight_info = [f"{name}: {mod_weights[sample_b, sample_t, i].item():.4f}" 
                                                          for i, name in enumerate(model.decoder.enabled_experts)]
                                    logging.info(f"原始专家权重: {original_weight_info}")
                                    logging.info(f"调整后专家权重: {modified_weight_info}")
                                
                                # 手动计算融合的logits
                                expert_idx = 0
                                final_logits = None
                                
                                # 行为专家
                                if 'behavior_expert' in model.decoder.enabled_experts:
                                    behavior_logits = model.decoder.behavior_expert_fc(args[0])
                                    weight = mod_weights[:, :, expert_idx].unsqueeze(-1)
                                    if final_logits is None:
                                        final_logits = weight * behavior_logits
                                    else:
                                        final_logits += weight * behavior_logits
                                    expert_idx += 1
                                
                                # 内容专家
                                if 'content_expert' in model.decoder.enabled_experts:
                                    content_query = None
                                    if hasattr(model.decoder, 'content_expert_attention'):
                                        content_context_vector, _ = model.decoder.content_expert_attention(
                                            query=args[0], key=args[1], value=args[1], 
                                            key_padding_mask=args[2] if len(args) > 2 else kwargs.get('memory_padding_mask')
                                        )
                                        content_query = model.decoder.content_attention_projection(content_context_vector)
                                    else:
                                        content_query = model.decoder.content_expert_fc(args[0])
                                    
                                    all_text_embeddings = model.decoder.text_embedding_matrix.transpose(0, 1)
                                    content_logits = torch.matmul(content_query, all_text_embeddings)
                                    
                                    weight = mod_weights[:, :, expert_idx].unsqueeze(-1)
                                    if final_logits is None:
                                        final_logits = weight * content_logits
                                    else:
                                        final_logits += weight * content_logits
                                    expert_idx += 1
                                
                                # 图像专家
                                if 'image_expert' in model.decoder.enabled_experts:
                                    visual_query = None
                                    if hasattr(model.decoder, 'image_expert_attention'):
                                        visual_context_vector, _ = model.decoder.image_expert_attention(
                                            query=args[0], key=args[1], value=args[1],
                                            key_padding_mask=args[2] if len(args) > 2 else kwargs.get('memory_padding_mask')
                                        )
                                        visual_query = model.decoder.image_attention_projection(visual_context_vector)
                                    else:
                                        visual_query = model.decoder.image_expert_fc(args[0])
                                    
                                    all_image_embeddings = model.decoder.image_embedding_matrix.transpose(0, 1)
                                    image_logits = torch.matmul(visual_query, all_image_embeddings)
                                    
                                    weight = mod_weights[:, :, expert_idx].unsqueeze(-1)
                                    if final_logits is None:
                                        final_logits = weight * image_logits
                                    else:
                                        final_logits += weight * image_logits
                                    expert_idx += 1
                                
                                return final_logits
                            
                            return logits
                        
                        # 临时替换forward方法
                        model.decoder.forward = modified_forward
                    
                    # 运行解码器获取物品评分
                    logits = model.decoder(
                        target_ids=dummy_targets, 
                        encoder_output=encoder_outputs, 
                        memory_padding_mask=source_padding_mask
                    )
                    
                    # 恢复原始forward方法（如果被替换了）
                    if idx == 0 and IMAGE_EXPERT_WEIGHT != 1.0:
                        model.decoder.forward = original_forward
                    
                    # 获取评分并进行排序
                    scores = logits[:, -1, :]  # [B, num_items] - 获取序列最后一个位置的评分
                    
                    # 对评分进行排序，获取前K个物品
                    _, sorted_indices = torch.sort(scores, dim=1, descending=True)
                    
                    # 计算HR和NDCG
                    hr_decoder = 0.0
                    ndcg_decoder = 0.0
                    valid_samples = 0
                    
                    for i in range(scores.size(0)):
                        target_id = target_item_ids[i].item()
                        if target_id >= 4:  # 跳过特殊标记
                            valid_samples += 1
                            # 获取前K个推荐物品
                            topk_items = sorted_indices[i, :10]
                            
                            # 检查目标物品是否在前K个推荐中
                            hit = (topk_items == target_id).any().item()
                            
                            # 计算排名
                            if hit:
                                rank = (topk_items == target_id).nonzero(as_tuple=True)[0][0].item() + 1
                                hr_decoder += 1
                                ndcg_decoder += 1.0 / math.log2(rank + 1)
                    
                    # 计算平均值
                    if valid_samples > 0:
                        hr_decoder /= valid_samples
                        ndcg_decoder /= valid_samples
                    
                    # 仅在第一批次输出详细日志
                    if idx == 0:
                        logging.info(f"专家融合评分方法 - HR@10: {hr_decoder:.4f}, NDCG@10: {ndcg_decoder:.4f}")
                
                # ================ 方法2：编码器嵌入相似度（传统方法）================
                # 使用compute_hr_ndcg_full计算编码器表示与物品的相似度
                hr_encoder, ndcg_encoder = compute_hr_ndcg_full(user_embeddings_encoder, all_item_embeddings, target_item_ids, k=10)
                
                if idx == 0:
                    logging.info(f"编码器嵌入方法 - HR@10: {hr_encoder:.4f}, NDCG@10: {ndcg_encoder:.4f}")
                    
                # 使用哪种方法的结果作为最终指标？
                # 这里我们选择专家融合评分方法（更贴合GeniusRec的完整架构）
                hr, ndcg = hr_decoder, ndcg_decoder
                    
            except Exception as e:
                logging.warning(f"使用专家融合评分方法失败: {e}")
                # 如果失败，回退到仅使用编码器的方法
                hr, ndcg = compute_hr_ndcg_full(user_embeddings_encoder, all_item_embeddings, target_item_ids, k=10)
                logging.warning("回退到编码器嵌入相似度方法")
            
            # 累加结果
            batch_size = source_ids.size(0)
            total_hr += hr * batch_size
            total_ndcg += ndcg * batch_size
            total_samples += batch_size
            
            if idx % 10 == 0:
                logging.info(f"进度: {idx}/{len(test_loader)}, 当前HR@10={hr:.4f}, NDCG@10={ndcg:.4f}")
                
                # 检查第一个样本的专家权重
                if idx == 0:
                    logging.info("分析并调整专家权重...")
                    expert_weights = check_expert_weights(source_ids[0])
                    
                    # 修改GenerativeDecoder的forward方法，临时覆盖专家权重
                    original_forward = model.decoder.forward
                    
                    def modified_forward(*args, **kwargs):
                        # 调用原始forward
                        logits = original_forward(*args, **kwargs)
                        
                        # 如果返回的是元组，说明原始函数返回了(logits, weights)
                        if isinstance(logits, tuple) and len(logits) == 2:
                            return logits[0], expert_weights
                        
                        # 否则仅返回logits
                        return logits
                    
                    # 只有当成功获取了权重时才应用修改
                    if expert_weights is not None:
                        model.decoder.forward = modified_forward
                        logging.info("已应用专家权重调整")
                # 打印一些示例结果以进行调试
                if idx == 0:
                    for i in range(min(5, len(target_item_ids))):
                        tid = target_item_ids[i].item()
                        logging.info(f"样本 {i}: 目标ID={tid}")
                        if tid >= 4:  # 有效物品ID
                            # 计算编码器嵌入排名
                            scores_i = torch.matmul(user_embeddings_encoder[i:i+1], all_item_embeddings.t())
                            _, sorted_indices_i = torch.sort(scores_i, dim=1, descending=True)
                            target_idx = tid - 4  # 物品ID转为索引（减4是因为我们跳过了4个特殊标记）
                            
                            # 更详细的诊断
                            logging.info(f"  - 目标物品: ID={tid}, 索引={target_idx}")
                            logging.info(f"  - 排序索引形状: {sorted_indices_i.shape}")
                            logging.info(f"  - all_item_embeddings形状: {all_item_embeddings.shape}")
                            
                            if 0 <= target_idx < all_item_embeddings.shape[0]:
                                positions = (sorted_indices_i[0] == target_idx).nonzero(as_tuple=True)[0]
                                rank = positions[0].item() + 1 if len(positions) > 0 else -1
                                logging.info(f"  - 在编码器评分排名中的位置: {rank}")
                                logging.info(f"  - 编码器评分前5个推荐的索引: {sorted_indices_i[0][:5].tolist()}")
                                logging.info(f"  - 编码器评分前5个推荐的物品ID: {[idx+4 for idx in sorted_indices_i[0][:5].tolist()]}")
                            else:
                                logging.warning(f"  - 目标索引 {target_idx} 超出范围 [0, {all_item_embeddings.shape[0]-1}]")
                        else:
                            logging.info(f"  - 无效目标ID (特殊标记)")
                      
    
    # 计算平均指标
    avg_hr = total_hr / total_samples
    avg_ndcg = total_ndcg / total_samples
    
    logging.info("=" * 50)
    logging.info(f"测试集结果 (总样本: {total_samples}):")
    logging.info(f"HR@10: {avg_hr:.4f}")
    logging.info(f"NDCG@10: {avg_ndcg:.4f}")
    logging.info("=" * 50)
    
    return {"hr@10": avg_hr, "ndcg@10": avg_ndcg}

def verify_special_tokens_handling():
    """验证特殊标记处理逻辑"""
    logging.info("检查特殊标记处理逻辑...")
    
    # 模拟嵌入和ID
    user_embed = torch.randn(2, 64)  # 2个用户
    item_embed = torch.randn(10, 64)  # 10个物品
    
    # 测试不同的目标ID
    for target_id in [4, 5, 10]:
        target_ids = torch.tensor([target_id, 0])
        hr, ndcg = compute_hr_ndcg_full(user_embed, item_embed, target_ids, k=5)
        logging.info(f"验证目标ID={target_id}: HR={hr:.4f}, NDCG={ndcg:.4f}")
    
    logging.info("特殊标记处理逻辑验证完成")

def main():
    logging.info("开始验证修改后的评估方法...")
    logging.info(f"图像专家权重因子: {IMAGE_EXPERT_WEIGHT}")
    
    verify_special_tokens_handling()
    
    # 获取配置
    config = get_config()
    device = setup_device()
    
    # 检查点路径
    checkpoint_path = project_root / "checkpoints" / "genius_rec_best.pth"
    if not checkpoint_path.exists():
        logging.error(f"找不到检查点文件: {checkpoint_path}")
        sys.exit(1)
    
    # 加载模型
    model, id_maps = load_model(config, device, checkpoint_path)
    
    # 验证模型中图像专家设置
    enabled_experts = [name for name, enabled in config['expert_system']['experts'].items() if enabled]
    logging.info(f"模型中启用的专家: {enabled_experts} (共{len(enabled_experts)}个)")
    
    # 运行评估
    results = run_evaluation(model, config, device)
    
    # 在结果中显示图像专家权重因子
    logging.info("=" * 50)
    logging.info(f"评估结果 (图像专家权重因子: {IMAGE_EXPERT_WEIGHT})")
    logging.info(f"HR@10: {results['hr@10']:.4f}")
    logging.info(f"NDCG@10: {results['ndcg@10']:.4f}")
    logging.info("=" * 50)
    
    logging.info("评估完成!")
    return results

if __name__ == "__main__":
    main()
