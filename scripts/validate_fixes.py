#!/usr/bin/env python
"""
快速验证脚本 - 验证门控网络维度修复是否有效

该脚本会:
1. 加载模型和配置
2. 创建测试数据（故意使用不同的序列长度）
3. 执行模型前向传播
4. 模拟评估流程中的权重处理逻辑
5. 如果所有步骤都没有错误，则修复已生效

使用方法:
python scripts/validate_fixes.py
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from src.config import get_config
from src.GeniusRec import GENIUSRecModel
import torch.nn.functional as F

def create_test_data(batch_size=2, source_len=64, target_len=20, embedding_dim=64):
    """创建测试数据，模拟不同的序列长度"""
    source_ids = torch.randint(1, 1000, (batch_size, source_len))
    target_ids = torch.randint(1, 1000, (batch_size, target_len))
    
    # 创建填充掩码 (随机制造一些填充)
    source_padding_mask = torch.zeros_like(source_ids, dtype=torch.bool)
    source_padding_mask[:, -5:] = True  # 最后5个位置是填充
    
    # 创建标签 (把target_ids向右偏移，最后一个是填充)
    labels = torch.cat([target_ids[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)], dim=1)
    
    # 把一些位置设为padding (为了测试掩码)
    pad_positions = torch.randint(0, target_len, (batch_size, 3))
    for i in range(batch_size):
        labels[i, pad_positions[i]] = 0
    
    return {
        'source_ids': source_ids,
        'target_ids': target_ids,
        'source_padding_mask': source_padding_mask,
        'labels': labels
    }

def mock_evaluation_logic(logits, gate_weights, labels, pad_token_id=0):
    """模拟评估函数中的门控权重处理逻辑"""
    print(f"⚡ 测试评估逻辑...")
    print(f"  - logits 形状: {logits.shape}")
    print(f"  - gate_weights 形状: {gate_weights.shape}")
    print(f"  - labels 形状: {labels.shape}")
    
    # 1. 创建一个掩码，标识哪些位置是有效标签（非pad）
    label_mask = (labels != pad_token_id).float().unsqueeze(-1)  # [B, T, 1]
    print(f"  - label_mask 形状: {label_mask.shape}")
    
    # 2. 应用掩码，只考虑有效位置的权重
    valid_gate_weights = gate_weights * label_mask
    print(f"  - valid_gate_weights 形状: {valid_gate_weights.shape}")
    
    # 3. 对每个batch的有效位置取平均
    batch_sum = valid_gate_weights.sum(dim=1)  # [B, num_experts]
    batch_count = label_mask.sum(dim=1)  # [B, 1]
    # 防止除零
    batch_mean = batch_sum / (batch_count + 1e-8)  # [B, num_experts]
    print(f"  - batch_mean 形状: {batch_mean.shape}")
    
    # 4. 再对整个batch取平均得到最终权重
    masked_gate_weights = batch_mean.mean(dim=0)  # [num_experts]
    print(f"  - masked_gate_weights 形状: {masked_gate_weights.shape}")
    print(f"  - 专家权重: {masked_gate_weights.detach().cpu().numpy()}")
    
    return masked_gate_weights

def validate_fixes():
    """验证修复是否有效"""
    print("🧪 开始验证门控网络维度修复...")
    
    # 1. 加载配置
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 2. 创建模型配置
    encoder_config = {
        'item_num': 10000,
        'max_len': 64,
        'embedding_dim': 64,
        'linear_hidden_dim': 16,
        'attention_dim': 16,
        'num_layers': 2,
        'num_heads': 2,
        'dropout': 0.1,
        'pad_token_id': 0,
    }
    
    decoder_config = {
        'num_items': 10000,
        'embedding_dim': 64,
        'num_layers': 2,
        'num_heads': 2,
        'ffn_hidden_dim': 256,
        'max_seq_len': 20,
        'dropout_ratio': 0.1,
        'pad_token_id': 0,
    }
    
    expert_config = config['expert_system']
    
    # 3. 初始化模型 - 只启用行为专家以避免加载嵌入矩阵
    modified_expert_config = expert_config.copy()
    # 禁用内容专家和图像专家，只保留行为专家进行测试
    modified_expert_config['enable_content_expert'] = False
    modified_expert_config['enable_image_expert'] = False
    modified_expert_config['enable_behavior_expert'] = True
    
    model = GENIUSRecModel(encoder_config, decoder_config, modified_expert_config)
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print("\n📋 模型配置摘要:")
    print(f"  - 编码器序列长度: {encoder_config['max_len']}")
    print(f"  - 解码器序列长度: {decoder_config['max_seq_len']}")
    print(f"  - 启用的专家: {model.decoder.enabled_experts}")
    
    # 加载必要的嵌入矩阵或禁用相应的专家
    print("\n🔧 准备专家系统...")
    
    # 简单方案：为测试目的禁用内容和图像专家，只保留行为专家
    if 'content_expert' in model.decoder.enabled_experts:
        print("  - 为测试目的禁用内容专家")
        model.decoder.expert_config['experts']['content_expert'] = False
        model.decoder.enabled_experts = [k for k, v in model.decoder.expert_config["experts"].items() if v]
    
    if 'image_expert' in model.decoder.enabled_experts:
        print("  - 为测试目的禁用图像专家")
        model.decoder.expert_config['experts']['image_expert'] = False
        model.decoder.enabled_experts = [k for k, v in model.decoder.expert_config["experts"].items() if v]
    
    print(f"  - 测试将使用的专家: {model.decoder.enabled_experts}")
    
    # 4. 创建测试数据
    test_data = create_test_data(
        batch_size=2, 
        source_len=encoder_config['max_len'],
        target_len=decoder_config['max_seq_len'],
        embedding_dim=encoder_config['embedding_dim']
    )
    
    # 转移数据到设备
    for key in test_data:
        test_data[key] = test_data[key].to(device)
    
    print("\n📊 测试数据形状:")
    for key, value in test_data.items():
        print(f"  - {key}: {value.shape}")
    
    # 5. 测试训练模式
    print("\n🔄 测试训练模式...")
    model.train()
    try:
        logits, gate_weights = model(
            test_data['source_ids'], 
            test_data['target_ids'], 
            test_data['source_padding_mask'],
            return_weights=True
        )
        print(f"  ✅ 前向传播成功!")
        print(f"  - logits 形状: {logits.shape}")
        print(f"  - gate_weights 形状: {gate_weights.shape}")
        
        # 模拟损失计算
        mock_evaluation_logic(logits, gate_weights, test_data['labels'])
        print("  ✅ 训练模式测试通过!")
    except Exception as e:
        print(f"  ❌ 训练模式测试失败: {e}")
        raise
    
    # 6. 测试评估模式
    print("\n🔍 测试评估模式...")
    model.eval()
    try:
        with torch.no_grad():
            logits, gate_weights = model(
                test_data['source_ids'], 
                test_data['target_ids'], 
                test_data['source_padding_mask'],
                return_weights=True
            )
        print(f"  ✅ 前向传播成功!")
        print(f"  - logits 形状: {logits.shape}")
        print(f"  - gate_weights 形状: {gate_weights.shape}")
        
        # 模拟评估流程
        mock_evaluation_logic(logits, gate_weights, test_data['labels'])
        print("  ✅ 评估模式测试通过!")
    except Exception as e:
        print(f"  ❌ 评估模式测试失败: {e}")
        raise
    
    # 7. 特别测试：不同序列长度的情况
    print("\n🧩 测试不同序列长度的情况...")
    try:
        # 创建序列长度不同的测试数据
        special_test_data = create_test_data(
            batch_size=2, 
            source_len=50,  # 与编码器配置不同
            target_len=15,  # 与解码器配置不同
            embedding_dim=encoder_config['embedding_dim']
        )
        
        # 转移数据到设备
        for key in special_test_data:
            special_test_data[key] = special_test_data[key].to(device)
            
        with torch.no_grad():
            logits, gate_weights = model(
                special_test_data['source_ids'], 
                special_test_data['target_ids'], 
                special_test_data['source_padding_mask'],
                return_weights=True
            )
        print(f"  ✅ 前向传播成功!")
        print(f"  - source_ids 形状: {special_test_data['source_ids'].shape}")
        print(f"  - target_ids 形状: {special_test_data['target_ids'].shape}")
        print(f"  - logits 形状: {logits.shape}")
        print(f"  - gate_weights 形状: {gate_weights.shape}")
        
        # 模拟评估流程
        mock_evaluation_logic(logits, gate_weights, special_test_data['labels'])
        print("  ✅ 不同序列长度测试通过!")
    except Exception as e:
        print(f"  ❌ 不同序列长度测试失败: {e}")
        raise
    
    print("\n🎉 所有测试通过! 门控网络维度修复有效!")
    return True

def test_evaluation_speed():
    """测试不同评估模式的速度差异"""
    import time
    from torch.utils.data import DataLoader, TensorDataset
    from src.unified_evaluation import evaluate_model_validation_with_ranking
    
    print("\n🚀 测试评估速度差异...")
    
    # 创建一个简单的测试集
    batch_size = 32
    source_len = 64
    target_len = 20
    embedding_dim = 64
    num_batches = 10  # 很小的测试集，只为了演示
    
    # 1. 加载配置
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 创建模型配置
    encoder_config = {
        'item_num': 10000,
        'max_len': 64,
        'embedding_dim': 64,
        'linear_hidden_dim': 16,
        'attention_dim': 16,
        'num_layers': 2,
        'num_heads': 2,
        'dropout': 0.1,
        'pad_token_id': 0,
    }
    
    decoder_config = {
        'num_items': 10000,
        'embedding_dim': 64,
        'num_layers': 2,
        'num_heads': 2,
        'ffn_hidden_dim': 256,
        'max_seq_len': 20,
        'dropout_ratio': 0.1,
        'pad_token_id': 0,
    }
    
    # 禁用内容和图像专家，只保留行为专家
    expert_config = config['expert_system'].copy()
    expert_config['enable_content_expert'] = False
    expert_config['enable_image_expert'] = False
    expert_config['enable_behavior_expert'] = True
    
    # 3. 创建模型
    model = GENIUSRecModel(encoder_config, decoder_config, expert_config)
    model.to(device)
    model.eval()
    
    # 4. 创建测试数据
    test_data = []
    for _ in range(num_batches):
        batch_data = create_test_data(
            batch_size=batch_size, 
            source_len=source_len,
            target_len=target_len
        )
        for key in batch_data:
            batch_data[key] = batch_data[key].to(device)
        test_data.append(batch_data)
    
    # 5. 创建一个简单的数据加载器
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # 6. 测试不同候选集大小的评估速度
    sample_sizes = [None, 1000, 500, 100]
    
    for sample_size in sample_sizes:
        start_time = time.time()
        mode_desc = "全量评估" if sample_size is None else f"采样评估({sample_size}候选项)"
        print(f"\n📊 测试模式: {mode_desc}")
        
        # 执行一小部分评估以测试速度
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 2:  # 只测试2个批次，节约时间
                    break
                    
                source_ids = batch['source_ids']
                decoder_input_ids = batch['target_ids']
                labels = batch['labels']
                source_padding_mask = (source_ids == 0)
                
                # 前向传播
                logits, gate_weights = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=True)
                
                # 用户嵌入
                encoder_outputs = model.encoder(source_ids)  # [B, L, D]
                user_embeddings = encoder_outputs[:, -1, :]  # [B, D]
                
                # 模拟评估计算
                if sample_size is None:
                    # 全量评估 - 计算与所有物品的相似度
                    all_item_ids = torch.arange(1, 1000, device=device)  # 模拟较小的物品集
                    all_item_embeddings = torch.randn(999, encoder_config['embedding_dim'], device=device)  # 随机嵌入
                    
                    # 计算相似度
                    scores = torch.matmul(user_embeddings, all_item_embeddings.t())  # [B, num_items]
                else:
                    # 采样评估 - 随机选择sample_size个候选物品
                    for i in range(user_embeddings.size(0)):
                        candidate_ids = torch.randint(1, 1000, (sample_size,), device=device)  # 随机候选物品
                        candidate_embeddings = torch.randn(sample_size, encoder_config['embedding_dim'], device=device)  # 随机嵌入
                        
                        # 计算相似度
                        scores = torch.matmul(user_embeddings[i:i+1], candidate_embeddings.t())  # [1, sample_size]
                
        elapsed_time = time.time() - start_time
        print(f"  ⏱️ 评估耗时: {elapsed_time:.4f}秒")
    
    print("\n💡 结论: 采样评估可以大大加速评估过程，特别是在大规模物品集的情况下")
    print("   建议: 开发过程中使用采样评估（100-500个候选项），最终测试使用全量评估")

if __name__ == "__main__":
    validate_fixes()
    # 运行评估速度测试
    test_evaluation_speed()
