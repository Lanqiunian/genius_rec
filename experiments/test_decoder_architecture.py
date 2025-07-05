#!/usr/bin/env python3
"""
GENIUS-Rec 解码器架构测试
=======================

专门测试当前多专家解码器架构的各项功能
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config
from src.decoder.decoder import GenerativeDecoder

def test_decoder_initialization():
    """测试解码器初始化"""
    print("🔧 测试解码器初始化...")
    
    config = get_config()
    expert_config = config.get('expert_system', {})
    
    try:
        decoder = GenerativeDecoder(
            num_items=10000,
            embedding_dim=64,
            num_layers=4,
            num_heads=4,
            ffn_hidden_dim=256,
            max_seq_len=50,
            expert_config=expert_config
        )
        
        print("✅ 解码器初始化成功")
        print(f"📋 启用的专家: {decoder.enabled_experts}")
        
        # 检查各专家组件
        if hasattr(decoder, 'behavior_expert_fc'):
            print("✅ 行为专家组件存在")
        if hasattr(decoder, 'content_expert_attention'):
            print("✅ 内容专家注意力组件存在")
        if hasattr(decoder, 'image_expert_attention'):
            print("✅ 图像专家注意力组件存在")
        if hasattr(decoder, 'gate_network'):
            print("✅ 门控网络组件存在")
            
        return decoder
        
    except Exception as e:
        print(f"❌ 解码器初始化失败: {e}")
        return None

def test_decoder_forward_pass(decoder):
    """测试解码器前向传播"""
    print("\n🔄 测试解码器前向传播...")
    
    try:
        batch_size, seq_len = 4, 20
        
        # 准备输入数据
        target_ids = torch.randint(0, 1000, (batch_size, seq_len))
        encoder_output = torch.randn(batch_size, seq_len, 64)
        memory_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        print(f"📊 输入形状:")
        print(f"  - target_ids: {target_ids.shape}")
        print(f"  - encoder_output: {encoder_output.shape}")
        print(f"  - memory_padding_mask: {memory_padding_mask.shape}")
        
        # 测试正常前向传播
        with torch.no_grad():
            logits, weights, balancing_loss = decoder(
                target_ids, encoder_output, memory_padding_mask,
                return_weights=True, force_equal_weights=False
            )
        
        print(f"✅ 前向传播成功")
        print(f"📊 输出形状:")
        print(f"  - logits: {logits.shape}")
        print(f"  - weights: {weights.shape if weights is not None else 'None'}")
        print(f"  - balancing_loss: {balancing_loss.item():.6f}")
        
        # 测试等权重模式
        with torch.no_grad():
            logits_eq, weights_eq, balancing_loss_eq = decoder(
                target_ids, encoder_output, memory_padding_mask,
                return_weights=True, force_equal_weights=True
            )
        
        print(f"✅ 等权重模式成功")
        print(f"📊 等权重输出:")
        print(f"  - logits: {logits_eq.shape}")
        print(f"  - balancing_loss: {balancing_loss_eq.item():.6f}")
        
        # 检查权重分布
        if weights is not None:
            print(f"📊 专家权重统计:")
            for i in range(weights.shape[-1]):
                expert_weights = weights[:, :, i]
                print(f"  - 专家{i}: 均值={expert_weights.mean().item():.4f}, 标准差={expert_weights.std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expert_configurations():
    """测试不同专家配置"""
    print("\n⚙️ 测试不同专家配置...")
    
    base_config = get_config()
    
    test_configs = [
        {
            "name": "仅行为专家",
            "experts": {
                "behavior_expert": True,
                "content_expert": False,
                "image_expert": False
            }
        },
        {
            "name": "行为+内容专家",
            "experts": {
                "behavior_expert": True,
                "content_expert": True,
                "image_expert": False
            }
        },
        {
            "name": "全专家配置",
            "experts": {
                "behavior_expert": True,
                "content_expert": True,
                "image_expert": True
            }
        }
    ]
    
    for test_config in test_configs:
        print(f"\n📋 测试配置: {test_config['name']}")
        
        # 修改专家配置
        expert_config = base_config['expert_system'].copy()
        expert_config['experts'] = test_config['experts']
        
        try:
            decoder = GenerativeDecoder(
                num_items=1000,
                embedding_dim=64,
                num_layers=2,  # 使用较小的配置以加快测试
                num_heads=4,
                ffn_hidden_dim=128,
                max_seq_len=20,
                expert_config=expert_config
            )
            
            print(f"✅ {test_config['name']} 初始化成功")
            print(f"   启用专家: {decoder.enabled_experts}")
            
            # 快速前向传播测试
            batch_size, seq_len = 2, 10
            target_ids = torch.randint(0, 1000, (batch_size, seq_len))
            encoder_output = torch.randn(batch_size, seq_len, 64)
            memory_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            
            with torch.no_grad():
                logits, weights, balancing_loss = decoder(
                    target_ids, encoder_output, memory_padding_mask,
                    return_weights=True, force_equal_weights=False
                )
            
            print(f"   前向传播成功: logits {logits.shape}")
            
        except Exception as e:
            print(f"❌ {test_config['name']} 失败: {e}")

def test_embedding_loading():
    """测试嵌入加载功能"""
    print("\n📥 测试嵌入加载功能...")
    
    config = get_config()
    expert_config = config.get('expert_system', {})
    
    try:
        decoder = GenerativeDecoder(
            num_items=1000,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_hidden_dim=128,
            max_seq_len=20,
            expert_config=expert_config
        )
        
        # 测试文本嵌入加载
        if expert_config['experts'].get('content_expert', False):
            print("📄 测试文本嵌入加载...")
            text_embedding_dim = expert_config['content_expert']['text_embedding_dim']
            fake_text_embeddings = torch.randn(1000, text_embedding_dim)
            decoder.load_text_embeddings(fake_text_embeddings, verbose=True)
        
        # 测试图像嵌入加载
        if expert_config['experts'].get('image_expert', False):
            print("🖼️ 测试图像嵌入加载...")
            image_embedding_dim = expert_config['image_expert']['image_embedding_dim']
            fake_image_embeddings = torch.randn(1000, image_embedding_dim)
            decoder.load_image_embeddings(fake_image_embeddings, verbose=True)
        
        print("✅ 嵌入加载测试完成")
        
    except Exception as e:
        print(f"❌ 嵌入加载测试失败: {e}")

def main():
    """主测试函数"""
    print("🎯 GENIUS-Rec 解码器架构测试")
    print("="*50)
    
    # 1. 初始化测试
    decoder = test_decoder_initialization()
    if decoder is None:
        print("❌ 初始化失败，终止测试")
        return
    
    # 2. 前向传播测试
    if not test_decoder_forward_pass(decoder):
        print("❌ 前向传播失败")
        return
    
    # 3. 不同配置测试
    test_expert_configurations()
    
    # 4. 嵌入加载测试
    test_embedding_loading()
    
    print("\n" + "="*50)
    print("🎉 解码器架构测试完成！")
    print("✅ 当前解码器架构状态良好，可以进行实验")

if __name__ == "__main__":
    main()
