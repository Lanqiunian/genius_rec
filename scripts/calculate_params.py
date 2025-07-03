#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型参数量计算脚本
用于分析GeniusRec模型的总参数数量
"""

import torch
import pickle
import sys
import os

# 添加项目根目录到路径
sys.path.append('/root/autodl-tmp/genius_rec-main')

from src.config import get_config
from src.GeniusRec import GENIUSRecModel

def count_parameters(model):
    """计算模型的总参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def format_number(num):
    """格式化数字，以B、M、K为单位"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def analyze_model_components(model):
    """分析模型各组件的参数量"""
    print("\n=== 模型组件参数详细分析 ===")
    
    # 分析编码器
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"🧠 编码器(HSTU)总参数: {format_number(encoder_params)}")
    
    # 编码器内部组件
    if hasattr(model.encoder, 'item_embedding'):
        item_emb_params = model.encoder.item_embedding.weight.numel()
        print(f"  └─ 物品嵌入层: {format_number(item_emb_params)}")
    
    if hasattr(model.encoder, 'encoder_layers'):
        layer_params = sum(p.numel() for p in model.encoder.encoder_layers.parameters())
        num_layers = len(model.encoder.encoder_layers)
        print(f"  └─ HSTU层 ({num_layers}层): {format_number(layer_params)}")
        print(f"      └─ 平均每层: {format_number(layer_params // num_layers)}")
    
    # 分析解码器
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n🎯 解码器(GenerativeDecoder)总参数: {format_number(decoder_params)}")
    
    # 解码器内部组件
    if hasattr(model.decoder, 'item_embedding'):
        decoder_item_emb_params = model.decoder.item_embedding.weight.numel()
        print(f"  └─ 物品嵌入层: {format_number(decoder_item_emb_params)}")
    
    if hasattr(model.decoder, 'pos_embedding'):
        pos_emb_params = model.decoder.pos_embedding.weight.numel()
        print(f"  └─ 位置嵌入层: {format_number(pos_emb_params)}")
    
    if hasattr(model.decoder, 'decoder_layers'):
        decoder_layer_params = sum(p.numel() for p in model.decoder.decoder_layers.parameters())
        num_decoder_layers = len(model.decoder.decoder_layers)
        print(f"  └─ 解码器层 ({num_decoder_layers}层): {format_number(decoder_layer_params)}")
        print(f"      └─ 平均每层: {format_number(decoder_layer_params // num_decoder_layers)}")
    
    # 分析专家系统
    expert_params = 0
    print(f"\n🔧 专家系统参数:")
    
    # 行为专家
    if hasattr(model.decoder, 'behavior_expert_fc'):
        behavior_params = sum(p.numel() for p in model.decoder.behavior_expert_fc.parameters())
        expert_params += behavior_params
        print(f"  └─ 行为专家: {format_number(behavior_params)}")
    
    # 内容专家
    if hasattr(model.decoder, 'text_embedding'):
        text_emb_params = model.decoder.text_embedding.weight.numel()
        expert_params += text_emb_params
        print(f"  └─ 文本嵌入: {format_number(text_emb_params)}")
    
    if hasattr(model.decoder, 'content_expert_attention'):
        content_attn_params = sum(p.numel() for p in model.decoder.content_expert_attention.parameters())
        expert_params += content_attn_params
        print(f"  └─ 内容注意力: {format_number(content_attn_params)}")
    
    if hasattr(model.decoder, 'content_attention_projection'):
        content_proj_params = sum(p.numel() for p in model.decoder.content_attention_projection.parameters())
        expert_params += content_proj_params
        print(f"  └─ 内容投影: {format_number(content_proj_params)}")
    
    # 图像专家（如果启用）
    if hasattr(model.decoder, 'image_embedding'):
        image_emb_params = model.decoder.image_embedding.weight.numel()
        expert_params += image_emb_params
        print(f"  └─ 图像嵌入: {format_number(image_emb_params)}")
    
    # 门控网络
    if hasattr(model.decoder, 'gate_network'):
        gate_params = sum(p.numel() for p in model.decoder.gate_network.parameters())
        expert_params += gate_params
        print(f"  └─ 门控网络: {format_number(gate_params)}")
    
    print(f"  🔧 专家系统总计: {format_number(expert_params)}")
    
    return encoder_params, decoder_params, expert_params

def main():
    print("🚀 开始计算GeniusRec模型参数量...")
    
    # 加载配置
    config = get_config()
    
    # 加载ID映射以获取物品数量
    id_maps_path = config["data"]["id_maps_file"]
    
    if not os.path.exists(id_maps_path):
        print(f"❌ 错误：找不到ID映射文件 '{id_maps_path}'")
        print("请先运行数据预处理脚本生成该文件")
        return
    
    print(f"📂 加载ID映射文件: {id_maps_path}")
    with open(id_maps_path, 'rb') as f:
        id_maps = pickle.load(f)
    
    num_items = len(id_maps['item_map'])
    num_users = len(id_maps['user_map'])
    
    print(f"📊 数据集统计:")
    print(f"  └─ 物品数量: {num_items:,}")
    print(f"  └─ 用户数量: {num_users:,}")
    
    # 构建模型配置
    encoder_config = config["encoder_model"].copy()
    encoder_config['item_num'] = num_items
    encoder_config['pad_token_id'] = config["pad_token_id"]
    
    decoder_config = config["decoder_model"].copy()
    decoder_config['num_items'] = num_items
    decoder_config['pad_token_id'] = config["pad_token_id"]
    
    expert_config = config["expert_system"]
    
    print(f"\n🔧 模型配置:")
    print(f"  编码器配置: {encoder_config}")
    print(f"  解码器配置: {decoder_config}")
    print(f"  专家系统配置: {expert_config}")
    
    # 创建模型
    print(f"\n🏗️  正在构建模型...")
    model = GENIUSRecModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        expert_config=expert_config
    )
    
    # 计算参数量
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n" + "="*60)
    print(f"📈 GeniusRec模型参数量统计")
    print(f"="*60)
    print(f"🔢 总参数数量: {total_params:,} ({format_number(total_params)})")
    print(f"🎯 可训练参数: {trainable_params:,} ({format_number(trainable_params)})")
    print(f"🔒 冻结参数: {total_params - trainable_params:,} ({format_number(total_params - trainable_params)})")
    
    # 详细组件分析
    encoder_params, decoder_params, expert_params = analyze_model_components(model)
    
    print(f"\n" + "="*60)
    print(f"📊 参数分布概览")
    print(f"="*60)
    print(f"编码器: {format_number(encoder_params)} ({encoder_params/total_params*100:.1f}%)")
    print(f"解码器: {format_number(decoder_params)} ({decoder_params/total_params*100:.1f}%)")
    print(f"  └─ 其中专家系统: {format_number(expert_params)} ({expert_params/total_params*100:.1f}%)")
    
    # 模型大小估算（FP32）
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
    print(f"\n💾 模型大小估算:")
    print(f"  └─ FP32: {model_size_mb:.1f} MB")
    print(f"  └─ FP16: {model_size_mb/2:.1f} MB")
    
    # 总结
    if total_params >= 1e9:
        scale = "B级"
        color = "🔥"
    elif total_params >= 1e8:
        scale = "百M级"
        color = "🚀"
    elif total_params >= 1e7:
        scale = "十M级"
        color = "💪"
    else:
        scale = "M级以下"
        color = "✨"
    
    print(f"\n{color} 总结: 您的GeniusRec模型是一个 {scale} 模型 ({format_number(total_params)})")
    
    return total_params, trainable_params

if __name__ == "__main__":
    main()
