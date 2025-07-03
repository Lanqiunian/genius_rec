#!/usr/bin/env python3
"""
权重迁移脚本：将预训练编码器权重迁移到新的ID系统

这个脚本的作用：
1. 加载旧的预训练编码器权重
2. 加载旧的和新的ID映射
3. 创建新的更大的嵌入矩阵（为特殊标记预留空间）
4. 将旧权重迁移到新的ID位置
5. 保存迁移后的权重

使用方法：
python scripts/migrate_weights.py --old_weights checkpoints/hstu_encoder.pth --old_id_maps data/processed/id_maps_old.pkl --new_id_maps data/processed/id_maps.pkl --output checkpoints/hstu_encoder_migrated.pth
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import get_config

def load_id_mapping(old_maps_path, new_maps_path):
    """
    加载旧的和新的ID映射，构建ASIN -> old_id -> new_id的映射关系
    
    Returns:
        dict: {asin: (old_id, new_id)} 的映射字典
    """
    print("正在加载ID映射...")
    
    # 加载旧的ID映射
    with open(old_maps_path, 'rb') as f:
        old_maps = pickle.load(f)
    
    # 加载新的ID映射  
    with open(new_maps_path, 'rb') as f:
        new_maps = pickle.load(f)
    
    old_item_map = old_maps['item_map']  # {asin: old_id}
    new_item_map = new_maps['item_map']  # {asin: new_id}
    
    # 构建迁移映射：{asin: (old_id, new_id)}
    migration_map = {}
    for asin in old_item_map:
        if asin in new_item_map:
            migration_map[asin] = (old_item_map[asin], new_item_map[asin])
        else:
            print(f"警告：ASIN {asin} 在新映射中不存在，跳过")
    
    print(f"成功映射 {len(migration_map)} 个物品")
    return migration_map, old_maps, new_maps

def migrate_embedding_weights(old_state_dict, migration_map, old_maps, new_maps, config):
    """
    迁移嵌入层权重到新的ID系统
    
    Args:
        old_state_dict: 旧的模型状态字典
        migration_map: ASIN映射字典 {asin: (old_id, new_id)}
        old_maps: 旧的ID映射信息
        new_maps: 新的ID映射信息
        config: 配置字典
        
    Returns:
        dict: 更新后的状态字典
    """
    print("正在迁移嵌入层权重...")
    
    # 如果state_dict是checkpoint格式，提取model_state_dict
    if 'model_state_dict' in old_state_dict:
        model_state = old_state_dict['model_state_dict']
        print("从checkpoint中提取model_state_dict")
    else:
        model_state = old_state_dict
    
    # 获取原始嵌入权重
    old_embedding_key = 'embedding.weight'  # 根据实际模型结构调整
    if old_embedding_key not in model_state:
        # 尝试其他可能的键名
        possible_keys = [k for k in model_state.keys() if 'embedding' in k and 'weight' in k]
        if possible_keys:
            old_embedding_key = possible_keys[0]
            print(f"使用嵌入层键名: {old_embedding_key}")
        else:
            raise KeyError(f"无法找到嵌入层权重，可用的键: {list(model_state.keys())}")
    
    old_embedding_weights = model_state[old_embedding_key]
    old_vocab_size, embedding_dim = old_embedding_weights.shape
    
    # 计算新的词汇表大小（包含特殊标记）
    new_vocab_size = new_maps['num_items'] + new_maps['num_special_tokens']
    
    print(f"旧词汇表大小: {old_vocab_size}")
    print(f"新词汇表大小: {new_vocab_size}")
    print(f"嵌入维度: {embedding_dim}")
    
    # 创建新的更大的嵌入矩阵
    new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim, dtype=old_embedding_weights.dtype)
    
    # 初始化特殊标记的嵌入（使用小的随机值）
    special_tokens = new_maps['special_tokens']
    for token_name, token_id in special_tokens.items():
        if token_id < new_vocab_size:
            # 使用正态分布初始化特殊标记
            new_embedding_weights[token_id] = torch.randn(embedding_dim) * 0.01
            print(f"初始化特殊标记 {token_name} (ID: {token_id})")
    
    # 迁移物品嵌入
    migrated_count = 0
    for asin, (old_id, new_id) in migration_map.items():
        if old_id < old_vocab_size and new_id < new_vocab_size:
            new_embedding_weights[new_id] = old_embedding_weights[old_id]
            migrated_count += 1
    
    print(f"成功迁移 {migrated_count} 个物品的嵌入向量")
    
    # 对于新出现的物品（在新数据中但不在旧数据中），使用随机初始化
    for new_id in range(new_maps['num_special_tokens'], new_vocab_size):
        if new_embedding_weights[new_id].sum() == 0:  # 如果还是零向量
            new_embedding_weights[new_id] = torch.randn(embedding_dim) * 0.01
    
    # 更新状态字典
    if 'model_state_dict' in old_state_dict:
        # 如果原来是checkpoint格式，保持格式
        new_state_dict = old_state_dict.copy()
        new_state_dict['model_state_dict'] = model_state.copy()
        new_state_dict['model_state_dict'][old_embedding_key] = new_embedding_weights
    else:
        # 如果原来就是纯状态字典
        new_state_dict = old_state_dict.copy()
        new_state_dict[old_embedding_key] = new_embedding_weights
    
    return new_state_dict

def main():
    parser = argparse.ArgumentParser(description='迁移预训练编码器权重到新的ID系统')
    parser.add_argument('--old_weights', type=str, required=True, help='旧的预训练权重文件路径')
    parser.add_argument('--old_id_maps', type=str, help='旧的ID映射文件路径')
    parser.add_argument('--new_id_maps', type=str, help='新的ID映射文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出的迁移后权重文件路径')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config()
    
    # 如果没有指定旧的ID映射，尝试找到备份文件
    if args.old_id_maps is None:
        # 假设在迁移前备份了旧的映射文件
        old_id_maps_path = Path(config['data']['processed_data_dir']) / 'id_maps_old.pkl'
        if not old_id_maps_path.exists():
            raise FileNotFoundError(f"请指定旧的ID映射文件路径，或确保存在备份文件：{old_id_maps_path}")
        args.old_id_maps = str(old_id_maps_path)
    
    if args.new_id_maps is None:
        args.new_id_maps = str(config['data']['id_maps_file'])
    
    print("=== 权重迁移脚本 ===")
    print(f"旧权重: {args.old_weights}")
    print(f"旧ID映射: {args.old_id_maps}")
    print(f"新ID映射: {args.new_id_maps}")
    print(f"输出: {args.output}")
    
    # 检查文件是否存在
    for file_path in [args.old_weights, args.old_id_maps, args.new_id_maps]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载旧的权重
    print(f"\n正在加载旧权重: {args.old_weights}")
    old_state_dict = torch.load(args.old_weights, map_location='cpu')
    print(f"权重文件包含的键: {list(old_state_dict.keys())}")
    
    # 加载ID映射
    migration_map, old_maps, new_maps = load_id_mapping(args.old_id_maps, args.new_id_maps)
    
    # 迁移权重
    new_state_dict = migrate_embedding_weights(old_state_dict, migration_map, old_maps, new_maps, config)
    
    # 保存新权重
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n正在保存迁移后的权重到: {output_path}")
    torch.save(new_state_dict, output_path)
    
    print("✅ 权重迁移完成！")
    print(f"新权重文件已保存到: {output_path}")
    
    # 验证迁移结果
    print("\n=== 迁移验证 ===")
    new_loaded = torch.load(output_path, map_location='cpu')
    
    # 处理checkpoint格式
    if 'model_state_dict' in new_loaded:
        state_dict_to_check = new_loaded['model_state_dict']
    else:
        state_dict_to_check = new_loaded
    
    embedding_keys = [k for k in state_dict_to_check.keys() if 'embedding' in k and 'weight' in k]
    if embedding_keys:
        embedding_key = embedding_keys[0]
        new_embedding_shape = state_dict_to_check[embedding_key].shape
        print(f"新嵌入矩阵形状: {new_embedding_shape}")
        print(f"预期词汇表大小: {new_maps['num_items'] + new_maps['num_special_tokens']}")
        
        if new_embedding_shape[0] == new_maps['num_items'] + new_maps['num_special_tokens']:
            print("✅ 迁移验证成功！")
        else:
            print("❌ 迁移验证失败！")
    else:
        print("❌ 无法找到嵌入层进行验证")

if __name__ == '__main__':
    main()
