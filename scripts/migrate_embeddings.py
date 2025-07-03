#!/usr/bin/env python3
"""
嵌入文件迁移脚本

用于将基于旧ID映射的嵌入文件迁移到新的ID映射系统。
主要处理：
1. book_gemini_embeddings_filtered.npy
2. book_image_embeddings.npy

作者：AI重构助手
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def load_id_maps(file_path):
    """加载ID映射文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_item_migration_map(old_maps, new_maps):
    """创建物品ID迁移映射"""
    migration_map = {}
    
    # 获取旧系统的物品映射（item_id -> index）
    if 'item_map' in old_maps:
        old_item_map = old_maps['item_map']
    elif 'item_id_to_index' in old_maps:
        old_item_map = old_maps['item_id_to_index']
    else:
        raise KeyError(f"旧映射中找不到物品映射，可用键: {list(old_maps.keys())}")
    
    # 获取新系统的物品映射
    if 'item_map' in new_maps:
        new_item_map = new_maps['item_map']
    elif 'item_id_to_index' in new_maps:
        new_item_map = new_maps['item_id_to_index']
    else:
        raise KeyError(f"新映射中找不到物品映射，可用键: {list(new_maps.keys())}")
    
    # 创建反向映射：旧index -> item_id
    old_index_to_item = {idx: item_id for item_id, idx in old_item_map.items()}
    
    # 创建迁移映射：旧index -> 新index
    migrated_count = 0
    for old_idx, item_id in old_index_to_item.items():
        if item_id in new_item_map:
            # 新系统中，物品索引需要为特殊标记预留空间
            new_idx = new_item_map[item_id] + new_maps['num_special_tokens']
            migration_map[old_idx] = new_idx
            migrated_count += 1
    
    print(f"成功映射 {migrated_count} 个物品索引")
    return migration_map

def migrate_embedding_file(input_file, output_file, migration_map, new_total_items, old_maps, new_maps):
    """迁移单个嵌入文件"""
    print(f"\n正在迁移: {input_file} -> {output_file}")
    
    try:
        # 尝试加载嵌入文件（允许pickle）
        data = np.load(input_file, allow_pickle=True)
        print(f"原始数据形状: {data.shape}")
        print(f"原始数据类型: {data.dtype}")
        
        # 如果是标量对象数组，提取实际的字典
        if data.shape == () and data.dtype == object:
            embedding_dict = data.item()
            print(f"检测到字典格式，条目数量: {len(embedding_dict)}")
            
            # 检查键和值的类型
            keys = list(embedding_dict.keys())
            if len(keys) > 0:
                first_key = keys[0]
                first_value = embedding_dict[first_key]
                print(f"键类型: {type(first_key)}, 值形状: {first_value.shape}")
                embedding_dim = first_value.shape[0]
                print(f"嵌入维度: {embedding_dim}")
                
                # 创建新的嵌入矩阵
                new_embeddings = np.zeros((new_total_items, embedding_dim), dtype=first_value.dtype)
                print(f"新嵌入矩阵形状: {new_embeddings.shape}")
                
                # 根据文件类型处理不同的键格式
                migrated_count = 0
                if 'gemini' in str(input_file):
                    # Gemini嵌入使用ASIN作为键，不需要迁移索引，直接保存为字典
                    print("处理Gemini嵌入（ASIN -> 保持字典格式）")
                    print("注意：Gemini嵌入保持原始字典格式，无需索引迁移")
                    
                    # 直接保存原始字典格式
                    np.save(output_file, embedding_dict)
                    migrated_count = len(embedding_dict)
                                
                elif 'image' in str(input_file):
                    # 图像嵌入使用整数ID作为键，保持字典格式但更新键值
                    print("处理图像嵌入（保持字典格式，更新整数键）")
                    # 原系统：{1: embedding, 2: embedding, ...}
                    # 新系统：{4: embedding, 5: embedding, ...} (为特殊标记预留0,1,2,3)
                    offset = 3  # 为SOS(1), EOS(2), MASK(3)预留空间
                    
                    new_embedding_dict = {}
                    for old_item_id, embedding in embedding_dict.items():
                        new_item_id = old_item_id + offset
                        new_embedding_dict[new_item_id] = embedding
                        migrated_count += 1
                    
                    # 保存为字典格式
                    np.save(output_file, new_embedding_dict)
                
                print(f"成功迁移 {migrated_count} 个嵌入向量")
                
                print(f"✅ 已保存到: {output_file}")
                
                return True
            else:
                print("❌ 空字典")
                return False
        else:
            print("❌ 不支持的文件格式")
            return False
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='迁移嵌入文件到新的ID系统')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据目录路径')
    parser.add_argument('--old_id_maps', type=str, default='data/processed/id_maps_old.pkl',
                        help='旧的ID映射文件路径')
    parser.add_argument('--new_id_maps', type=str, default='data/processed/id_maps.pkl',
                        help='新的ID映射文件路径')
    
    args = parser.parse_args()
    
    print("=== 嵌入文件迁移脚本 ===")
    
    # 要迁移的文件列表
    embedding_files = [
        'book_gemini_embeddings_filtered.npy',
        'book_image_embeddings.npy'
    ]
    
    # 检查文件是否存在
    data_dir = Path(args.data_dir)
    for filename in embedding_files:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return
        print(f"✅ 找到文件: {file_path}")
    
    # 加载ID映射
    print(f"\n正在加载ID映射...")
    old_maps = load_id_maps(args.old_id_maps)
    new_maps = load_id_maps(args.new_id_maps)
    
    print(f"旧系统物品数量: {old_maps.get('num_items', len(old_maps.get('item_map', {})))}")
    print(f"新系统物品数量: {new_maps['num_items']}")
    print(f"特殊标记数量: {new_maps['num_special_tokens']}")
    
    # 创建迁移映射
    migration_map = create_item_migration_map(old_maps, new_maps)
    
    # 计算新的总物品数量（包括特殊标记）
    new_total_items = new_maps['num_items'] + new_maps['num_special_tokens']
    
    # 迁移每个嵌入文件
    success_count = 0
    for filename in embedding_files:
        input_file = data_dir / filename
        output_file = data_dir / f"{filename.replace('.npy', '_migrated.npy')}"
        
        if migrate_embedding_file(input_file, output_file, migration_map, new_total_items, old_maps, new_maps):
            success_count += 1
    
    print(f"\n=== 迁移完成 ===")
    print(f"成功迁移 {success_count}/{len(embedding_files)} 个文件")
    
    if success_count == len(embedding_files):
        print("✅ 所有嵌入文件迁移成功！")
        print("\n迁移后的文件:")
        for filename in embedding_files:
            output_file = data_dir / f"{filename.replace('.npy', '_migrated.npy')}"
            print(f"  - {output_file}")
    else:
        print("❌ 部分文件迁移失败，请检查错误信息")

if __name__ == '__main__':
    main()
