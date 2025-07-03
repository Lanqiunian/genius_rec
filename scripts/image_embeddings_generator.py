#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像嵌入生成器 - 支持CLIP等多种视觉模型
使用单卡4090高效生成图像嵌入，支持批处理和内存优化
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from tqdm import tqdm
import gc

class ImageDataset(Dataset):
    """高效的图像数据集，支持延迟加载"""
    
    def __init__(self, image_paths, asins, preprocess):
        self.image_paths = image_paths
        self.asins = asins
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        asin = self.asins[idx]
        
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image)
            return image_tensor, asin, True  # True表示加载成功
        except Exception as e:
            # 如果图像损坏，返回零张量
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, asin, False  # False表示加载失败

class ImageEmbeddingGenerator:
    """图像嵌入生成器"""
    
    def __init__(self, model_type='clip', device='cuda'):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.preprocess = None
        
    def load_model(self):
        """加载指定的视觉模型"""
        print(f"正在加载 {self.model_type} 模型...")
        
        if self.model_type == 'clip':
            # 加载CLIP模型
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            self.model = model
            self.preprocess = preprocess
            print(f"✅ CLIP ViT-B/32 模型加载成功 (设备: {self.device})")
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def generate_embeddings(self, image_dir, batch_size=32, num_workers=4):
        """
        批量生成图像嵌入
        
        Args:
            image_dir: 图像目录路径
            batch_size: 批处理大小 (4090可以使用较大的batch_size)
            num_workers: 数据加载器工作线程数
            
        Returns:
            dict: {asin: embedding_vector}
        """
        if self.model is None:
            self.load_model()
            
        # 扫描图像文件
        print("正在扫描图像文件...")
        image_paths = []
        asins = []
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                asin = filename.split('.')[0]  # 去掉扩展名得到ASIN
                image_path = os.path.join(image_dir, filename)
                image_paths.append(image_path)
                asins.append(asin)
        
        if not image_paths:
            print("❌ 未找到任何图像文件")
            return {}
            
        print(f"📁 找到 {len(image_paths)} 个图像文件")
        
        # 创建数据集和数据加载器
        dataset = ImageDataset(image_paths, asins, self.preprocess)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        # 批量生成嵌入
        embeddings = {}
        failed_count = 0
        failed_images = []  # 记录失败的图像
        
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="生成图像嵌入")
            
            for batch_images, batch_asins, batch_success in progress_bar:
                batch_images = batch_images.to(self.device, non_blocking=True)
                
                # 生成图像嵌入
                if self.model_type == 'clip':
                    image_features = self.model.encode_image(batch_images)
                    # 归一化嵌入向量
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 转换为CPU numpy数组并保存
                image_features_cpu = image_features.cpu().numpy()
                
                for i, (asin, success) in enumerate(zip(batch_asins, batch_success)):
                    if success:
                        embeddings[asin] = image_features_cpu[i]
                    else:
                        failed_count += 1
                        failed_images.append(asin)  # 记录失败的ASIN
                
                # 更新进度条信息
                progress_bar.set_postfix({
                    ' 成功': len(embeddings),
                    '失败': failed_count,
                    'GPU内存': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
                })
                
                # 定期清理GPU内存
                if len(embeddings) % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        print(f"\n✅ 嵌入生成完成!")
        print(f"  - 成功生成: {len(embeddings)} 个")
        print(f"  - 失败: {failed_count} 个")
        print(f"  - 嵌入维度: {list(embeddings.values())[0].shape if embeddings else 'N/A'}")
        
        # 如果有失败的图像，记录到文件
        if failed_images:
            failed_log_file = os.path.join(os.path.dirname(image_dir), 'failed_image_embeddings.txt')
            with open(failed_log_file, 'w') as f:
                f.write("失败生成嵌入的图像列表:\n")
                f.write("=" * 40 + "\n")
                for asin in failed_images:
                    f.write(f"{asin}\n")
            print(f"  - 失败图像列表已保存到: {failed_log_file}")
        
        return embeddings

def map_asins_to_item_ids(embeddings, mapping_file):
    """将ASIN键转换为item_id键，用于与训练代码兼容"""
    if not os.path.exists(mapping_file):
        print(f"⚠️  映射文件不存在: {mapping_file}")
        print("   将使用ASIN作为键返回嵌入")
        return embeddings
    
    print("正在加载ASIN到item_id的映射...")
    with open(mapping_file, 'rb') as f:
        asin_to_itemid = pickle.load(f)
    
    # 转换键从ASIN到item_id
    itemid_embeddings = {}
    missing_count = 0
    
    for asin, embedding in embeddings.items():
        if asin in asin_to_itemid:
            item_id = asin_to_itemid[asin]
            itemid_embeddings[item_id] = embedding
        else:
            missing_count += 1
    
    print(f"✅ 键转换完成:")
    print(f"  - 成功映射: {len(itemid_embeddings)} 个")
    print(f"  - 映射缺失: {missing_count} 个")
    
    return itemid_embeddings

def main():
    parser = argparse.ArgumentParser(description='图像嵌入生成器')
    parser.add_argument('--model_type', type=str, default='clip', 
                       choices=['clip'], help='视觉模型类型')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入图像目录')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出嵌入文件路径 (.npy)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批处理大小 (4090建议32-64)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作线程数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--use_item_id_keys', action='store_true',
                       help='使用item_id作为键 (用于训练兼容性)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，将使用CPU (速度较慢)")
        args.device = 'cpu'
    
    print("🚀 图像嵌入生成器启动")
    print("=" * 50)
    print(f"模型类型: {args.model_type}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出文件: {args.output_file}")
    print(f"批处理大小: {args.batch_size}")
    print(f"计算设备: {args.device}")
    if args.device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 50)
    
    # 创建生成器并生成嵌入
    generator = ImageEmbeddingGenerator(args.model_type, args.device)
    embeddings = generator.generate_embeddings(
        args.input_dir, 
        args.batch_size, 
        args.num_workers
    )
    
    if not embeddings:
        print("❌ 未生成任何嵌入，程序退出")
        return
    
    # 如果需要，转换为item_id键
    if args.use_item_id_keys:
        mapping_file = os.path.join(args.input_dir, 'asin_to_itemid_mapping.pkl')
        embeddings = map_asins_to_item_ids(embeddings, mapping_file)
    
    # 保存嵌入文件
    print(f"\n💾 正在保存嵌入到: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, embeddings, allow_pickle=True)
    
    print("🎉 图像嵌入生成完成!")
    print(f"📊 最终统计:")
    print(f"  - 嵌入数量: {len(embeddings)}")
    print(f"  - 文件大小: {os.path.getsize(args.output_file) / 1024**2:.1f} MB")
    
    # 内存清理
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
