#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像嵌入生成器 - 使用CLIP模型为书籍封面生成嵌入向量

功能：
1. 扫描book_covers_enhanced目录下的图像文件
2. 使用预训练CLIP模型生成图像嵌入
3. 支持批处理和GPU加速
4. 生成book_image_embeddings.npy嵌入文件

使用方法：
1. 确保已安装依赖：pip install torch torchvision clip-by-openai pillow tqdm
2. 准备图像文件到data/book_covers_enhanced/目录
3. 运行：python scripts/image_embeddings_generator.py

注意：需要GPU支持以获得最佳性能
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
            pin_memory=True if self.device == 'cuda' else False
        )
        
        # 生成嵌入
        embeddings_dict = {}
        failed_count = 0
        
        print("开始生成图像嵌入...")
        self.model.eval()
        
        with torch.no_grad():
            for batch_images, batch_asins, batch_success in tqdm(dataloader, desc="处理图像批次"):
                # 移动到指定设备
                batch_images = batch_images.to(self.device)
                
                # 使用CLIP模型生成图像特征
                if self.model_type == 'clip':
                    image_features = self.model.encode_image(batch_images)
                    # L2归一化
                    image_features = F.normalize(image_features, p=2, dim=1)
                
                # 转换为numpy并存储
                image_features_np = image_features.cpu().numpy()
                
                for i, (asin, success) in enumerate(zip(batch_asins, batch_success)):
                    if success:
                        embeddings_dict[asin] = image_features_np[i]
                    else:
                        failed_count += 1
                
                # 清理GPU缓存
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        print(f"✅ 嵌入生成完成!")
        print(f"   - 成功处理: {len(embeddings_dict)} 个图像")
        print(f"   - 处理失败: {failed_count} 个图像")
        
        return embeddings_dict


def main():
    """主函数 - 解析参数并执行图像嵌入生成"""
    # 默认配置
    IMAGE_DIR = 'data/book_covers_enhanced'
    OUTPUT_FILE = 'data/book_image_embeddings.npy'
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    print("=== 图像嵌入生成器 ===\n")
    
    # 检查输入目录
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 图像目录不存在: {IMAGE_DIR}")
        print("请确保已将书籍封面图像放入该目录")
        return
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️  未检测到CUDA，将使用CPU (处理速度较慢)")
    
    print(f"配置信息:")
    print(f"  - 图像目录: {IMAGE_DIR}")
    print(f"  - 输出文件: {OUTPUT_FILE}")
    print(f"  - 批处理大小: {BATCH_SIZE}")
    print(f"  - 计算设备: {device}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  - GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print()
    
    try:
        # 创建生成器并生成嵌入
        generator = ImageEmbeddingGenerator(model_type='clip', device=device)
        embeddings = generator.generate_embeddings(
            image_dir=IMAGE_DIR,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        
        if not embeddings:
            print("❌ 未生成任何嵌入")
            return
        
        # 保存嵌入到文件
        print(f"\n正在保存嵌入到 {OUTPUT_FILE}...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        np.save(OUTPUT_FILE, embeddings)
        
        print("✅ 图像嵌入生成完成!")
        print(f"   - 总嵌入数: {len(embeddings)}")
        sample_embedding = next(iter(embeddings.values()))
        print(f"   - 嵌入维度: {sample_embedding.shape}")
        print(f"   - 已保存至: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ 生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
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
