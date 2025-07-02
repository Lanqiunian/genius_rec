# generate_image_embeddings.py
"""
生成书封面图像嵌入的脚本（预留）
用于为图像专家准备数据

使用方法:
python generate_image_embeddings.py --model_type clip --input_dir data/book_covers_enhanced --output_file data/book_image_embeddings.npy

支持的模型:
- clip: 使用OpenAI CLIP模型
- resnet: 使用ResNet-50预训练模型
- vit: 使用Vision Transformer
"""

import argparse
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    from torchvision.models import resnet50, vit_b_16
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("⚠️  TorchVision not available. Install with: pip install torchvision")


class ImageEmbeddingGenerator:
    """图像嵌入生成器"""
    
    def __init__(self, model_type="clip", device="cuda"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.preprocess = None
        
        self._load_model()
    
    def _load_model(self):
        """加载指定的图像编码模型"""
        if self.model_type == "clip":
            if not CLIP_AVAILABLE:
                raise ImportError("CLIP not available")
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.embedding_dim = 512
            
        elif self.model_type == "resnet":
            if not TORCHVISION_AVAILABLE:
                raise ImportError("TorchVision not available")
            self.model = resnet50(pretrained=True)
            self.model.fc = nn.Identity()  # 移除最后的分类层
            self.model = self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = 2048
            
            # ResNet预处理
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        elif self.model_type == "vit":
            if not TORCHVISION_AVAILABLE:
                raise ImportError("TorchVision not available")
            self.model = vit_b_16(pretrained=True)
            self.model.heads = nn.Identity()  # 移除分类头
            self.model = self.model.to(self.device)
            self.model.eval()
            self.embedding_dim = 768
            
            # ViT预处理
            self.preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        print(f"✅ 加载{self.model_type.upper()}模型成功，嵌入维度: {self.embedding_dim}")
    
    def extract_embedding(self, image_path):
        """从单张图像提取嵌入"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            if self.model_type == "clip":
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.model.encode_image(image_tensor)
                    embedding = embedding.cpu().numpy().flatten()
            else:
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.model(image_tensor)
                    embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"❌ 处理图像失败 {image_path}: {e}")
            return None
    
    def batch_extract_embeddings(self, image_dir, supported_formats=('.jpg', '.jpeg', '.png', '.bmp')):
        """批量提取图像嵌入"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        
        # 查找所有图像文件
        image_files = []
        for ext in supported_formats:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"🔍 找到 {len(image_files)} 张图像")
        
        embeddings_dict = {}
        failed_count = 0
        
        for image_path in tqdm(image_files, desc="提取图像嵌入"):
            # 使用文件名作为ASIN（假设文件名就是ASIN）
            asin = image_path.stem
            
            embedding = self.extract_embedding(image_path)
            if embedding is not None:
                embeddings_dict[asin] = embedding
            else:
                failed_count += 1
        
        print(f"✅ 成功提取 {len(embeddings_dict)} 个图像嵌入")
        if failed_count > 0:
            print(f"⚠️  失败 {failed_count} 个图像")
        
        return embeddings_dict


def main():
    parser = argparse.ArgumentParser(description="Generate image embeddings for book covers")
    parser.add_argument('--model_type', type=str, default='clip', 
                       choices=['clip', 'resnet', 'vit'],
                       help='Type of image encoder model')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing book cover images')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output file path for embeddings (.npy)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--id_maps_file', type=str, 
                       default='data/processed/id_maps.pkl',
                       help='ID maps file for filtering valid ASINs')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 初始化图像嵌入生成器
    try:
        generator = ImageEmbeddingGenerator(args.model_type, args.device)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return
    
    # 批量提取嵌入
    print("开始提取图像嵌入...")
    embeddings_dict = generator.batch_extract_embeddings(args.input_dir)
    
    if not embeddings_dict:
        print("❌ 没有成功提取任何嵌入")
        return
    
    # 如果提供了ID maps文件，进行过滤
    if os.path.exists(args.id_maps_file):
        print(f"🔍 使用ID maps过滤有效的ASIN...")
        with open(args.id_maps_file, 'rb') as f:
            id_maps = pickle.load(f)
        
        valid_asins = set(id_maps['item_map'].keys())
        filtered_embeddings = {asin: emb for asin, emb in embeddings_dict.items() if asin in valid_asins}
        
        print(f"📊 过滤前: {len(embeddings_dict)}, 过滤后: {len(filtered_embeddings)}")
        embeddings_dict = filtered_embeddings
    
    # 检查是否有ASIN到item_id的映射文件
    mapping_file = Path(args.input_dir) / 'asin_to_itemid_mapping.pkl'
    if mapping_file.exists():
        print(f"🔗 发现ASIN到item_id映射文件，转换为item_id索引...")
        with open(mapping_file, 'rb') as f:
            asin_to_itemid = pickle.load(f)
        
        # 将embeddings字典的键从ASIN转换为item_id
        itemid_embeddings = {}
        converted_count = 0
        for asin, embedding in embeddings_dict.items():
            if asin in asin_to_itemid:
                item_id = asin_to_itemid[asin]
                itemid_embeddings[item_id] = embedding
                converted_count += 1
        
        print(f"📈 成功转换 {converted_count} 个嵌入的索引 (ASIN → item_id)")
        embeddings_dict = itemid_embeddings
    
    # 保存结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, embeddings_dict)
    print(f"✅ 图像嵌入已保存到: {output_path}")
    print(f"📊 最终包含 {len(embeddings_dict)} 个有效的图像嵌入")
    print(f"🎯 嵌入维度: {generator.embedding_dim}")


if __name__ == "__main__":
    main()
