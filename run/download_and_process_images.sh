#!/bin/bash
# download_and_process_images.sh
# 一键下载5-core子集图片并生成图像嵌入的脚本

set -e

echo "🚀 开始5-core子集图片下载和处理流程"
echo "=" * 60

# 步骤1: 下载5-core过滤后的图片
echo "📥 步骤1: 下载5-core过滤后的图片..."
cd /root/autodl-tmp/genius_rec-main
python scripts/data_enhancer.py

# 检查下载结果
IMAGE_DIR="data/book_covers_enhanced"
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ 图片目录不存在，下载可能失败"
    exit 1
fi

IMAGE_COUNT=$(find "$IMAGE_DIR" -name "*.jpg" | wc -l)
echo "📊 已下载图片数量: $IMAGE_COUNT"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "⚠️  没有下载到任何图片，请检查网络连接和数据文件"
    exit 1
fi

# 步骤2: 生成图像嵌入 (使用CLIP)
echo "🖼️  步骤2: 生成图像嵌入 (CLIP模型)..."
python generate_image_embeddings.py \
    --model_type clip \
    --input_dir "$IMAGE_DIR" \
    --output_file data/book_image_embeddings.npy \
    --batch_size 32 \
    --use_item_id_keys

# 检查生成结果
if [ -f "data/book_image_embeddings.npy" ]; then
    echo "✅ 图像嵌入生成成功: data/book_image_embeddings.npy"
else
    echo "❌ 图像嵌入生成失败"
    exit 1
fi

# 步骤3: 验证生成的嵌入
echo "🔍 步骤3: 验证生成的嵌入..."
python -c "
import numpy as np
import pickle

# 加载嵌入文件
embeddings = np.load('data/book_image_embeddings.npy', allow_pickle=True).item()
print(f'✅ 成功加载图像嵌入')
print(f'📊 嵌入数量: {len(embeddings)}')

# 检查嵌入格式
if len(embeddings) > 0:
    sample_key = next(iter(embeddings.keys()))
    sample_embedding = embeddings[sample_key]
    print(f'🎯 嵌入维度: {sample_embedding.shape}')
    print(f'📝 样例键: {sample_key} (类型: {type(sample_key)})')
    
    # 检查是否使用了item_id作为键
    if isinstance(sample_key, int):
        print('✅ 使用item_id作为键，与训练代码兼容')
    else:
        print('⚠️  使用ASIN作为键，可能需要额外转换')

# 加载id_maps验证一致性
with open('data/processed/id_maps.pkl', 'rb') as f:
    id_maps = pickle.load(f)

total_items = id_maps['num_items']
coverage = len(embeddings) / total_items * 100
print(f'📈 覆盖率: {coverage:.1f}% ({len(embeddings)}/{total_items})')
"

echo "🎉 图片下载和处理流程完成！"
echo ""
echo "📋 生成的文件:"
echo "  - 图片目录: $IMAGE_DIR/"
echo "  - 图像嵌入: data/book_image_embeddings.npy"
echo "  - 映射文件: $IMAGE_DIR/asin_to_itemid_mapping.pkl"
echo ""
echo "💡 下一步: 可以在训练中启用图像专家"
echo "  python -m src.train_GeniusRec --enable_image_expert --image_embeddings_path data/book_image_embeddings.npy"
