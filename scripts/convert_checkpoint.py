# upgrade_checkpoint.py (已修复BUG的最终版)

import torch
import os
from pathlib import Path

# ==============================================================================
#   配置区域：请根据需要修改这里的路径和参数
# ==============================================================================

# 1. 定义检查点所在的目录
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# 2. 定义输入和输出文件名
INPUT_CHECKPOINT_NAME = "hstu_encoder4win_old.pth"
OUTPUT_CHECKPOINT_NAME = "hstu_encoder_upgraded.pth" # 建议使用新名字以避免混淆

# 3. 定义新旧词汇表的特殊符号数量
OLD_NUM_SPECIAL_TOKENS = 1  # 旧检查点只有1个特殊token (<PAD>)
NEW_NUM_SPECIAL_TOKENS = 4  # 当前模型有4个特殊token

# 4. 定义新的总词汇表大小
NEW_TOTAL_VOCAB_SIZE = 506949

# 5. 嵌入层在state_dict中的键名
EMBEDDING_KEY = "item_embedding.weight"

# ==============================================================================

def upgrade_checkpoint_embedding_size_fixed():
    """
    一个独立的、已修复BUG的脚本，用于精确升级检查点中嵌入层的尺寸，
    并根据新的特殊token数量，正确地、分段地迁移权重。
    """
    print("--- 开始升级检查点 (已修复BUG的最终版) ---")
    
    input_path = CHECKPOINT_DIR / INPUT_CHECKPOINT_NAME
    output_path = CHECKPOINT_DIR / OUTPUT_CHECKPOINT_NAME

    if not input_path.exists():
        print(f"❌ 错误：输入文件未找到 -> {input_path}")
        return

    print(f"▶️ 正在加载旧的检查点: {input_path}")
    # 加载时明确指定 weights_only=False 以读取包含pickle对象的旧文件
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # 兼容两种检查点格式：直接的state_dict或包含它的外层字典
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if EMBEDDING_KEY not in state_dict:
        print(f"❌ 错误：在检查点中未找到键名 '{EMBEDDING_KEY}'。")
        return

    old_embedding_tensor = state_dict[EMBEDDING_KEY]
    old_vocab_size, embedding_dim = old_embedding_tensor.shape
    
    print(f"   - 旧的词汇表大小: {old_vocab_size}")
    print(f"   - 新的目标大小: {NEW_TOTAL_VOCAB_SIZE}")
    print(f"   - 嵌入维度: {embedding_dim}")

    if old_vocab_size >= NEW_TOTAL_VOCAB_SIZE:
        print("✅ 警告：旧的检查点尺寸大于或等于目标尺寸，无需升级。")
        return

    # --- 【核心修复】创建零初始化的新矩阵，并分段迁移 ---
    
    # 1. 创建一个全零的新矩阵，这对于PAD token是正确的
    new_embedding_tensor = torch.zeros(NEW_TOTAL_VOCAB_SIZE, embedding_dim, dtype=old_embedding_tensor.dtype)
    print(f"   - 已创建新的零初始化尺寸为 [{NEW_TOTAL_VOCAB_SIZE}, {embedding_dim}] 的嵌入层。")

    # 2. 迁移特殊Tokens (例如, 将旧的PAD token向量拷贝到新PAD token的位置)
    num_special_to_copy = min(OLD_NUM_SPECIAL_TOKENS, NEW_NUM_SPECIAL_TOKENS)
    new_embedding_tensor[:num_special_to_copy] = old_embedding_tensor[:num_special_to_copy]
    print(f"   - ✅ 已精确拷贝 {num_special_to_copy} 个特殊Token的权重。")

    # 3. 迁移所有真实物品的嵌入向量
    old_items_embedding = old_embedding_tensor[OLD_NUM_SPECIAL_TOKENS:]
    num_items_to_copy = min(old_items_embedding.shape[0], NEW_TOTAL_VOCAB_SIZE - NEW_NUM_SPECIAL_TOKENS)

    new_embedding_tensor[NEW_NUM_SPECIAL_TOKENS : NEW_NUM_SPECIAL_TOKENS + num_items_to_copy] = \
        old_items_embedding[:num_items_to_copy]
    
    print(f"   - ✅ 已精确将 {num_items_to_copy} 个真实物品的权重从旧索引[{OLD_NUM_SPECIAL_TOKENS}:]拷贝到新索引[{NEW_NUM_SPECIAL_TOKENS}:]")

    # 用修复后的新嵌入层权重替换掉旧的
    state_dict[EMBEDDING_KEY] = new_embedding_tensor

    # 根据原始检查点格式，正确地保存
    if 'model_state_dict' in checkpoint:
        checkpoint['model_state_dict'] = state_dict
        torch.save(checkpoint, output_path)
    else:
        torch.save(state_dict, output_path)

    print(f"💾 正在保存修复后的检查点到: {output_path}")
    print("--- ✅ 检查点升级成功！现在可以安全使用新文件进行训练。 ---")


if __name__ == '__main__':
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    upgrade_checkpoint_embedding_size_fixed()