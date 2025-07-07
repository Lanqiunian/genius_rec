# upgrade_checkpoint.py (最终修复版)

import torch
import os
from pathlib import Path

# ==============================================================================
#  配置区域：请根据需要修改这里的路径和尺寸
# ==============================================================================

# 1. 定义检查点所在的目录
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# 2. 定义输入和输出文件名
INPUT_CHECKPOINT_NAME = "hstu_encoder4win_old.pth"
OUTPUT_CHECKPOINT_NAME = "hstu_encoder4win_upgraded.pth"

# 3. 定义新旧词汇表的特殊符号数量
OLD_NUM_SPECIAL_TOKENS = 1  # 假设旧的检查点只有1个特殊token (<PAD>)
NEW_NUM_SPECIAL_TOKENS = 4  # 当前模型有4个特殊token

# 4. 定义新的总词汇表大小
NEW_TOTAL_VOCAB_SIZE = 506949

# 5. 嵌入层在state_dict中的键名
EMBEDDING_KEY = "item_embedding.weight"

# ==============================================================================

def upgrade_checkpoint_embedding_size():
    """
    一个独立的脚本，用于升级检查点中嵌入层尺寸，并根据新的特殊token数量，
    正确地迁移物品向量的位置。
    """
    print("--- 开始升级检查点 (最终修复版) ---")
    
    input_path = CHECKPOINT_DIR / INPUT_CHECKPOINT_NAME
    output_path = CHECKPOINT_DIR / OUTPUT_CHECKPOINT_NAME

    if not input_path.exists():
        print(f"❌ 错误：输入文件未找到 -> {input_path}")
        return

    print(f"▶️ 正在加载旧的检查点: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

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

    # 创建一个新的、尺寸正确的嵌入层张量
    new_embedding_tensor = torch.empty(NEW_TOTAL_VOCAB_SIZE, embedding_dim, dtype=old_embedding_tensor.dtype)
    torch.nn.init.normal_(new_embedding_tensor, mean=0, std=0.02)
    print(f"   - 已创建新的尺寸为 [{NEW_TOTAL_VOCAB_SIZE}, {embedding_dim}] 的嵌入层。")

    # --- 【核心修复】执行正确的迁移逻辑 ---
    # 1. 计算旧的真实物品嵌入部分
    old_items_embedding = old_embedding_tensor[OLD_NUM_SPECIAL_TOKENS:]
    num_old_items = old_items_embedding.shape[0]

    # 2. 计算新嵌入矩阵中，可以被填充的物品数量
    num_new_items_to_fill = min(num_old_items, NEW_TOTAL_VOCAB_SIZE - NEW_NUM_SPECIAL_TOKENS)

    # 3. 执行精确的拷贝
    #    将旧的物品向量，拷贝到新矩阵中从第4个位置开始的地方
    new_embedding_tensor[NEW_NUM_SPECIAL_TOKENS : NEW_NUM_SPECIAL_TOKENS + num_new_items_to_fill] = \
        old_items_embedding[:num_new_items_to_fill]
    
    print(f"   - ✅ 已成功将 {num_new_items_to_fill} 个真实物品的权重，从旧索引 {OLD_NUM_SPECIAL_TOKENS}: "
          f"拷贝到新索引 {NEW_NUM_SPECIAL_TOKENS}:")

    # 用新的嵌入层权重替换掉旧的
    state_dict[EMBEDDING_KEY] = new_embedding_tensor

    print(f"💾 正在保存升级后的检查点到: {output_path}")
    torch.save(checkpoint, output_path)

    print("--- ✅ 检查点升级成功！ ---")


if __name__ == '__main__':
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    upgrade_checkpoint_embedding_size()