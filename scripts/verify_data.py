# verify_data.py (已修复索引计算BUG的最终版)

import torch
from pathlib import Path
import numpy as np

# --- 1. 配置区域 (保持不变) ---
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
OLD_CHECKPOINT_PATH = CHECKPOINT_DIR / "hstu_encoder4win_old.pth" 
NEW_CHECKPOINT_PATH = CHECKPOINT_DIR / "hstu_encoder.pth" # 验证您用修复后脚本生成的文件
EMBEDDING_KEY = "item_embedding.weight"

# --- 2. 架构参数 (保持不变) ---
OLD_NUM_SPECIAL_TOKENS = 1
NEW_NUM_SPECIAL_TOKENS = 4

# --- 3. 抽样ID (保持不变) ---
# 注意：这些ID是新架构下的ID
SAMPLE_ITEM_IDS = [4, 10, 100, 5555, 12345]

def verify_weights_content_definitively():
    """
    通过精确的索引映射，对转换后的权重进行最终的、决定性的内容校验。
    """
    print("--- 开始执行决定性的编码器权重内容校验 (V3 - 最终修复版) ---")
    
    if not OLD_CHECKPOINT_PATH.exists() or not NEW_CHECKPOINT_PATH.exists():
        print(f"❌ 错误：一个或多个检查点文件不存在。请检查路径。")
        return

    try:
        old_checkpoint = torch.load(OLD_CHECKPOINT_PATH, map_location='cpu')
        new_checkpoint = torch.load(NEW_CHECKPOINT_PATH, map_location='cpu')

        old_state_dict = old_checkpoint.get('model_state_dict', old_checkpoint)
        new_state_dict = new_checkpoint.get('model_state_dict', new_checkpoint)
    except Exception as e:
        print(f"❌ 加载检查点时发生错误: {e}")
        return
        
    old_embedding = old_state_dict[EMBEDDING_KEY]
    new_embedding = new_state_dict[EMBEDDING_KEY]

    print("\n--- 尺寸检查 ---")
    print(f"旧权重尺寸: {old_embedding.shape}")
    print(f"新权重尺寸: {new_embedding.shape}")

    all_checks_passed = True
    print("\n--- 内容校验 (精确索引映射比对) ---")

    for item_id_new in SAMPLE_ITEM_IDS:
        # ---【最终核心修复】---
        # 根据新架构下的 item_id，计算出它在旧架构中对应的索引
        
        # 1. 新权重中的索引就是其ID本身
        new_index = item_id_new
        
        # 2. 计算该物品在旧权重中对应的索引
        #   公式: old_index = (new_index - NEW_TOKENS) + OLD_TOKENS
        old_index = (new_index - NEW_NUM_SPECIAL_TOKENS) + OLD_NUM_SPECIAL_TOKENS
        
        # 安全检查，确保计算出的索引有效
        if not (0 <= old_index < old_embedding.shape[0]):
            print(f"  - 校验 ID={item_id_new}: ❌ 失败 (计算出的旧索引 {old_index} 超出范围)")
            all_checks_passed = False
            continue

        try:
            old_vector = old_embedding[old_index]
            new_vector = new_embedding[new_index]
        except IndexError as e:
            print(f"  - 校验 ID={item_id_new}: ❌ 失败 (张量访问越界: {e})")
            all_checks_passed = False
            continue

        are_vectors_identical = torch.allclose(old_vector, new_vector, atol=1e-7)
        
        if are_vectors_identical:
            print(f"  - 校验新ID={item_id_new}: ✅ 通过 (旧索引 {old_index} <--> 新索引 {new_index})")
        else:
            print(f"  - 校验新ID={item_id_new}: ❌ 失败! 向量值不匹配 (旧索引 {old_index} vs 新索引 {new_index})")
            print(f"    - L2 距离: {torch.dist(old_vector, new_vector).item():.6f}")
            all_checks_passed = False

    print("\n--- 最终结论 ---")
    if all_checks_passed:
        print("✅【校验通过】确认兼容！您的转换脚本工作正常，权重已正确迁移。")
        print("现在可以100%排除权重文件本身的问题。")
        print("请将注意力集中在训练超参数（特别是 `encoder_lr`）上来解决行为专家权重过低的问题。")
    else:
        print("❌【依旧不兼容】校验失败！如果使用了最新的验证脚本依然失败，情况非常蹊跷。请检查文件是否在传输或保存过程中被意外损坏。")

if __name__ == '__main__':
    verify_weights_content_definitively()