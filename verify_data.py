# verify_data.py (高精度最终版)

import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np

# 这是一个独立的脚本，我们直接在这里定义配置路径，避免因运行位置不同导致的问题
# --- 请根据您的项目结构确认这里的路径 ---
try:
    # 假设脚本在项目根目录 (与 src 同级)
    ROOT_DIR = Path(__file__).parent 
    PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
    
    # 检查目录是否存在
    if not PROCESSED_DATA_DIR.exists():
        # 如果不存在，尝试上一级目录，兼容从 src 目录运行的情况
        ROOT_DIR = Path(__file__).parent.parent
        PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
        if not PROCESSED_DATA_DIR.exists():
            print(f"❌ 致命错误: 无法在 {Path(__file__).parent} 或 {Path(__file__).parent.parent} 下找到 data/processed 目录。")
            sys.exit(1) # 退出脚本

    ID_MAPS_FILE = PROCESSED_DATA_DIR / "id_maps.pkl"
    TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
    VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.parquet"
except Exception as e:
    print(f"❌ 致命错误: 路径配置失败: {e}")
    sys.exit(1)


def find_max_id_in_parquet(file_path: Path) -> int:
    """
    使用最基础的循环来精确查找 Parquet 文件中 history 列表/数组里的最大ID。
    这个版本能正确处理 history 列中元素为 Numpy 数组的情况。
    """
    try:
        df = pd.read_parquet(file_path)
        if 'history' not in df.columns:
            print(f"   ⚠️ 警告: 在 {file_path.name} 中未找到 'history' 列。")
            return 0

        global_max_id = 0
        
        for history_container in tqdm(df['history'], desc=f"   分析 {file_path.name}"):
            # 【核心修正】
            # 不再使用 'if history_container:' 这种模糊的判断
            # 直接检查容器类型和长度
            if isinstance(history_container, (np.ndarray, list, tuple)) and len(history_container) > 0:
                local_max = np.max(history_container) # 使用 np.max() 更通用
                if local_max > global_max_id:
                    global_max_id = local_max
        
        return int(global_max_id)

    except FileNotFoundError:
        print(f"   ❌ 错误: 找不到文件 {file_path}。")
        return -1 
    except Exception as e:
        print(f"   ❌ 错误: 加载或分析文件时发生未知错误: {e}")
        return -2


def main():
    """
    主函数，执行所有验证步骤。
    """
    print("="*80)
    print("🚀 开始执行高精度数据一致性验证脚本 (Numpy兼容版)...")
    print(f"[*] 将从以下目录读取处理后的数据: {PROCESSED_DATA_DIR.resolve()}")
    print("="*80)

    # 1. 检查 id_maps.pkl 文件
    print(f"--- 步骤 1: 检查 ID 映射文件: {ID_MAPS_FILE.name} ---")
    
    expected_max_id = -1
    
    try:
        with open(ID_MAPS_FILE, 'rb') as f:
            id_maps = pickle.load(f)
        print("✅ Pickle 文件加载成功!")

        if 'max_item_id' in id_maps:
            pkl_max_item_id = int(id_maps['max_item_id'])
            print(f"   检测到 'max_item_id' (新版格式)，值为: {pkl_max_item_id}")
            expected_max_id = pkl_max_item_id
        elif 'num_items' in id_maps:
            pkl_num_items = int(id_maps['num_items'])
            num_special = int(id_maps.get('num_special_tokens', 4))
            print(f"   检测到 'num_items' (旧版格式)，值为: {pkl_num_items}")
            expected_max_id = pkl_num_items + num_special - 1
            print(f"   根据 'num_items' 推断出期望的最大 ID 应为: {expected_max_id}")
        else:
            print("❌ 致命错误: 在 id_maps.pkl 中既没有找到 'max_item_id' 也没有 'num_items'。")
            return

    except FileNotFoundError:
        print(f"❌ 致命错误: 找不到 {ID_MAPS_FILE.name}。请先运行预处理。")
        return
    except Exception as e:
        print(f"❌ 致命错误: 加载或读取 Pickle 文件时发生错误: {e}")
        return

    print("-" * 80)
    
    # 2. 检查 Parquet 文件
    print("--- 步骤 2: 精确分析 Parquet 文件中的真实最大 ID ---")
    true_max_id_train = find_max_id_in_parquet(TRAIN_FILE)
    true_max_id_val = find_max_id_in_parquet(VALIDATION_FILE)
    
    print("\n   分析完成。真实最大ID如下:")
    print(f"   - 训练集 (train.parquet): {true_max_id_train if true_max_id_train >= 0 else '读取失败'}")
    print(f"   - 验证集 (validation.parquet): {true_max_id_val if true_max_id_val >= 0 else '读取失败'}")
    
    print("="*80)

    # 3. 最终诊断 (逻辑不变)
    print("--- 步骤 3: 最终诊断 ---")
    if expected_max_id == -1:
        print("无法进行诊断，因为未能从 .pkl 文件中确定期望的最大ID。")
        return

    all_true_max_ids = [mid for mid in [true_max_id_train, true_max_id_val] if mid >= 0]
    if not all_true_max_ids:
        print("未能成功读取任何 Parquet 文件，无法进行比较。")
        return
        
    overall_true_max_id = max(all_true_max_ids)

    print(f"[*] 综合所有文件，得出的关键值如下:")
    print(f"   - 根据 id_maps.pkl, 期望的最大 ID 是: {expected_max_id}")
    print(f"   - 在所有 .parquet 文件中, 实际找到的最大 ID 是: {overall_true_max_id}")
    
    print("\n[*] 诊断结论:")
    if overall_true_max_id == expected_max_id:
        print("🎉 恭喜！元数据与真实数据完全一致！")
        print("   如果仍然报错，问题非常可能出在您的训练代码读取/处理数据的逻辑中，而非文件本身。")
    elif overall_true_max_id > expected_max_id:
        print("❌ 确诊: 严重不一致！(这是导致索引越界的原因)")
        print(f"   数据文件中的最大ID ({overall_true_max_id}) 超出了元数据声明的范围 ({expected_max_id})。")
        print("   这表明 `preprocess.py` 脚本生成了相互矛盾的文件。")
    else:
        print("⚠️ 确诊: 数据不一致。")
        print(f"   数据文件中的最大ID ({overall_true_max_id}) 小于元数据声明的范围 ({expected_max_id})。")
        print("   这虽不会导致索引越界，但表明您的预处理流程存在缺陷，可能过滤掉了包含最大ID的数据。")
        
    print("="*80)


if __name__ == "__main__":
    main()
