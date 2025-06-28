# preprocess_aligned.py (对齐官方代码版本)

import pandas as pd
import gzip
from tqdm import tqdm
import pickle
import json
from pathlib import Path

# --- 配置 (对齐官方代码) ---
ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"
# 为对齐版本创建新的输出目录
PROCESSED_DATA_DIR = DATA_DIR / "processed_aligned"
# 注意：官方代码使用ratings_Books.csv，我们用jsonl文件模拟，只取需要的列
REVIEW_FILE = DATA_DIR / "Books.jsonl.gz" 
# 输出文件
TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.parquet"
TEST_FILE = PROCESSED_DATA_DIR / "test.parquet"
ID_MAPS_FILE = PROCESSED_DATA_DIR / "id_maps.pkl"

# 对齐参数
K_CORE = 5
MIN_SEQ_LEN = 5
SEED = 42

def parse_jsonl_to_df(file_path):
    """解析JSONL并直接返回只含所需列的DataFrame，以节省内存"""
    records = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"解析 {Path(file_path).name}"):
            try:
                data = json.loads(line)
                # 只提取我们需要的字段
                records.append({
                    'user_id': data.get('user_id'),
                    'item_id': data.get('asin'),
                    'timestamp': data.get('timestamp')
                })
            except (json.JSONDecodeError, AttributeError):
                continue
    return pd.DataFrame(records)

def main():
    """完全对齐官方代码库中AmazonDataProcessor的预处理逻辑"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("--- 1. 加载原始评论数据 ---")
    ratings = parse_jsonl_to_df(REVIEW_FILE)
    ratings.dropna(inplace=True) # 丢弃无效行
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'])
    
    print(f"原始数据点: {len(ratings)}")
    print(f"原始用户数: {ratings['user_id'].nunique()}")
    print(f"原始物品数: {ratings['item_id'].nunique()}")

    print(f"--- 2. 执行 {K_CORE}-core 过滤 ---")
    while True:
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['item_id'].value_counts()
        initial_rows = len(ratings)
        user_mask = user_counts[user_counts >= K_CORE].index
        item_mask = item_counts[item_counts >= K_CORE].index
        ratings = ratings[ratings['user_id'].isin(user_mask) & ratings['item_id'].isin(item_mask)]
        if initial_rows == len(ratings):
            break
            
    print(f"过滤后数据点: {len(ratings)}")
    print(f"过滤后用户数: {ratings['user_id'].nunique()}")
    print(f"过滤后物品数: {ratings['item_id'].nunique()}")

    # --- 3. ID 重映射 (对齐官方) ---
    print("--- 3. 将user_id和item_id重映射为连续整数 ---")
    
    # 【关键修正】重映射后的ID从1开始，为pad_token_id=0留出空间
    ratings['user_id_remap'] = pd.Categorical(ratings['user_id']).codes + 1
    ratings['item_id_remap'] = pd.Categorical(ratings['item_id']).codes + 1
    
    # 【关键修正】保存新的映射关系
    user_map = {original: remap_id for original, remap_id in zip(ratings['user_id'], ratings['user_id_remap'])}
    item_map = {original: remap_id for original, remap_id in zip(ratings['item_id'], ratings['item_id_remap'])}
    
    id_maps = {
        'user_map': user_map,
        'item_map': item_map,
        'num_users': len(user_map),
        'num_items': len(item_map)
    }

    print("--- 4. 按用户分组生成序列 (使用重映射后的ID) ---")
    df_sorted = ratings.sort_values(by=['user_id_remap', 'timestamp'])
    grouped = df_sorted.groupby('user_id_remap')['item_id_remap'].apply(list)
    
    print(f"--- 5. 过滤掉长度小于 {MIN_SEQ_LEN} 的序列 ---")
    user_history = grouped[grouped.apply(len) >= MIN_SEQ_LEN].reset_index()
    user_history.rename(columns={'item_id_remap': 'history'}, inplace=True)
    
    print(f"最终有效用户数: {len(user_history)}")
    
    print("--- 6. 划分训练/验证/测试集 ---")
    user_ids = user_history['user_id_remap'].unique()
    import numpy as np
    np.random.seed(SEED)
    np.random.shuffle(user_ids)
    
    train_size = int(len(user_ids) * 0.8)
    val_size = int(len(user_ids) * 0.1)
    
    train_users = user_ids[:train_size]
    val_users = user_ids[train_size : train_size + val_size]
    test_users = user_ids[train_size + val_size:]
    
    train_df = user_history[user_history['user_id_remap'].isin(train_users)]
    validation_df = user_history[user_history['user_id_remap'].isin(val_users)]
    test_df = user_history[user_history['user_id_remap'].isin(test_users)]

    print("--- 7. 保存处理后的数据 ---")
    train_df[['history']].to_parquet(TRAIN_FILE)
    validation_df[['history']].to_parquet(VALIDATION_FILE)
    test_df[['history']].to_parquet(TEST_FILE)
    with open(ID_MAPS_FILE, 'wb') as f:
        pickle.dump(id_maps, f)
    print("预处理完成！")

if __name__ == '__main__':
    main()