# preprocess.py (重构版)

import pandas as pd
import gzip
from tqdm import tqdm
import pickle
import json
from pathlib import Path
from src.config import get_config

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
    config = get_config()
    
    # 使用配置中的路径
    DATA_DIR = config['data']['data_dir']
    PROCESSED_DATA_DIR = config['data']['processed_data_dir']
    
    # 输入文件
    REVIEW_FILE = DATA_DIR / "Books.jsonl.gz" 
    
    # 输出文件
    TRAIN_FILE = config['data']['train_file']
    VALIDATION_FILE = config['data']['validation_file']
    TEST_FILE = config['data']['test_file']
    ID_MAPS_FILE = config['data']['id_maps_file']
    
    # 配置参数
    K_CORE = config['k_core']
    MIN_SEQ_LEN = config['min_seq_len']
    SEED = config['seed']
    
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
    
    # 获取特殊标记配置
    pad_token_id = config['pad_token_id']
    sos_token_id = config['sos_token_id'] 
    eos_token_id = config['eos_token_id']
    mask_token_id = config['mask_token_id']
    
    # 计算特殊标记的数量，为其预留ID空间
    special_token_ids = [pad_token_id, sos_token_id, eos_token_id, mask_token_id]
    num_special_tokens = max(special_token_ids) + 1
    
    # 🔧 安全检查：确保特殊token ID连续且从0开始
    expected_special_ids = list(range(len(special_token_ids)))
    if sorted(special_token_ids) != expected_special_ids:
        raise ValueError(f"特殊token ID必须连续且从0开始，当前: {sorted(special_token_ids)}, 期望: {expected_special_ids}")
    
    # 【关键修正】重映射后的ID从num_special_tokens开始，为特殊标记留出空间
    ratings['user_id_remap'] = pd.Categorical(ratings['user_id']).codes + num_special_tokens
    ratings['item_id_remap'] = pd.Categorical(ratings['item_id']).codes + num_special_tokens
    
    # 【关键修正】保存新的映射关系
    user_map = {original: remap_id for original, remap_id in zip(ratings['user_id'], ratings['user_id_remap'])}
    item_map = {original: remap_id for original, remap_id in zip(ratings['item_id'], ratings['item_id_remap'])}
    
    id_maps = {
        'user_map': user_map,
        'item_map': item_map,
        'num_users': len(user_map),
        'num_items': len(item_map),
        'num_special_tokens': num_special_tokens,
        'special_tokens': {
            'pad_token_id': pad_token_id,
            'sos_token_id': sos_token_id,
            'eos_token_id': eos_token_id,
            'mask_token_id': mask_token_id
        }
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