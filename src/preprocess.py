# src/preprocess.py
import pickle
from pathlib import Path
from tqdm import tqdm
import gzip
import json
import logging
from collections import Counter, defaultdict

# --- 设置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置 ---
class Config:
    DATA_DIR = Path("./data")
    REVIEW_FILE = DATA_DIR / "Books.jsonl.gz"
    OUTPUT_DIR = DATA_DIR / "processed"
    PROCESSED_DATA_PKL = OUTPUT_DIR / "processed_data.pkl"
    K_CORE = 10 # K值可以根据需要调整

def stream_jsonl(filename: Path):
    """一个生成器，逐行读取并解析jsonl.gz文件"""
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def preprocess_data_efficient_kcore():
    """
    内存安全且K-core迭代高效的数据预处理流程。
    """
    cfg = Config()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("启动内存安全且高效的预处理流程...")

    # --- 1. 第一遍（唯一一次）扫描: 加载所有交互并获取初始计数 ---
    user_counts = Counter()
    item_counts = Counter()
    # interactions 存储所有有效的 (user_id_str, item_id_str) 对
    interactions = []
    
    logging.info(f"唯一一次全量扫描: 正在从 {cfg.REVIEW_FILE} 加载交互数据...")
    for review in tqdm(stream_jsonl(cfg.REVIEW_FILE), desc="全量扫描"):
        uid = review.get('user_id')
        iid = review.get('parent_asin')
        if uid and iid:
            user_counts[uid] += 1
            item_counts[iid] += 1
            interactions.append((uid, iid))
    
    logging.info(f"初始数据加载完成。用户数: {len(user_counts)}, 物品数: {len(item_counts)}, 总交互数: {len(interactions)}")

    # --- 2. 在内存中高效进行K-core迭代 ---
    logging.info(f"开始在内存中进行高效的 {cfg.K_CORE}-core 迭代过滤...")
    while True:
        # a. 找出不满足条件的user和item
        inactive_users = {u for u, c in user_counts.items() if c < cfg.K_CORE}
        inactive_items = {i for i, c in item_counts.items() if c < cfg.K_CORE}

        # 如果没有需要移除的，说明已经收敛，退出循环
        if not inactive_users and not inactive_items:
            logging.info("K-core已收敛。")
            break
        
        logging.info(f"本轮迭代: 正在移除 {len(inactive_users)} 个非活跃用户和 {len(inactive_items)} 个非活跃物品...")

        # b. 过滤交互列表，只保留活跃用户和活跃物品的交互
        # 同时更新下一轮的计数器
        next_interactions = []
        next_user_counts = Counter()
        next_item_counts = Counter()

        for uid, iid in interactions:
            if uid not in inactive_users and iid not in inactive_items:
                next_interactions.append((uid, iid))
                next_user_counts[uid] += 1
                next_item_counts[iid] += 1
        
        interactions = next_interactions
        user_counts = next_user_counts
        item_counts = next_item_counts
        
        logging.info(f"迭代后剩余: 用户数: {len(user_counts)}, 物品数: {len(item_counts)}, 交互数: {len(interactions)}")

    active_users = set(user_counts.keys())
    active_items = set(item_counts.keys())
    logging.info(f"K-core 过滤完成。")

    # --- 3. 构建最终序列 (基于过滤后的交互) ---
    logging.info("正在加载有效交互数据并构建最终序列...")
    user_interactions_dict = defaultdict(list)
    # 我们需要原始的时间戳来进行排序，所以需要再次扫描文件，但这次只处理活跃的交互
    for review in tqdm(stream_jsonl(cfg.REVIEW_FILE), desc="加载时间戳"):
        uid = review.get('user_id')
        iid = review.get('parent_asin')
        timestamp = review.get('timestamp')
        if uid in active_users and iid in active_items:
            user_interactions_dict[uid].append({'item_id': iid, 'timestamp': timestamp})
    
    # --- 4. 创建ID映射和最终序列 ---
    logging.info("正在创建ID映射并构建最终序列...")
    user2id = {uid: i + 1 for i, uid in enumerate(active_users)}
    item2id = {iid: i + 1 for i, iid in enumerate(active_items)}
    id2user = {i: uid for uid, i in user2id.items()}
    id2item = {i: iid for iid, i in item2id.items()}
    
    user_sequences_list = []
    for user_id_str, interactions_list in tqdm(user_interactions_dict.items(), desc="构建最终序列"):
        sorted_interactions = sorted(interactions_list, key=lambda x: x['timestamp'])
        sequence = [item2id[interaction['item_id']] for interaction in sorted_interactions]
        user_sequences_list.append({
            'user_id': user2id[user_id_str],
            'sequence': sequence
        })

    # --- 5. 保存处理好的数据 ---
    logging.info(f"正在将处理好的数据保存到 {cfg.PROCESSED_DATA_PKL}...")
    data_to_save = {
        'user_sequences': user_sequences_list,
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        'num_users': len(user2id),
        'num_items': len(item2id)
    }
    with open(cfg.PROCESSED_DATA_PKL, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    logging.info("数据预处理全部完成！")

if __name__ == '__main__':
    preprocess_data_efficient_kcore()