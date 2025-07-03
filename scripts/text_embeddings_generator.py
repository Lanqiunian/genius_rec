# generate_book_embeddings_final_production.py

import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import time
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading

# ===================================================================
# 1. 配置
# ===================================================================

# 【注意】建议使用环境变量来管理您的密钥以提高安全性。
GOOGLE_API_KEY = "AIzaSyDSR_G4ESL3CtkVcmxjUwT5QpS_wCvKO_c"

# --- 模型和并发配置 ---
EMBEDDING_MODEL = "models/text-embedding-004"
# Gemini API通常有速率限制（如每分钟150次），并发数不宜过高，15-30是比较安全的选择
MAX_CONCURRENT_REQUESTS = 5
# Gemini API支持的批量大小，最大为100
BATCH_SIZE = 100
# 为防止意外错误，对输入文本进行字符截断
MAX_CHARS_PER_INPUT = 2000 

# --- 文件路径配置 ---
INPUT_PARQUET_FILE = 'data/book_text_data.parquet'
MISSING_ASINS_FILE = 'missing_text_embeddings.txt'  # 缺失嵌入的ASIN列表
EXISTING_EMBEDDINGS_FILE = 'data/book_gemini_embeddings_filtered_migrated.npy'  # 现有嵌入文件
STREAMING_OUTPUT_FILE = 'data/missing_embeddings_progress.jsonl'  # 新生成嵌入的进度文件
FINAL_NUMPY_FILE = 'data/book_gemini_embeddings_filtered_migrated.npy'  # 更新后的最终文件

# ===================================================================
# 2. 【核心优化】客户端速率限制器
# ===================================================================
class RateLimiter:
    """
    一个简单的令牌桶速率限制器，用于控制API请求频率。
    """
    def __init__(self, requests_per_minute):
        # API配额是每分钟150次，我们设置一个安全值
        self.requests_per_minute = requests_per_minute 
        self.refill_rate_per_second = self.requests_per_minute / 60.0
        # 使用信号量作为令牌桶，初始令牌为最大值
        self._semaphore = threading.Semaphore(self.requests_per_minute)
        self.running = True
        
        # 启动一个后台线程来定期补充令牌
        self.refill_thread = threading.Thread(target=self._refill, daemon=True)
        self.refill_thread.start()

    def _refill(self):
        """后台补充令牌的函数"""
        while self.running:
            try:
                self._semaphore.release()
            except ValueError:
                pass # 信号量已满时会报错，忽略即可
            time.sleep(1 / self.refill_rate_per_second)
            
    def acquire(self):
        """获取一个令牌，如果桶空了则阻塞等待。"""
        self._semaphore.acquire()

    def stop(self):
        """停止补充令牌的后台线程"""
        self.running = False

# ===================================================================
# 3. 核心函数
# ===================================================================

def test_api_connection():
    """发送一个简单的请求来测试API密钥和服务的连通性。"""
    print("--- 正在执行API连通性预检测试 ---")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content="API connection test",
            task_type="RETRIEVAL_DOCUMENT"
        )
        if 'embedding' in result and len(result['embedding']) > 0:
            print("✅ API预检测试成功！服务连接正常。")
            return True
        else:
            print(f"❌ API预检测试失败: 服务返回了空的结果。返回内容: {result}")
            return False
    except Exception as e:
        print(f"❌ API预检测试失败: 发生严重错误。")
        print(f"   - 错误详情: {e}")
        print("   - 请检查您的API密钥是否正确、账户是否有效以及网络连接（包括代理设置）。")
        return False

def load_processed_asins(filepath):
    """读取已完成的进度文件，返回一个包含所有已处理ASIN的集合。"""
    if not os.path.exists(filepath):
        return set()
    print(f"检测到已存在的进度文件 '{filepath}', 正在加载已完成的记录...")
    processed_asins = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                processed_asins.add(json.loads(line)['asin'])
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"已加载 {len(processed_asins)} 条已处理的记录。")
    return processed_asins

def load_missing_asins(filepath):
    """从文本文件中加载缺失的ASIN列表。"""
    if not os.path.exists(filepath):
        print(f"警告: 未找到缺失ASIN文件 '{filepath}'")
        return set()
    
    print(f"正在加载缺失的ASIN列表: '{filepath}'")
    missing_asins = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            asin = line.strip()
            if asin:
                missing_asins.add(asin)
    
    print(f"已加载 {len(missing_asins)} 个缺失的ASIN")
    return missing_asins

def process_batch(batch_data, rate_limiter):
    """
    处理单个批次的函数，会先从速率限制器获取令牌。
    """
    batch_asins, batch_texts = batch_data
    
    # 在发送请求前，先从速率限制器获取一个“许可”
    rate_limiter.acquire()
    
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch_texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return batch_asins, result['embedding']
    except Exception as e:
        # 即使有速率限制器，也保留错误处理以应对突发网络问题等
        return batch_asins, f"Error: {str(e)}"

# ===================================================================
# 4. 主逻辑
# ===================================================================
def main():
    # --- 步骤 1: 配置并执行API预检 ---
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"API密钥配置环节出错: {e}")
        return

    if not test_api_connection():
        print("\n程序因API预检失败而终止。")
        return
        
    print("\n--- 开始执行缺失嵌入补充任务 ---")

    # --- 步骤 2: 加载数据和缺失ASIN列表 ---
    try:
        books_df = pd.read_parquet(INPUT_PARQUET_FILE)
        print(f"成功加载 {len(books_df)} 条书籍数据。")
    except FileNotFoundError:
        print(f"错误: 未找到 '{INPUT_PARQUET_FILE}'。请先运行数据增强脚本。")
        return

    # 加载缺失的ASIN列表
    missing_asins = load_missing_asins(MISSING_ASINS_FILE)
    if not missing_asins:
        print("没有缺失的ASIN需要处理。")
        return

    # 加载已处理的ASIN（从进度文件）
    processed_asins = load_processed_asins(STREAMING_OUTPUT_FILE)
    
    # --- 步骤 3: 准备任务批次（只处理缺失的ASIN） ---
    print("正在准备需要处理的缺失ASIN任务...")
    books_df.dropna(subset=['full_text', 'asin'], inplace=True)
    books_df = books_df[books_df['full_text'].str.strip() != '']
    
    # 筛选出缺失的且未处理的ASIN
    target_asins = missing_asins - processed_asins
    tasks_df = books_df[books_df['asin'].isin(target_asins)]
    
    if tasks_df.empty:
        print("所有缺失的ASIN均已处理完毕或无对应文本数据！")
    else:
        text_list = [text[:MAX_CHARS_PER_INPUT] for text in tasks_df['full_text']]
        asin_list = tasks_df['asin'].tolist()
        
        batches = []
        for i in range(0, len(text_list), BATCH_SIZE):
            batch_texts = text_list[i:i + BATCH_SIZE]
            batch_asins = asin_list[i:i + BATCH_SIZE]
            batches.append((batch_asins, batch_texts))
            
        print(f"共需处理 {len(text_list)} 条缺失记录，分为 {len(batches)} 个批次。")

        # --- 步骤 4: 初始化速率限制器并执行并发任务 ---
        rate_limiter = RateLimiter(requests_per_minute=140) # 设置为比150略低的安全值
        
        try:
            with open(STREAMING_OUTPUT_FILE, 'a', encoding='utf-8') as f:
                with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                    future_to_batch = {executor.submit(process_batch, batch, rate_limiter): batch for batch in batches}
                    
                    progress = tqdm(as_completed(future_to_batch), total=len(batches), desc="处理缺失嵌入批次")
                    for future in progress:
                        batch_asins, result = future.result()
                        if isinstance(result, list) and len(batch_asins) == len(result):
                            for asin, embedding in zip(batch_asins, result):
                                output_record = {"asin": asin, "embedding": embedding}
                                f.write(json.dumps(output_record) + '\n')
                            f.flush()
                        else:
                            print(f"\n处理一个批次时发生错误: {result}")
        finally:
            # 确保无论如何都停止速率限制器的后台线程
            rate_limiter.stop()

    # --- 步骤 5: 合并现有嵌入和新生成的嵌入 ---
    print("\n正在合并现有嵌入和新生成的嵌入...")
    
    # 加载现有嵌入
    existing_embeddings = {}
    if os.path.exists(EXISTING_EMBEDDINGS_FILE):
        existing_embeddings = np.load(EXISTING_EMBEDDINGS_FILE, allow_pickle=True).item()
        print(f"已加载 {len(existing_embeddings)} 个现有嵌入")
    
    # 加载新生成的嵌入
    new_embeddings = {}
    if os.path.exists(STREAMING_OUTPUT_FILE):
        with open(STREAMING_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    new_embeddings[record['asin']] = np.array(record['embedding'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"已加载 {len(new_embeddings)} 个新生成的嵌入")

    # 合并嵌入（新生成的会覆盖现有的）
    final_embeddings = existing_embeddings.copy()
    final_embeddings.update(new_embeddings)

    # 保存合并后的最终文件
    np.save(FINAL_NUMPY_FILE, final_embeddings)
    print(f"\n✅ 嵌入补充完成！")
    print(f"   - 现有嵌入: {len(existing_embeddings)}")
    print(f"   - 新增嵌入: {len(new_embeddings)}")
    print(f"   - 最终总数: {len(final_embeddings)}")
    print(f"   - 已保存至: '{FINAL_NUMPY_FILE}'")

    # 清理进度文件（可选）
    if os.path.exists(STREAMING_OUTPUT_FILE):
        os.remove(STREAMING_OUTPUT_FILE)
        print(f"已清理临时进度文件: '{STREAMING_OUTPUT_FILE}'")


if __name__ == "__main__":
    main()