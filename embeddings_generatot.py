# generate_book_embeddings_eas.py

import pandas as pd
import requests
import json
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================================================================
# 1. 请在这里配置您的服务信息
# ===================================================================

# 您的服务访问地址
ENDPOINT_URL = ""

# 您的Header信息
HEADERS = {
    "Authorization": "",
    "Content-Type": "application/json"
}

# --- 其他配置 ---
# 并发数，可以根据您的服务规格调整，例如50
MAX_CONCURRENT_REQUESTS = 50 
# 为了防止文本过长导致API错误，设置一个保守的字符上限
# GME-2B模型通常有较大的上下文窗口，但设置上限是好习惯
MAX_CHARS_PER_INPUT = 2000 


# ===================================================================
# 2. 加载数据
# ===================================================================

input_parquet_file = 'book_text_data.parquet'
try:
    books_df = pd.read_parquet(input_parquet_file)
    print(f"成功加载 {len(books_df)} 条书籍数据。")
except FileNotFoundError:
    print(f"错误: 未找到 '{input_parquet_file}'。请先运行数据增强脚本。")
    exit()

# ===================================================================
# 3. 准备文本、截断并分批
# ===================================================================

print("正在准备用于嵌入的文本...")
books_df.dropna(subset=['full_text'], inplace=True)
books_df = books_df[books_df['full_text'].str.strip() != '']

# 截断超长文本
text_list = [text[:MAX_CHARS_PER_INPUT] for text in books_df['full_text']]

# GME模型通常支持批量处理，但为了简单和稳定，我们先按单条文本一个请求来并发
# 将文本列表和索引一起打包
tasks_with_indices = list(enumerate(text_list))

print(f"文本已准备完成，共 {len(text_list)} 条有效文本 (超长部分已截断)，将发起 {len(tasks_with_indices)} 个并发请求。")

# ===================================================================
# 4. 使用线程池并发调用API
# ===================================================================

all_embeddings_map = {} # 使用字典存储结果 {索引: embedding}
print(f"开始使用阿里云EAS服务并发生成嵌入 (并发数: {MAX_CONCURRENT_REQUESTS})...")

def process_single_text(task_with_index):
    """处理单条文本的函数，用于多线程调用"""
    index, text = task_with_index
    
    # 构建请求体
    request_body = {
        "input": {
            "contents": [{"text": text}]
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=HEADERS, data=json.dumps(request_body), timeout=20)
        response.raise_for_status() # 如果状态码不是2xx，则抛出异常
        
        response_json = response.json()
        if response_json.get('status_code') == 200:
            embeddings = response_json.get('output', {}).get('embeddings', [])
            if embeddings:
                # 假设每个文本只返回一个嵌入向量
                return index, embeddings[0].get('embedding')
        
        # 如果响应中没有找到有效的嵌入
        return index, f"Error: No embedding in response - {response_json.get('message', 'Unknown error')}"

    except Exception as e:
        return index, f"Error: {str(e)}"

with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
    future_to_task = {executor.submit(process_single_text, task): task for task in tasks_with_indices}
    
    progress = tqdm(as_completed(future_to_task), total=len(tasks_with_indices), desc="并发生成嵌入")
    for future in progress:
        index, result = future.result()
        if isinstance(result, list): # 检查结果是否是列表（即成功的嵌入）
            all_embeddings_map[index] = result
        else:
            # 记录失败（但不打印，以免刷屏），后续可以分析
            all_embeddings_map[index] = None


# ===================================================================
# 5. 整合结果并保存
# ===================================================================

print("\n正在整合所有嵌入结果...")
# 按照原始索引排序，确保顺序正确
sorted_embeddings_list = [all_embeddings_map.get(i) for i in range(len(text_list))]

if len(sorted_embeddings_list) == len(books_df):
    books_df['embedding'] = sorted_embeddings_list
    
    successful_embeddings_df = books_df.dropna(subset=['embedding'])
    
    # 创建 asin -> embedding 的映射字典
    book_asin_to_embedding = {
        row['asin']: np.array(row['embedding']) 
        for _, row in successful_embeddings_df.iterrows()
    }
    
    # 保存为numpy文件
    output_file_path = 'book_eas_gme_embeddings.npy'
    np.save(output_file_path, book_asin_to_embedding)
    
    print(f"\n处理完成！成功生成 {len(book_asin_to_embedding)} / {len(books_df)} 个嵌入。")
    print(f"已将 ASIN -> Embedding 的字典保存至: {output_file_path}")
else:
    print(f"\n错误：生成的嵌入数量与书籍数量不匹配。数据未保存。")