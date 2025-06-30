import pandas as pd
import os
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

# =============================================================================
# 辅助函数（下载、行处理、分块）
# =============================================================================

def process_chunk(chunk_data):
    """工作进程函数：处理传入的字节数据块。"""
    results = []
    for line in chunk_data.decode('utf-8').splitlines():
        if not line:
            continue
        try:
            item = json.loads(line.strip())
            asin = item.get('parent_asin')
            title = item.get('title')
            if not asin or not title:
                continue

            subtitle = item.get('subtitle', '')
            description_list = item.get('description', [])
            description_text = ' '.join(description_list) if isinstance(description_list, list) else ''
            features_list = item.get('features', [])
            features_text = ' '.join(features_list) if isinstance(features_list, list) else ''
            full_text = f"标题: {title} <SEP> 副标题: {subtitle} <SEP> 特点: {features_text} <SEP> 描述: {description_text}"

            image_url = None
            image_list = item.get('images', [])
            if image_list and isinstance(image_list, list):
                for img_info in image_list:
                    if isinstance(img_info, dict) and 'large' in img_info and img_info['large']:
                        image_url = img_info['large']
                        break
            
            results.append({
                'asin': asin,
                'full_text': ' '.join(full_text.split()),
                'image_url': image_url
            })
        except (json.JSONDecodeError, AttributeError):
            continue
    return results

def read_chunks(file_path):
    """智能地将大文件按行分割成适合并行处理的块。"""
    file_size = os.path.getsize(file_path)
    num_chunks = min(cpu_count() * 4, int(file_size / (1024 * 1024 * 64)))
    num_chunks = max(num_chunks, 1)
    chunk_size = file_size // num_chunks

    with open(file_path, 'rb') as f:
        start = 0
        while start < file_size:
            end = min(start + chunk_size, file_size)
            if end < file_size:
                f.seek(end)
                f.readline()
                end = f.tell()
            f.seek(start)
            chunk_data = f.read(end - start)
            yield chunk_data
            start = end
            if start >= file_size:
                break

def download_image_task(url, save_path, asin):
    """单个图片下载任务"""
    if not url or not isinstance(url, str) or not url.startswith('http'):
        return False, asin
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as handler:
            for chunk in response.iter_content(chunk_size=8192):
                handler.write(chunk)
        return True, asin
    except requests.RequestException:
        return False, asin

# =============================================================================
# 主逻辑函数
# =============================================================================

def create_full_text_parquet(metadata_path, output_path):
    """第一阶段：处理全量元数据，生成文本Parquet文件"""
    print("--- 阶段 1: 开始处理全量元数据 ---")
    if not os.path.exists(metadata_path):
        print(f"错误：找不到元数据文件 '{metadata_path}'。")
        print("请先执行 'gzip -d meta_Books.jsonl.gz' 命令解压文件。")
        return False
    
    tasks = list(read_chunks(metadata_path))
    all_results = []

    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=len(tasks), desc="多进程解析进度") as pbar:
            for result in pool.imap_unordered(process_chunk, tasks):
                all_results.extend(result)
                pbar.update(1)

    all_books_df = pd.DataFrame(all_results)
    all_books_df.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\n✅ 阶段 1 完成！共 {len(all_books_df)} 条记录已保存至 '{output_path}'")
    return True

def download_image_subset(text_parquet_path, image_dir, limit, max_workers):
    """第二阶段：从Parquet文件中读取数据，下载指定数量的图片"""
    print(f"\n--- 阶段 2: 开始下载指定数量 ({limit}张) 的图片 ---")
    if not os.path.exists(text_parquet_path):
        print(f"错误：找不到文本数据文件 '{text_parquet_path}'。请先完成阶段1。")
        return

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"已创建图片保存目录: {image_dir}")

    df = pd.read_parquet(text_parquet_path)
    
    # 筛选出需要下载的子集
    subset_df = df.head(limit).dropna(subset=['image_url'])

    download_tasks = [
        (row['image_url'], os.path.join(image_dir, f"{row['asin']}.jpg"), row['asin'])
        for _, row in subset_df.iterrows()
        if not os.path.exists(os.path.join(image_dir, f"{row['asin']}.jpg"))
    ]
    
    if not download_tasks:
        print("所有指定范围内的图片均已存在，无需下载。")
        print("✅ 阶段 2 完成！")
        return

    print(f"共需下载 {len(download_tasks)} / {len(subset_df)} 张新图片。")
    successful_downloads = 0
    failed_downloads_asin = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image_task, url, path, asin): asin for url, path, asin in download_tasks}
        progress = tqdm(as_completed(futures), total=len(futures), desc="图片下载进度")
        for future in progress:
            success, asin = future.result()
            if success:
                successful_downloads += 1
            else:
                failed_downloads_asin.append(asin)

    print("\n--------------------------------------------------")
    print("✅ 阶段 2 图片下载完成！")
    print(f"  - 成功下载: {successful_downloads} 张")
    print(f"  - 下载失败: {len(failed_downloads_asin)} 张")
    if failed_downloads_asin:
        print("  - 失败的ASIN列表 (前20个):", failed_downloads_asin[:20])
    print("--------------------------------------------------")

def main():
    # --- 全局参数设置 ---
    RAW_METADATA_PATH = 'data/meta_Books.jsonl'
    TEXT_PARQUET_PATH = 'book_text_data.parquet'
    IMAGE_OUTPUT_DIR = 'book_covers_enhanced'
    
    # !!! 在这里设置您想要下载的图片数量 !!!
    IMAGE_DOWNLOAD_LIMIT = 200000 
    
    MAX_DOWNLOAD_WORKERS = 50

    # --- 执行工作流 ---
    # 如果全量文本文件不存在，则创建它
    if not os.path.exists(TEXT_PARQUET_PATH):
        success = create_full_text_parquet(RAW_METADATA_PATH, TEXT_PARQUET_PATH)
        if not success:
            return # 如果第一步失败，则终止
    else:
        print(f"--- 阶段 1: 跳过 ---")
        print(f"已找到全量文本文件 '{TEXT_PARQUET_PATH}'。")


    # 执行第二步，下载指定数量的图片
    download_image_subset(TEXT_PARQUET_PATH, IMAGE_OUTPUT_DIR, IMAGE_DOWNLOAD_LIMIT, MAX_DOWNLOAD_WORKERS)
    
    print("\n所有任务已执行完毕。")

if __name__ == '__main__':
    main()