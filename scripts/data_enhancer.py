import pandas as pd
import os
import json
import pickle
import requests
import time
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

def download_image_task(url, save_path, asin, max_retries=2):
    """单个图片下载任务，带重试机制"""
    if not url or not isinstance(url, str) or not url.startswith('http'):
        return False, asin
    
    for attempt in range(max_retries):
        try:
            # 设置请求头模拟浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }
            
            response = requests.get(url, timeout=15, stream=True, headers=headers)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png']):
                continue  # 尝试下次重试
            
            with open(save_path, 'wb') as handler:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handler.write(chunk)
            
            # 验证文件大小
            if os.path.getsize(save_path) < 1024:
                os.remove(save_path)
                continue  # 尝试下次重试
            
            return True, asin
            
        except requests.RequestException:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # 递增延迟
                continue
    
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

def download_filtered_images(text_parquet_path, image_dir, id_maps_path, max_workers):
    """第二阶段：基于5-core过滤后的物品子集下载图片"""
    print(f"\n--- 阶段 2: 开始下载5-core过滤后子集的图片 ---")
    
    # 检查必要文件
    if not os.path.exists(text_parquet_path):
        print(f"错误：找不到文本数据文件 '{text_parquet_path}'。请先完成阶段1。")
        return
    
    if not os.path.exists(id_maps_path):
        print(f"错误：找不到ID映射文件 '{id_maps_path}'。")
        return

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"已创建图片保存目录: {image_dir}")

    # 加载5-core过滤后的物品列表
    print("正在加载5-core过滤后的物品映射...")
    with open(id_maps_path, 'rb') as f:
        id_maps = pickle.load(f)
    
    valid_asins = set(id_maps['item_map'].keys())
    print(f"5-core过滤后有效物品数量: {len(valid_asins)}")

    # 加载全量文本数据并过滤
    print("正在加载和过滤文本数据...")
    df = pd.read_parquet(text_parquet_path)
    
    # 只保留在5-core子集中的物品
    filtered_df = df[df['asin'].isin(valid_asins)].dropna(subset=['image_url'])
    print(f"过滤后有图片URL的物品数量: {len(filtered_df)}")

    # 生成下载任务列表，文件名使用ASIN
    download_tasks = []
    asin_to_itemid = {}  # 记录ASIN到item_id的映射
    
    for _, row in filtered_df.iterrows():
        asin = row['asin']
        item_id = id_maps['item_map'][asin]
        image_path = os.path.join(image_dir, f"{asin}.jpg")
        
        # 如果图片还不存在，加入下载任务
        if not os.path.exists(image_path):
            download_tasks.append((row['image_url'], image_path, asin))
        
        # 记录映射关系
        asin_to_itemid[asin] = item_id
    
    # 保存ASIN到item_id的映射文件，供后续embedding生成使用
    mapping_file = os.path.join(image_dir, 'asin_to_itemid_mapping.pkl')
    with open(mapping_file, 'wb') as f:
        pickle.dump(asin_to_itemid, f)
    print(f"已保存ASIN到item_id映射文件: {mapping_file}")
    
    if not download_tasks:
        print("所有5-core子集图片均已存在，无需下载。")
        print("✅ 阶段 2 完成！")
        return

    print(f"共需下载 {len(download_tasks)} / {len(filtered_df)} 张新图片。")
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
    print(f"  - 5-core有效物品: {len(valid_asins)}")
    print(f"  - 有图片URL的物品: {len(filtered_df)}")
    print(f"  - 成功下载: {successful_downloads} 张")
    print(f"  - 下载失败: {len(failed_downloads_asin)} 张")
    print(f"  - 图片保存位置: {image_dir}")
    print(f"  - 映射文件: {mapping_file}")
    if failed_downloads_asin:
        print("  - 失败的ASIN列表 (前20个):", failed_downloads_asin[:20])
    print("--------------------------------------------------")

def main():
    # --- 全局参数设置 ---
    RAW_METADATA_PATH = 'data/meta_Books.jsonl'
    TEXT_PARQUET_PATH = 'data/book_text_data.parquet'
    IMAGE_OUTPUT_DIR = 'data/book_covers_enhanced'
    ID_MAPS_PATH = 'data/processed/id_maps.pkl'  # 5-core过滤后的物品映射
    
    MAX_DOWNLOAD_WORKERS = 50

    print("=== 基于5-core过滤子集的图片下载器 ===")
    print(f"输入元数据: {RAW_METADATA_PATH}")
    print(f"文本数据: {TEXT_PARQUET_PATH}")
    print(f"ID映射文件: {ID_MAPS_PATH}")
    print(f"图片保存目录: {IMAGE_OUTPUT_DIR}")
    print("=" * 50)

    # --- 执行工作流 ---
    # 如果全量文本文件不存在，则创建它
    if not os.path.exists(TEXT_PARQUET_PATH):
        success = create_full_text_parquet(RAW_METADATA_PATH, TEXT_PARQUET_PATH)
        if not success:
            return # 如果第一步失败，则终止
    else:
        print(f"--- 阶段 1: 跳过 ---")
        print(f"已找到全量文本文件 '{TEXT_PARQUET_PATH}'。")

    # 执行第二步，下载5-core过滤后的图片
    download_filtered_images(TEXT_PARQUET_PATH, IMAGE_OUTPUT_DIR, ID_MAPS_PATH, MAX_DOWNLOAD_WORKERS)
    
    print("\n🎉 所有任务已执行完毕！")
    print("\n📋 生成的文件:")
    print(f"  - 图片目录: {IMAGE_OUTPUT_DIR}/*.jpg")
    print(f"  - 映射文件: {IMAGE_OUTPUT_DIR}/asin_to_itemid_mapping.pkl")
    print("\n💡 下一步:")
    print("  可以使用generate_image_embeddings.py生成图像嵌入")
    print(f"  python generate_image_embeddings.py --input_dir {IMAGE_OUTPUT_DIR} --output_file data/book_image_embeddings.npy")

if __name__ == '__main__':
    main()