import requests
import json
import base64
import os

# ===================================================================
# 1. 请在这里配置您的服务信息
# ===================================================================
# 您的服务访问地址
ENDPOINT_URL = "http://1234294571044664.cn-shanghai.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20250630_mz2i"

# 您的Header信息
HEADERS = {
    "Authorization": "Y2RlOTNiOWMxNWYxYjBhMzJjNWI0YjcxOGQ2ZmE0NDVlNjEwODJkZA==",
    "Content-Type": "application/json"
}

# ===================================================================
# 2. 定义测试函数
# ===================================================================

def call_embedding_service(input_contents):
    """
    一个通用的函数，用于调用您的EAS嵌入服务。
    
    Args:
        input_contents (list): 一个包含输入块（如{"text": "..."}）的列表。
    """
    # 构建完全符合官方示例的请求体
    request_body = {
        "input": {
            "contents": input_contents
        }
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=HEADERS, data=json.dumps(request_body), timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        print("响应内容:")
        try:
            # 格式化打印JSON响应
            response_json = response.json()
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            # 检查并打印关键信息
            if response_json.get('status_code') == 200 and response_json.get('output'):
                embeddings = response_json['output'].get('embeddings', [])
                print(f"\n✅ 分析: 成功获取到 {len(embeddings)} 个嵌入向量。")
                if embeddings:
                    print(f"   - 第一个嵌入向量的维度: {len(embeddings[0].get('embedding', []))}")
        except json.JSONDecodeError:
            print(response.text)
            
    except requests.RequestException as e:
        print(f"请求发生错误: {e}")

def encode_image_to_data_uri(image_path):
    """将图片文件编码为包含Data URI scheme的Base64字符串"""
    if not os.path.exists(image_path):
        print(f"错误: 图片文件 '{image_path}' 不存在。")
        return None
        
    # 获取图片格式
    image_format = os.path.splitext(image_path)[1].lower().replace('.', '')
    if image_format == 'jpg':
        image_format = 'jpeg' # 标准MIME类型是jpeg
        
    with open(image_path, "rb") as image_file:
        base64_encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:image/{image_format};base64,{base64_encoded_string}"


# ===================================================================
# 3. 执行测试
# ===================================================================
if __name__ == "__main__":
    # --- 测试1: 纯文本嵌入 (最符合您当前需求的场景) ---
    print("--- 开始测试 1: 纯文本嵌入 ---")
    text_input = [
        {"text": "一只可爱的小猫在睡觉"}
    ]
    call_embedding_service(text_input)

    # --- 测试2: 纯图片嵌入 ---
    print("\n\n--- 开始测试 2: 纯图片嵌入 ---")
    image_path = "test_image.jpg" # <--- 请确保这个图片文件存在
    image_data_uri = encode_image_to_data_uri(image_path)
    if image_data_uri:
        image_input = [
            {"image": image_data_uri}
        ]
        call_embedding_service(image_input)
    
    # --- 测试3: 图文联合嵌入 ---
    print("\n\n--- 开始测试 3: 图文联合嵌入 ---")
    if image_data_uri:
        multimodal_input = [
            {"image": image_data_uri},
            {"text": "这是一张测试图片"}
        ]
        call_embedding_service(multimodal_input)