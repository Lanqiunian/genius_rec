import requests
import json

# ===================================================================
# 1. 配置信息 (请确保这些是正确的)
# ===================================================================
ENDPOINT_URL = "http://1234294571044664.cn-shanghai.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20250630_mz2i"
HEADERS = {
    "Authorization": "Y2RlOTNiOWMxNWYxYjBhMzJjNWI0YjcxOGQ2ZmE0NDVlNjEwODJkZA==",
    "Content-Type": "application/json"
}

# ===================================================================
# 2. 准备一次性的请求数据
# ===================================================================
# 要发送的文本
text_to_embed = "这是一本关于人工智能的书"

# 根据我们确认的官方格式，构建请求体
request_body = {
    "input": {
        "contents": [{"text": text_to_embed}]
    }
}

# 将请求体转换为JSON字符串
request_data_json = json.dumps(request_body)


# ===================================================================
# 3. 打印所有请求信息，然后发送
# ===================================================================
print("--- 准备发送单次请求 ---")
print(f"请求地址 (URL): {ENDPOINT_URL}")
print("请求头 (Headers):")
print(json.dumps(HEADERS, indent=2))
print("请求体 (Body):")
print(json.dumps(request_body, indent=2, ensure_ascii=False))
print("--------------------------")

try:
    print("\n--- 正在发送请求... ---")
    response = requests.post(ENDPOINT_URL, headers=HEADERS, data=request_data_json, timeout=60) # 延长超时时间

    # ===================================================================
    # 4. 打印所有收到的响应信息，无论成功与否
    # ===================================================================
    print("\n--- 已收到响应 ---")
    print(f"响应状态码: {response.status_code}")
    
    print("\n响应头 (Response Headers):")
    print(json.dumps(dict(response.headers), indent=2))
    
    print("\n原始响应内容 (Raw Response Text):")
    print(response.text)
    
    print("\n--- 响应内容解析 ---")
    # 尝试将响应解析为JSON
    try:
        response_json = response.json()
        print("成功解析为JSON格式:")
        print(json.dumps(response_json, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("无法将响应解析为JSON格式。")

except requests.RequestException as e:
    print(f"\n!!!!!! 请求发生严重错误 !!!!!!")
    print(e)