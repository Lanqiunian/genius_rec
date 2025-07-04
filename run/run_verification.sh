#!/bin/bash
# 运行验证脚本

# 确保在正确的目录
cd /root/autodl-tmp/genius_rec-main

# 添加执行权限
chmod +x scripts/verify_eval.py

# 运行验证脚本
echo "开始验证评估方法..."
python scripts/verify_eval.py

echo "验证完成!"
