#!/bin/bash
# 运行验证脚本

# 确保在正确的目录
cd /root/autodl-tmp/genius_rec-main

# 添加执行权限
chmod +x scripts/verify_eval.py

# 创建日志目录
mkdir -p logs

# 获取当前时间作为日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/verification_${TIMESTAMP}.log"

echo "开始验证评估方法..."
echo "日志文件: $LOG_FILE"

# 运行验证脚本并保存输出到日志文件
python scripts/verify_eval.py 2>&1 | tee "$LOG_FILE"

echo "验证完成!"
echo "查看完整日志: $LOG_FILE"
