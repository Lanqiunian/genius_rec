#!/bin/bash
# 运行验证脚本并保存输出到日志文件

# 确保在正确的目录
cd /root/autodl-tmp/genius_rec-main

# 添加执行权限
chmod +x scripts/verify_eval.py

# 创建日志目录
mkdir -p logs

# 处理命令行参数
IMAGE_EXPERT_WEIGHT="1.0"  # 默认使用完整的图像专家权重

# 检查命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --image-expert-weight=*)
      IMAGE_EXPERT_WEIGHT="${1#*=}"
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--image-expert-weight=值]"
      echo "  值范围从0.0到1.0，1.0表示使用原始权重，0.0表示完全忽略图像专家"
      exit 1
      ;;
  esac
done

# 运行验证脚本，将输出保存到日志文件
echo "开始验证评估方法... (图像专家权重因子: ${IMAGE_EXPERT_WEIGHT})"
LOG_FILE="logs/verification_$(date +%Y%m%d_%H%M%S)_image_weight_${IMAGE_EXPERT_WEIGHT}.log"
python scripts/verify_eval.py --image-expert-weight=${IMAGE_EXPERT_WEIGHT} 2>&1 | tee $LOG_FILE

echo "验证完成! 日志已保存至 $LOG_FILE"
