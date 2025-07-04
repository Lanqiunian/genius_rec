#!/bin/bash
# 图像专家权重对比测试脚本

# 创建日志目录
mkdir -p logs/verification

# 运行基准测试 - 使用原始权重
echo "===== 运行基准测试 - 使用原始专家权重 (image_expert_weight=1.0) ====="
python scripts/verify_eval.py --image-expert-weight 1.0 2>&1 | tee logs/verification/verify_weight_1.0.log

# 测试0.5权重 - 图像专家权重降低一半
echo "===== 测试图像专家权重降低一半 (image_expert_weight=0.5) ====="
python scripts/verify_eval.py --image-expert-weight 0.5 2>&1 | tee logs/verification/verify_weight_0.5.log

# 测试0权重 - 完全忽略图像专家
echo "===== 测试完全忽略图像专家 (image_expert_weight=0.0) ====="
python scripts/verify_eval.py --image-expert-weight 0.0 2>&1 | tee logs/verification/verify_weight_0.0.log

# 测试1.5权重 - 增加图像专家权重
echo "===== 测试增加图像专家权重 (image_expert_weight=1.5) ====="
python scripts/verify_eval.py --image-expert-weight 1.5 2>&1 | tee logs/verification/verify_weight_1.5.log

# 对比结果
echo "===== 对比不同权重下的评估结果 ====="
echo "原始权重 (1.0):"
grep "HR@10\|NDCG@10" logs/verification/verify_weight_1.0.log | grep -v "编码器嵌入方法" | tail -3

echo "降低一半权重 (0.5):"
grep "HR@10\|NDCG@10" logs/verification/verify_weight_0.5.log | grep -v "编码器嵌入方法" | tail -3

echo "忽略图像专家 (0.0):"
grep "HR@10\|NDCG@10" logs/verification/verify_weight_0.0.log | grep -v "编码器嵌入方法" | tail -3

echo "增加图像专家权重 (1.5):"
grep "HR@10\|NDCG@10" logs/verification/verify_weight_1.5.log | grep -v "编码器嵌入方法" | tail -3
