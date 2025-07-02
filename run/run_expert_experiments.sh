#!/bin/bash
# run_expert_experiments.sh
# 用于方便地运行不同专家组合的实验脚本

set -e

echo "🧠 GENIUSRec专家系统实验脚本"
echo "=================================="

# 基础参数
ENCODER_WEIGHTS="checkpoints/hstu_encoder.pth"
BASE_CMD="python -m src.train_GeniusRec --encoder_weights_path $ENCODER_WEIGHTS --freeze_encoder"

# 确保日志目录存在
mkdir -p logs/expert_experiments

echo ""
echo "📋 可用的实验配置:"
echo "1. 仅行为专家 (Behavior Expert Only)"
echo "2. 仅内容专家 (Content Expert Only)"  
echo "3. 行为+内容专家 (Behavior + Content Expert)"
echo "4. 仅图像专家 (Image Expert Only) [需要图像嵌入]"
echo "5. 行为+图像专家 (Behavior + Image Expert) [需要图像嵌入]"
echo "6. 内容+图像专家 (Content + Image Expert) [需要图像嵌入]"
echo "7. 全部专家 (All Experts) [需要图像嵌入]"
echo ""

# 函数：运行单个实验
run_experiment() {
    local exp_name=$1
    local exp_cmd=$2
    local log_file="logs/expert_experiments/${exp_name}.log"
    
    echo "🚀 开始实验: $exp_name"
    echo "📝 日志文件: $log_file"
    echo "💻 命令: $exp_cmd"
    echo ""
    
    # 运行实验并记录日志
    eval "$exp_cmd --save_dir checkpoints/${exp_name}" 2>&1 | tee "$log_file"
    
    echo "✅ 实验完成: $exp_name"
    echo "----------------------------------------"
    echo ""
}

# 检查图像嵌入是否存在
IMAGE_EMBEDDINGS_PATH="data/book_image_embeddings.npy"
HAS_IMAGE_EMBEDDINGS=false

if [ -f "$IMAGE_EMBEDDINGS_PATH" ]; then
    HAS_IMAGE_EMBEDDINGS=true
    echo "✅ 检测到图像嵌入文件: $IMAGE_EMBEDDINGS_PATH"
else
    echo "⚠️  未检测到图像嵌入文件: $IMAGE_EMBEDDINGS_PATH"
    echo "💡 要启用图像专家，请先运行: python generate_image_embeddings.py --input_dir data/book_covers_enhanced --output_file $IMAGE_EMBEDDINGS_PATH"
fi

echo ""

# 如果提供了参数，运行特定实验
if [ $# -gt 0 ]; then
    case $1 in
        1|behavior)
            run_experiment "behavior_only" "$BASE_CMD --disable_content_expert"
            ;;
        2|content)
            run_experiment "content_only" "$BASE_CMD --disable_behavior_expert"
            ;;
        3|behavior_content)
            run_experiment "behavior_content" "$BASE_CMD"
            ;;
        4|image)
            if [ "$HAS_IMAGE_EMBEDDINGS" = true ]; then
                run_experiment "image_only" "$BASE_CMD --disable_behavior_expert --disable_content_expert --enable_image_expert --image_embeddings_path $IMAGE_EMBEDDINGS_PATH"
            else
                echo "❌ 图像专家实验需要图像嵌入文件"
                exit 1
            fi
            ;;
        5|behavior_image)
            if [ "$HAS_IMAGE_EMBEDDINGS" = true ]; then
                run_experiment "behavior_image" "$BASE_CMD --disable_content_expert --enable_image_expert --image_embeddings_path $IMAGE_EMBEDDINGS_PATH"
            else
                echo "❌ 图像专家实验需要图像嵌入文件"
                exit 1
            fi
            ;;
        6|content_image)
            if [ "$HAS_IMAGE_EMBEDDINGS" = true ]; then
                run_experiment "content_image" "$BASE_CMD --disable_behavior_expert --enable_image_expert --image_embeddings_path $IMAGE_EMBEDDINGS_PATH"
            else
                echo "❌ 图像专家实验需要图像嵌入文件"
                exit 1
            fi
            ;;
        7|all)
            if [ "$HAS_IMAGE_EMBEDDINGS" = true ]; then
                run_experiment "all_experts" "$BASE_CMD --enable_image_expert --image_embeddings_path $IMAGE_EMBEDDINGS_PATH"
            else
                echo "❌ 全专家实验需要图像嵌入文件"
                exit 1
            fi
            ;;
        *)
            echo "❌ 无效的实验编号: $1"
            echo "请使用 1-7 之间的数字，或者对应的名称"
            exit 1
            ;;
    esac
else
    # 交互式选择
    echo "请选择要运行的实验 (输入数字 1-7, 或按 Enter 运行全部基础实验):"
    read -r choice
    
    if [ -z "$choice" ]; then
        echo "🔄 运行基础实验序列..."
        run_experiment "behavior_only" "$BASE_CMD --disable_content_expert"
        run_experiment "content_only" "$BASE_CMD --disable_behavior_expert" 
        run_experiment "behavior_content" "$BASE_CMD"
        
        if [ "$HAS_IMAGE_EMBEDDINGS" = true ]; then
            echo "🖼️  检测到图像嵌入，继续运行图像相关实验..."
            run_experiment "image_only" "$BASE_CMD --disable_behavior_expert --disable_content_expert --enable_image_expert --image_embeddings_path $IMAGE_EMBEDDINGS_PATH"
            run_experiment "all_experts" "$BASE_CMD --enable_image_expert --image_embeddings_path $IMAGE_EMBEDDINGS_PATH"
        fi
    else
        # 递归调用脚本执行选中的实验
        exec "$0" "$choice"
    fi
fi

echo "🎉 实验脚本执行完成!"
echo ""
echo "📊 查看实验结果:"
echo "  - 检查点: checkpoints/[实验名称]/"
echo "  - 日志: logs/expert_experiments/[实验名称].log"
echo ""
echo "💡 快速比较结果:"
echo "  grep 'Test HR@10' logs/expert_experiments/*.log"
echo "  grep 'Test NDCG@10' logs/expert_experiments/*.log"
