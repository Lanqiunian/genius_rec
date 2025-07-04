#!/usr/bin/env python
"""
加速评估脚本 - 用于快速评估模型性能

该脚本修改默认的评估配置，通过以下方式加速评估过程：
1. 使用采样评估模式而非全量评估
2. 可配置候选物品数量，默认为500（可调整）

使用方法:
python scripts/fast_eval.py --sample_size 500  # 使用500个候选物品
python scripts/fast_eval.py --sample_size 1000  # 使用1000个候选物品
"""

import os
import sys
import argparse
import logging
import torch
import subprocess
from pathlib import Path
import time

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GENIUS-Rec 快速评估脚本")
    parser.add_argument('--sample_size', type=int, default=500, help='采样评估的候选物品数量')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/genius_rec_best.pth", 
                        help='要评估的检查点路径')
    parser.add_argument('--encoder_weights_path', type=str, default="checkpoints/hstu_encoder.pth",
                        help='编码器权重路径')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    sample_size = args.sample_size
    
    logger.info("=" * 50)
    logger.info(f"🚀 启动加速评估 - 使用{sample_size}个候选物品")
    logger.info("=" * 50)
    
    # 构建评估命令
    cmd = [
        "python", "-m", "src.train_GeniusRec",
        "--encoder_weights_path", args.encoder_weights_path,
        "--resume_from", args.checkpoint_path,
        "--sample_eval_size", str(sample_size),
        # 添加其他需要的参数
    ]
    
    logger.info(f"📋 执行命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行命令
    logger.info("⏱️ 开始评估...")
    process = subprocess.Popen(
        cmd,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # 实时显示输出
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
        if not line:
            break
    
    # 等待进程完成
    process.wait()
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    logger.info(f"✅ 评估完成！耗时: {elapsed_time:.2f}秒")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
