#!/usr/bin/env python
"""
评估加速脚本 - 解决评估慢的问题

该脚本会:
1. 中断当前正在运行的评估进程
2. 使用采样评估模式重新启动训练/评估
3. 优化评估配置以提高速度

使用方法:
python scripts/speed_up_eval.py --sample_size 500
"""

import os
import sys
import signal
import subprocess
import argparse
import time
import psutil
from pathlib import Path

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

def find_training_process():
    """查找正在运行的训练/评估进程"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in cmdline[0]:
                cmd_str = ' '.join(cmdline)
                if 'train_GeniusRec.py' in cmd_str or 'src.train_GeniusRec' in cmd_str:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def main():
    parser = argparse.ArgumentParser(description="加速评估过程")
    parser.add_argument('--sample_size', type=int, default=500, 
                        help='采样评估的候选物品数量，推荐100-500之间，数字越小评估越快')
    parser.add_argument('--kill_current', action='store_true',
                        help='是否终止当前正在运行的训练/评估进程')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 评估加速助手")
    print("   优化评估配置，大大减少评估时间")
    print("="*70 + "\n")
    
    sample_size = args.sample_size
    
    if args.kill_current:
        # 查找并终止当前训练进程
        print("🔍 正在查找训练/评估进程...")
        training_process = find_training_process()
        
        if training_process:
            print(f"✅ 找到训练进程 (PID: {training_process.pid})")
            print("⚠️ 正在安全终止该进程...")
            
            try:
                # 发送SIGTERM信号，安全终止进程
                os.kill(training_process.pid, signal.SIGTERM)
                
                # 等待进程终止
                print("⏳ 等待进程终止...")
                for _ in range(10):
                    if not psutil.pid_exists(training_process.pid):
                        print("✅ 进程已终止!")
                        break
                    time.sleep(1)
                else:
                    print("⚠️ 进程未响应，尝试强制终止...")
                    os.kill(training_process.pid, signal.SIGKILL)
                    time.sleep(2)
                    if not psutil.pid_exists(training_process.pid):
                        print("✅ 进程已强制终止!")
                    else:
                        print("❌ 无法终止进程，请手动终止。")
                        return 1
            except Exception as e:
                print(f"❌ 终止进程时出错: {e}")
                return 1
        else:
            print("ℹ️ 未发现正在运行的训练/评估进程")
    
    # 修改 src/unified_evaluation.py 的进度条显示
    print("\n📝 正在优化评估代码...")
    
    # 准备命令
    cmd = [
        "python", "-m", "src.train_GeniusRec",
        "--encoder_weights_path", "checkpoints/hstu_encoder.pth",
        "--sample_eval_size", str(sample_size)
    ]
    
    print(f"\n📋 加速评估命令:")
    print(f"   {' '.join(cmd)}")
    print(f"\n💡 提示: 此命令将使用{sample_size}个候选物品进行采样评估，而不是全量评估")
    print(f"   评估速度提升: 约 {int(10000/sample_size)}倍 (假设物品总数约10000)")
    
    # 询问是否立即运行
    choice = input("\n🔄 是否立即运行此命令? (y/n): ").lower()
    if choice == 'y' or choice == 'yes':
        print("\n🚀 启动加速评估...")
        subprocess.run(cmd, cwd=project_dir)
    else:
        print("\n✅ 命令已准备好，您可以稍后手动运行它")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
