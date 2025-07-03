#!/usr/bin/env python3
"""
GENIUS-Rec 实验启动器
====================

这是一个友好的实验启动脚本，帮助你选择合适的实验模式。

使用方法:
    python start_experiments.py

或者直接指定实验类型:
    python start_experiments.py --mode quick     # 快速验证（推荐开始）
    python start_experiments.py --mode expert    # 专家系统消融
    python start_experiments.py --mode full      # 完整实验套件
    python start_experiments.py --mode baseline  # 基线对比
"""

import argparse
import subprocess
import sys
from pathlib import Path

def print_banner():
    """打印实验横幅"""
    print("\n" + "="*70)
    print("🎯 GENIUS-Rec 实验启动器")
    print("   下一代生成式推荐系统实验平台")
    print("="*70)

def print_experiment_options():
    """打印实验选项"""
    print("\n📋 可用的实验模式:")
    print()
    print("1. 🚀 quick       - 快速验证实验 (30-60分钟)")
    print("   验证核心假设，快速得出初步结论")
    print("   推荐：首次运行时选择此模式")
    print()
    print("2. 🧠 expert      - 专家系统消融实验 (2-4小时)")
    print("   深入测试不同专家组合的效果")
    print()
    print("3. 🏗️  architecture - 架构配置实验 (1-3小时)")
    print("   测试不同的模型架构配置")
    print()
    print("4. 📊 baseline    - 基线对比实验 (1-2小时)")
    print("   与传统推荐算法进行性能对比")
    print()
    print("5. 🎛️  hyperparameter - 超参数搜索 (4-8小时)")
    print("   系统性搜索最优超参数组合")
    print()
    print("6. 🔬 full        - 完整实验套件 (6-12小时)")
    print("   运行所有实验，获得完整的实验报告")
    print()

def get_user_choice():
    """获取用户选择"""
    while True:
        choice = input("请选择实验模式 (输入数字或名称, 或 'q' 退出): ").strip().lower()
        
        if choice == 'q' or choice == 'quit':
            print("👋 退出实验启动器")
            sys.exit(0)
        
        mode_map = {
            '1': 'quick',
            '2': 'expert', 
            '3': 'architecture',
            '4': 'baseline',
            '5': 'hyperparameter',
            '6': 'full',
            'quick': 'quick',
            'expert': 'expert',
            'architecture': 'architecture', 
            'baseline': 'baseline',
            'hyperparameter': 'hyperparameter',
            'full': 'full'
        }
        
        if choice in mode_map:
            return mode_map[choice]
        else:
            print("❌ 无效选择，请重新输入")

def check_prerequisites():
    """检查实验前提条件"""
    print("\n🔍 检查实验前提条件...")
    
    base_dir = Path("/root/autodl-tmp/genius_rec-main")
    
    # 检查必要文件
    required_files = [
        "src/train_GeniusRec.py",
        "src/config.py", 
        "checkpoints/hstu_encoder.pth",
        "data/processed/train.parquet",
        "data/processed/validation.parquet",
        "data/processed/test.parquet",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (base_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 请确保:")
        print("   1. 已完成数据预处理 (python -m src.preprocess)")
        print("   2. 已预训练编码器 (python -m src.encoder.train_encoder)")
        return False
    
    # 检查图像嵌入（可选）
    image_embeddings = base_dir / "data/book_image_embeddings.npy"
    if image_embeddings.exists():
        print("✅ 找到图像嵌入文件，可以测试视觉专家")
    else:
        print("⚠️  未找到图像嵌入文件，视觉专家将被禁用")
    
    print("✅ 前提条件检查通过")
    return True

def run_experiment(mode: str):
    """运行指定模式的实验"""
    
    if not check_prerequisites():
        print("❌ 前提条件不满足，无法开始实验")
        return False
    
    print(f"\n🚀 启动 {mode} 模式实验...")
    
    try:
        if mode == 'quick':
            # 快速验证实验
            cmd = ["python", "experiments/quick_validation.py"]
            print("📝 运行快速验证实验...")
            
        elif mode == 'expert':
            # 专家系统消融实验
            cmd = ["python", "experiments/run_experiments.py", "--experiment_suite", "expert_ablation"]
            print("🧠 运行专家系统消融实验...")
            
        elif mode == 'architecture':
            # 架构配置实验
            cmd = ["python", "experiments/run_experiments.py", "--experiment_suite", "architecture"]
            print("🏗️ 运行架构配置实验...")
            
        elif mode == 'baseline':
            # 基线对比实验
            cmd = ["python", "experiments/run_experiments.py", "--experiment_suite", "baseline_comparison"]
            print("📊 运行基线对比实验...")
            
        elif mode == 'hyperparameter':
            # 超参数搜索实验
            cmd = ["python", "experiments/run_experiments.py", "--experiment_suite", "hyperparameter"]
            print("🎛️ 运行超参数搜索实验...")
            
        elif mode == 'full':
            # 完整实验套件
            cmd = ["python", "experiments/run_experiments.py", "--experiment_suite", "all"]
            print("🔬 运行完整实验套件...")
        
        else:
            print(f"❌ 未知的实验模式: {mode}")
            return False
        
        print(f"📋 执行命令: {' '.join(cmd)}")
        print("⏱️ 实验开始，请耐心等待...\n")
        
        # 运行实验
        result = subprocess.run(cmd, cwd="/root/autodl-tmp/genius_rec-main")
        
        if result.returncode == 0:
            print("\n🎉 实验成功完成!")
            print("📄 请查看 experiments/ 目录下的结果报告")
            return True
        else:
            print(f"\n❌ 实验失败，退出码: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断实验")
        return False
    except Exception as e:
        print(f"\n💥 实验运行异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="GENIUS-Rec 实验启动器")
    parser.add_argument(
        "--mode", 
        choices=["quick", "expert", "architecture", "baseline", "hyperparameter", "full"],
        help="直接指定实验模式"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.mode:
        # 直接运行指定模式
        mode = args.mode
        print(f"🎯 直接运行 {mode} 模式")
    else:
        # 交互式选择
        print_experiment_options()
        mode = get_user_choice()
    
    print(f"\n✅ 选择的实验模式: {mode}")
    
    # 确认运行
    if mode in ['full', 'hyperparameter']:
        print("⚠️  这是一个长时间运行的实验，可能需要数小时完成")
        confirm = input("确认继续? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("👋 取消实验")
            return
    
    # 运行实验
    success = run_experiment(mode)
    
    if success:
        print("\n🎊 实验完成! 感谢使用GENIUS-Rec实验平台")
    else:
        print("\n😞 实验未能成功完成，请检查错误信息")

if __name__ == "__main__":
    main()
