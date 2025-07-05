#!/usr/bin/env python3
"""
GENIUS-Rec 实验预修复脚本
========================

自动修复运行 `python start_experiments.py --mode full` 前的关键问题

使用方法:
    python fix_experiment_issues.py
"""

import os
import shutil
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_fix_paths():
    """检查和修复路径问题"""
    logger.info("🔍 检查路径配置...")
    
    current_dir = Path.cwd()
    logger.info(f"当前工作目录: {current_dir}")
    
    # 检查是否在正确的项目目录
    required_files = ["src/config.py", "experiments/run_experiments.py", "start_experiments.py"]
    missing_files = []
    
    for file_path in required_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ 不在正确的项目目录，缺少文件: {missing_files}")
        logger.error("请切换到 Genius_Rec 项目根目录后重新运行此脚本")
        return False
    
    logger.info("✅ 路径检查通过")
    return True

def fix_encoder_weights():
    """修复编码器权重文件问题"""
    logger.info("🔍 检查编码器权重文件...")
    
    checkpoints_dir = Path("checkpoints")
    target_file = checkpoints_dir / "hstu_encoder.pth"
    
    if target_file.exists():
        logger.info("✅ hstu_encoder.pth 已存在")
        return True
    
    # 查找可用的编码器文件
    candidate_files = [
        "hstu_official_aligned_best.pth",
        "hstu_encoder_migrated.pth",
        "baseline_transformer_best.pth"
    ]
    
    source_file = None
    for candidate in candidate_files:
        candidate_path = checkpoints_dir / candidate
        if candidate_path.exists():
            source_file = candidate_path
            logger.info(f"找到候选文件: {candidate}")
            break
    
    if source_file is None:
        logger.error("❌ 未找到任何可用的编码器权重文件")
        logger.error("请确保checkpoints目录中有以下文件之一:")
        for candidate in candidate_files:
            logger.error(f"  - {candidate}")
        return False
    
    try:
        # 创建软链接或复制文件
        if hasattr(os, 'symlink'):
            try:
                os.symlink(source_file.absolute(), target_file)
                logger.info(f"✅ 创建软链接: {source_file.name} -> {target_file.name}")
            except OSError:
                # Windows可能没有权限创建软链接，改用复制
                shutil.copy2(source_file, target_file)
                logger.info(f"✅ 复制文件: {source_file.name} -> {target_file.name}")
        else:
            shutil.copy2(source_file, target_file)
            logger.info(f"✅ 复制文件: {source_file.name} -> {target_file.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 修复编码器权重失败: {e}")
        return False

def check_data_files():
    """检查必要的数据文件"""
    logger.info("🔍 检查数据文件...")
    
    required_data_files = [
        "data/processed/train.parquet",
        "data/processed/validation.parquet", 
        "data/processed/test.parquet",
        "data/processed/id_maps.pkl"
    ]
    
    missing_files = []
    for file_path in required_data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("❌ 缺少必要的数据文件:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        logger.error("请先运行数据预处理: python -m src.preprocess")
        return False
    
    logger.info("✅ 数据文件检查通过")
    return True

def check_image_embeddings():
    """检查图像嵌入文件"""
    logger.info("🔍 检查图像嵌入文件...")
    
    image_files = [
        "data/book_image_embeddings_migrated.npy",
        "data/book_gemini_embeddings_filtered.npy"
    ]
    
    found_files = []
    for file_path in image_files:
        if Path(file_path).exists():
            found_files.append(file_path)
            logger.info(f"✅ 找到图像嵌入: {file_path}")
    
    if not found_files:
        logger.warning("⚠️ 未找到图像嵌入文件，图像专家实验将被跳过")
        return False
    
    return True

def create_experiment_directories():
    """创建实验目录"""
    logger.info("🔍 创建实验目录...")
    
    experiment_dirs = [
        "experiments/checkpoints",
        "experiments/checkpoints/expert_ablation",
        "experiments/checkpoints/architecture", 
        "experiments/checkpoints/hyperparameter",
        "experiments/checkpoints/data_augmentation",
        "experiments/checkpoints/quick_validation"
    ]
    
    for dir_path in experiment_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ 创建目录: {dir_path}")

def update_config_for_current_environment():
    """更新配置以适应当前环境"""
    logger.info("🔍 检查配置文件...")
    
    config_file = Path("src/config.py")
    if not config_file.exists():
        logger.error("❌ 配置文件不存在")
        return False
    
    # 读取配置文件内容
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否需要更新路径
    current_dir = Path.cwd()
    if str(current_dir) not in content:
        logger.info("✅ 配置文件使用相对路径，无需修改")
    
    return True

def check_system_resources():
    """检查系统资源"""
    logger.info("🔍 检查系统资源...")
    
    # 检查磁盘空间
    import shutil
    free_bytes = shutil.disk_usage('.').free
    free_gb = free_bytes / (1024**3)
    
    if free_gb < 5:
        logger.warning(f"⚠️ 磁盘空间不足: {free_gb:.1f}GB，建议至少10GB")
    else:
        logger.info(f"✅ 磁盘空间充足: {free_gb:.1f}GB")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU可用: {gpu_name} ({gpu_memory:.1f}GB)")
            
            if gpu_memory < 6:
                logger.warning("⚠️ GPU内存可能不足，建议至少8GB")
        else:
            logger.warning("⚠️ GPU不可用，将使用CPU训练（速度较慢）")
    except ImportError:
        logger.warning("⚠️ PyTorch未安装")
    
    return True

def main():
    """主修复流程"""
    logger.info("🚀 开始GENIUS-Rec实验预修复...")
    
    success = True
    
    # 1. 检查路径
    if not check_and_fix_paths():
        success = False
    
    # 2. 修复编码器权重
    if not fix_encoder_weights():
        success = False
    
    # 3. 检查数据文件
    if not check_data_files():
        success = False
    
    # 4. 检查图像嵌入
    check_image_embeddings()  # 非致命错误
    
    # 5. 创建实验目录
    create_experiment_directories()
    
    # 6. 更新配置
    if not update_config_for_current_environment():
        success = False
    
    # 7. 检查系统资源
    check_system_resources()  # 仅警告
    
    if success:
        logger.info("🎉 预修复完成！可以运行实验了")
        logger.info("建议先运行: python start_experiments.py --mode quick")
        logger.info("验证无误后再运行: python start_experiments.py --mode full")
    else:
        logger.error("❌ 修复过程中发现严重问题，请查看上方错误信息并手动修复")
    
    return success

if __name__ == "__main__":
    main()
