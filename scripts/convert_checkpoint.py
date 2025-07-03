#!/usr/bin/env python3
"""
Windows to Linux Checkpoint Converter for PyTorch Models

This script converts PyTorch checkpoint files from Windows format to Linux-compatible format,
fixing path separator issues and ensuring proper serialization compatibility.

Usage:
    python convert_checkpoint.py --input checkpoints/hstu_encoder.pth --output checkpoints/hstu_encoder_linux.pth
    python convert_checkpoint.py --input checkpoints/hstu_encoder.pth --backup --overwrite
"""

import argparse
import logging
import os
import shutil
from pathlib import Path, PosixPath, WindowsPath
import torch
import platform
import pickle
import io

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class PathPickleCompatibility:
    """处理Windows和Linux之间的路径兼容性问题"""
    
    @staticmethod
    def path_constructor(loader, node):
        """自定义路径构造器，将WindowsPath转换为PosixPath"""
        path_str = loader.construct_scalar(node)
        # 将Windows路径分隔符转换为Unix格式
        unix_path = path_str.replace('\\', '/')
        return PosixPath(unix_path)
    
    @staticmethod
    def setup_pickle_compatibility():
        """设置pickle兼容性以处理跨平台路径问题"""
        # 注册路径类型的兼容性处理
        def windows_path_constructor(self, *args, **kwargs):
            # 将WindowsPath转换为字符串，然后创建PosixPath
            if args:
                path_str = str(args[0]).replace('\\', '/')
                return PosixPath(path_str)
            return PosixPath()
        
        # 临时替换WindowsPath类
        original_windows_path = None
        try:
            import pathlib
            original_windows_path = pathlib.WindowsPath
            pathlib.WindowsPath = lambda *args, **kwargs: PosixPath(str(args[0]).replace('\\', '/') if args else "")
        except:
            pass
        
        return original_windows_path

class CustomUnpickler(pickle.Unpickler):
    """自定义unpickler来处理跨平台兼容性问题"""
    
    def find_class(self, module, name):
        # 处理WindowsPath类
        if module == 'pathlib' and name == 'WindowsPath':
            return PosixPath
        elif module == 'pathlib' and name == 'PosixPath':
            return PosixPath
        # 其他路径相关的类也进行转换
        elif 'Path' in name and 'pathlib' in module:
            return PosixPath
        
        return super().find_class(module, name)

def backup_checkpoint(file_path):
    """创建原文件的备份"""
    backup_path = str(file_path) + '.backup'
    shutil.copy2(file_path, backup_path)
    logging.info(f"✅ 原文件已备份到: {backup_path}")
    return backup_path

def fix_state_dict_keys(state_dict):
    """
    修复state_dict中可能存在的路径相关问题
    """
    if isinstance(state_dict, dict):
        fixed_dict = {}
        for key, value in state_dict.items():
            # 只对字符串类型的键进行路径分隔符替换
            if isinstance(key, str):
                fixed_key = key.replace('\\', '/')
            else:
                fixed_key = key
            
            if isinstance(value, dict):
                fixed_dict[fixed_key] = fix_state_dict_keys(value)
            else:
                fixed_dict[fixed_key] = value
        return fixed_dict
    return state_dict

def convert_paths_in_object(obj):
    """
    递归转换对象中的所有路径对象为Linux兼容格式
    """
    try:
        if isinstance(obj, (WindowsPath, PosixPath)):
            # 将任何路径对象转换为PosixPath
            path_str = str(obj).replace('\\', '/')
            return PosixPath(path_str)
        elif isinstance(obj, dict):
            converted_dict = {}
            for key, value in obj.items():
                # 只对字符串类型的键进行处理
                if isinstance(key, str):
                    converted_key = key.replace('\\', '/')
                else:
                    converted_key = key
                converted_dict[converted_key] = convert_paths_in_object(value)
            return converted_dict
        elif isinstance(obj, (list, tuple)):
            converted = [convert_paths_in_object(item) for item in obj]
            return type(obj)(converted)
        elif isinstance(obj, str):
            # 修复字符串中可能的Windows路径分隔符，但只处理看起来像路径的字符串
            if '\\' in obj and ('/' in obj or ':' in obj or len(obj.split('\\')) > 1):
                return obj.replace('\\', '/')
            return obj
        else:
            return obj
    except Exception as e:
        # 如果转换过程中出现任何问题，返回原对象
        logging.warning(f"转换对象时出现错误: {e}, 返回原对象")
        return obj

def convert_checkpoint(input_path, output_path=None, overwrite=False, create_backup=True):
    """
    转换checkpoint文件从Windows格式到Linux兼容格式
    
    Args:
        input_path (str): 输入checkpoint文件路径
        output_path (str, optional): 输出文件路径，如果为None则覆盖原文件
        overwrite (bool): 是否覆盖原文件
        create_backup (bool): 是否创建备份
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 确定输出路径
    if output_path is None:
        if overwrite:
            output_path = input_path
        else:
            output_path = input_path.parent / f"{input_path.stem}_linux{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # 创建备份（如果需要且要覆盖原文件）
    backup_path = None
    if create_backup and (overwrite or output_path == input_path):
        backup_path = backup_checkpoint(input_path)
    
    try:
        logging.info(f"🔄 开始转换checkpoint: {input_path}")
        
        # 尝试加载checkpoint，使用不同的兼容性设置
        checkpoint_loaded = False
        checkpoint = None
        
        # 方法1: 使用自定义unpickler处理路径兼容性
        try:
            logging.info("🔧 尝试使用自定义unpickler解决路径兼容性问题...")
            with open(input_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                checkpoint = unpickler.load()
            checkpoint_loaded = True
            logging.info("✅ 使用自定义unpickler成功加载checkpoint")
        except Exception as e:
            logging.warning(f"自定义unpickler方法失败: {e}")
        
        # 方法2: 设置路径兼容性后标准加载
        if not checkpoint_loaded:
            try:
                logging.info("🔧 尝试设置路径兼容性后加载...")
                path_compat = PathPickleCompatibility()
                original_path = path_compat.setup_pickle_compatibility()
                
                checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
                checkpoint_loaded = True
                logging.info("✅ 使用路径兼容性方法成功加载checkpoint")
                
            except Exception as e:
                logging.warning(f"路径兼容性方法失败: {e}")
        
        # 方法3: 使用pickle模块直接处理
        if not checkpoint_loaded:
            try:
                logging.info("🔧 尝试使用pickle模块直接处理...")
                with open(input_path, 'rb') as f:
                    # 设置pickle的find_global来处理路径类
                    def safe_find_global(module, name):
                        if module == 'pathlib':
                            if name in ['WindowsPath', 'PosixPath']:
                                return PosixPath
                        return getattr(__import__(module, fromlist=['']), name)
                    
                    original_find_global = pickle.Unpickler.find_class
                    pickle.Unpickler.find_class = lambda self, module, name: safe_find_global(module, name)
                    
                    try:
                        checkpoint = pickle.load(f)
                        checkpoint_loaded = True
                        logging.info("✅ 使用pickle直接处理成功加载checkpoint")
                    finally:
                        pickle.Unpickler.find_class = original_find_global
                        
            except Exception as e:
                logging.warning(f"pickle直接处理方法失败: {e}")
        
        # 方法4: 最后尝试强制转换
        if not checkpoint_loaded:
            try:
                logging.info("🔧 尝试强制转换方法...")
                # 临时monkey patch pathlib
                import pathlib
                original_windows_path = getattr(pathlib, 'WindowsPath', None)
                
                # 创建一个兼容的WindowsPath类
                class CompatWindowsPath:
                    def __new__(cls, *args, **kwargs):
                        if args:
                            path_str = str(args[0]).replace('\\', '/')
                            return PosixPath(path_str)
                        return PosixPath()
                
                pathlib.WindowsPath = CompatWindowsPath
                
                try:
                    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
                    checkpoint_loaded = True
                    logging.info("✅ 使用强制转换方法成功加载checkpoint")
                finally:
                    if original_windows_path:
                        pathlib.WindowsPath = original_windows_path
                        
            except Exception as e:
                logging.warning(f"强制转换方法失败: {e}")
        
        if not checkpoint_loaded:
            raise RuntimeError(f"所有加载方法都失败了。这个checkpoint可能严重损坏或使用了不兼容的格式。")
        
        # 检查checkpoint格式并进行转换
        if isinstance(checkpoint, dict):
            # 修复可能的路径问题
            checkpoint = fix_state_dict_keys(checkpoint)
            
            # 递归处理所有可能的路径对象
            checkpoint = convert_paths_in_object(checkpoint)
            
            # 报告checkpoint内容
            if 'model_state_dict' in checkpoint:
                logging.info("🔍 检测到完整checkpoint格式 (包含model_state_dict)")
                if 'epoch' in checkpoint:
                    logging.info(f"   - Epoch: {checkpoint['epoch']}")
                if 'optimizer_state_dict' in checkpoint:
                    logging.info("   - 包含optimizer状态")
                if 'scheduler_state_dict' in checkpoint:
                    logging.info("   - 包含scheduler状态")
                    
                # 检查模型参数数量
                model_params = checkpoint['model_state_dict']
                param_count = len(model_params)
                logging.info(f"   - 模型参数数量: {param_count}")
                
            else:
                logging.info("🔍 检测到纯权重格式")
                param_count = len(checkpoint)
                logging.info(f"   - 参数数量: {param_count}")
        else:
            logging.warning("⚠️ 未知的checkpoint格式")
            # 仍然尝试转换可能的路径对象
            checkpoint = convert_paths_in_object(checkpoint)
        
        # 保存转换后的checkpoint
        logging.info(f"💾 保存转换后的checkpoint到: {output_path}")
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用Linux兼容的保存方式
        torch.save(checkpoint, output_path, pickle_protocol=4)
        
        logging.info("✅ Checkpoint转换完成!")
        
        # 验证转换后的文件
        logging.info("🔍 验证转换后的文件...")
        try:
            verification_checkpoint = torch.load(output_path, map_location='cpu', weights_only=False)
            logging.info("✅ 转换后的文件验证成功!")
            
            # 比较文件大小
            original_size = input_path.stat().st_size / (1024 * 1024)  # MB
            converted_size = output_path.stat().st_size / (1024 * 1024)  # MB
            
            logging.info(f"📊 文件大小对比:")
            logging.info(f"   - 原文件: {original_size:.2f} MB")
            logging.info(f"   - 转换后: {converted_size:.2f} MB")
            
            size_diff_percent = abs(converted_size - original_size) / original_size * 100
            if size_diff_percent < 5:  # 5%以内的差异是正常的
                logging.info(f"   - 大小差异: {size_diff_percent:.2f}% (正常)")
            else:
                logging.warning(f"   - 大小差异: {size_diff_percent:.2f}% (可能有问题)")
            
        except Exception as e:
            logging.error(f"❌ 转换后文件验证失败: {e}")
            if backup_path and output_path == input_path:
                logging.info(f"🔄 恢复备份文件...")
                shutil.copy2(backup_path, input_path)
            raise
        
        return str(output_path)
        
    except Exception as e:
        logging.error(f"❌ 转换失败: {e}")
        if backup_path and output_path == input_path:
            logging.info(f"🔄 恢复备份文件...")
            shutil.copy2(backup_path, input_path)
        raise

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint from Windows to Linux format")
    parser.add_argument('--input', '-i', required=True, help='Input checkpoint file path')
    parser.add_argument('--output', '-o', help='Output checkpoint file path (optional)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the original file')
    parser.add_argument('--backup', action='store_true', default=True, help='Create backup of original file (default: True)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup')
    
    args = parser.parse_args()
    
    # 处理备份选项
    create_backup = args.backup and not args.no_backup
    
    logging.info("=== PyTorch Checkpoint转换工具 ===")
    logging.info(f"🖥️  当前系统: {platform.system()} {platform.release()}")
    logging.info(f"🐍 Python版本: {platform.python_version()}")
    logging.info(f"🔥 PyTorch版本: {torch.__version__}")
    
    try:
        output_file = convert_checkpoint(
            args.input, 
            args.output, 
            args.overwrite, 
            create_backup
        )
        
        logging.info("=" * 50)
        logging.info("🎉 转换成功完成!")
        logging.info(f"📁 输出文件: {output_file}")
        logging.info("💡 现在可以在Linux环境下正常使用该checkpoint了")
        
        # 提供使用建议
        if 'hstu_encoder' in args.input:
            logging.info("\n📋 使用建议:")
            logging.info("   # 端到端微调（推荐）:")
            logging.info("   python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth")
            logging.info("   # 冻结编码器（对比实验）:")
            logging.info("   python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --freeze_encoder")
        
    except Exception as e:
        logging.error(f"💥 转换失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
