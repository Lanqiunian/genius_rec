#!/usr/bin/env python3
# checkpoint_manager.py - 检查点管理工具

import torch
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

def list_checkpoints(directory):
    """列出目录中的所有检查点文件"""
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return []
    
    checkpoint_files = []
    for file in os.listdir(directory):
        if file.endswith('.pth'):
            file_path = os.path.join(directory, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            checkpoint_files.append({
                'name': file,
                'path': file_path,
                'size_mb': file_size,
                'modified': mod_time
            })
    
    # 按修改时间排序
    checkpoint_files.sort(key=lambda x: x['modified'], reverse=True)
    return checkpoint_files

def backup_checkpoint(src_path, backup_dir):
    """备份检查点文件"""
    if not os.path.exists(src_path):
        print(f"❌ 源文件不存在: {src_path}")
        return False
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # 创建带时间戳的备份文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_name = os.path.basename(src_path)
    backup_name = f"{timestamp}_{src_name}"
    backup_path = os.path.join(backup_dir, backup_name)
    
    try:
        shutil.copy2(src_path, backup_path)
        print(f"✅ 备份成功: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        return False

def clean_old_checkpoints(directory, keep_latest=5, keep_best=True):
    """清理旧的检查点文件，保留最新的几个和最佳的"""
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        print("没有找到检查点文件")
        return
    
    print(f"🧹 开始清理检查点，保留最新 {keep_latest} 个文件")
    
    # 分离最佳和最新检查点
    best_checkpoints = [ckpt for ckpt in checkpoints if 'best' in ckpt['name']]
    latest_checkpoints = [ckpt for ckpt in checkpoints if 'latest' in ckpt['name']]
    other_checkpoints = [ckpt for ckpt in checkpoints if 'best' not in ckpt['name'] and 'latest' not in ckpt['name']]
    
    files_to_delete = []
    
    # 保留最佳检查点
    if keep_best and best_checkpoints:
        print(f"🏆 保留最佳检查点: {len(best_checkpoints)} 个")
    
    # 保留最新的几个latest检查点
    if len(latest_checkpoints) > keep_latest:
        files_to_delete.extend(latest_checkpoints[keep_latest:])
        print(f"📂 保留最新检查点: {keep_latest} 个，删除 {len(latest_checkpoints) - keep_latest} 个")
    
    # 保留最新的几个其他检查点
    if len(other_checkpoints) > keep_latest:
        files_to_delete.extend(other_checkpoints[keep_latest:])
        print(f"📄 保留其他检查点: {keep_latest} 个，删除 {len(other_checkpoints) - keep_latest} 个")
    
    # 执行删除
    if files_to_delete:
        print(f"\n将删除 {len(files_to_delete)} 个文件:")
        total_size = 0
        for ckpt in files_to_delete:
            print(f"  - {ckpt['name']} ({ckpt['size_mb']:.2f} MB)")
            total_size += ckpt['size_mb']
        
        print(f"总计释放空间: {total_size:.2f} MB")
        
        confirm = input("\n确认删除这些文件吗? (y/N): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for ckpt in files_to_delete:
                try:
                    os.remove(ckpt['path'])
                    deleted_count += 1
                    print(f"✅ 已删除: {ckpt['name']}")
                except Exception as e:
                    print(f"❌ 删除失败 {ckpt['name']}: {e}")
            
            print(f"\n🎉 清理完成! 成功删除 {deleted_count} 个文件")
        else:
            print("取消删除操作")
    else:
        print("✅ 没有需要清理的文件")

def show_checkpoints_summary(directory):
    """显示检查点摘要信息"""
    checkpoints = list_checkpoints(directory)
    if not checkpoints:
        print(f"❌ 目录 {directory} 中没有找到检查点文件")
        return
    
    print(f"📋 检查点摘要 - 目录: {directory}")
    print("=" * 80)
    
    total_size = sum(ckpt['size_mb'] for ckpt in checkpoints)
    print(f"文件总数: {len(checkpoints)}")
    print(f"总大小: {total_size:.2f} MB")
    print()
    
    print(f"{'文件名':<30} {'大小(MB)':<10} {'修改时间':<20}")
    print("-" * 80)
    
    for ckpt in checkpoints:
        print(f"{ckpt['name']:<30} {ckpt['size_mb']:<10.2f} {ckpt['modified'].strftime('%Y-%m-%d %H:%M:%S'):<20}")

def main():
    if len(sys.argv) < 2:
        print("检查点管理工具")
        print("使用方法:")
        print("  python checkpoint_manager.py list <directory>           # 列出检查点")
        print("  python checkpoint_manager.py clean <directory>          # 清理旧检查点")
        print("  python checkpoint_manager.py backup <file> <backup_dir> # 备份检查点")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        if len(sys.argv) < 3:
            print("请指定目录路径")
            sys.exit(1)
        show_checkpoints_summary(sys.argv[2])
    
    elif command == "clean":
        if len(sys.argv) < 3:
            print("请指定目录路径")
            sys.exit(1)
        
        directory = sys.argv[2]
        keep_latest = 3  # 默认保留3个最新的
        
        if len(sys.argv) >= 4:
            try:
                keep_latest = int(sys.argv[3])
            except ValueError:
                print("保留数量必须是整数")
                sys.exit(1)
        
        clean_old_checkpoints(directory, keep_latest=keep_latest)
    
    elif command == "backup":
        if len(sys.argv) < 4:
            print("请指定源文件和备份目录")
            sys.exit(1)
        
        src_file = sys.argv[2]
        backup_dir = sys.argv[3]
        backup_checkpoint(src_file, backup_dir)
    
    else:
        print(f"未知命令: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
