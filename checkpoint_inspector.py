#!/usr/bin/env python3
# checkpoint_inspector.py - 检查点查看工具

import torch
import os
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """检查并显示检查点的详细信息"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return False
    
    try:
        print(f"📋 检查点信息: {checkpoint_path}")
        print("=" * 60)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 基本训练信息
        print("🏃 训练状态:")
        print(f"  轮次: {checkpoint.get('epoch', 'N/A')}")
        print(f"  耐心计数: {checkpoint.get('patience_counter', 'N/A')}")
        
        # 性能指标
        print("\n📊 性能指标:")
        print(f"  当前困惑度: {checkpoint.get('current_perplexity', 'N/A'):.4f}")
        print(f"  最佳困惑度: {checkpoint.get('best_perplexity', 'N/A'):.4f}")
        print(f"  训练损失: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"  验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        # 模型状态
        print("\n🧠 模型状态:")
        if 'model_state_dict' in checkpoint:
            model_keys = list(checkpoint['model_state_dict'].keys())
            print(f"  模型参数数量: {len(model_keys)}")
            print(f"  参数示例: {model_keys[:3]}...")
        else:
            print("  ❌ 未找到模型状态")
        
        # 优化器状态
        print("\n⚡ 优化器状态:")
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            print(f"  参数组数量: {len(opt_state.get('param_groups', []))}")
            if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
                print(f"  学习率: {opt_state['param_groups'][0].get('lr', 'N/A')}")
        else:
            print("  ❌ 未找到优化器状态")
        
        # 调度器状态
        print("\n📅 调度器状态:")
        if 'scheduler_state_dict' in checkpoint:
            print("  ✅ 包含调度器状态")
        else:
            print("  ❌ 未找到调度器状态")
        
        # 配置信息
        print("\n⚙️ 配置信息:")
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  设备: {config.get('device', 'N/A')}")
            if 'finetune' in config:
                print(f"  批次大小: {config['finetune'].get('batch_size', 'N/A')}")
                print(f"  总轮次: {config['finetune'].get('num_epochs', 'N/A')}")
        else:
            print("  ❌ 未找到配置信息")
        
        # 参数信息
        print("\n🔧 训练参数:")
        if 'args' in checkpoint:
            args = checkpoint['args']
            print(f"  保存目录: {args.get('save_dir', 'N/A')}")
            print(f"  冻结编码器: {args.get('freeze_encoder', 'N/A')}")
            print(f"  编码器权重路径: {args.get('encoder_weights_path', 'N/A')}")
        else:
            print("  ❌ 未找到训练参数")
        
        # 文件大小
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"\n💾 文件大小: {file_size:.2f} MB")
        
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("使用方法: python checkpoint_inspector.py <checkpoint_path>")
        print("或者: python checkpoint_inspector.py <directory> (检查目录下所有检查点)")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # 检查单个文件
        inspect_checkpoint(path)
    elif os.path.isdir(path):
        # 检查目录下的所有检查点文件
        checkpoint_files = []
        for file in os.listdir(path):
            if file.endswith('.pth'):
                checkpoint_files.append(os.path.join(path, file))
        
        if not checkpoint_files:
            print(f"❌ 目录 {path} 中未找到检查点文件 (.pth)")
            return
        
        print(f"🔍 在目录 {path} 中找到 {len(checkpoint_files)} 个检查点文件:")
        print()
        
        for i, ckpt_file in enumerate(checkpoint_files, 1):
            print(f"\n[{i}/{len(checkpoint_files)}] 检查点:")
            inspect_checkpoint(ckpt_file)
            if i < len(checkpoint_files):
                input("按回车键继续...")
    else:
        print(f"❌ 路径不存在: {path}")

if __name__ == "__main__":
    main()
