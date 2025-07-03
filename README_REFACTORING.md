# GENIUS-Rec 重构指南

## 重构概述

本次重构实现了以下主要改进：

### 1. 集中化配置系统 ✅
- **文件**: `src/config.py`
- **改进**: 所有配置集中在一个函数 `get_config()` 中
- **优势**: 
  - 统一管理所有超参数
  - 跨平台路径支持（使用 `pathlib.Path`）
  - 易于维护和调试

### 2. 特殊标记支持 ✅
- **新增标记**:
  - `pad_token_id: 0` (填充标记)
  - `sos_token_id: 1` (序列开始标记)
  - `eos_token_id: 2` (序列结束标记)
  - `mask_token_id: 3` (掩码标记)

### 3. 数据预处理改进 ✅
- **文件**: `src/preprocess.py`
- **改进**: 
  - 为特殊标记预留ID空间
  - 正确的ID重映射
  - 保存特殊标记配置信息

### 4. 权重迁移系统 ✅
- **文件**: `scripts/migrate_weights.py`
- **功能**: 
  - 自动迁移预训练权重到新ID系统
  - 处理词汇表大小变化
  - 初始化特殊标记嵌入

### 5. 数据集改进 ✅
- **文件**: `src/dataset.py`
- **改进**:
  - 正确的SOS/EOS标记处理
  - 配置驱动的初始化
  - 改进的序列构造逻辑

### 6. 模型架构更新 ✅
- **文件**: `src/GeniusRec.py`, `src/encoder/encoder.py`, `src/decoder/decoder.py`
- **改进**:
  - 配置字典驱动的初始化
  - 更好的参数传递
  - 向前兼容性

## 使用指南

### 快速开始

1. **运行重构设置脚本**:
```bash
cd /root/autodl-tmp/genius_rec-main
python setup_refactored_system.py
```

2. **手动步骤**（如果需要）:

#### 步骤 1: 数据预处理
```bash
python -m src.preprocess
```

#### 步骤 2: 权重迁移（如果有预训练权重）
```bash
python scripts/migrate_weights.py \
    --old_weights checkpoints/hstu_encoder.pth \
    --output checkpoints/hstu_encoder_migrated.pth
```

#### 步骤 3: 训练模型
```bash
# 使用迁移后的权重
python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder_migrated.pth

# 或从头开始训练
python -m src.train_GeniusRec
```

### 配置说明

#### 特殊标记配置
```python
{
    "pad_token_id": 0,    # 填充标记
    "sos_token_id": 1,    # 序列开始
    "eos_token_id": 2,    # 序列结束  
    "mask_token_id": 3    # 掩码标记
}
```

#### 专家系统配置
```python
{
    "expert_system": {
        "experts": {
            "behavior_expert": True,   # 行为专家
            "content_expert": False,   # 内容专家
            "image_expert": False      # 图像专家
        }
    }
}
```

### 文件结构变化

```
genius_rec-main/
├── src/
│   ├── config.py              # 🆕 集中化配置
│   ├── preprocess.py          # ✅ 更新：特殊标记支持
│   ├── dataset.py             # ✅ 更新：SOS/EOS处理
│   ├── train_GeniusRec.py     # ✅ 更新：配置驱动
│   ├── GeniusRec.py           # ✅ 更新：配置传递
│   └── ...
├── scripts/
│   └── migrate_weights.py     # 🆕 权重迁移脚本
├── setup_refactored_system.py # 🆕 设置指南
└── README_REFACTORING.md      # 🆕 本文档
```

## 重要变化说明

### 1. ID映射变化
- **之前**: 物品ID从1开始
- **现在**: 物品ID从4开始（为特殊标记预留0-3）

### 2. 词汇表大小计算
- **之前**: `num_items + 1`
- **现在**: `num_items + num_special_tokens`

### 3. 数据集初始化
- **之前**: 多个独立参数
- **现在**: 配置字典

### 4. 权重加载
- **之前**: 直接加载原始权重
- **现在**: 优先使用迁移后的权重

## 故障排除

### 常见问题

1. **维度不匹配错误**
   - **原因**: 使用了未迁移的权重
   - **解决**: 运行权重迁移脚本

2. **找不到配置参数**
   - **原因**: 配置格式变化
   - **解决**: 检查 `src/config.py` 中的参数名

3. **特殊标记ID冲突**
   - **原因**: 旧数据与新配置不匹配
   - **解决**: 重新运行数据预处理

### 调试技巧

1. **检查配置**:
```python
from src.config import get_config
config = get_config()
print(config['pad_token_id'])  # 应该是 0
```

2. **验证ID映射**:
```python
import pickle
with open('data/processed/id_maps.pkl', 'rb') as f:
    id_maps = pickle.load(f)
print(id_maps['special_tokens'])
```

3. **检查权重形状**:
```python
import torch
weights = torch.load('checkpoints/hstu_encoder_migrated.pth')
print([k for k in weights.keys() if 'embedding' in k])
```

## 向后兼容性

- ✅ 保持原有的训练脚本接口
- ✅ 支持旧的命令行参数
- ✅ 自动处理配置格式转换
- ⚠️ 需要重新预处理数据和迁移权重

## 性能优化

重构后的系统包含以下性能优化：

1. **内存优化**: 更高效的嵌入矩阵管理
2. **计算优化**: 改进的注意力机制
3. **IO优化**: 更快的数据加载
4. **可扩展性**: 模块化的专家系统

## 未来扩展

重构后的系统为以下扩展提供了基础：

1. **多模态专家**: 轻松添加新的专家类型
2. **动态配置**: 运行时配置调整
3. **分布式训练**: 配置驱动的分布式设置
4. **自动超参数调优**: 基于配置的参数搜索

---

## 联系与支持

如果在使用过程中遇到问题，请：

1. 检查本文档的故障排除部分
2. 运行 `setup_refactored_system.py` 进行诊断
3. 查看日志文件 `logs/` 目录
4. 检查配置文件 `src/config.py`

**祝您使用愉快！** 🚀
