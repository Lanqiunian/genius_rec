# GENIUS-Rec 实验框架

## 概述

本实验框架基于严格的科学实验设计原则，旨在系统性地验证GENIUS-Rec模型的核心技术贡献。通过精心设计的对照实验、消融研究和基线比较，我们力求回答关键的研究问题并提供可重现的实验结果。

## 脚本概览

| 脚本文件 | 功能描述 | 适用场景 |
|---------|---------|---------|
| `start_experiments.py` | 交互式实验启动器 | 日常实验，推荐使用 |
| `run_experiments.py` | 批量实验管理器 | 自动化批量实验 |
| `quick_validation.py` | 快速验证脚本 | 调试和初步验证 |
| `config_generator.py` | 配置文件生成器 | 高级自定义实验 |
| `analyze_results.py` | 结果分析器 | 实验结果可视化分析 |

## 快速开始

```bash
# 推荐的实验流程
cd /root/autodl-tmp/genius_rec-main

# 1. 快速验证环境和基本功能
python start_experiments.py --mode quick

# 2. 运行核心消融实验
python start_experiments.py --mode expert

# 3. 分析实验结果  
python experiments/analyze_results.py
```

## 研究假设与实验设计

### 核心研究假设
1. **H1**: 多专家混合架构(MoE)在序列推荐任务中显著优于单一模型架构
2. **H2**: 多模态信息融合(文本+视觉)能够显著提升推荐性能  
3. **H3**: 生成式Seq2Seq范式相比传统点预测方法具有优势
4. **H4**: 预训练的序列编码器能够为推荐任务提供有效的表示学习

### 实验设计原则
- **对照性**: 通过严格的对照组确保实验结果的可靠性
- **消融性**: 逐步移除/添加组件以量化各部分的贡献
- **可重现性**: 固定随机种子，详细记录实验配置和环境
- **统计显著性**: 采用适当的统计检验方法验证结果显著性

## 实验模块

### 1. 专家系统消融实验 (`expert_ablation`)

**目标**: 验证多专家架构的有效性并量化各专家的贡献

**实验配置**:
```
- behavior_only: 仅启用行为专家（基线）
- content_only: 仅启用内容专家  
- image_only: 仅启用图像专家
- behavior_content: 行为+内容专家
- behavior_image: 行为+图像专家
- content_image: 内容+图像专家
- all_experts: 完整多专家系统
```

**关键指标**: HR@10, NDCG@10, 验证损失

### 2. 架构配置实验 (`architecture`)

**目标**: 验证关键架构设计选择的合理性

**实验配置**:
- 编码器策略: 端到端微调 vs 冻结
- 解码器深度: 2层, 4层, 6层
- 注意力机制: 不同头数的影响

### 3. 基线比较实验 (`baseline_comparison`)

**目标**: 与现有推荐方法进行公平比较

**基线模型**:
- Transformer-based序列推荐模型
- 传统协同过滤方法
- 其他序列推荐SOTA方法
### 4. 超参数敏感性分析 (`hyperparameter`)

**目标**: 评估模型对关键超参数的鲁棒性

**参数范围**:
- 学习率: [1e-4, 5e-4, 1e-3, 2e-3]
- 批次大小: [32, 64, 128]  
- 解码器层数: [2, 4, 6]
- 注意力头数: [4, 8, 12]

### 5. 数据增强效果验证 (`data_augmentation`)

**目标**: 量化不同数据增强策略的效果

**策略对比**:
- 无图像嵌入 vs 有图像嵌入
- 不同文本嵌入质量的影响
- 数据集规模对性能的影响

## 使用方法

### 命令行接口

#### 基础用法
```bash
# 运行特定实验套件
python experiments/run_experiments.py --experiment_suite expert_ablation

# 运行所有实验
python experiments/run_experiments.py --experiment_suite all
```

#### 交互式启动器
```bash
# 启动交互式界面
python start_experiments.py

# 直接指定模式
python start_experiments.py --mode quick
python start_experiments.py --mode expert  
python start_experiments.py --mode full
```

#### 快速验证
```bash
# 快速验证核心假设
python experiments/quick_validation.py
```

### 高级配置

#### 自定义实验
```bash
# 直接调用训练脚本进行自定义实验（默认端到端微调）
python -m src.train_GeniusRec \
    --encoder_weights_path checkpoints/hstu_encoder.pth \
    --enable_image_expert \
    --save_dir custom_experiment

# 或者冻结编码器进行对比实验
python -m src.train_GeniusRec \
    --encoder_weights_path checkpoints/hstu_encoder.pth \
    --enable_image_expert \
    --save_dir custom_experiment \
    --freeze_encoder
```

#### 配置文件生成
```bash
# 生成不同的配置变体
python experiments/config_generator.py
```

## 结果分析

### 自动分析
```bash
# 分析最新实验结果
python experiments/analyze_results.py

# 分析特定结果文件
python experiments/analyze_results.py --results_file path/to/results.json
```

### 输出文件结构
```
experiments/
├── checkpoints/           # 模型检查点
│   ├── expert_ablation/   
│   ├── architecture/      
│   └── baseline/          
├── quick_results/         # 快速验证结果
├── analysis/              # 分析报告和可视化
├── configs/               # 动态生成的配置
└── logs/                  # 详细日志文件
```

## 评估指标

### 主要指标
- **Hit Ratio @ K (HR@K)**: 前K个推荐中包含目标物品的用户比例
- **Normalized Discounted Cumulative Gain @ K (NDCG@K)**: 考虑位置权重的推荐质量指标
- **Validation Loss**: 模型在验证集上的交叉熵损失

### 统计分析
- 所有实验结果均报告均值和标准差
- 采用配对t检验评估统计显著性 (p < 0.05)
- 提供95%置信区间

## 实验环境要求

### 硬件要求
- GPU: 建议12GB以上显存 (NVIDIA V100/A100/RTX 3090及以上)
- 内存: 32GB以上
- 存储: 100GB可用空间

### 软件要求
- Python 3.9+
- PyTorch 1.12+
- CUDA 11.6+
- 详细依赖见 `requirements.txt`

### 数据要求
- 已预处理的Amazon Books数据集
- 预训练的HSTU编码器权重
- 可选: 图像嵌入文件 (用于视觉专家)

## 最佳实践

### 实验流程建议
1. **环境验证**: 运行快速验证确保环境配置正确
2. **基线建立**: 首先运行单专家实验建立性能基线
3. **消融分析**: 系统性地添加组件验证各部分贡献
4. **超参数优化**: 针对最优配置进行精细调优
5. **结果验证**: 多次运行确保结果稳定性

### 调试指南
- 检查日志文件定位错误: `experiments/logs/`
- 验证数据文件完整性和格式
- 确认GPU内存充足，必要时减少批次大小
- 使用快速验证模式调试配置问题

### 性能优化
- 使用混合精度训练节省显存
- 启用数据并行处理加速训练
- 合理设置检查点保存频率
- 定期清理中间文件释放存储空间

## 故障排除

### 常见问题

**Q: 实验运行失败，提示CUDA内存不足**
A: 1. 减少配置文件中的batch_size
   2. 启用gradient checkpointing
   3. 使用较小的模型配置

**Q: 图像专家无法启用**  
A: 1. 确认图像嵌入文件存在: `data/book_image_embeddings.npy`
   2. 检查文件格式和键值对应关系
   3. 验证嵌入维度匹配

**Q: 实验结果差异较大**
A: 1. 检查随机种子设置
   2. 增加训练轮次确保收敛
   3. 验证数据预处理的一致性

**Q: 基线比较结果异常**
A: 1. 确保使用相同的数据划分
   2. 验证评估指标计算的一致性  
   3. 检查模型初始化和训练过程

## 技术支持

- 详细日志文件: `experiments/logs/`
- 错误报告模板: `experiments/error_report_template.md`
- 配置文件验证: `python experiments/validate_config.py`
- 社区讨论: 项目Issues页面

---

**注意**: 所有实验结果应当在论文发表前进行充分的验证和统计分析。建议在最终提交前运行完整的实验套件并进行多次重复验证。
