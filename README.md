# GENIUS-Rec: Generative Sequential Recommendation with Large Language Models

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Lanqiunian/genius_rec/blob/main/LICENSE)

## 🎯 项目简介

GENIUS-Rec是一个基于Transformer架构的生成式序列推荐系统，结合了多专家混合架构和多模态嵌入（文本+图像），能够为用户生成个性化的推荐序列。

### 主要特性

- **🔄 生成式架构**: 使用编码器-解码器结构，支持序列到序列的推荐生成
- **🧠 多专家系统**: 
  - 行为专家：基于用户历史行为模式
  - 内容专家：基于物品文本语义信息
  - 图像专家：基于物品视觉特征
- **📝 多模态嵌入**: 支持Google Gemini文本嵌入和CLIP图像嵌入
- **⚡ 高效训练**: 支持GPU加速，内存优化，批处理等推荐系统

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Lanqiunian/genius_rec/blob/main/LICENSE)

**GENIUS-Rec** 是一个旨在突破传统推荐范式局限的开源研究项目。它不仅仅预测用户可能点击的“下一个物品”，而是致力于成为一个能够理解用户多模态兴趣，并直接**生成一个连贯、高质量推荐会话（Session）**的AI智能顾问。

---

## 核心理念与技术架构

本项目针对传统推荐存在的两大核心痛点进行设计：
1.  **语义鸿沟**: 传统模型难以理解物品（如书籍、电影）的深层文本和视觉内涵。
2.  **范式局限**: “单点预测”无法满足用户对连贯、探索式体验的需求。

为实现此目标，GENIUS-Rec采用了一个标准的**Encoder-Decoder（编码器-解码器）**架构：

* **编码器 (Encoder)**: 为了追求极致的序列理解能力，完整复现并预训练了Meta AI提出的SOTA模型 **HSTU (Hierarchical Sequential Transduction Unit)**，使其能深度编码用户的行为历史。
* **解码器 (Decoder)**: 在编码器提供的用户理解之上，构建了一个标准的**生成式Transformer解码器**,融入MoE，负责自回归地生成推荐序列。

## 项目进展与核心成果

项目采取分阶段验证的策略，目前已取得以下成果：

* **第一阶段 (可行性验证):**
    * **思想**: 模仿前沿研究（字节的HLLM）思想，验证大模型外部知识注入的价值。
    * **实现**: 通过调用大语言模型（LLM）API为电影生成高质量语义向量，并结合序列模型进行推荐。
    * **成果**: 在 **MovieLens** 数据集上，使 **NDCG@10 指标提升了18%**，成功证明了引入外部知识的巨大潜力。

* **第二阶段 (编码器预训练):**
    * **实现**: 在 **Amazon Books** 数据集上完整复现并预训练了HSTU编码器。
    * **成果**: 预训练复现指标为 **HR@10: 0.0596, NDCG@10: 0.0317**，为后续的生成任务提供了坚实的序列编码基础。

* **第三阶段 (架构可行性实现):**
    * **实现**: 将预训练好的HSTU编码器与生成式解码器集成为一个完整的 **Seq2Seq 模型**。
    * **成果**: 端到端微调后，在Amazon Books数据集上的指标为**Perplexity：34.5**。

* **后续计划:**
    * **模型**: 融入多专家模型，搭建出完整的**GENIUS-Rec模型**。
    * **实验**: 尝试进行DPO。
    * **实验**: 设计并完成完成充分的实验，验证模型性能。

    
## 快速开始 (Quick Start)

### 1. 环境搭建

```bash
# 克隆项目仓库
git clone [https://github.com/Lanqiunian/genius_rec.git](https://github.com/Lanqiunian/genius_rec.git)
cd genius_rec

# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # on Windows, use `.venv\Scripts\activate`

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据预处理

在开始训练前，您需要先下载原始数据并执行预处理脚本。

```bash
# 假设原始数据文件 Books.jsonl.gz 已放置在 data/ 目录下
python -m src.preprocess
```
该脚本会自动执行K-core过滤、ID重映射，并将处理好的数据保存在 `data/processed/` 目录下。

### 3. 如何训练

本项目包含两个主要的训练阶段，请按顺序执行。

#### 阶段一：预训练HSTU编码器

*此步骤旨在复现论文中的Next-Item Prediction任务，为编码器提供初始权重。*

```bash
python -m src.encoder.train_encoder 
```

#### 阶段二：微调完整的Seq2Seq模型

*此步骤是项目的核心，用于训练完整的生成式推荐模型。*

```bash
# 指定预训练好的编码器权重路径，开始端到端微调训练
python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth

# 或者如果需要冻结编码器参数进行对比实验
python -m src.train_GeniusRec --encoder_weights_path checkpoints/hstu_encoder.pth --freeze_encoder
```

## 未来规划

在当前Seq2Seq模型稳定收敛后，项目将向着最终的目标演进：

1.  **集成多模态混合专家 (MoE)**: 在解码器中引入一个可解释的MoE框架，动态融合来自**文本(Gemini/TinyLlama)**和**视觉(CLIP)**专家的信息。
2.  **引入直接偏好优化 (DPO)**: 借鉴业界成功经验（如快手OneRec），引入DPO，使模型生成的推荐列表与**新颖性、多样性**等更复杂的人类偏好对齐。

## 致谢

本项目的构思与实现，受到了以下优秀工作的启发，在此表示感谢：
* Meta AI: **Actions Speak Louder than Words (HSTU)**
* Google Research: **HLLM: Unlocking the Power of Large Language Models for Recommendation**
* Kuaishou: **OneRec: A Self-Sufficient and Composable Recommender System**

## 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源。
