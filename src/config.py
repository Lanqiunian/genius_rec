# src/config.py (最终对齐版)

import torch
from pathlib import Path

def get_config():
    """
    返回一个包含所有项目配置的字典。
    此版本为对齐官方Amazon Books数据集的HSTU编码器任务而设计。
    """
    ROOT_DIR = Path(__file__).parent.parent 
    
    config = {
        # --- 1. 文件与路径配置 ---
        "data_dir": ROOT_DIR / "data",
        "processed_data_dir": ROOT_DIR / "data" / "processed", # 指向对齐版预处理的输出目录
        "log_dir": ROOT_DIR / "logs",

        # 对齐版预处理脚本生成的Parquet文件和ID映射文件
        "train_file": ROOT_DIR / "data" / "processed" / "train.parquet",
        "validation_file": ROOT_DIR / "data" / "processed" / "validation.parquet",
        "test_file": ROOT_DIR / "data" / "processed" / "test.parquet",
        "id_maps_file": ROOT_DIR / "data" / "processed" / "id_maps.pkl",
        
        "checkpoint_dir": ROOT_DIR / "checkpoints", # 为对齐版模型设置新的checkpoint目录
        "log_file": ROOT_DIR / "training.log",
        
        # --- 2. 数据预处理配置 (与preprocess.py对齐) ---
        "k_core": 5,
        "min_seq_len": 5,

        # --- 3. 模型超参数 (完全对齐官方Amazon Books配置) ---
        "max_seq_len": 50,         # 官方Amazon Books使用50
        "embedding_dim": 64,       # 官方Amazon Books配置：64
        "linear_hidden_dim": 16,   # 官方Amazon Books配置：dv=16
        "attention_dim": 16,       # 官方Amazon Books配置：dqk=16
        "num_encoder_layers": 4,   # 官方Amazon Books配置：4层
        "nhead": 4,                # 官方Amazon Books配置：4个头
        "dropout": 0.1,            # 官方Amazon Books的dropout率

        
        
        # --- 4. 训练超参数 (对齐官方Amazon Books) ---
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "num_epochs": 201,         
        "batch_size": 256,         
        "learning_rate": 1e-3,     # 官方默认1e-3
        "weight_decay": 0,         # 官方Amazon Books使用0
        "early_stopping_patience": 10,
        "num_workers": 10,
        
        # --- 5. 评估参数 ---
        "top_k": 10,
        "num_neg_samples": 512,    # 官方Amazon Books配置：512
        "temperature": 0.05,       # 官方默认温度

        # --- 6. 特殊Token ID ---
        "pad_token_id": 0 # 注意：我们的物品ID是从0开始的，所以需要一个专用的pad_id。
    }
    
    return config