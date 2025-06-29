# src/config.py (重构版)

import torch
from pathlib import Path

def get_config():
    """
    返回一个包含所有项目配置的字典。
    此版本为支持Encoder-Decoder微调而重构。
    """
    ROOT_DIR = Path(__file__).parent.parent 
    
    config = {
        # =================================================================
        # 1. 通用与路径配置 (General & Path Config)
        # =================================================================
        "data": {
            "data_dir": ROOT_DIR / "data",
            "processed_data_dir": ROOT_DIR / "data" / "processed",
            "log_dir": ROOT_DIR / "logs",
            "checkpoint_dir": ROOT_DIR / "checkpoints",
            
            "train_file": ROOT_DIR / "data" / "processed" / "train.parquet",
            "validation_file": ROOT_DIR / "data" / "processed" / "validation.parquet",
            "test_file": ROOT_DIR / "data" / "processed" / "test.parquet",
            "id_maps_file": ROOT_DIR / "data" / "processed" / "id_maps.pkl",
        },
        
        "k_core": 5,
        "min_seq_len": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "pad_token_id": 0,
        
        # =================================================================
        # 2. 模型超参数配置 (Model Hyperparameters)
        # =================================================================
        "encoder_model": {
            "max_len": 50,
            "embedding_dim": 64,
            "linear_hidden_dim": 16, # dv
            "attention_dim": 16,     # dqk
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.1,
            # item_num 和 pad_token_id 在训练脚本中动态传入
        },

        "decoder_model": {
            "max_seq_len": 50, # 解码器也需要知道最大长度
            "embedding_dim": 64, # 维度通常与编码器保持一致
            "num_layers": 4,     # 解码器层数，可以与编码器不同
            "num_heads": 4,
            "ffn_hidden_dim": 64 * 4, # 前馈网络隐藏层维度，通常是embedding_dim的4倍
            "dropout_ratio": 0.1,
             # num_items 在训练脚本中动态传入
        },

        # =================================================================
        # 3. 训练阶段配置 (Training Phase Config)
        # =================================================================
        
        # --- 阶段一: 编码器预训练配置 ---
        "pretrain": {
            "log_file": "pretrain_encoder.log",
            "num_epochs": 501,         
            "batch_size": 256,         
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "early_stopping_patience": 20,
            "num_workers": 10,
            "num_neg_samples": 512, # 负采样数量
            "temperature": 0.05,    # Sampled Softmax温度
        },
        
        # --- 阶段二: Encoder-Decoder 微调配置 ---
        "finetune": {
            "log_file": "finetune_genius_rec.log",
            "num_epochs": 50, # 微调通常不需要太多轮次
            "batch_size": 64, # 由于模型更大，可能需要减小batch_size
            "learning_rate": {
                "decoder_lr": 1e-3, # 解码器使用较大的学习率
                "encoder_lr": 5e-6, # 编码器使用更小的学习率
            },
            "warmup_steps": 1000, # 学习率预热步数
            "weight_decay": 0.01,
            "early_stopping_patience": 5,
            "num_workers": 10,
            "split_ratio": 0.5, # 数据集分割比例
        },
        
        # =================================================================
        # 4. 评估参数 (Evaluation Config)
        # =================================================================
        "evaluation": {
            "top_k": 10,
        }
    }
    
    return config