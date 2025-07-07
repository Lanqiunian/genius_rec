# src/config.py (重构版)

import torch
from pathlib import Path

def get_config():

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
        "sos_token_id": 1,
        "eos_token_id": 2,
        "mask_token_id": 3,
        "num_special_tokens": 4,
        
        # =================================================================
        # 2. 模型超参数配置 (Model Hyperparameters)
        # =================================================================
        "encoder_model": {
            "max_len": 50,
            "embedding_dim": 64,         # 建议: 128 或 256
            "linear_hidden_dim": 16,     # dv, 建议: 32 或 64
            "attention_dim": 16,         # dqk, 建议: 32 或 64
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.1,
        },

        "decoder_model": {
            "max_seq_len": 50,
            "embedding_dim": 64,         # 建议: 128 或 256
            "num_layers": 4,
            "num_heads": 4,
            "ffn_hidden_dim": 64 * 4,    # 建议随 embedding_dim 调整, e.g., 128 * 4
            "dropout_ratio": 0.2,        # 建议: 0.1 或 0.2
        },

        # =================================================================
        # 3. 训练阶段配置 (Training Phase Config)
        # =================================================================
        
        # --- 阶段一: 编码器预训练配置 ---
        "pretrain": {
            "log_file": "pretrain_encoder.log",
            "num_epochs": 501,         
            "batch_size": 256,         
            "learning_rate": 1e-3,
            "weight_decay": 0.1,
            "early_stopping_patience": 20,
            "num_workers": 10,
            "num_neg_samples": 512, # 负采样数量
            "temperature": 0.05,    # Sampled Softmax温度
        },
        
        # --- 阶段二: Encoder-Decoder 微调配置 ---
        "finetune": {
            "log_file": "finetune_genius_rec.log",
            "num_epochs": 50,
            "batch_size": 16,
            "learning_rate": {
                "decoder_lr": 1e-4,  # 解码器学习率, 建议: 3e-4 或 1e-4
                "encoder_lr": 5e-6,  # 保持不变，用于精调
                "embedding_lr": 1e-4, # 嵌入层学习率, 建议: 1e-4 或 3e-4
                "gate_lr": 1e-4      # 门控网络学习率
            },
            "balancing_loss_alpha": 0.1, # 负载均衡损失的系数, 建议: 0.01 或 0.05
            "label_smoothing": 0,
            "warmup_steps": 1000,
            "weight_decay": 0.01,    
            "early_stopping_patience": 4,
            "num_workers": 10,

            "use_stochastic_length": False,
            "stochastic_threshold": 20,
            "stochastic_prob": 0.5,
              
        },
                
        # =================================================================
        # 4. 评估参数 (Evaluation Config)
        # =================================================================
        "evaluation": {
            "top_k": 10,
        },
        
        # =================================================================
        # 5. 专家系统配置 (Expert System Config) 
        # =================================================================
        "expert_system": {
            # 专家启用开关
            "experts": {
                "behavior_expert": True,     # 行为专家（基于用户序列行为）
                "content_expert": True,      # 内容专家（基于文本嵌入）
                "image_expert": True,        # 图像专家（基于书封面）🎨 启用视觉专家！
            },
            
            # 门控网络配置
            "gate_config": {
                "gate_type": "mlp",       # 门控类型：'simple'(原始), 'mlp'(新增)
                "gate_hidden_dim": 64,       # MLP门控的隐藏层维度（仅gate_type='mlp'时使用）
                "temperature": 1.0,          # softmax温度参数（预留）
                "noise_epsilon": 0.8,         # 门控网络噪声参数，用于对抗专家极化
            },
            
            # 内容专家配置
            "content_expert": {
                "attention_heads": 4,        # 交叉注意力头数
                "use_cross_attention": True, # 是否使用交叉注意力
                "text_projection_type": "mlp",     # 文本投影类型：'simple'（原始）, 'mlp'
                "text_embedding_dim": 768,   # 文本嵌入维度
                "trainable_embeddings": False 
            },
            
            # 图像专家配置
            "image_expert": {
                "attention_heads": 4,        # 交叉注意力头数  
                "use_cross_attention": True, # 是否使用交叉注意力
                "image_embedding_dim": 512,  # 图像嵌入维度（CLIP ViT-B/32）
                "image_encoder": "clip",     # 图像编码器类型
                "use_adaptive_pooling": True, # 使用自适应池化适配不同维度
                "image_projection_type": "mlp", # 图像投影类型：'simple', 'mlp'
                "trainable_embeddings": False
            }
        }
    }
    
    # 🔧 新增：配置验证
    special_ids = [config['pad_token_id'], config['sos_token_id'], 
                   config['eos_token_id'], config['mask_token_id']]
    if len(set(special_ids)) != len(special_ids):
        raise ValueError("Special token IDs must be unique!")
    if config['pad_token_id'] != 0:
        raise ValueError("pad_token_id must be 0 for PyTorch compatibility!")
    
    return config