# src/config.py
import torch
from pathlib import Path

def get_config():
    """
    返回一个包含所有项目配置的字典。
    """
    
    # 获取项目根目录
    ROOT_DIR = Path(__file__).parent.parent 
    
    config = {
        # --- 1. 文件与路径配置 ---
        "data_dir": ROOT_DIR / "data",
        "review_file": ROOT_DIR / "data" / "Books.jsonl.gz", # 原始评论数据
        "meta_file": ROOT_DIR / "data" / "meta_Books.jsonl.gz", # 原始元数据
        
        "processed_data_dir": ROOT_DIR / "data" / "processed",
        "processed_data_path": ROOT_DIR / "data" / "processed" / "processed_data.pkl", # 预处理后数据的保存路径
        
        "checkpoint_dir": ROOT_DIR / "checkpoints" / "hstu_encoder_phase1", # 模型checkpoint的保存目录
        "log_file": ROOT_DIR / "encoder_training.log", # 日志文件
        
        # --- 2. 数据预处理配置 ---
        "k_core": 10,  # K-core过滤的阈值，过滤掉交互少于10次的用户和物品

        # --- 3. 模型超参数 ---
        "embedding_dim": 128,      # 物品嵌入向量的维度
        "max_seq_len": 200,        # 输入给模型的序列最大长度
        "num_encoder_layers": 4,   # HSTU编码器中Transformer层的数量
        "nhead": 4,                # Transformer中的多头注意力头数 (必须能被embedding_dim整除)
        "dim_feedforward": 512,    # Transformer中前馈网络的隐藏层维度
        "dropout": 0.1,            # Dropout的比率
        
        # --- 4. 训练超参数 ---
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs": 50,          # 训练的总轮数
        "batch_size": 256,         # 批处理大小
        "learning_rate": 0.001,    # 学习率
        "weight_decay": 0.0,       # 权重衰减 (L2正则化)
        "num_workers": 4,          # DataLoader使用的工作进程数，可以加快数据加载
        "early_stopping_patience": 5,  # 新增：如果验证集NDCG连续5轮没有提升，则提前停止训练

        # --- 5. 评估参数 ---
        "top_k": 10,               # 计算Recall@K和NDCG@K时的K值

        # --- 6. 特殊Token ID (这些值将在train脚本中被动态填充) ---
        "num_items": None,         # 将在加载预处理数据后填充
        "sep_token_id": None,      # 将在加载预处理数据后填充
        "pad_token_id": 0          # 我们约定0为padding token
    }
    
    return config