# src/train_encoder.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import logging
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- 导入我们自己项目中的模块 ---
from .dataset import HSTUDataset
from .model import Standalone_HSTU_Model
from .evaluate_encoder import evaluate_model
from .config import get_config

# Trainer 类的实现保持不变
class Trainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader):
        self.config = config
        self.device = torch.device(config['device'])
        logging.info(f"使用设备: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=config['pad_token_id']) 

        self.best_ndcg = -1.0
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.epochs_no_improve = 0
        self.patience = config.get('early_stopping_patience', 5)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"训练中 (Epoch {self.epoch+1})", leave=False)

        for batch in progress_bar:
            input_seq = batch['input_seq'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()
            scores = self.model(input_seq)
            loss = self.criterion(scores, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def train(self):
        # logging.info(f"开始训练, 配置: learning_rate={self.config['learning_rate']:.5f}, dropout={self.config['dropout']:.2f}, nhead={self.config['nhead']}")
        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            train_loss = self._train_epoch()
            
            val_metrics = evaluate_model(self.model, self.val_loader, self.device, self.config['top_k'])
            val_ndcg = val_metrics[f'NDCG@{self.config["top_k"]}']
            
            logging.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"训练损失: {train_loss:.4f} | "
                f"验证 NDCG@{self.config['top_k']}: {val_ndcg:.4f}"
            )

            if val_ndcg > self.best_ndcg:
                self.best_ndcg = val_ndcg
                self.epochs_no_improve = 0
                checkpoint_path = self.checkpoint_dir / "best_model.ckpt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'best_ndcg': self.best_ndcg,
                    'config': self.config
                }, checkpoint_path)
                logging.info(f"找到更优模型 (NDCG={val_ndcg:.4f})，已保存。")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logging.info(f"验证集性能连续 {self.patience} 轮未提升，提前停止训练。")
                    break 

        logging.info(f"训练完成！最佳验证NDCG: {self.best_ndcg:.4f}")
        return self.best_ndcg

# --- 核心修复点在这里 ---
def main(config_override=None):
    # 1. 获取配置
    # 如果有外部传入的配置（来自调优脚本），则使用它
    if config_override:
        config = config_override
    # 否则，加载默认配置
    else:
        config = get_config()
    
    # 设置日志
    log_file_path = Path(config['log_file'])
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    # 2. 加载预处理好的数据
    logging.info(f"从 {config['processed_data_path']} 加载数据...")
    with open(config['processed_data_path'], 'rb') as f:
        data = pickle.load(f)
    
    user_sequences = data['user_sequences']
    
    # 3. 确定词汇表大小
    model_vocab_size = data['num_items'] + 1
    # 将真实的物品数量（不含padding）保存到config，方便后续使用
    config['num_items'] = data['num_items']

    # 4. 划分数据集
    logging.info("正在划分训练、验证、测试集...")
    train_val_sequences, test_sequences = train_test_split(user_sequences, test_size=0.1, random_state=42)
    train_sequences, val_sequences = train_test_split(train_val_sequences, test_size=1/9, random_state=42)
    
    # 5. 创建Dataset和DataLoader
    logging.info("正在创建DataLoaders...")
    train_dataset = HSTUDataset(train_sequences, config)
    val_dataset = HSTUDataset(val_sequences, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # 6. 实例化模型
    model = Standalone_HSTU_Model(model_vocab_size, config)
    
    # 7. 实例化并启动Trainer
    # 我们在调优时不需要测试集，所以最后一个参数传None
    trainer = Trainer(config, model, train_loader, val_loader, None)
    best_ndcg = trainer.train()
    return best_ndcg

if __name__ == '__main__':
    main()