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

def main():
    # 1. 获取配置
    config = get_config()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config['log_file']),
            logging.StreamHandler()
        ]
    )

    # 2. 加载预处理好的数据
    logging.info(f"从 {config['processed_data_path']} 加载数据...")
    try:
        with open(config['processed_data_path'], 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"错误: 未找到预处理数据文件 {config['processed_data_path']}。请先运行 preprocess.py。")
        return

    user_sequences = data['user_sequences']
    
    # 3. 动态填充配置项
    # 我们的物品ID从1开始，0是padding，所以总的物品数是 data['num_items']
    # 嵌入层的总大小需要是 data['num_items'] + 1
    config['num_items'] = data['num_items'] + 1
    model_vocab_size = config['num_items']

    # 4. 划分数据集
    logging.info("正在划分训练、验证、测试集...")
    train_val_sequences, test_sequences = train_test_split(
        user_sequences, test_size=0.1, random_state=42
    )
    train_sequences, val_sequences = train_test_split(
        train_val_sequences, test_size=1/9, random_state=42
    )
    logging.info(f"数据集划分完成: 训练集 {len(train_sequences)}, 验证集 {len(val_sequences)}, 测试集 {len(test_sequences)}")

    # 5. 创建Dataset和DataLoader
    logging.info("正在创建DataLoaders...")
    train_dataset = HSTUDataset(train_sequences, config)
    val_dataset = HSTUDataset(val_sequences, config)
    test_dataset = HSTUDataset(test_sequences, config)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # 6. 实例化模型
    # --- 核心修正点：调用现在是匹配的 ---
    model = Standalone_HSTU_Model(model_vocab_size, config)
    
    # 7. 实例化并启动Trainer
    trainer = Trainer(config, model, train_loader, val_loader, test_loader)
    trainer.train()

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
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def train(self):
        logging.info("开始训练 HSTU 编码器...")
        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            train_loss = self._train_epoch()
            
            val_metrics = evaluate_model(self.model, self.val_loader, self.device, self.config['top_k'])
            val_recall = val_metrics[f'Recall@{self.config["top_k"]}']
            val_ndcg = val_metrics[f'NDCG@{self.config["top_k"]}']
            
            elapsed_time = time.time() - start_time
            
            logging.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"训练损失: {train_loss:.4f} | "
                f"验证 Recall@{self.config['top_k']}: {val_recall:.4f} | "
                f"验证 NDCG@{self.config['top_k']}: {val_ndcg:.4f} | "
                f"用时: {elapsed_time:.2f}s"
            )

            if val_ndcg > self.best_ndcg:
                self.best_ndcg = val_ndcg
                checkpoint_path = self.checkpoint_dir / "best_model.ckpt"
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.hstu_encoder.state_dict(),
                    'best_ndcg': self.best_ndcg,
                    'config': self.config
                }, checkpoint_path)
                
                logging.info(f"在Epoch {epoch+1} 找到更优模型 (NDCG={val_ndcg:.4f})，已保存到 {checkpoint_path}")

        logging.info("训练完成！")
        
        logging.info("在测试集上进行最终评估...")
        best_model_path = self.checkpoint_dir / "best_model.ckpt"
        if best_model_path.exists():
            final_model = Standalone_HSTU_Model(
                self.config['num_items'],
                self.config
            ).to(self.device)
            checkpoint = torch.load(best_model_path)
            final_model.hstu_encoder.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"已加载最佳模型: {best_model_path}")

            test_metrics = evaluate_model(final_model, self.test_loader, self.device, self.config['top_k'])
            logging.info(f"最终测试集评估结果: ")
            for metric, value in test_metrics.items():
                logging.info(f"  {metric}: {value:.4f}")
        else:
            logging.warning("未找到最佳模型checkpoint，无法进行最终评估。")

if __name__ == '__main__':
    main()