# tune_hyperparams.py

import itertools
import logging
from pathlib import Path
import copy

# 导入我们改造过的main函数和基础配置
from src.config import get_config
from src.train_encoder import main as train_main

def tune():
    # --- 1. 定义超参数的搜索空间 ---
    # 我们选择对训练速度影响最小，但对性能影响最大的参数
    param_grid = {
        'learning_rate': [5e-4, 1e-3, 2e-3],
        'dropout': [0.1, 0.2, 0.3],
        'nhead': [2, 4, 8]
    }

    # 获取所有参数组合
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"开始超参数调优，共计 {len(experiments)} 组实验。")

    best_score = -1
    best_params = {}
    
    # --- 2. 循环执行每一组实验 ---
    for i, params in enumerate(experiments):
        logging.info(f"\n{'='*20} 实验 {i+1}/{len(experiments)} {'='*20}")
        
        # 加载基础配置
        config = get_config()
        
        # 更新当前实验的参数
        config.update(params)
        
        # 为每次实验创建独立的日志和模型保存路径
        run_name = f"lr_{params['learning_rate']}_do_{params['dropout']}_nh_{params['nhead']}"
        config['checkpoint_dir'] = Path(config['checkpoint_dir']).parent / f"tune_{run_name}"
        config['log_file'] = Path(config['log_file']).parent / f"tune_{run_name}.log"

        # 创建一个深拷贝的配置，避免互相影响
        current_config = copy.deepcopy(config)

        try:
            # 调用训练主函数
            score = train_main(current_config)
            
            # 记录最佳结果
            if score > best_score:
                best_score = score
                best_params = params
                logging.info(f"*** 新的最佳分数: {best_score:.4f}，参数: {best_params} ***")
        except Exception as e:
            logging.error(f"实验失败，参数: {params}，错误: {e}")

    logging.info(f"\n{'='*20} 调优完成 {'='*20}")
    logging.info(f"最佳验证集 NDCG 分数: {best_score:.4f}")
    logging.info(f"对应的最佳超参数: {best_params}")

if __name__ == '__main__':
    tune()