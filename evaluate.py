import torch
import argparse
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 导入您项目中的所有核心模块
from src.config import get_config
from src.GeniusRec import GENIUSRecModel
from src.metrics import get_metrics
from src.encoder.encoder import Hstu
from src.decoder.decoder import GenerativeDecoder
from src.dataset import Seq2SeqRecDataset

# --- 数据集类保持不变 ---
class LeaveOneOutDataset(Dataset):
    def __init__(self, data_path, max_len, pad_token_id=0):
        self.data = pd.read_parquet(data_path)
        self.max_len = max_len
        self.pad_token_id = pad_token_id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        full_seq = self.data.iloc[idx]['history']
        if len(full_seq) > self.max_len:
            full_seq = full_seq[-self.max_len:]
        input_seq = full_seq[:-1]
        ground_truth_item = full_seq[-1]
        source_ids = np.full(self.max_len - 1, self.pad_token_id, dtype=np.int64)
        if len(input_seq) > 0:
            source_ids[-len(input_seq):] = input_seq
        return {'input_ids': torch.tensor(source_ids, dtype=torch.long), 'ground_truth': torch.tensor(ground_truth_item, dtype=torch.long)}

# --- 评估函数保持不变 ---
def run_scientific_evaluation(model, config, device, batch_size):
    print("\n--- Running Evaluation Mode 1: Scientific (Leave-One-Out) ---")
    pad_token_id, top_k, max_len = config['pad_token_id'], config['evaluation']['top_k'], config['encoder_model']['max_len']
    test_dataset = LeaveOneOutDataset(config['data']['test_file'], max_len, pad_token_id=pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config['finetune']['num_workers'])
    model.eval()
    all_hr_scores, all_ndcg_scores = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Scientific Eval"):
            input_ids, ground_truth = batch['input_ids'].to(device), batch['ground_truth'].to(device)
            encoder_output = model.encoder(input_ids)
            source_padding_mask = (input_ids == pad_token_id)
            decoder_start_token = torch.full((input_ids.size(0), 1), pad_token_id, dtype=torch.long, device=device)
            batch_logits = model.decoder(decoder_start_token, encoder_output, source_padding_mask)
            pseudo_labels = ground_truth.unsqueeze(1)
            hr_k, ndcg_k = get_metrics(batch_logits, pseudo_labels, k=top_k, pad_token_id=-1)
            all_hr_scores.extend(hr_k)
            all_ndcg_scores.extend(ndcg_k)
    return np.mean(all_hr_scores), np.mean(all_ndcg_scores)

def run_training_style_evaluation(model, config, device, batch_size):
    print("\n--- Running Evaluation Mode 2: Training-Style (In-Sequence Prediction) ---")
    pad_token_id, top_k, max_len = config['pad_token_id'], config['evaluation']['top_k'], config['decoder_model']['max_seq_len']
    test_dataset = Seq2SeqRecDataset(config['data']['test_file'], max_len, pad_token_id=pad_token_id, split_ratio=config['finetune']['split_ratio'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config['finetune']['num_workers'])
    model.eval()
    all_hr_scores, all_ndcg_scores = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Training-Style Eval"):
            source_ids, decoder_input_ids, labels = batch['source_ids'].to(device), batch['decoder_input_ids'].to(device), batch['labels'].to(device)
            source_padding_mask = (source_ids == pad_token_id)
            logits = model(source_ids, decoder_input_ids, source_padding_mask, return_weights=False)
            hr_k, ndcg_k = get_metrics(logits, labels, k=top_k, pad_token_id=pad_token_id)
            all_hr_scores.extend(hr_k)
            all_ndcg_scores.extend(ndcg_k)
    return np.mean(all_hr_scores), np.mean(all_ndcg_scores)


# --- 主函数 (【已重构】: 支持为不同评估模式设置不同批次大小) ---
# --- 请将您的 evaluate.py 中的 main 函数，完整替换为这个版本 ---

def main():
    parser = argparse.ArgumentParser(description="Run a comparative evaluation for GENIUS-Rec.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the best model checkpoint file.')
    parser.add_argument('--sci_batch_size', type=int, default=512, help='Batch size for Scientific evaluation.')
    parser.add_argument('--train_style_batch_size', type=int, default=64, help='Batch size for Training-Style evaluation.')
    args = parser.parse_args()

    # --- 1. 加载配置和模型 ---
    print("--- 1. Loading Config and Model ---")
    config = get_config()
    device = torch.device(config['device'])

    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
    
    # --- 【决定性修正】 ---
    # 必须使用和训练时完全一致的 `num_items` 定义
    # 训练时，num_items = 原始物品数 + 1 (为padding)
    num_items_for_init = id_maps['num_items'] + 1
    print(f"✅ Correctly setting item vocabulary size for initialization to: {num_items_for_init}")

    # 将修正后的词典大小，应用到模型配置中
    config['encoder_model']['item_num'] = num_items_for_init
    config['decoder_model']['num_items'] = num_items_for_init
    
    # 文本嵌入的维度获取逻辑保持不变
    try:
        text_embedding_file = config['data']['data_dir'] / 'book_gemini_embeddings_filtered.npy'
        temp_dict = np.load(text_embedding_file, allow_pickle=True).item()
        config['decoder_model']['text_embedding_dim'] = next(iter(temp_dict.values())).shape[0]
    except Exception:
        config['decoder_model']['text_embedding_dim'] = 768

    # 使用修正后的配置来创建模型“骨架”
    model = GENIUSRecModel(config['encoder_model'], config['decoder_model']).to(device)

    try:
        # 现在，模型的“骨架”和存档的“灵魂”尺寸完全匹配，可以成功加载
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Model state loaded successfully from '{args.checkpoint_path}'!")
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    except Exception as e:
        print(f"❌ Error loading model state: {e}")
        return

    # --- 2. 依次执行两种评估 ---
    # 后续的评估函数调用和结果打印逻辑，在我们之前的版本中已经修正，保持不变即可
    hr_sci, ndcg_sci = run_scientific_evaluation(model, config, device, args.sci_batch_size)
    hr_train, ndcg_train = run_training_style_evaluation(model, config, device, args.train_style_batch_size)
    top_k = config['evaluation']['top_k']

    # --- 3. 打印最终对比结果 ---
    print("\n" + "="*70)
    print(" " * 20 + "EVALUATION METRICS COMPARISON")
    print("="*70)
    print(f"| {'Metric':<25} | {'Scientific (Corrected)':<25} | {'Training-Style (In-Sequence)':<25} |")
    print(f"|{'-'*27}|{'-'*27}|{'-'*27}|")
    print(f"| {'HR@' + str(top_k):<25} | {hr_sci:<25.4f} | {hr_train:<25.4f} |")
    print(f"| {'NDCG@' + str(top_k):<25} | {ndcg_sci:<25.4f} | {ndcg_train:<25.4f} |")
    print("="*70)
    print("\nℹ️  'Scientific' now reflects the true next-item prediction performance.")


if __name__ == '__main__':
    main()