import torch
import argparse
import pickle
from torch.utils.data import DataLoader

# 导入您项目中的核心模块
from src.config import get_config
from src.dataset import Seq2SeqRecDataset
from src.GeniusRec import GENIUSRecModel
# 1. 最大长度

# 循环次数达到上限
# 保证程序一定会停，是最后的底线。

# 2. 重复停止

# 同一个物品ID连续出现N次
# 防止陷入无意义的循环，提高生成序列的质量。这是当前最主要的启发式停止策略。

# 3. 填充符停止

# 模型主动生成pad_token_id
# 理想中的模型“自然”停止方式，但在当前训练设置下基本不起作用。

def generate_sequence(model, source_ids, max_len, pad_token_id, device):
    """
    【已优化】使用贪心策略自回归地生成序列。
    新增了重复检测机制来提前停止。
    """
    model.eval()
    
    encoder_output = model.encoder(source_ids)
    source_padding_mask = (source_ids == pad_token_id)
    decoder_input_ids = torch.full((1, 1), pad_token_id, dtype=torch.long, device=device)
    
    # 用于检测重复的变量
    last_generated_id = -1
    repetition_count = 0
    REPETITION_THRESHOLD = 3 # 如果同一个ID连续出现3次，就停止

    for _ in range(max_len - 1):
        logits = model.decoder(decoder_input_ids, encoder_output, source_padding_mask)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        
        # 检查是否生成了PAD token (虽然可能性小，但保留)
        if next_token_id.item() == pad_token_id:
            break
            
        # 【核心优化】检查并更新重复计数
        if next_token_id.item() == last_generated_id:
            repetition_count += 1
        else:
            repetition_count = 1 # 重置计数
        last_generated_id = next_token_id.item()
        
        # 如果达到重复阈值，则停止
        if repetition_count >= REPETITION_THRESHOLD:
            # print(f" (Stopping due to {REPETITION_THRESHOLD} repetitions of token {last_generated_id})") # for debugging
            break

        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
            
    return decoder_input_ids.squeeze(0)[1:].tolist()


def main():
    parser = argparse.ArgumentParser(description="Visualize GENIUS-Rec model predictions.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (e.g., checkpoints/checkpoints_genius_rec/genius_rec_best.pth).')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize from the validation set.')
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    print("--- 1. Loading Config ---")
    config = get_config()
    device = torch.device(config['device'])
    pad_token_id = config['pad_token_id']
    
    split_point = int(config['decoder_model']['max_seq_len'] * config['finetune']['split_ratio'])
    decoder_len = config['decoder_model']['max_seq_len'] - split_point

    # --- 2. 加载验证数据集 ---
    print("--- 2. Loading Validation Dataset ---")
    with open(config['data']['id_maps_file'], 'rb') as f:
        id_maps = pickle.load(f)
        num_items = id_maps['num_items'] + 1

    val_dataset = Seq2SeqRecDataset(
        config['data']['validation_file'],
        config['decoder_model']['max_seq_len'],
        pad_token_id=pad_token_id,
        split_ratio=config['finetune']['split_ratio']
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    print(f"Validation set loaded. Size: {len(val_dataset)}")

    # --- 3. 初始化模型并加载检查点 ---
    print(f"--- 3. Loading Model from: {args.checkpoint_path} ---")
    encoder_config = {**config['encoder_model'], 'item_num': num_items, 'pad_token_id': pad_token_id}
    decoder_config = {**config['decoder_model'], 'num_items': num_items, 'pad_token_id': pad_token_id}
    model = GENIUSRecModel(encoder_config=encoder_config, decoder_config=decoder_config).to(device)
    
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # --- 4. 进行预测并可视化 ---
    print("\n" + "="*50)
    print(" " * 15 + "VISUALIZING PREDICTIONS")
    print("="*50 + "\n")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_samples:
                break

            source_ids = batch['source_ids'].to(device)
            labels = batch['labels'].squeeze(0)

            predicted_ids = generate_sequence(
                model, 
                source_ids, 
                max_len=decoder_len, 
                pad_token_id=pad_token_id, 
                device=device
            )

            history_ids = [pid for pid in source_ids.squeeze(0).tolist() if pid != pad_token_id]
            actual_ids = [pid for pid in labels.tolist() if pid != pad_token_id]

            print(f"--- Sample #{i+1} ---")
            print(f"👤 用户交互历史 (输入编码器部分):\n   {history_ids}\n")
            print(f"🤖 模型预测结果 (生成的交互序列):\n   {predicted_ids}\n")
            print(f"✅ 实际交互历史:\n   {actual_ids}\n")
            print("-" * 25)

if __name__ == '__main__':
    main()