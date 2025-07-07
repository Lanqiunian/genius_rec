# ==============================================================================
# 修改 src/GeniusRec.py 中的 forward 方法
# ==============================================================================
import torch
import torch.nn as nn
from src.encoder.encoder import Hstu
from src.decoder.decoder import GenerativeDecoder

class GENIUSRecModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, expert_config=None):
        super().__init__()
        
        self.encoder = Hstu(**encoder_config)
        
        decoder_config_with_expert = decoder_config.copy()
        if expert_config:
            decoder_config_with_expert['expert_config'] = expert_config
            
        self.decoder = GenerativeDecoder(**decoder_config_with_expert)
        
        encoder_dim = encoder_config['embedding_dim']
        decoder_dim = decoder_config['embedding_dim']
        
        if encoder_dim != decoder_dim:
            print(f"⚠️  编码器维度({encoder_dim}) != 解码器维度({decoder_dim})，添加投影层")
            self.encoder_projection = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.encoder_projection = None

    def forward(self, input_ids, padding_mask, return_weights: bool = False):
        """
        【最终重构版】模型的前向传播。
        统一使用 input_ids 作为编码器和解码器的输入。
        """
        # 编码器对完整输入序列进行编码，获取全局上下文
        encoder_output = self.encoder(input_ids)
        
        if self.encoder_projection is not None:
            encoder_output = self.encoder_projection(encoder_output)
        
        # 解码器接收相同的输入序列(target_ids=input_ids)，
        # 并利用编码器的输出作为记忆（memory），同时处理padding_mask
        return self.decoder(
            target_ids=input_ids,
            encoder_output=encoder_output,
            memory_padding_mask=padding_mask,
            return_weights=return_weights
        )