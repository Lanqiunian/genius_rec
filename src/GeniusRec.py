# src/GeniusRec.py

import torch.nn as nn
# 确保导入正确的类
from src.encoder.encoder import Hstu         
from src.decoder.decoder import GenerativeDecoder 

class GENIUSRecModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, expert_config=None):
        super().__init__()
        self.encoder = Hstu(**encoder_config)
        
        # 将专家配置传递给解码器
        decoder_config_with_expert = decoder_config.copy()
        if expert_config:
            decoder_config_with_expert['expert_config'] = expert_config
            
        self.decoder = GenerativeDecoder(**decoder_config_with_expert)

    # 【关键修正】
    def forward(self, source_ids, target_ids, source_padding_mask, return_weights: bool = False):
        encoder_output = self.encoder(source_ids)
        
        return self.decoder(
            target_ids=target_ids, 
            encoder_output=encoder_output, 
            memory_padding_mask=source_padding_mask,
            return_weights=return_weights
        )