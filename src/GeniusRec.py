# src/GeniusRec.py

import torch.nn as nn
# 确保导入正确的类
from src.encoder.encoder import Hstu      
from src.decoder.decoder import GenerativeDecoder 

class GENIUSRecModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, expert_config=None):
        super().__init__()
        self.encoder = Hstu(**encoder_config)
        
        # 将专家配置传递给解码器 (这部分逻辑保持不变)
        decoder_config_with_expert = decoder_config.copy()
        if expert_config:
            decoder_config_with_expert['expert_config'] = expert_config
            
        self.decoder = GenerativeDecoder(**decoder_config_with_expert)

    # ==================== ✨ 关键修正 ✨ ====================
    # 在 forward 方法签名中添加 force_equal_weights 参数
    def forward(self, source_ids, target_ids, source_padding_mask, 
                force_equal_weights: bool = False, 
                return_weights: bool = False):
        
        encoder_output = self.encoder(source_ids)
        
        # 将参数一路传递给解码器
        return self.decoder(
            target_ids=target_ids, 
            encoder_output=encoder_output, 
            memory_padding_mask=source_padding_mask,
            force_equal_weights=force_equal_weights, 
            return_weights=return_weights
        )