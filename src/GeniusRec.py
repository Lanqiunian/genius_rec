import torch
import torch.nn as nn
from src.encoder.encoder import Hstu
from src.decoder.decoder import GenerativeDecoder

class GENIUSRecModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, expert_config=None):
        """
        【最终版】初始化GENIUS-Rec模型
        """
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

    def forward(self, source_ids, target_ids, source_padding_mask, return_weights: bool = False):
        """
        【最终版】模型的前向传播。
        拥有一个简洁的接口，正确传递参数和返回值。
        """
        encoder_output = self.encoder(source_ids)
        
        if self.encoder_projection is not None:
            encoder_output = self.encoder_projection(encoder_output)
        
        # 直接调用解码器，并返回其所有的输出
        # 解码器会返回 (final_logits, weights, balancing_loss)
        return self.decoder(
            target_ids=target_ids,
            encoder_output=encoder_output,
            memory_padding_mask=source_padding_mask,
            return_weights=return_weights
        )