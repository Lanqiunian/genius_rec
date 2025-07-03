# src/GeniusRec.py

import torch
import torch.nn as nn
# 确保导入正确的类
from src.encoder.encoder import Hstu      
from src.decoder.decoder import GenerativeDecoder 

class GENIUSRecModel(nn.Module):
    def __init__(self, encoder_config, decoder_config, expert_config=None):
        """
        初始化GENIUS-Rec模型
        
        Args:
            encoder_config: 编码器配置字典
            decoder_config: 解码器配置字典
            expert_config: 专家系统配置字典
        """
        super().__init__()
        
        # 初始化编码器
        self.encoder = Hstu(**encoder_config)
        
        # 将专家配置传递给解码器
        decoder_config_with_expert = decoder_config.copy()
        if expert_config:
            decoder_config_with_expert['expert_config'] = expert_config
            
        self.decoder = GenerativeDecoder(**decoder_config_with_expert)
        
        # 🔧 修复：添加维度匹配检查和投影层
        encoder_dim = encoder_config['embedding_dim']
        decoder_dim = decoder_config['embedding_dim']
        
        if encoder_dim != decoder_dim:
            print(f"⚠️  编码器维度({encoder_dim}) != 解码器维度({decoder_dim})，添加投影层")
            self.encoder_projection = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.encoder_projection = None

    def forward(self, source_ids, target_ids, source_padding_mask, 
                force_equal_weights: bool = False, 
                return_weights: bool = False):
        
        encoder_output = self.encoder(source_ids)
        
        # 🔧 修复：处理维度不匹配
        if self.encoder_projection is not None:
            encoder_output = self.encoder_projection(encoder_output)
        
        # 将参数一路传递给解码器
        return self.decoder(
            target_ids=target_ids, 
            encoder_output=encoder_output, 
            memory_padding_mask=source_padding_mask,
            force_equal_weights=force_equal_weights, 
            return_weights=return_weights
        )