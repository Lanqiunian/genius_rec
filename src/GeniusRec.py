# src/GeniusRec.py (修改版)

import torch
import torch.nn as nn
from src.encoder.encoder import Hstu
from src.decoder.decoder import GenerativeDecoder # 假设新的解码器在此文件

class GENIUSRecModel(nn.Module):
    def __init__(self, config): # 简化构造函数，直接传入总配置
        super().__init__()
        self.config = config # 保存总配置
        self.encoder = Hstu(**config['encoder_model'])
        
        # 将总配置传给解码器
        self.decoder = GenerativeDecoder(config)
        
        encoder_dim = self.encoder.item_embedding.embedding_dim
        decoder_dim = self.decoder.embedding_dim
        if encoder_dim != decoder_dim:
            self.encoder_projection = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.encoder_projection = None

    def forward(self, source_ids, decoder_input_ids, source_padding_mask, target_padding_mask, **kwargs):
        """
        【Sampled Softmax 修改版】模型的前向传播。
        通过kwargs将 labels 和 negative_samples 传递给解码器。
        """
        encoder_output = self.encoder(source_ids)
        
        if self.encoder_projection is not None:
            encoder_output = self.encoder_projection(encoder_output)
        
        # 将所有额外参数都传递给解码器
        return self.decoder(
            target_ids=decoder_input_ids,
            encoder_output=encoder_output,
            memory_padding_mask=source_padding_mask,
            target_padding_mask=target_padding_mask,
            **kwargs
        )