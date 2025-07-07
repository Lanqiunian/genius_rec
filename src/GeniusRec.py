# src/GeniusRec.py (最终修复版)

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
        
        encoder_dim = self.encoder.item_embedding.embedding_dim
        decoder_dim = self.decoder.embedding_dim
        if encoder_dim != decoder_dim:
            self.encoder_projection = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.encoder_projection = None

    def forward(self, source_ids, decoder_input_ids, source_padding_mask, **kwargs):
        """
        【最终修复版】模型的前向传播。
        严格区分source和target，彻底杜绝数据泄露。
        """
        encoder_output = self.encoder(source_ids)
        
        if self.encoder_projection is not None:
            encoder_output = self.encoder_projection(encoder_output)
        
        return self.decoder(
            target_ids=decoder_input_ids,
            encoder_output=encoder_output,
            memory_padding_mask=source_padding_mask,
            **kwargs
        )