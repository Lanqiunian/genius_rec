# In src/seq2seq_model.py
import torch.nn as nn
from src.encoder.encoder import Hstu         
from src.decoder.decoder import GenerativeDecoder 

class GENIUSRecModel(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = Hstu(**encoder_config)
        self.decoder = GenerativeDecoder(**decoder_config)

    def forward(self, source_ids, target_ids, source_padding_mask):
        """
        接收 source_padding_mask 并将其传递给解码器.
        """
        # 1. 调用编码器
        encoder_output = self.encoder(source_ids)
        
        # 2. 调用解码器, 现在传入了需要的 memory_padding_mask
        logits = self.decoder(
            target_ids=target_ids, 
            encoder_output=encoder_output, 
            memory_padding_mask=source_padding_mask
        )
        return logits