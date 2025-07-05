import torch
import torch.nn as nn
import math
class FM(nn.Module):
    def __init__(self,field_nums,emb_dims = 16):
        super().__init__()
        self.bias = nn.Parameter(torch.zero((1,)))
        self.linear = nn.Parameter(field_nums,1)
        self.embedding = nn.Embedding(field_nums,emb_dims)

    def forward(self,x):
        emb_x = self.embedding(x)

        square_of_sum = emb_x.sum(dim=1).pow(2).sum(dim = 1)

        sum_of_square = emb_x.pow(2).sum(dim = 1).sum(dim = 1)

        interaction_term = (square_of_sum - sum_of_square) / 2


        linear_term = self.linear(X).squeeze(-1)

        output = self.bias + linear_term + interaction_term

        return torch.sigmoid(output).unsqueeze(-1)
        


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,dropout = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads = 0,"注意"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model / num_heads

        self.W_qkv = nn.Linear(d_model,d_model*3)

        self.W_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(p = dropout)

    def scaled_dot_product_attention(self,Q,K,V, mask = None):
        attn_scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_filled(mask ==0, -1e9)
        
        attn_weight = torch.softmax(attn_scores,dim = -1)
        attn_weight = self.dropout(attn_weight)

        output = torch.matmul(attn_scores,V)

        return output, attn_weight
    
    def forward(self, x mask = None):
        batch_size,seq_len,_ = x.shape
        qkv = self.W_qkv
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, self.dk*3)

        qkv = qkv.permute(0,2,1,3)

        Q,K,V = qkv.chunk(3, dim = -1)

        context, attn_weights = self.scaled_dot_product_attention(Q,K,V,None)

        context = context.permute(0,2,1,3).contiguous()

        context = context.reshape(batch_size, seq_len, self.d_model)

        output = self.W_o(context)

        return output




    



