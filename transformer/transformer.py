import torch 
import torch.nn as nn 
import torch.nn.functional as F

"""
In this code, we assume the format of hidden dimension and  
the embedding dimension is like [1, seq_len, hidden_dim].
ref:1. https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/
    2. Attention Is All You Need.
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros((max_len, d_model))
        """
        for i in range(max_len):
            coeff = 2 * torch.arange(0, d_model//2).float() / d_model
            pe[i, 0::2] = torch.sin(i / torch.pow(10000, coeff))
            pe[i, 1::2] = torch.cos(i / torch.pow(10000, coeff))
        """
        coeff = 2 * torch.arange(0, d_model//2).float() / d_model
        inv = torch.arange(0, max_len).float().unsqueeze(1)
        pe[:, 0::2] = torch.sin(inv / torch.pow(10000, coeff).unsqueeze(0))
        pe[:, 1::2] = torch.cos(inv / torch.pow(10000, coeff).unsqueeze(0))
        pe = pe.unsqueeze(0) # shape [1, seq, dim]
        self.weight = nn.Parameter(pe, requires_grad=False)
    
    """
    input x dim is : [1, seq_len, hidden_dim]
    """
    def forward(self, x):
        return self.weight[:, :x.size(1), :]

class ScaledDotAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(2)
        scaled = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(d_k)
        scaled = torch.exp(scaled)
        """
        mask is a upper matrix
        """
        if mask:
            scaled = scaled.masked_fill(mask, 0)
        scaled = scaled / scaled.sum(-1, keepdim=True)
        scaled = self.dropout(scaled)
        output = torch.matmul(scaled, v)
        return output

class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, dropout=0.1):
        super().__init__() 
        self.d_model = d_model
        self.d_k = d_k
        self.q_linear = nn.Linear(d_model, d_k)
        self.k_linear = nn.Linear(d_model, d_k)
        self.v_linear = nn.Linear(d_model, d_k)
        self.attn = ScaledDotAttention(dropout)

    def forward(self, q, k, v, mask=None):
        q_ = self.q_linear(q)
        k_ = self.k_linear(k)
        v_ = self.v_linear(v)
        output = self.attn(q_, k_, v_, mask)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, nheads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k 
        self.nheads = nheads
        self.attHeads = nn.ModuleList([
            AttentionHead(d_model, d_k, dropout) for _ in range(nheads)
        ])
        self.linear = nn.Linear(d_k*nheads, d_model)

    def forward(self, q, k, v, mask=None):
        heads = []
        for m in self.attHeads:
            heads.append(m(q, k, v, mask))
        heads = torch.cat(heads, dim=-1) 
        output = self.linear(heads)
        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_ff, nheads=8, dropout=0.1):
        super().__init__()
        self.multiAttn = MultiHeadAttention(d_model, d_k, nheads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.pos_fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, input):
        x = input 
        attn = self.multiAttn(input, input, input)
        x = x + self.dropout1(attn)
        x = self.layernorm1(x)
        res = x 
        x = self.pos_fc(x)
        x = res + self.dropout2(x)
        x = self.layernorm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_k, d_ff, nblocks=6, nheads=8, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(d_model, d_k, d_ff, nheads, dropout) \
                for _ in range(nblocks)
        ])

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_ff, nheads=8, dropout=0.1):
        super().__init__()
        self.multiAttn1 = MultiHeadAttention(d_model, d_k, nheads, dropout)
        self.multiAttn2 = MultiHeadAttention(d_model, d_k, nheads, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.pos_fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, input, encoder_output, tgt_mask=None):
        res = input 
        x = self.multiAttn1(input, input, input, mask=tgt_mask)
        x = res + self.dropout1(x)
        x = self.layernorm1(x)
        res = x
        x = self.multiAttn2(q=x, k=encoder_output, v=encoder_output)
        x = res + self.dropout2(x)
        x = self.layernorm2(x)
        res = x
        x = self.pos_fc(x)
        x = res + self.dropout3(x)
        x = self.layernorm3(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, d_k, d_ff, nblocks=6, nheads=8, dropout=0.1):
        super().__init__()
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(d_model, d_k, d_ff, nheads, dropout) \
                for _ in range(nblocks)
        ])
    
    def forward(self, x, encoder_output, tgt_mask):
        for decoder in self.decoders:
            x = decoder(x, encoder_output, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_ff=2048, n_heads=8, nblocks=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nheads = n_heads
        self.nblocks = nblocks
        self.encoder = TransformerEncoder(d_model, d_k, d_ff, nblocks, nheads, dropout)
        self.decoder = TransformerDecoder(d_model, d_k, d_ff, nblocks, nheads, dropout)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def generate_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 1
        return mask
    
    def forward(self, src, tgt):
        x = src + self.pos_encoding(src)
        x = self.dropout1(x)
        enc_output = self.encoder(x)
        tgt_seq = tgt.size(1)
        tgt_mask = self.generate_mask(tgt_seq)
        y = tgt + self.pos_encoding(tgt)
        y = self.dropout2(y)
        y = self.decoder(y, enc_output, tgt_mask)
        return y