import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model, dropout_res=0.1):
        super().__init__()

        # Multi-layer Perceptron (Same as Vaswani et al. 2017)
        #   transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
        #   transformer.h.0.mlp.c_fc.bias torch.Size([3072])
        #   transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
        #   transformer.h.0.mlp.c_proj.bias torch.Size([768])
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout_res)
        
    def forward(self, X):
        X = self.fc(X)
        X = self.gelu(X)
        X = self.proj(X)
        return self.dropout(X)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len=1024, attn_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_size = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

        # State dict but not optimized
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0))
        
    def forward(self, X):
        b, seqlen, embed_size = X.size()

        # attention query key value
        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.d_model, dim=2)

        def reshape_multihead(v):
            return v.view(b, seqlen, self.n_head, self.head_size).transpose(1, 2)

        k = reshape_multihead(k)
        q = reshape_multihead(q)
        v = reshape_multihead(v)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Causal Mask
        attn = attn.masked_fill(self.bias[:, :, :seqlen, :seqlen] == 0, float('-inf'))

        attn = F.softmax(attn, dim=1)
        attn = self.attn_dropout(attn)

        y = attn @ v

        y = y.transpose(1, 2).reshape(b, seqlen, embed_size)

        return self.residual_dropout(self.c_proj(y))

class GPT2Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        # The weights contain the following:
        #
        # Layer Normalization
        #   transformer.h.0.ln_1.weight torch.Size([768])
        #   transformer.h.0.ln_1.bias torch.Size([768])
        # Multi-head Self-attention
        #   transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
        #   transformer.h.0.attn.c_attn.bias torch.Size([2304])
        #   transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
        #   transformer.h.0.attn.c_proj.bias torch.Size([768])
        # Layer Normalizatoin again
        #   transformer.h.0.ln_2.weight torch.Size([768])
        #   transformer.h.0.ln_2.bias torch.Size([768])
        # Multi-layer Perceptron (Same as Vaswani et al. 2017)
        #   transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
        #   transformer.h.0.mlp.c_fc.bias torch.Size([3072])
        #   transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
        #   transformer.h.0.mlp.c_proj.bias torch.Size([768])

        # Vaswani et al has LayerNorm on output of each subblock.
        # GPT-2 movies them to the inputs.
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = MultiheadSelfAttention(d_model, n_head)

        self.mlp = MLP(d_model)
        
    def forward(self, X):
        X_norm = self.ln1(X)
        X = X + self.attn(X_norm)
        X_norm = self.ln2(X)
        X = X + self.mlp(X_norm)
        return X

class GPT2(nn.Module):
    def __init__(self,
                 n_decoder_layers=12,
                 d_model=768,
                 max_content_len=1024,
                 vocab_size=50257,
                 n_head=12):

        super().__init__()

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_content_len, d_model)

        self.h = nn.ModuleList([
            GPT2Block(d_model, n_head) for _ in range(n_decoder_layers)
        ])

        # Final normalization layer
        self.ln = nn.LayerNorm(d_model)

        
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        # https://martinlwx.github.io/en/an-explanation-of-weight-tying/
        self.head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        b, seqlen = idx.size()

        # Token + Pos Embedding
        pos = torch.arange(0, seqlen, dtype=torch.long, device=self.device)

        # b, seqlen, d_model
        tok_emb = self.wte(idx)
        # seqlen, d_model
        pos_emb = self.wpe(pos)

        x = self.dropout(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln(x)
        return self.head(x)
