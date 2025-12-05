import torch
import torch.nn

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # Multi-layer Perceptron (Same as Vaswani et al. 2017)
        #   transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
        #   transformer.h.0.mlp.c_fc.bias torch.Size([3072])
        #   transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
        #   transformer.h.0.mlp.c_proj.bias torch.Size([768])
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(d_model * 4, d_model)
    def forward(self, X):
        X = self.fc(X)
        X = self.gelu(x)
        X = self.proj(x)
        return X

class GPT2Block(nn.Module):
    def __init__(self, d_model):
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

        self.attn = MultiheadSelfAttention()

        self.mlp = MLP(nn.Module)
        
    def forward(self, X):
        X_norm = self.ln1(X)
        X = X + self.attn(X_norm)
        X_norm = self.ln2(X)
        X = X + self.mlp(X_norm)
        return x

class GPT2(nn.Module):
    def __init__(self,
                 n_decoder_layers=12,
                 d_model=768,
                 max_content_len=1024,
                 vocab_size=50257):
        super().__init__()

        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(max_content_len, d_model)

        
    
        
