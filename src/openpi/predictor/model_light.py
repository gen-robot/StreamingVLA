from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiTPredictor(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 1024, emb_len: int = 256, action_dim: int = 6,
                 n_layers: int = 15, n_heads: int = 16, dim_feedforward: int = 3072, 
                 dropout: float = 0.1, cond_dim: int = 256):
        super().__init__()
        
        self.in_dim = in_dim        
        self.hidden_dim = hidden_dim 
        self.emb_len = emb_len
        self.n_layers = n_layers
        self.cond_dim = cond_dim

        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )

        
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, cond_dim * n_layers),
            nn.ReLU(inplace=True)
        )

        
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(emb_c=hidden_dim, n_heads=n_heads, dim_feedforward=dim_feedforward,
                                        dropout=dropout, cond_dim=cond_dim)
            for _ in range(n_layers)
        ])

        
        self.out_proj = nn.Linear(hidden_dim, in_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, image1_emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        image1_emb: (B, 256, 2048)
        action: (B, 6)
        """
        B, L, C = image1_emb.shape
        x = self.input_proj(image1_emb) 

        conds = self.action_proj(action) 
        conds = conds.view(B, self.n_layers, self.cond_dim)

        for i, block in enumerate(self.blocks):
            cond_i = conds[:, i, :]
            x = block(x, cond_i) 

        
        return self.out_proj(x) 

class ConditionalLayerNorm(nn.Module):
    def __init__(self, normalized_shape, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.cond_to_affine = nn.Sequential(
            nn.Linear(cond_dim, normalized_shape * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x_norm = self.norm(x)
        affine = self.cond_to_affine(cond) 
        affine = affine.view(B, 2, C) 
        gamma = affine[:, 0, :].unsqueeze(1) 
        beta = affine[:, 1, :].unsqueeze(1) 
        return x_norm * (1.0 + gamma) + beta

class ConditionalTransformerBlock(nn.Module):
    def __init__(self, emb_c: int, n_heads: int, dim_feedforward: int, dropout: float, cond_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.emb_c = emb_c
        
        self.qkv_proj = nn.Linear(emb_c, emb_c * 3)
        self.out_proj = nn.Linear(emb_c, emb_c)
        
        self.ln1 = ConditionalLayerNorm(emb_c, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_c, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, emb_c)
        )
        self.ln2 = ConditionalLayerNorm(emb_c, cond_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        res = x
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, L, C)
        attn_out = self.out_proj(attn_out)
        
        x = self.ln1(res + self.dropout(attn_out), cond)
        x = self.ln2(x + self.dropout(self.mlp(x)), cond)
        return x