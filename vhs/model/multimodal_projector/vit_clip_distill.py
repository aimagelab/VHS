import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class PatchEmbed(nn.Module):
    """Image to Patch Embedding via Conv2d."""
    def __init__(self, in_chans: int, patch_size: int | Tuple[int, int], embed_dim: Optional[int] = None):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        if embed_dim is None:
            embed_dim = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # (B, C, H, W) -> (B, N, D)
        x = self.proj(x)                      # (B, D, H/ps, W/ps)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x, (H, W)


class CustomTransformerLayer(nn.Module):
    """Custom transformer layer that can use external Q and K tensors."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def attention(self, x: torch.Tensor, external_q: Optional[torch.Tensor] = None, 
                  external_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        
        # Use external Q and K if provided, otherwise compute them
        if external_q is not None:
            q = external_q
        else:
            q = self.q_proj(x)
            
        if external_k is not None:
            k = external_k
        else:
            k = self.k_proj(x)
            
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        
        return self.out_proj(attn_output)
    
    def forward(self, x: torch.Tensor, external_q: Optional[torch.Tensor] = None,
                external_k: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x), external_q, external_k)
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        #num_classes: int = 0,
        dropout: float = 0.1,
        global_pool: str = "cls",
    ):
        super().__init__()
        assert (global_pool in {"cls", "mean", "all"}), "global_pool must be 'cls', 'mean', or an integer > 0"
        self.global_pool = global_pool
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans, patch_size, embed_dim)

        # Number of patches for nominal size
        Hn, Wn = image_size
        Hn //= patch_size
        Wn //= patch_size
        self.grid_size_nominal = (Hn, Wn)
        num_patches_nominal = Hn * Wn

        # Class token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_nominal + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder (custom layers)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        #self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _interpolate_pos_embed(self, pos_embed, old_hw, new_hw):
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        D = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, old_hw[0], old_hw[1], D).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=new_hw, mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_hw[0] * new_hw[1], D)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x, external_qk: Optional[Dict[int, Dict[str, torch.Tensor]]] = None):
        """
        Args:
            x: Input tensor (B, C, H, W)
            external_qk: Dict mapping layer indices to {"Q": tensor, "K": tensor}
                        from CLIP model. If provided, these will be used instead of
                        computing Q and K from scratch.
        """
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)
        
        # Positional embeddings (interpolated if needed)
        if (Hp, Wp) != self.grid_size_nominal:
            pos_embed = self._interpolate_pos_embed(self.pos_embed, self.grid_size_nominal, (Hp, Wp))
        else:
            pos_embed = self.pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, D)
        x = x + pos_embed
        x = self.pos_drop(x)

        # Transformer stack with optional external Q, K
        for i, layer in enumerate(self.encoder_layers):
            external_q = None
            external_k = None
            
            if external_qk is not None and i in external_qk:
                external_q = external_qk[i].get("Q")
                external_k = external_qk[i].get("K")
                
                # Ensure the tensors have the right shape (B, N, D)
                if external_q is not None and external_q.shape[1] != x.shape[1]:
                    # If CLIP doesn't have CLS token, add zeros for CLS position
                    if external_q.shape[1] == x.shape[1] - 1:
                        cls_q = torch.zeros(B, 1, external_q.shape[-1], 
                                          device=external_q.device, dtype=external_q.dtype)
                        external_q = torch.cat([cls_q, external_q], dim=1)
                
                if external_k is not None and external_k.shape[1] != x.shape[1]:
                    if external_k.shape[1] == x.shape[1] - 1:
                        cls_k = torch.zeros(B, 1, external_k.shape[-1], 
                                          device=external_k.device, dtype=external_k.dtype)
                        external_k = torch.cat([cls_k, external_k], dim=1)
            
            x = layer(x, external_q, external_k)
            
        x = self.norm(x)
        return x[:, 1:]  # Return patch tokens without CLS token


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 224, 224
    embed_dim = 512
    model = ViT(
        image_size=(224, 224),
        patch_size=16,
        in_chans=C,
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=6,
        global_pool="all",  # return all tokens
    )
    x = torch.randn(B, C, H, W)
    
    # Without external Q, K (normal operation)
    tokens = model(x)
    print("Tokens shape:", tokens.shape)
    
    # With external Q, K (from CLIP)
    # Example: using external Q, K for layers 0 and 1
    num_patches = (H // 16) * (W // 16)  # 14 * 14 = 196
    external_qk = {
        0: {
            "Q": torch.randn(B, num_patches, embed_dim),  # Without CLS token
            "K": torch.randn(B, num_patches, embed_dim),
        },
        1: {
            "Q": torch.randn(B, num_patches, embed_dim),
            "K": torch.randn(B, num_patches, embed_dim),
        }
    }
    
    tokens_with_clip = model(x, external_qk)
    print("Tokens with CLIP Q,K shape:", tokens_with_clip.shape)
