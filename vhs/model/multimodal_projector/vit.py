import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Optional


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

        # Transformer encoder (pre-implemented)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,  # (B, N, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

    def forward(self, x): #return_tokens=False
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

        # Transformer stack
        x = self.encoder(x)
        x = self.norm(x)

        #if return_tokens:
        return x[:, 1:]  # (B, 1+N, D)

        #if self.global_pool == "cls":
        #    pooled = x[:, 0]
        #    pooled = self.head(pooled) 
        #elif self.global_pool == "mean":
        #    pooled = x[:, 1:].mean(dim=1)
        #    pooled = self.head(pooled)
        #else:
        #    pooled = x[:, 1:]

        #   return pooled


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 32, 32, 32
    model = ViT(
        image_size=(32, 32),
        patch_size=4,
        in_chans=C,
        embed_dim=512,
        num_heads=4,
        num_layers=6,
        #num_classes=10,  # classification head
        global_pool="all",  # return all tokens
    )
    x = torch.randn(B, C, H, W)
    logits = model(x)
    print("Logits:", logits.shape)          # (B, num_classes)
    tokens = model(x, return_tokens=True)
    print("Tokens:", tokens.shape)          # (B, 1+N, D)
