import torch
import torch.nn as nn
import re
from .vit import ViT


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, model_args=None, delay_load=False, **kwargs):
    if model_args is None:
        swap=getattr(config, "swap", False)
    else:
        swap=getattr(model_args, "swap")
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    apply_norm = False
    if projector_type.startswith('norm'):
        apply_norm = True
        projector_type = projector_type[len('norm'):]
        if projector_type.startswith('_'):
            projector_type = projector_type[1:]
        if not projector_type:
            raise ValueError("Projector type 'norm' must specify a base adapter type.")

    def add_pre_norm(module: nn.Module) -> nn.Module:
        if not apply_norm:
            return module
        return nn.Sequential(nn.LayerNorm(config.mm_hidden_size), module)

    modules = []
    if projector_type == 'linear':
        return add_pre_norm(nn.Linear(config.mm_hidden_size, config.hidden_size))
    if projector_type.startswith("conv"):
        projector_type=projector_type.split("conv_")[-1]
        from .downsample_conv import LearnableCompressor
        compressor = LearnableCompressor(0.5, config.mm_hidden_size, config.mm_hidden_size)
        modules.append(compressor)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return add_pre_norm(nn.Sequential(*modules))

    attn_match = re.match(
        r'^mlp(\d+)x_gelu_attn_layers_(\d+)_heads_(\d+)$', projector_type
    )
    if attn_match:
        mlp_depth = int(attn_match.group(1))
        num_layers = int(attn_match.group(2))
        num_heads = int(attn_match.group(3))

        modules.append(nn.Linear(config.mm_hidden_size, config.hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=2e-2)  # very small std
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size if not swap else config.mm_hidden_size,
            nhead=num_heads,
            batch_first=True,
            activation='gelu',  # <-- hardcoded GELU
            norm_first=True,
            dropout=0,

        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        if swap:
            modules = [transformer_encoder] + modules
        else:
            modules.append(transformer_encoder)

        return add_pre_norm(nn.Sequential(*modules))

    # Match ViT + MLP pattern: vit_p_4_d_512_h_4_l_6_mlp2x_gelu
    vit_mlp_match = re.match(
        r'^vit_p_(\d+)_d_(\d+)_h_(\d+)_l_(\d+)_mlp(\d+)x_gelu$', projector_type
    )
    if vit_mlp_match:
        patch_size = int(vit_mlp_match.group(1))
        embed_dim = int(vit_mlp_match.group(2))
        num_heads = int(vit_mlp_match.group(3))
        num_layers = int(vit_mlp_match.group(4))
        mlp_depth = int(vit_mlp_match.group(5))
        
        # Create ViT with hardcoded parameters
        vit = ViT(
            image_size=(32, 32),
            patch_size=patch_size,
            in_chans=32,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            global_pool="all"  # Return all patch tokens
        )
        
        # Create MLP projector that takes ViT output as input
        mlp_modules = [nn.Linear(embed_dim, config.hidden_size)]
        for _ in range(1, mlp_depth):
            mlp_modules.append(nn.GELU())
            mlp_modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        
        # Combine ViT and MLP in sequential
        return add_pre_norm(nn.Sequential(vit, nn.Sequential(*mlp_modules)))

    if projector_type == 'identity':
        return add_pre_norm(IdentityMap())

    raise ValueError(f'Unknown projector type: {projector_type}')
