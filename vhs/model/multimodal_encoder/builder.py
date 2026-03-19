import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .vae_encoder import VAEVisionTower
from .dummy_encoder import DummyVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    vae_image_size = getattr(vision_tower_cfg, "vae_image_size", 512)


    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("timm") or vision_tower.startswith("google") or "ShareGPT4V" in vision_tower or 'facebook' in vision_tower or "vae" in vision_tower or "dummy" in vision_tower or "hidden" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        if "vae" in vision_tower and "clip" not in vision_tower:
            if "dynamic" in vision_tower:
                return VAEVisionTower(vision_tower=vision_tower, args=vision_tower_cfg, dynamic_resolution=True)
            else:
                return VAEVisionTower(vision_tower=vision_tower, args=vision_tower_cfg, image_size=vae_image_size)
        if "hidden" in vision_tower:
            from .dummy_encoder_hidden_layers import DummyHiddenVisionTower
            return DummyHiddenVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if "dummy" in vision_tower:
            return DummyVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
