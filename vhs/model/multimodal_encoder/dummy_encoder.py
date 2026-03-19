import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from argparse import Namespace
from transformers.image_processing_utils import BaseImageProcessor


class DummyImageProcessor(BaseImageProcessor):
    def __init__(self, image_size=512, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.size = {'shortest_edge': image_size}
        self.crop_size = {
            "width": image_size,
            "height": image_size
        }
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        # Define the transformation - similar to VAE but just basic preprocessing
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
        ])

    def preprocess(self, images, return_tensors=None, **kwargs):
        if not isinstance(images, list):
            images = [images]

        processed_images = [self.transform(image) for image in images]

        if return_tensors == "pt":
            return {"pixel_values": [torch.stack(processed_images)]}
        return processed_images

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


class DummyConfig:
    """Dummy configuration that mimics the structure expected by the vision tower"""
    def __init__(self, image_size=512):
        self.latent_channels = 32  # Same as input channels for RGB
        self.patch_size = 32
        self.image_size = image_size
        self._class_name = "DummyEncoder"


class DummyVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, 'mm_vision_select_feature', 0)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.image_size = getattr(args, 'image_size', 512)
        self.flatten_vae_output = getattr(args, 'flatten_vae_output', True)
        
        # Create dummy config
        self.cfg_only = DummyConfig(self.image_size)
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, model_args=None, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = DummyImageProcessor(self.image_size)
        
        # Create a dummy "vision tower" that's just an identity module
        self.vision_tower = nn.Identity()
        self.vision_tower.config = DummyConfig(self.image_size)
        self.vision_tower.dtype = torch.float32
        self.vision_tower.device = torch.device('cuda')
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        return image_forward_outs

    @torch.no_grad()
    def forward(self, images):
        """
        Forward pass that returns the input tensor unchanged.
        Images tensor should be of shape (batch, 3, H, W)
        """
        if type(images) is list:
            image_features = []
            for image in images:
                # Simply reshape the image to match expected output format
                # Flatten spatial dimensions and move channel to last
                image_reshaped = self.dummy_encode(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_reshaped).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.dummy_encode(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features
    
    def dummy_encode(self, images):
        """
        Dummy encoding that just reshapes the input tensor to match expected output format.
        Converts from (batch, 3, H, W) to (batch, H*W, 3) to match VAE output format.
        """
        batch_size, channels, height, width = images.shape
        # Reshape to match VAE output: (batch, num_patches, channels)
        if self.flatten_vae_output:
            images = images.reshape(batch_size, channels, -1).permute(0, 2, 1)
        return images
    


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            self.vision_tower.config.patch_size = 32
            self.vision_tower.config.image_size = self.image_size
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.latent_channels  # 3 for RGB

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


if __name__ == "__main__":
    args = Namespace(
        image_size=256,
    )

    # Test the dummy encoder
    encoder = DummyVisionTower("dummy", args=args)
    
    # Create a dummy tensor that simulates processed image input
    # Shape: (batch=1, channels=3, height=256, width=256)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = encoder(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test decoding
    decoded = encoder.dummy_decode(output)
    print(f"Decoded shape: {decoded.shape}")
    
    # Check if input and decoded are close (should be identical for dummy encoder)
    print(f"Input and decoded are close: {torch.allclose(dummy_input, decoded)}")
    
    print(f"Hidden size: {encoder.hidden_size}")
    print(f"Num patches: {encoder.num_patches}")
    print(f"Device: {encoder.device}")
    print(f"Dtype: {encoder.dtype}")
