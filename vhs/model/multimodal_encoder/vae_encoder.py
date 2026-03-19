import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from diffusers import SanaPipeline, AutoencoderDC
from PIL import Image
from torchvision import transforms as T
from argparse import Namespace
from transformers.image_processing_utils import BaseImageProcessor
from diffusers.image_processor import PixArtImageProcessor
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN
from torchvision.transforms.functional import resize
import json
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

#COMPATIBLE_BINS = [[12, 1], [3, 4], [4, 3], [3, 1], [5, 1], [2, 2], [1, 6], [2, 5], [1, 3], [1, 9], [1, 12], [6, 2], [7, 1], [4, 2], [3, 3], [9, 1], [11, 1], [2, 4], [1, 2], [2, 1], [1, 5], [1, 11], [6, 1], [1, 8], [3, 2], [4, 1], [5, 2], [8, 1], [1, 1], [10, 1], [1, 4], [2, 3], [1, 7], [2, 6], [1, 10]]
COMPATIBLE_BINS_ = {12.0: [12, 1], 0.75: [3, 4], 1.3333333333333333: [4, 3], 3.0: [6, 2], 5.0: [5, 1], 1.0: [1, 1], 0.16666666666666666: [1, 6], 0.4: [2, 5], 0.3333333333333333: [2, 6], 0.1111111111111111: [1, 9], 0.08333333333333333: [1, 12], 7.0: [7, 1], 2.0: [2, 1], 9.0: [9, 1], 11.0: [11, 1], 0.5: [1, 2], 0.2: [1, 5], 0.09090909090909091: [1, 11], 6.0: [6, 1], 0.125: [1, 8], 1.5: [3, 2], 4.0: [4, 1], 2.5: [5, 2], 8.0: [8, 1], 10.0: [10, 1], 0.25: [1, 4], 0.6666666666666666: [2, 3], 0.14285714285714285: [1, 7], 0.1: [1, 10]}
INTERSECTED_BINS = {0.25: [512.0, 2048.0], 1/3: [576.0, 1728.0], 0.4: [640.0, 1600.0], 0.5: [704.0, 1408.0], 1.0: [1024.0, 1024.0], 2.0: [1408.0, 704.0], 2.5: [1600.0, 640.0], 3.0: [1728.0, 576.0], 4.0: [2048.0, 512.0]}
FILTERED_RATIOS = [(1, 1), (1, 2), (2, 1), (3, 1), (1, 3), (4, 1), (1, 4), (2, 5), (5, 2)]
class VAEImageProcessor(BaseImageProcessor):
    def __init__(self, image_size=512, dynamic_resolution=False, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.size = {'shortest_edge': image_size}
        self.crop_size = {
            "width": image_size,
            "height": image_size
        }
        self.dynamic_resolution = dynamic_resolution
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        # Define the transformation
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        self.no_resize_transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
    def preprocess(self, images, return_tensors=None, bins = INTERSECTED_BINS, return_bins = False, use_nvila_ratios=True, **kwargs):
        if not isinstance(images, list):
            images = [images]
        heights = [image.width for image in images]
        widths = [image.height for image in images]
        new_heights = []
        new_widths = []
        my_bins = []
        for h, w in zip(heights, widths):
            if not use_nvila_ratios:
                new_h, new_w = PixArtImageProcessor.classify_height_width_bin(h, w, ratios=bins)
                ratio = new_h/new_w
                bin = COMPATIBLE_BINS_.get(ratio, None)
            else:
                target_ratio = find_closest_aspect_ratio(h/w, FILTERED_RATIOS, w, h, 448)
                new_h, new_w = INTERSECTED_BINS[target_ratio[1]/target_ratio[0]]
                new_h, new_w = int(new_h), int(new_w)
                bin=target_ratio[1]/target_ratio[0]
            new_heights.append(new_h)
            new_widths.append(new_w)
            my_bins.append(bin)
        if self.dynamic_resolution:
            images = [resize(image, (height, width)) for image, height, width in zip(images, new_widths, new_heights)]
            processed_images = [self.no_resize_transform(image) for image in images]
        else:
            processed_images = [self.transform(image) for image in images]
        if return_tensors == "pt":
            return {"pixel_values": [torch.stack(processed_images)]} if not return_bins else {"pixel_values": [torch.stack(processed_images)],
                                                                                                        "bins": my_bins}
        return processed_images if not return_bins else processed_images, bins

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)



def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io

        images = [log["images"][0] for log in image_caption]
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i : i + images_per_row][0] for i in range(0, len(images), images_per_row))

        new_im = Image.new("RGB", (max_width * images_per_row, total_height))

        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i : i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)

        return new_im


class VAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, dynamic_resolution=False, image_size=512):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, 'mm_vision_select_feature', 0)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.flatten_output = getattr(args, "flatten_vae_output", True)
        self.image_size = image_size
        self.dynamic_resolution=dynamic_resolution
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            pass
            #self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)
    def load_model(self, model_args=None, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = VAEImageProcessor(image_size=self.image_size, dynamic_resolution=self.dynamic_resolution)
        vae = AutoencoderDC.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            low_cpu_memory_usage=False,
            device_map=device_map,
        )
        self.vision_tower = vae
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        return image_forward_outs

    @torch.no_grad()
    def forward(self, images): # images tensor of shape (batch, 3, 336, 336)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vae_encode(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vae_encode(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features
    
    #def get_image_processor(self):
    #    transform = [
    #        T.Lambda(lambda img: img.convert("RGB")),
    #        T.Resize(self.image_size),  # Image.BICUBIC
    #        T.CenterCrop(self.image_size),
    #        # T.RandomHorizontalFlip(),
    #        T.ToTensor(),
    #        T.Normalize([0.5], [0.5]),
    #    ]
    #    return T.Compose(transform)
    def vae_encode(self, images, sample_posterior=True):
        name = self.vision_tower.config._class_name
        if name == "sdxl" or name == "sd3":
            posterior = self.vision_tower.encode(images.to(self.vision_tower.device)).latent_dist
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
            z = (z - self.vision_tower.config.shift_factor) * self.vision_tower.config.scaling_factor
        elif "dc-ae" in name:
            ae = self.vision_tower
            scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
            z = ae.encode(images.to(self.vision_tower.device))
            z = z * scaling_factor
        elif "AutoencoderDC" in name:
            scaling_factor = self.vision_tower.config.scaling_factor if self.vision_tower.config.scaling_factor else 0.41407
            z = self.vision_tower.encode(images.to(self.vision_tower.device))[0]
            z = z * scaling_factor
        else:
            print("error load vae")
            exit()
        if self.flatten_output:
            z = z.reshape(z.shape[0], z.shape[1], -1).permute(0,2,1) #flatten and channel dim last
        return z
    
    def vae_decode(self, latent):
        name = self.vision_tower.config._class_name
        vae = self.vision_tower
        if name == "sdxl" or name == "sd3":
            latent = (latent.detach() / vae.config.scaling_factor) + vae.config.shift_factor
            samples = vae.decode(latent).sample
        elif "dc-ae" in name:
            ae = vae
            vae_scale_factor = (
                2 ** (len(ae.config.encoder_block_out_channels) - 1)
                if hasattr(ae, "config") and ae.config is not None
                else 32
            )
            scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
            if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
                from patch_conv import convert_model

                ae = convert_model(ae, splits=4)
            samples = ae.decode(latent.detach() / scaling_factor)
        elif "AutoencoderDC" in name:
            ae = vae
            scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
            try:
                samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
            except torch.cuda.OutOfMemoryError as e:
                print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
                ae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)
                samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
        else:
            print("error load vae")
            exit()
        return samples
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype # torch.bfloat16

    @property
    def device(self):
        return self.vision_tower.device # device(type='cuda', index=0)

    @property
    def config(self):
        if self.is_loaded:
            self.vision_tower.config.patch_size = 32
            self.vision_tower.config.image_size = self.image_size
            return self.vision_tower.config # CLIPVisionConfig
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.latent_channels # 1024

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size # (518 // 14) = 37

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2 # 1369
    
    
if __name__ == "__main__":
    args = Namespace(
        image_size=256,
    )

    encoder = VAEVisionTower("vae", args=args).to("cuda")
    image = Image.open("images/image.png").convert("RGB")
    image = encoder.image_processor(image).unsqueeze(0)  # Add batch dimension
    image = image.to("cuda", dtype=torch.float16)
    latent = encoder(image)
    samples = encoder.vae_decode(latent)

    #image = processor(image).unsqueeze(0)  # Add batch dimension
    #image = image.to("cuda", dtype=torch.float16)
    #latent = vae_encode(vae.config._class_name, vae, image, sample_posterior=False, device="cuda")
    #samples = vae_decode(vae.config._class_name, vae, latent)
    samples = (
        torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    )
    decoded_image = Image.fromarray(samples)
    decoded_image.save("decoded_image.png")

