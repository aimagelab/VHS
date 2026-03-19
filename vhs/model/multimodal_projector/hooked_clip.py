import torch
from transformers import CLIPModel, CLIPProcessor
from .vit_clip_distill import ViT  # Import our custom ViT

# -> transformers.models.clip.modeling_clip.CLIPEncoderLayer

class QKExtractor:
    """Class to extract Q and K matrices from CLIP encoder layers."""
    
    def __init__(self):
        self.qk_per_layer = {}  # {layer_idx: {"Q": tensor, "K": tensor}}
        self.hooks = []  # Store hook handles for cleanup
    
    def make_hook(self, layer_idx):
        """Create a hook function for the specified layer index."""
        def hook(module, inputs):
            # inputs[0] is hidden_states: (B, N, D)
            # For encoder layers, the first input is the hidden_states
            hidden_states = inputs[0]
            # Access the self-attention module and compute Q, K
            self_attn = module.self_attn
            Q = self_attn.q_proj(hidden_states)
            K = self_attn.k_proj(hidden_states)
            self.qk_per_layer[layer_idx] = {"Q": Q.detach(), "K": K.detach()}
        return hook
    
    def register_hooks(self, vision_encoder):
        """Register hooks on all layers of the vision encoder."""
        for i, layer in enumerate(vision_encoder.encoder.layers):
            hook_handle = layer.register_forward_pre_hook(self.make_hook(i))
            self.hooks.append(hook_handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_qk_data(self):
        """Get the extracted Q, K data."""
        return self.qk_per_layer
    
    def clear_data(self):
        """Clear the stored Q, K data."""
        self.qk_per_layer.clear()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    vision_encoder = model.vision_model.encoder  # all the Transformer layers
    print(type(vision_encoder.layers[0]))
    
    # Create QK extractor and register hooks
    qk_extractor = QKExtractor()
    qk_extractor.register_hooks(vision_encoder)

    from PIL import Image
    img = Image.open("images/dog.jpg").convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model.vision_model(**inputs)   # forward through vision encoder only

    # Get the extracted Q, K data
    qk_per_layer = qk_extractor.get_qk_data()
    for idx, qk in qk_per_layer.items():
        print(f"Layer {idx}: Q {qk['Q'].shape}, K {qk['K'].shape}")

    # Clean up hooks
    qk_extractor.remove_hooks()

    # Now use the extracted Q, K with our custom ViT
    print("\n" + "="*50)
    print("Using extracted Q, K with custom ViT")
    print("="*50)

    # Create custom ViT model
    vit_model = ViT(
        image_size=(224, 224),
        patch_size=16,
        in_chans=3,
        embed_dim=1024,  # CLIP Large has 1024 dim
        num_heads=16,    # CLIP Large has 16 heads
        num_layers=12,   # Use fewer layers for this example
        dropout=0.1,
    ).to(device)

    # Prepare image tensor for ViT (convert from PIL to tensor)
    import numpy as np
    image_array = np.array(img)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Adapt CLIP Q, K for ViT (remove CLS token since CLIP has it at position 0)
    adapted_qk = {}
    for layer_idx, qk in qk_per_layer.items():
        if layer_idx < 6:  # Use only first 6 layers
            # Remove CLS token (position 0) to match ViT patch tokens
            Q_patches = qk["Q"][:, 1:]  # Remove CLS token
            K_patches = qk["K"][:, 1:]  # Remove CLS token
            adapted_qk[layer_idx] = {"Q": Q_patches, "K": K_patches}

    print(f"Adapted Q, K for {len(adapted_qk)} layers")

    # Run ViT with CLIP features
    with torch.no_grad():
        vit_output = vit_model(image_tensor, adapted_qk)

    print(f"ViT output shape: {vit_output.shape}")
    print("Successfully integrated CLIP Q, K with custom ViT!")
    
    # Optional: Clear the stored data
    qk_extractor.clear_data()