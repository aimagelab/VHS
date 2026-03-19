#!/usr/bin/env python3
"""
Standalone model instantiation script for LLaVA models.
Extracted from the training script for evaluation purposes.
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import transformers
from packaging import version
# Import LLaVA modules
from . import conversation as conversation_lib
from .model import *
from .utils import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model.language_model.llava_qwen import *

logger = get_logger(__name__)

# For compatibility with training script
def rank0_print(*args):
    """Print function that respects distributed training rank"""
    print(*args)

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(transformers.__version__) >= version.parse('4.14')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_vision_tower: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_vision_normalize: bool = field(default=True)
    do_layer_19_norm: bool = field(default=False)
    s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default=None)
    siglip: bool = field(default=False)
    use_cache: bool = field(default=True)
    vae_image_size: Optional[int] = field(default=512)
    flatten_vae_output: bool = field(default=True)
    swap: bool = field(default=False)
    hidden_dim: int = field(default=2240)
    normalization_mean_path: Optional[str] = field(default=None)
    normalization_variance_path: Optional[str] = field(default=None)
    sanity_check: bool = field(default=False)

@dataclass
class DataArguments:
    image_aspect_ratio: str = 'square'
    mm_use_im_start_end: bool = field(default=False)
    is_multimodal: bool = False

@dataclass
class InferenceArguments:
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=512)
    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    device: str = field(default="cuda")
    llm_backbone: Optional[str] = field(default=None)
    llm_pad_token: Optional[str] = field(default=None)
    lora_enable: bool = field(default=False)
    lora_weights: Optional[str] = field(default=None)
    attn_implementation: Optional[str] = field(default=None)
    # Training-specific arguments
    mpt_attn_impl: Optional[str] = field(default="triton")
    gradient_checkpointing: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_vision_tower: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(default=None)
    fsdp: Optional[str] = field(default=None)
    local_rank: int = field(default=-1)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def find_all_linear_names(model):
    """Find all linear layer names for LoRA adaptation."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def load_model_and_tokenizer(
    model_args: ModelArguments,
    data_args: DataArguments,
    inference_args: InferenceArguments,
    for_training: bool = False
):
    """
    Load and initialize LLaVA model and tokenizer.
    
    Args:
        model_args: Model configuration
        data_args: Data configuration  
        inference_args: Inference/training configuration
        for_training: Whether loading for training (enables training-specific features)
    
    Returns:
        tuple: (model, tokenizer, image_processor)
    """
    
    # Determine compute dtype
    compute_dtype = (
        torch.float16 if inference_args.fp16 else 
        (torch.bfloat16 if inference_args.bf16 else torch.float32)
    )
    
    # Setup quantization if needed
    bnb_model_from_pretrained_args = {}
    if inference_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": inference_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=inference_args.bits == 4,
                load_in_8bit=inference_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=inference_args.double_quant,
                bnb_4bit_quant_type=inference_args.quant_type
            )
        ))

    # Load model based on type
    if model_args.vision_tower is not None:
        if 'vhs' in model_args.model_name_or_path.lower():
            if "qwen3" in model_args.model_name_or_path.lower():
                config = LlavaQwen3Config.from_pretrained(model_args.model_name_or_path)
            else:
                config = LlavaQwenConfig.from_pretrained(model_args.model_name_or_path)
            if not hasattr(config, "swap"):
                setattr(config, "swap", model_args.swap)
            if not hasattr(config, "flatten_vae_output"):
                setattr(config, "flatten_vae_output", model_args.flatten_vae_output)
            if 'hidden' in model_args.vision_tower:
                setattr(config, "mm_hidden_size", model_args.hidden_dim)
                setattr(config, "hidden_dim", model_args.hidden_dim)
            setattr(config, 'sanity_check', model_args.sanity_check)
            if model_args.sanity_check:
                print(f"Running Sanity Check: {config.sanity_check}")
            if "qwen3" in model_args.model_name_or_path.lower():
                model = LlavaQwen3ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=inference_args.cache_dir,
                    attn_implementation=inference_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if inference_args.bf16 else None),
                    **bnb_model_from_pretrained_args,
                )
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=inference_args.cache_dir,
                    attn_implementation=inference_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if inference_args.bf16 else None),
                    **bnb_model_from_pretrained_args,
                )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=inference_args.cache_dir,
                attn_implementation=inference_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if inference_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        raise "porcodio porcamadonna"
        # if "verifier" in model_args.model_name_or_path.lower():
        #     image_processor = transformers.AutoProcessor.from_pretrained(model_args.model_name_or_path, 
        #                                                     trust_remote_code=True, 
        #                                                     local_files_only=True, 
        #                                                     use_fast=False)
        #     model = transformers.AutoModel.from_pretrained(
        #             model_args.model_name_or_path, 
        #             trust_remote_code=True, 
        #             attn_implementation=inference_args.attn_implementation,
        #             torch_dtype=(torch.bfloat16 if inference_args.bf16 else None),
        #             **bnb_model_from_pretrained_args            
        #             )
 
        # model = transformers.LlamaForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     cache_dir=inference_args.cache_dir,
        #     attn_implementation=inference_args.attn_implementation,
        #     torch_dtype=(torch.bfloat16 if inference_args.bf16 else None),
        #     **bnb_model_from_pretrained_args
        # )
    
    # Enable/disable cache based on use case
    model.config.use_cache = model_args.use_cache if not for_training else False
    
    # Training-specific configurations
    if for_training:
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)

        if inference_args.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            model.config.torch_dtype = (
                torch.float32 if inference_args.fp16 else 
                (torch.bfloat16 if inference_args.bf16 else torch.float32)
            )
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=inference_args.gradient_checkpointing
            )

        if inference_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if inference_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=inference_args.lora_r,
                lora_alpha=inference_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=inference_args.lora_dropout,
                bias=inference_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if inference_args.bits == 16:
                if inference_args.bf16:
                    model.to(torch.bfloat16)
                if inference_args.fp16:
                    model.to(torch.float16)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)

    # Load tokenizer
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=inference_args.cache_dir,
            model_max_length=inference_args.model_max_length,
            padding_side="right"
        )
    elif "Minerva" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=inference_args.cache_dir,
            model_max_length=inference_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    elif "vhs" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=inference_args.cache_dir,
            model_max_length=inference_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    # Setup tokenizer padding - following train.py logic exactly
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif 'gemma' in model_args.model_name_or_path:
        tokenizer.pad_token_id = 0
    elif 'Phi' in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token
    elif 'Minerva' in model_args.model_name_or_path:
        print(f"Pad token is already set")
    else:
        # Handle Llama variants
        if (inference_args.llm_backbone is None or
            ("llama_3" not in inference_args.llm_backbone and
             "qwen" not in model_args.model_name_or_path.lower())):
            if tokenizer.unk_token is None:
                if for_training:
                    print("resize embedding dimesion")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(unk_token="[UNK]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        
        if inference_args.llm_backbone == "llama_3_1":
            if for_training:
                print(f"pad token: {inference_args.llm_pad_token}")
            if inference_args.llm_pad_token == 'end_of_text':
                tokenizer.pad_token_id = 128001
            elif inference_args.llm_pad_token == 'eot':
                tokenizer.pad_token_id = 128009
            elif inference_args.llm_pad_token == 'pad':
                tokenizer.pad_token_id = 128004
            elif "vhs" in model_args.model_name_or_path.lower():
                pass
            else:
                raise ValueError(f"Unknown llm_pad_token: {inference_args.llm_pad_token}")
        elif inference_args.llm_backbone == "llama_3":
            if inference_args.llm_pad_token == 'eos':
                tokenizer.pad_token = tokenizer.eos_token
            elif inference_args.llm_pad_token == 'pad':
                tokenizer.pad_token_id = 128003
            else:
                tokenizer.pad_token = tokenizer.unk_token
        elif "vhs" in model_args.model_name_or_path.lower():
            pass
        else:
            tokenizer.pad_token = tokenizer.unk_token

    # Setup conversation template
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    conversation_lib.default_conversation.tokenizer = tokenizer

    # Initialize vision modules if needed
    image_processor = None
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=inference_args.fsdp if for_training else None
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if inference_args.bf16 else torch.float16, 
            device=inference_args.device
        )

        image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_aspect_ratio == 'anyres':
            base_size = vision_tower.config.image_size
            grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            model.config.image_grid_pinpoints = [
                [g[0]*base_size, g[1]*base_size] for g in grids
            ]
        
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        
        # Training-specific vision configurations
        if for_training:
            model.config.tune_mm_mlp_adapter = inference_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_vision_tower = inference_args.tune_vision_tower = model_args.tune_vision_tower
            if model_args.tune_mm_mlp_adapter:
                model.requires_grad_(False)
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_vision_tower:
                model.get_vision_tower().requires_grad_(True)
            model.config.freeze_mm_mlp_adapter = inference_args.freeze_mm_mlp_adapter
            if inference_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.mm_projector_lr = inference_args.mm_projector_lr
        
        if model_args.s2:
            model.config.s2 = model_args.s2
            model.config.s2_scales = model_args.s2_scales
        if model_args.siglip:
            model.config.siglip = model_args.siglip
            
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        if inference_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=inference_args.device)

    # Load LoRA weights if specified (for inference)
    if not for_training and inference_args.lora_enable and inference_args.lora_weights:
        from peft import PeftModel
        print(f"Loading LoRA weights from {inference_args.lora_weights}")
        model = PeftModel.from_pretrained(model, inference_args.lora_weights)
        if "non_lora_trainables.bin" in os.listdir(inference_args.lora_weights):
            non_lora_state_dict = torch.load(
                os.path.join(inference_args.lora_weights, 'non_lora_trainables.bin'), 
                map_location='cpu',
                weights_only=True
            )
            missing, unexpected = model.load_state_dict(non_lora_state_dict, strict=False)
            print("Loaded non-LoRA weights. Missing:", missing, "Unexpected:", unexpected)

    # Additional training-specific post-processing
    if for_training and inference_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if inference_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if inference_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    # Move model to device (only if not using quantized model)
    if not bnb_model_from_pretrained_args and not for_training:
        model = model.to(inference_args.device)
    
    # Set to eval mode for inference
    if not for_training:
        model.eval()
    
    if inference_args.local_rank in [0, -1]:
        print(f"Model loaded successfully!")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        if for_training:
            count_par_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {count_par_trainable:,}")
    
    return model, tokenizer, image_processor

def load_conversation_template(model_architecture: str):
    return conversation_lib.conv_templates.get(model_architecture, conversation_lib.conv_vicuna_v1)

def load_teacher_model_for_distillation(
    teacher_name_or_path: str,
    teacher_vision_tower: Optional[str] = None,
    device: str = "cuda",
    model_args: Optional[ModelArguments] = None,
    data_args: Optional[DataArguments] = None
):
    """
    Load teacher model for distillation training.
    """
    if model_args is None or data_args is None:
        raise ValueError("model_args and data_args required for teacher model loading")

    teacher_model, _, _ = create_model_from_args(
        model_name_or_path=teacher_name_or_path,
        vision_tower=teacher_vision_tower,
        version=model_args.version,
        device=device,
        bf16=True,
        model_max_length=4096,  # Default for teacher
        use_cache=False,
        mm_projector_type=model_args.mm_projector_type,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        mm_use_im_start_end=model_args.mm_use_im_start_end,
        mm_use_im_patch_token=model_args.mm_use_im_patch_token,
        image_aspect_ratio=data_args.image_aspect_ratio,
    )

    return teacher_model, None

def create_model_from_args(**kwargs):
    """
    Convenience function to create model from keyword arguments.
    
    Example usage:
        model, tokenizer, image_processor = create_model_from_args(
            model_name_or_path="path/to/model",
            vision_tower="openai/clip-vit-large-patch14",
            version="v1",
            device="cuda"
        )
    """
    # Set default values
    defaults = {
        'model_name_or_path': None,
        'version': 'v1',
        'vision_tower': None,
        'device': 'cuda',
        'bf16': True,
        'model_max_length': 4096,
        'bits': 16,
        'vae_image_size': 512,
        'flatten_vae_output': True,
        'swap': False,
        'for_training': False,  # New parameter
        'hidden_dim': 2240
    }
    
    # Update with provided kwargs
    defaults.update(kwargs)
    
    # Create argument objects
    model_args = ModelArguments(
        model_name_or_path=defaults['model_name_or_path'],
        version=defaults['version'],
        vision_tower=defaults['vision_tower'],
        pretrain_mm_mlp_adapter=defaults.get('pretrain_mm_mlp_adapter', None),
        mm_projector_type=defaults.get('mm_projector_type', 'linear'),
        mm_vision_select_layer=defaults.get('mm_vision_select_layer', -1),
        mm_use_im_start_end=defaults.get('mm_use_im_start_end', False),
        mm_use_im_patch_token=defaults.get('mm_use_im_patch_token', True),
        s2=defaults.get('s2', False),
        s2_scales=defaults.get('s2_scales', None),
        siglip=defaults.get('siglip', False),
        use_cache=defaults.get('use_cache', True),
        vae_image_size=defaults["vae_image_size"],
        flatten_vae_output=defaults["flatten_vae_output"],
        swap=defaults["swap"],
        freeze_backbone=defaults.get('freeze_backbone', False),
        tune_mm_mlp_adapter=defaults.get('tune_mm_mlp_adapter', False),
        hidden_dim=defaults.get('hidden_dim', 2240),
        normalization_mean_path=defaults.get('normalization_mean_path', None),
        normalization_variance_path=defaults.get('normalization_variance_path', None)
    )
    
    data_args = DataArguments(
        image_aspect_ratio=defaults.get('image_aspect_ratio', 'square'),
        mm_use_im_start_end=defaults.get('mm_use_im_start_end', False)
    )
    
    inference_args = InferenceArguments(
        device=defaults['device'],
        bf16=defaults.get('bf16', True),
        fp16=defaults.get('fp16', False),
        model_max_length=defaults['model_max_length'],
        bits=defaults['bits'],
        llm_backbone=defaults.get('llm_backbone', None),
        llm_pad_token=defaults.get('llm_pad_token', None),
        lora_enable=defaults.get('lora_enable', False),
        lora_weights=defaults.get('lora_weights', None),
        attn_implementation=defaults.get('attn_implementation', None),
        mpt_attn_impl=defaults.get('mpt_attn_impl', 'triton'),
        gradient_checkpointing=defaults.get('gradient_checkpointing', False),
        freeze_backbone=defaults.get('freeze_backbone', False),
        lora_r=defaults.get('lora_r', 64),
        lora_alpha=defaults.get('lora_alpha', 16),
        lora_dropout=defaults.get('lora_dropout', 0.05),
        lora_bias=defaults.get('lora_bias', 'none'),
        tune_mm_mlp_adapter=defaults.get('tune_mm_mlp_adapter', False),
        freeze_mm_mlp_adapter=defaults.get('freeze_mm_mlp_adapter', False),
        mm_projector_lr=defaults.get('mm_projector_lr', None),
        fsdp=defaults.get('fsdp', None),
        local_rank=defaults.get('local_rank', -1)
    )
    
    return load_model_and_tokenizer(
        model_args, data_args, inference_args, for_training=defaults['for_training']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load LLaVA model for inference")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("--vision_tower", type=str, default=None,
                       help="Path to vision tower")
    parser.add_argument("--version", type=str, default="v1",
                       help="Model version")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to load model on")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", 
                       help="Use float16")
    parser.add_argument("--bits", type=int, default=16,
                       help="Quantization bits (4, 8, or 16)")
    parser.add_argument("--lora_weights", type=str, default=None,
                       help="Path to LoRA weights")
    parser.add_argument("--llm_backbone", type=str, default=None,
                       help="LLM backbone type")
    parser.add_argument("--llm_pad_token", type=str, default=None,
                       help="Padding token for LLM")
    
    args = parser.parse_args()
    
    # Create model
    model, tokenizer, image_processor = create_model_from_args(
        model_name_or_path=args.model_name_or_path,
        vision_tower=args.vision_tower,
        version=args.version,
        device=args.device,
        bf16=args.bf16,
        fp16=args.fp16,
        bits=args.bits,
        lora_enable=args.lora_weights is not None,
        lora_weights=args.lora_weights,
        llm_backbone=args.llm_backbone,
        llm_pad_token=args.llm_pad_token
    )
    
    print("Model loaded successfully! You can now use it for inference.")
    print(f"Model type: {type(model)}")
    print(f"Tokenizer type: {type(tokenizer)}")
    if image_processor:
        print(f"Image processor type: {type(image_processor)}")
