import sys
import os
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)
import torch
import numpy as np
from diffusers import SanaSprintPipeline
from PIL import Image
import argparse
import re
import pandas as pd
import accelerate
from train_scripts.geneval_utils import SegmentationFeedback
from train_scripts.modeling.utils import set_seed
from latent_verifier_dict import LATENT_VERIFIER
from tqdm import tqdm
import yaml
import json
from inference_scripts.sana_activation_catcher import ActivationCatcher    

def load_config_from_yaml(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_args_from_config(config):
    """Create argparse.Namespace from config dictionary"""
    # Set defaults
    
    args = argparse.Namespace()
    
    # Set all arguments with defaults, then override with config values
    args.ckpt = config.get('ckpt', "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
    args.output_path = os.path.join(config.get('output_path', 'outputs_geneval'), config.get('name', 'geneval_experiment'))
    args.name = config.get('name', 'none')
    args.num_samples = config.get('num_samples', 20)
    args.cont = config.get('cont', False)
    args.dataset = config.get('dataset', None)
    args.fmt = config.get('fmt', 'png')
    args.seed = config.get('seed', 42)
    args.re_eval = config.get('re_eval', False)
    args.generation_mode = config.get('generation_mode', 'dev')
    args.annotation = config.get('annotation', 'geneval/prompts/evaluation_metadata.jsonl')
    args.sana_lora_weights = config.get('sana_lora_weights', None)
    args.segmentation_config_path = config.get('segmentation_config_path', config.get('segmentation_feedback_config_path', None))
    args.segmentation_ckpt_path = config.get('segmentation_ckpt_path', config.get('segmentation_feedback_ckpt_path', None))
    
    return args

def setup_argument_parser():
    """Setup the traditional argument parser as fallback"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')
    parser.add_argument('--ckpt', type=str, default="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
    parser.add_argument('--output_path', type=str, default='outputs_geneval')
    parser.add_argument('--name', type=str, default='none')
    parser.add_argument('-n', '--num_samples', type=int, default=20)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['geneval', 'dpg', 'custom', 'token_compose'])
    parser.add_argument('--fmt', type=str, default='png')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible experiments')
    parser.add_argument('--re_eval', action='store_true', default=False)
    parser.add_argument('--generation-mode', dest='generation_mode', type=str, default='dev', choices=['dev', 'prod'],
                        help='Generation mode. "dev" keeps the current behavior. "prod" enables Sana hidden-layer fast path.')
    parser.add_argument('-i', '--annotation', default='geneval/prompts/evaluation_metadata.jsonl', type=str)
    parser.add_argument('--sana-lora-weights', dest='sana_lora_weights', type=str, default=None, help='Path to LoRA weights to load into the Sana-Sprint pipeline')
    parser.add_argument('--segmentation-config-path', dest='segmentation_config_path', type=str, default=None,
                        help='Path to mmdet config file used by SegmentationFeedback')
    parser.add_argument('--segmentation-ckpt-path', dest='segmentation_ckpt_path', type=str, default=None,
                        help='Path to mmdet checkpoint used by SegmentationFeedback')
    parser.add_argument("--n_proc", type=int, default=1, help="Number of processes")
    parser.add_argument("--proc_id", type=int, default=0, help="Process ID")
    
    return parser

def get_best_index(score_list: list):
    scores = [score['score'] for score in score_list]
    return scores.index(max(scores))

def get_annotation_filenames(ROOT, latent_verifier_name, proc_id):
    """Get the annotation filenames based on the naming convention"""
    verifier_name = latent_verifier_name if latent_verifier_name is not None else "default"
    all_annotations_path = os.path.join(ROOT, f'all_annotations_proc_{verifier_name}_id_proc_{proc_id}.json')
    best_annotations_path = os.path.join(ROOT, f'best_annotations_proc_{verifier_name}_id_proc_{proc_id}.json')
    return all_annotations_path, best_annotations_path


class _PassthroughTransformerBlock(torch.nn.Module):
    """Pass-through block used to skip tail DiT blocks in prod mode candidate generation."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class TransformerSplitRunner:
    """Temporarily keeps only transformer blocks up to split_idx active."""

    def __init__(self, transformer, split_idx):
        self.transformer = transformer
        self.split_idx = split_idx
        self._replaced_blocks = []

    def __enter__(self):
        n_blocks = len(self.transformer.transformer_blocks)
        if self.split_idx < 0 or self.split_idx >= n_blocks:
            raise ValueError(f"split_idx={self.split_idx} is out of range for {n_blocks} transformer blocks")
        for idx in range(self.split_idx + 1, n_blocks):
            original_block = self.transformer.transformer_blocks[idx]
            self._replaced_blocks.append((idx, original_block))
            first_param = next(original_block.parameters(), None)
            target_device = first_param.device if first_param is not None else None
            target_dtype = first_param.dtype if first_param is not None else None
            passthrough_block = _PassthroughTransformerBlock()
            if target_device is not None or target_dtype is not None:
                passthrough_block = passthrough_block.to(device=target_device, dtype=target_dtype)
            self.transformer.transformer_blocks[idx] = passthrough_block
        return self

    def __exit__(self, exc_type, exc, tb):
        for idx, original_block in self._replaced_blocks:
            self.transformer.transformer_blocks[idx] = original_block
        self._replaced_blocks = []


def _parse_hidden_block_index(vision_tower):
    if not vision_tower or "hidden" not in vision_tower:
        return None
    match = re.search(r"(\d+)(?!.*\d)", vision_tower)
    if match is None:
        raise ValueError(f"Could not parse hidden block index from vision_tower='{vision_tower}'")
    return int(match.group(1))

def SanaSprint_forward(pipe, *args, **kwargs):
    return pipe(
        intermediate_timesteps= None if num_inference_steps!= 2 else intermediate_timesteps,
        use_resolution_binning=use_resolution_binning, 
        *args, 
        **kwargs
        )

accelerator = accelerate.Accelerator()
# Parse arguments
parser = setup_argument_parser()
parsed_args = parser.parse_args()
experiment_name = parsed_args.name
config = {}

# Check if config file is provided
if parsed_args.config:
    try:
        print(f"Loading configuration from {parsed_args.config}")
        config = load_config_from_yaml(parsed_args.config)
        args = create_args_from_config(config)
        if args.name == 'none':
            args.name = experiment_name
        print("Configuration loaded successfully from YAML file: ", parsed_args.config)
    except FileNotFoundError:
        print(f"Config file {parsed_args.config} not found. Using command line arguments.")
        args = parsed_args
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file: {e}. Using command line arguments.")
        args = parsed_args
    except Exception as e:
        print(f"Error loading config file: {e}. Using command line arguments.")
        args = parsed_args
else:
    print("No config file provided. Using command line arguments.")
    args = parsed_args

args.proc_id = parsed_args.proc_id
args.n_proc = parsed_args.n_proc
if parsed_args.generation_mode != 'dev':
    args.generation_mode = parsed_args.generation_mode
elif not hasattr(args, 'generation_mode'):
    args.generation_mode = 'dev'
if getattr(parsed_args, 'segmentation_config_path', None):
    args.segmentation_config_path = parsed_args.segmentation_config_path
if getattr(parsed_args, 'segmentation_ckpt_path', None):
    args.segmentation_ckpt_path = parsed_args.segmentation_ckpt_path

# Set seed for reproducible experiments
set_seed(args.seed)

# Generate n_samples random numbers and create corresponding generators
rng_main = np.random.default_rng(args.seed)
sample_seeds = rng_main.integers(0, 2**32 - 1, size=args.num_samples)
sample_generators = [torch.Generator().manual_seed(int(seed)) for seed in sample_seeds]

num_inference_steps = config.get("num_inference_steps", 1)
intermediate_timesteps = config.get("intermediate_timesteps", 1.3)
generator = config.get("generator", "Sana-Sprint")
imgs_folder = config.get("imgs_path", None)

pipeline = SanaSprintPipeline.from_pretrained(
    args.ckpt,
    torch_dtype=torch.bfloat16
)
if args.sana_lora_weights:
    pipeline.load_lora_weights(args.sana_lora_weights)
generation_forward = SanaSprint_forward
catcher = ActivationCatcher(pipeline.transformer)
hidden_dim = 2240

pipeline.to(accelerator.device)

image_width = config.get("image_width", 1024)
image_height = config.get("image_height", 1024)

verifier = None
if getattr(args, 'segmentation_config_path', None) or getattr(args, 'segmentation_ckpt_path', None):
    verifier = SegmentationFeedback(
        accelerator.device,
        detector_config_path=getattr(args, 'segmentation_config_path', None),
        detector_ckpt_path=getattr(args, 'segmentation_ckpt_path', None),
    )  # GT of GeneVal
gt_score_field = "score"

feedback = None  # Initialize with None
scorer = None    # Initialize with None
latent_verifier = config.get("latent_verifier", None)
latent_verifier_name = None  # Initialize with default value
vision_tower = ""
if latent_verifier is not None:
    latent_verifier_name = config.get("latent_verifier_name", None)
    latent_verifier_path = LATENT_VERIFIER[f"{latent_verifier_name}"]["latent_verifier_path"]
    assert latent_verifier_path is not None
    vision_tower = LATENT_VERIFIER[f"{latent_verifier_name}"]["vision_tower"]
    os.environ["TOKENIZER_PATH"] = config.get("tokenizer_path", latent_verifier_path)
    if vision_tower is not None:
        from train_scripts.latent_verifier import LatentGemmaVerifier
        generation_prompt = config.get("use_generation_prompt", True)
        mm_projector_type=config.get("mm_projector_type", "mlp2x_gelu")
        feedback = LatentGemmaVerifier("cuda",
                                       latent_verifier_path,
                                       vision_tower=vision_tower,
                                       lora_weights=config.get("lora_weights", None),
                                       use_generation_prompt=generation_prompt,
                                       mm_projector_type=mm_projector_type,
                                       hidden_dim=hidden_dim,
                                       normalization_mean_path=config.get("normalization_mean_path", None),
                                       normalization_variance_path=config.get("normalization_variance_path", None),
                                       )
    else:
        from train_scripts.latent_verifier import LatentGemmaVerifier
        feedback = LatentGemmaVerifier("cuda", latent_verifier_path, lora_weights=config.get("lora_weights", None))
external_verifier = config.get("external_verifier", None)
if feedback is None and scorer is None and args.dataset != 'token_compose':
    raise ValueError(f"No verifier provided")
use_resolution_binning = config.get("use_resolution_binning", None)
if "hidden" in vision_tower:
        mm_vision_select_feature = f"block_{vision_tower.split('_')[-1]}"

hidden_block_idx = _parse_hidden_block_index(vision_tower) if "hidden" in vision_tower else None
if hidden_block_idx is not None:
    mm_vision_select_feature = f"block_{hidden_block_idx}"

prod_hidden_sana_mode = (
    args.generation_mode == "prod"
    and ("Sana" in generator)
    and ("hidden" in vision_tower)
    and bool(latent_verifier)
)

clip_based_verifier = (
    bool(latent_verifier)
    and (
        "clip" in str(latent_verifier_name).lower()
        or "clip" in str(vision_tower).lower()
    )
)

prod_clip_sana_mode = (
    args.generation_mode == "prod"
    and ("Sana" in generator)
    and clip_based_verifier
    and ("hidden" not in vision_tower)
)

if prod_hidden_sana_mode:
    print(f"[generation-mode=prod] Enabling fast Sana hidden split at block {hidden_block_idx}")
if prod_clip_sana_mode:
    print("[generation-mode=prod] Enabling clip verifier fast path: skip per-sample saves, save only final selected sample")

if args.dataset == 'geneval':
    with open(args.annotation) as fp:
            print(f"loading {args.annotation}")
            metadatas = json.load(fp)
            ## if the file is not a valid json, it might be a jsonl file
            #fp.seek(0)
            #metadatas = [json.loads(line) for line in fp]
elif args.dataset == 'custom':
    metadatas = pd.read_csv(args.annotation).to_dict(orient='records')
n = len(metadatas)

indices = np.arange(len(metadatas))
rng = np.random.default_rng(args.seed)
shuffled_indices = rng.permutation(indices)
print(shuffled_indices[:100])

ROOT=os.path.join(args.output_path,f'gen_eval_{args.name}/')
#divide the work among n_proc processes
os.makedirs(ROOT, exist_ok=True)

# Get annotation file paths
all_annotations_path, best_annotations_path = get_annotation_filenames(ROOT, latent_verifier_name, args.proc_id)

# Load existing annotations if continuing and files exist
all_evaluations = []
selected_evaluations = []
if args.cont and not args.re_eval:
    if os.path.exists(all_annotations_path):
        print(f"Loading existing all_annotations from {all_annotations_path}")
        try:
            with open(all_annotations_path, 'r') as f:
                all_evaluations = json.load(f)
            print(f"Loaded {len(all_evaluations)} existing all_evaluations")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading all_annotations: {e}. Starting with empty list.")
            all_evaluations = []
    else:
        print(f"No existing all_annotations file found at {all_annotations_path}")
    
    if os.path.exists(best_annotations_path):
        print(f"Loading existing best_annotations from {best_annotations_path}")
        try:
            with open(best_annotations_path, 'r') as f:
                selected_evaluations = json.load(f)
            print(f"Loaded {len(selected_evaluations)} existing selected_evaluations")
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading best_annotations: {e}. Starting with empty list.")
            selected_evaluations = []
    else:
        print(f"No existing best_annotations file found at {best_annotations_path}")
else:
    print("Starting with empty annotations")
    all_evaluations = []
    selected_evaluations = []

# Split work evenly among processes
items_per_proc = n // args.n_proc
remainder = n % args.n_proc

# Calculate start and end indices for this process
start_idx = args.proc_id * items_per_proc
if args.proc_id < remainder:
    start_idx += args.proc_id
    end_idx = start_idx + items_per_proc + 1
else:
    start_idx += remainder
    end_idx = start_idx + items_per_proc

print(f"Process {args.proc_id}/{args.n_proc}: processing items {start_idx} to {end_idx-1} (total: {end_idx-start_idx})")

for i in tqdm(range(start_idx, end_idx)):
    if i >= len(metadatas):
        continue 
    i = shuffled_indices[i]
    outpath = os.path.join(ROOT,f"{i:0>5}") # padded with zeros
    metadata = metadatas[i]
    if imgs_folder is not None and args.re_eval:
        imgs_path = os.path.join(imgs_folder,f"{i:0>5}", "samples")
    else:
        imgs_path = os.path.join(outpath, "samples")
    sample_path = os.path.join(outpath, "samples")

    try:
        os.makedirs(sample_path, exist_ok=True)
    except:
        pass

    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "metadata.jsonl"), mode="w") as fp:
        json.dump(metadata, fp)
    
    prompt = metadata.get('prompt') or metadata.get('text')
    if prompt is None:
        raise ValueError("Metadata entry missing 'prompt' or 'text' field")
    metadata['prompt'] = prompt


    print(f"Prompt:{prompt}")
    all_scores = []
    imgs = []
    sample_eval_cache = {}
    sample_rng_states = {}
    for j in range(args.num_samples):
        outpath_file = os.path.join(imgs_path, f"img_{j:05}.{args.fmt}")
        outpath_file_json = os.path.join(sample_path, f"{j:05}_metadata.json")
        outpath_latents = os.path.join(sample_path, f"{j:05}_latent.pt")
        outpath_file_abs = os.path.abspath(outpath_file)
        if prod_clip_sana_mode and not args.re_eval and not args.cont and not external_verifier:
            # Snapshot RNG state so the selected sample can be regenerated exactly once for final save.
            sample_rng_states[j] = sample_generators[j].get_state()
            generated = generation_forward(
                pipe=pipeline,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                width=image_width,
                height=image_height,
                generator=sample_generators[j],
            ).images
            img = generated[0] if isinstance(generated, list) else generated
            evaluation_results = feedback.evaluate_image(img, metadata)
            all_scores.append(evaluation_results)
            sample_eval_cache[j] = evaluation_results
            all_evaluations.append({
                'correct_verifier': evaluation_results['correct'],
                'score': evaluation_results['score'].item() if hasattr(evaluation_results['score'], 'item') else float(evaluation_results['score']),
                'prompt': prompt,
                'gen_idx_gt': j,
                'filename': outpath_file,
            })
            continue
        if prod_hidden_sana_mode and not args.re_eval and not args.cont and hidden_block_idx is not None:
            # Snapshot RNG state so best-sample full generation can replay exactly the same noise stream as dev mode.
            sample_rng_states[j] = sample_generators[j].get_state()
            split_runner = TransformerSplitRunner(pipeline.transformer, hidden_block_idx)
            with split_runner:
                _ = generation_forward(
                    pipe=pipeline,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    output_type="latent",
                    width=image_width,
                    height=image_height,
                    generator=sample_generators[j],
                ).images
            activations = catcher.pop_activations()
            act = activations[mm_vision_select_feature]
            evaluation_results = feedback.evaluate_image(act, metadata)
            all_scores.append(evaluation_results)
            sample_eval_cache[j] = evaluation_results
            sample_annotation = {
                'correct_verifier': evaluation_results['correct'],
                'score': evaluation_results['score'].item() if hasattr(evaluation_results['score'], 'item') else float(evaluation_results['score']),
                'prompt': prompt,
                'gen_idx_gt': j,
                'filename': outpath_file,
            }
            all_evaluations.append(sample_annotation)
            with open(outpath_file_json, 'w') as f:
                json.dump(sample_annotation, f)
            continue
        if  os.path.exists(outpath_file):
            if args.re_eval:
                print(f"Re-evaluating existing file: {outpath_file}")
                imgs.append(Image.open(outpath_file))
            else:
                if args.cont:
                    print(f"Skipping existing file: {outpath_file}")
                    continue
                latents = generation_forward(pipe=pipeline,
                    prompt=prompt, 
                    num_inference_steps=num_inference_steps, 
                    output_type="latent" if latent_verifier and not vision_tower else "pil",
                    width=image_width, height=image_height,
                    generator=sample_generators[j]
                ).images
                if "hidden" in vision_tower:
                    activations = catcher.pop_activations()
                    act = activations[mm_vision_select_feature]
                if isinstance(latents, torch.Tensor):
                    imgs.append(latents)
                    torch.save(latents, outpath_latents)
                    with torch.no_grad():
                        decoded_image = pipeline.decode_latent(latents.to(pipeline.dtype), use_resolution_binning=use_resolution_binning, orig_height=image_height, orig_width=image_width)
                    decoded_image.images[0].save(outpath_file)
                else:
                    img=latents[0]
                    img.save(outpath_file)
                    imgs.append(img)
        else:
            latents = generation_forward(pipe=pipeline,
                    prompt=prompt, 
                    num_inference_steps=num_inference_steps, 
                    output_type="latent"if latent_verifier and not vision_tower else "pil",
                    width=image_width, height=image_height,
                    generator=sample_generators[j]
                ).images
            if "hidden" in vision_tower:
                    activations = catcher.pop_activations()
                    act = activations[mm_vision_select_feature]
            if isinstance(latents, torch.Tensor):
                imgs.append(latents)
                torch.save(latents, outpath_latents)
                with torch.no_grad():
                    decoded_image = pipeline.decode_latent(latents.to(pipeline.dtype), use_resolution_binning=use_resolution_binning, orig_height=image_height, orig_width=image_width)
                decoded_image.images[0].save(outpath_file)
            else:
                img=latents[0]
                img.save(outpath_file)
                imgs.append(img)
        evaluation_results = None
        score_result = None
        sample_annotation = None
        if latent_verifier:
            if "hidden" in vision_tower:
                evaluation_results = feedback.evaluate_image(act,metadata)
            else:
                fd = os.open(outpath_file, os.O_RDONLY)
                os.close(fd)
                evaluation_results = feedback.evaluate_image(outpath_file if vision_tower and vision_tower != "dummy" else latents,metadata)
            score_result = evaluation_results
        # 1. model uses `feedback` to generate image
        # 2. `verifier` is used to generate evaluation metrics, saved in variable `actual_results`
        # 3. in actual_results, we add two new fields `correct_verifier` and text_feedback_verifier,
        #    In this context,thet come from `feedback` model in python code, not `verifier` model
        # 4. `verifier`  would be a dummpy object if using custom prompts

        if latent_verifier:
            if os.path.exists(outpath_file_json):
                with open(outpath_file_json, 'r') as f:
                    actual_results = json.load(f)
                    #gt_score_field = "score_verifier"
            else:
                if verifier is not None:
                    actual_results = verifier.evaluate_image(outpath_file_abs, metadata)
                else:
                    actual_results = {}
            if gt_score_field in actual_results:
                actual_results["score_verifier"] = actual_results[gt_score_field]
            actual_results['correct_verifier'] = evaluation_results['correct']
            actual_results['score'] = evaluation_results["score"].item()
            actual_results['prompt'] = prompt
            actual_results['gen_idx_gt'] = j
            actual_results['filename'] = outpath_file
            sample_annotation = actual_results

        if score_result is not None:
            all_scores.append(score_result)
        if sample_annotation is not None:
            all_evaluations.append(sample_annotation)
            with open(outpath_file_json,'w') as f:
                json.dump(sample_annotation, f)

    if prod_clip_sana_mode and not args.re_eval and not args.cont and not external_verifier:
        best_index = get_best_index(all_scores)
        replay_generator = torch.Generator()
        replay_generator.set_state(sample_rng_states[best_index])
        best_generation = generation_forward(
            pipe=pipeline,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            width=image_width,
            height=image_height,
            generator=replay_generator,
        ).images
        best_image = best_generation[0] if isinstance(best_generation, list) else best_generation
        outpath_file = os.path.join(sample_path, f"best_{best_index:05}.{args.fmt}")
        best_image.save(outpath_file)
        actual_results = verifier.evaluate_image(outpath_file, metadata) if verifier is not None else {}
        selected_evaluations.append(actual_results)

        # Save only selected-sample metadata in clip-prod mode.
        best_eval = sample_eval_cache.get(best_index)
        if best_eval is not None:
            outpath_file_json = os.path.join(sample_path, f"{best_index:05}_metadata.json")
            best_sample_annotation = {
                'correct_verifier': best_eval['correct'],
                'score': best_eval['score'].item() if hasattr(best_eval['score'], 'item') else float(best_eval['score']),
                'prompt': prompt,
                'gen_idx_gt': best_index,
                'filename': outpath_file,
            }
            if gt_score_field in actual_results:
                best_sample_annotation['score_verifier'] = actual_results[gt_score_field]
            with open(outpath_file_json, 'w') as f:
                json.dump(best_sample_annotation, f)

        try:
            with open(all_annotations_path, 'w') as f:
                json.dump(all_evaluations, f)
            with open(best_annotations_path, 'w') as f:
                json.dump(selected_evaluations, f)
        except Exception as e:
            print(f"Warning: Failed to save annotations during iteration: {e}")
        continue

    if prod_hidden_sana_mode and not args.re_eval and not args.cont and hidden_block_idx is not None and not external_verifier:
        best_index = get_best_index(all_scores)
        best_outpath_file = os.path.join(imgs_path, f"img_{best_index:05}.{args.fmt}")
        replay_generator = torch.Generator()
        replay_generator.set_state(sample_rng_states[best_index])
        best_generation = generation_forward(
            pipe=pipeline,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            width=image_width,
            height=image_height,
            generator=replay_generator,
        ).images
        best_image = best_generation[0] if isinstance(best_generation, list) else best_generation
        best_image.save(best_outpath_file)
        outpath_file = os.path.join(sample_path, f"best_{best_index:05}.{args.fmt}")
        best_image.save(outpath_file)
        actual_results = verifier.evaluate_image(outpath_file, metadata) if verifier is not None else {}
        selected_evaluations.append(actual_results)

        # Persist best-sample metadata for prod mode
        best_eval = sample_eval_cache.get(best_index)
        if best_eval is not None:
            outpath_file_json = os.path.join(sample_path, f"{best_index:05}_metadata.json")
            best_sample_annotation = {
                'correct_verifier': best_eval['correct'],
                'score': best_eval['score'].item() if hasattr(best_eval['score'], 'item') else float(best_eval['score']),
                'prompt': prompt,
                'gen_idx_gt': best_index,
                'filename': best_outpath_file,
            }
            if gt_score_field in actual_results:
                best_sample_annotation['score_verifier'] = actual_results[gt_score_field]
            with open(outpath_file_json, 'w') as f:
                json.dump(best_sample_annotation, f)

        # Save annotations after each iteration
        try:
            with open(all_annotations_path, 'w') as f:
                json.dump(all_evaluations, f)
            with open(best_annotations_path, 'w') as f:
                json.dump(selected_evaluations, f)
        except Exception as e:
            print(f"Warning: Failed to save annotations during iteration: {e}")
        continue

    if external_verifier:
        best_index, best_image = scorer.get_top_k(sample_path.split("samples")[0], 1, 
                                                  args.num_samples,
                                                    None, batch_size=None, return_filename=True, 
                                                    use_tournament=config.get("use_tournament", True))
        best_image=Image.open(best_image[0])
        best_index=best_index[0]
    else:
        best_index = get_best_index(all_scores)
        best_latent = imgs[best_index]
        if isinstance(best_latent, torch.Tensor):
            with torch.no_grad():
                best_image = pipeline.decode_latent(best_latent.to(pipeline.dtype), use_resolution_binning=use_resolution_binning, orig_height=image_height, orig_width=image_width)
        else:
            best_image=best_latent
    outpath_file = os.path.join(sample_path, f"best_{best_index:05}.{args.fmt}")    
    try:
        best_image.images[0].save(outpath_file)
    except:
        best_image.save(outpath_file)
    actual_results = verifier.evaluate_image(outpath_file, metadata) if verifier is not None else {}
    selected_evaluations.append(actual_results)
    
    # Save annotations after each iteration
    try:
        with open(all_annotations_path, 'w') as f:
            json.dump(all_evaluations, f)
        with open(best_annotations_path, 'w') as f:
            json.dump(selected_evaluations, f)
    except Exception as e:
        print(f"Warning: Failed to save annotations during iteration: {e}")
    
# Final save of all annotations (redundant but kept for consistency)
try:
    with open(all_annotations_path, 'w') as f:
        json.dump(all_evaluations, f)
    with open(best_annotations_path, 'w') as f:
        json.dump(selected_evaluations, f)
    print(f"Final annotations saved to {all_annotations_path} and {best_annotations_path}")
except Exception as e:
    print(f"Error saving final annotations: {e}")
        
accelerator.wait_for_everyone()
