import torch
from diffusers import SanaSprintPipeline
import argparse
import json
import math
import h5py
from tqdm import tqdm
import os
from verifier_scripts.modeling.utils import set_seed
import random

class ActivationCatcher:
    def __init__(self, model, modules="blocks"):
        """
        model: nn.Module (e.g. pipeline.transformer)
        modules: "all" for every submodule, or "blocks" for only transformer blocks
        """
        self.model = model
        self.handles = []
        self.activations = {}

        if modules == "all":
            for name, module in model.named_modules():
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        elif modules == "blocks":
            for i, block in enumerate(model.transformer_blocks):
                self.handles.append(block.register_forward_hook(self._make_hook(f"block_{i}")))
        else:
            raise ValueError("modules must be 'all' or 'blocks'")

    def _make_hook(self, name):
        def hook(module, input, output):
            # store detached activations (to CPU by default)
            self.activations[name] = output.detach().cpu()
        return hook

    def clear(self):
        """Clear stored activations without returning them."""
        self.activations = {}

    def pop_activations(self):
        """Return activations and clear them afterwards."""
        activs = self.activations
        self.clear()
        return activs

    def remove(self):
        """Remove all hooks permanently."""
        for h in self.handles:
            h.remove()
        self.handles = []



def main(args):
    batch_size = args.batch_size
    process_id = args.process_id
    num_processes = args.num_processes
    set_seed(42)
    pipeline = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16
        ).to("cuda")

    # Attach hooks
    catcher = ActivationCatcher(pipeline.transformer, modules="blocks")
    print(f"Loading annotations from {args.input_json}")
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    #process just the first 50000 samples:
    #data = data[:50000]
    # Split data among processes
    total_samples = len(data)
    samples_per_process = math.ceil(total_samples / num_processes)
    start_idx = process_id * samples_per_process
    end_idx = min(start_idx + samples_per_process, total_samples)
    output_hdf5 = args.output_hdf5.replace(".hdf5", f"_part{process_id}.hdf5")
    args.output_hdf5 = output_hdf5
    data = data[start_idx:end_idx]
    print(f"Process {process_id}/{num_processes}: Processing samples {start_idx} to {end_idx-1} ({len(data)} samples)")

    output_data = []
    samples_processed = 0
    
    # Determine output JSON path
    input_json_dir = os.path.dirname(args.input_json)
    input_json_filename = os.path.basename(args.input_json)
    input_json_name, input_json_ext = os.path.splitext(input_json_filename)
    output_json_path = os.path.join(input_json_dir, f"{input_json_name}_layer6_8_processed_part{process_id}{input_json_ext}")
    
    # Create HDF5 file for saving activations
    os.makedirs(os.path.dirname(args.output_hdf5), exist_ok=True)
    #data = data[:100]

    with h5py.File(args.output_hdf5, 'w') as hdf5_file:
        batch_count = 0
        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            #new_seed = random.randint(0, 1000000)
            #set_seed(new_seed)
            batch = data[i:i + batch_size]
            ids = [item["id"] for item in batch]
            captions = [item["re_cap_conversations"][1]["value"] for item in batch]
            output_image_paths = [item["image"] for item in batch]
            output_image_paths = [os.path.join(args.output_image_folder, path) for path in output_image_paths]
            basedirs = [os.path.dirname(path) for path in output_image_paths]
            [os.makedirs(basedir, exist_ok=True) for basedir in basedirs]
            
            output_images = pipeline(prompt=captions, num_inference_steps=1, intermediate_timesteps=None).images
            activations = catcher.pop_activations()
            
            # Save images
            for j in range(len(batch)):
                output_images[j].save(output_image_paths[j])
            
            # Update batch items with HDF5 path
            for j in range(len(batch)):
                batch[j]["hdf5_path"] = args.output_hdf5
                output_data.append(batch[j])
            
            # Save activations to HDF5
            for j in tqdm(range(len(batch))):
                sample_id = str(ids[j])
                # Create a group for this sample ID
                sample_group = hdf5_file.create_group(sample_id)
                # Pre-allocate list for mean calculation
                activations_for_mean = []
                # Save each block's activation
                for block_name, activation in activations.items():
                    # Extract activation for this sample from the batch
                    if block_name in ["block_6", "block_8"]:
                        sample_activation = activation[j]
                        # Convert to numpy once and reuse
                        sample_np = sample_activation.view(torch.uint16).cpu().numpy()
                        sample_group.create_dataset(block_name, data=sample_np, )
                                                   #compression="gzip", compression_opts=4)
                    if args.compute_mean:
                        activations_for_mean.append(activation[j])
                # Calculate mean more efficiently without stacking
                if activations_for_mean:
                    mean_activation = torch.stack(activations_for_mean).mean(dim=0)
                    sample_group.create_dataset("mean_activation", data=mean_activation.view(torch.uint16).cpu().numpy(),)
                                               #compression="gzip", compression_opts=4)

            samples_processed += len(batch)
            batch_count += 1

            print(f"Process {process_id}: Processed {samples_processed}/{len(data)} samples")
            print(f"Process {process_id}: Activations saved to {args.output_hdf5}")
            
            # Save JSON periodically
            if batch_count % args.save_every_n_batches == 0:
                with open(output_json_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"Process {process_id}: Updated JSON saved to {output_json_path} (after {batch_count} batches)")
    
    # Save updated JSON with HDF5 paths (final save)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Process {process_id}: Final JSON saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-caption images using Qwen2.5-VL model")
    
    # Required arguments
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to input JSON annotation file")
    parser.add_argument("--output_image_folder", type=str, required=True,
                        help="Base folder for the output images")
    parser.add_argument("--output_hdf5", type=str, required=True,
                        help="Path to save the output HDF5 file")
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference (default: 8)")
    # Distributed processing arguments
    parser.add_argument("--process_id", type=int, default=0,
                        help="ID of the current process (0-indexed, default: 0)")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="Total number of processes (default: 1)")
    parser.add_argument("--compute_mean", action="store_true", default=False)
    parser.add_argument("--save_every_n_batches", type=int, default=16,
                        help="Save updated JSON every N batches (default: 16)")
    args = parser.parse_args()
    
    # Validate distributed processing arguments
    if args.process_id < 0 or args.process_id >= args.num_processes:
        raise ValueError(f"process_id must be between 0 and {args.num_processes-1}")
    
    main(args)
