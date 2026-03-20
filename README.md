<div id="top"></div>

<br />
<div align="center">

  <img src="assets/logo.png" width="100"/>

  <h1 align="center">Tiny Inference-Time Scaling with Latent Verifiers</h1>

  <h3 align="center">
   CVPR 2026 Findings
  </h3>


<div align="center">
  <a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=190">Davide Bucciarelli*</a>
  •
  <a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=185">Evelyn Turri*</a>
  •
  <a href="https://lorenzbaraldi.github.io/">Lorenzo Baraldi</a>
  <br />
  <a href="https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=90">Marcella Cornia</a>
  •
  <a href="https://www.lorenzobaraldi.com/">Lorenzo Baraldi</a>
  •
  <a href="https://www.ritacucchiara.it/">Rita Cucchiara</a>
 <p align="center"> 
  </p>
</div>
<div align="center">
  <p align="center">
    <a href="" target="_blank"><img src="https://img.shields.io/badge/-arxiv-grey" alt="Paper"></a>
  </p>
</div>


# VHS: Verifier on Hidden States

**VHS** is a latent verifier framework for best-of-N text-to-image generation. It scores candidate images — or their intermediate latent representations — against a text prompt using a lightweight multimodal language model (Qwen2.5-0.5B + LLaVA), enabling efficient selection of the best generation without running a full evaluator on every sample.

This release provides the inference pipeline and two pre-trained verifiers for evaluating [Sana-Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) (1-step diffusion) on the [GeneVal](https://github.com/djghosh13/geneval) compositional benchmark.

---

## Overview

The pipeline works as follows:

1. **Generate** N=32 candidate images from a text prompt with Sana-Sprint.
2. **Score** each candidate with a latent verifier — either at the hidden-layer level (fast) or using CLIP embeddings.
3. **Select** the highest-scoring image.
4. **Evaluate** the selected image on GeneVal (object detection + color + spatial relations via Mask2Former).

Two verifier variants are provided:

| Verifier | Vision input | HF checkpoint | Speed |
|---|---|---|---|
| **Hidden-layer** | Sana transformer block 7 activations | [`aimagelab/vhs-hidden-verifier`](https://huggingface.co/aimagelab/vhs-hidden-verifier) | Fast — evaluates before full decode |
| **CLIP** | CLIP ViT-L/14@336 embeddings | [`aimagelab/vhs-mllm-clip-verifier`](https://huggingface.co/aimagelab/vhs-mllm-clip-verifier) | Slow — evaluates on completed images |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins `transformers` and `diffusers` to specific commits for reproducibility.

---

## Checkpoints

Latent verifier weights are downloaded automatically from the Hugging Face Hub on first use.

Download the Mask2Former detector and normalization statistics required for GeneVal evaluation:

```bash
# Mask2Former detector (required for GeneVal evaluation)
huggingface-cli download aimagelab/vhs-checkpoints \
  mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth \
  --repo-type dataset --local-dir ckpts/geneval-det/

# Hidden-layer activation normalization statistics (required for the hidden-layer verifier)
huggingface-cli download aimagelab/vhs-checkpoints \
  block_7_mean_bf16.pt block_7_variance_bf16.pt \
  --repo-type dataset --local-dir ckpts/normalization/
```

---

## Running Inference

All commands should be run from the `vhs-release/` root.

### Hidden-layer verifier (fastest)

Uses Sana hidden activations at transformer block 7. Generation is interrupted early, scored, and only the winner is decoded at full quality.

```bash
python inference_scripts/sana_sprint_best_of_n_comple.py \
  --config configs/config_hidden-7_Qwen2.5-0.5B_mlp_train-val-best-CEfocal-loss_alpha-0.37_gamma-0.0_step_1_BO32_step_1_sana_sprint.yaml
```

### CLIP verifier

Uses CLIP ViT-L/14@336 embeddings on completed images.

```bash
python inference_scripts/sana_sprint_best_of_n_comple.py \
  --config configs/config_clip_orig_Qwen2.5-0.5B_mlp_train-val-best-CE-loss_step_1_BO32_sana_sprint.yaml
```

---

## Configuration

Configs are YAML files in `configs/`. Key parameters:

| Parameter | Description |
|---|---|
| `latent_verifier_name` | Which verifier to use (must match a key in `inference_scripts/latent_verifier_dict.py`) |
| `vision_tower` | Vision encoder: `hidden_7` for the latent verifier, or a CLIP model ID |
| `num_inference_steps` | Diffusion steps (1 for Sana-Sprint) |
| `image_width` / `image_height` | Output resolution (default 1024×1024) |
| `generation_mode` | `prod` enables production optimizations (early stopping + winner-only decode for hidden mode) |
| `num_samples` | Number of candidates per prompt (N in best-of-N, default 32) |

To add a new verifier, register it in `inference_scripts/latent_verifier_dict.py`:

```python
LATENT_VERIFIER = {
    "my-verifier-name": {
        "latent_verifier_path": "org/my-hf-model",
        "vision_tower": "hidden_7"          # or a CLIP model ID
    },
}
```

---

## Outputs

Results are written to `outputs/<run_name>/gen_eval_<run_name>/<prompt_id>/`:

```
<prompt_id>/
├── samples/
│   ├── img_00000.jpg … img_00031.jpg   # all N candidates
│   ├── best_XXXXX.jpg                  # selected best image
│   └── XXXXX_metadata.json             # per-sample scores and feedback
```

Aggregated result files per process:
- `all_annotations_proc_<verifier>_id_proc_<N>.json` — scores for all candidates
- `best_annotations_proc_<verifier>_id_proc_<N>.json` — GeneVal metrics for best-selected images

---

## Repository Structure

```
vhs-release/
├── configs/                        # YAML inference configs
├── data/
│   └── evaluation_metadata.json   # GeneVal prompts (100+ structured text-image pairs)
├── inference_scripts/
│   ├── sana_sprint_best_of_n_comple.py   # Main inference + evaluation script
│   ├── sana_activation_catcher.py        # PyTorch hook for capturing hidden activations
│   ├── latent_verifier_dict.py           # Registry of available verifier checkpoints
│   └── object_names.txt                  # COCO-80 class names for GeneVal
├── train_scripts/
│   ├── latent_verifier.py                # LatentGemmaVerifier / LatentGemmaFeedback classes
│   └── geneval_utils.py                  # GeneVal evaluator (Mask2Former + CLIP)
├── vhs/                            # Core VHS package
│   ├── model/
│   │   ├── llava_arch.py                 # LLaVA base architecture
│   │   ├── language_model/
│   │   │   └── llava_qwen.py             # Qwen2/Qwen3 multimodal LLM
│   │   ├── multimodal_encoder/           # Vision towers (CLIP, VAE, hidden-layer)
│   │   └── multimodal_projector/         # Vision-to-LLM projection heads
│   ├── model_loader.py                   # Model instantiation utilities
│   ├── conversation.py                   # Conversation formatting helpers
│   └── mm_utils.py                       # Multimodal utilities
├── requirements.txt
└── README.md
```

---

## Acknowledgements

This codebase builds on:

- [LLaVA](https://github.com/haotian-liu/LLaVA) — multimodal language model architecture (Apache 2.0)
- [Sana-Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers) — 1-step text-to-image diffusion model
- [GeneVal](https://github.com/djghosh13/geneval) — compositional text-to-image evaluation benchmark
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) — instance segmentation for object detection
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — CLIP vision encoders

---

## License

The VHS package (`vhs/`) is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), consistent with its upstream dependencies (LLaVA, Transformers). See individual file headers for attribution.
