# ZImageSR

FluxSR-style super-resolution pipeline for Z-Image Turbo:
- **Stage 0** — generate offline `(eps, z0, x0)` pairs and placeholder LR images
- **FTD Training** — Flow Trajectory Distillation with LoRA on the Z-Image transformer
- **SR inference** — one-step or multi-step latent refinement from a degraded latent
- inspect tensor/image/model shapes for debugging
- upload/download dataset bundles with S3
- orchestrate runs with minimal ZenML wrappers

## Current Status (Important)

This repository is a research reproduction in progress.

- Current practical result from long Phase 2 runs: outputs are still mostly blurry.
- Possible causes: undertraining, objective weighting, or implementation mismatch vs paper details.
- No confirmed paper-quality reproduction yet due limited additional H100 budget.
- Keep this in mind when planning experiments and expectations.

### Known Limitations / Negative Results

- Multiple Phase 2 runs (from short to long schedules) improved color/structure but did not consistently recover sharp high-frequency detail.
- Some later checkpoints produced over-textured/noisy outputs instead of clean sharpening.
- Best qualitative checkpoints were often mid-run; final checkpoints were not reliably best.
- Current implementation should be treated as an experimental baseline, not a validated FluxSR reproduction.

## 1. Setup

```bash
uv sync
uv pip install -e .
```

For FTD training, install the optional training dependencies:

```bash
uv pip install -e ".[training]"
```

This adds `peft` (LoRA), `lpips` (perceptual loss), and `wandb` (native Phase 2 logging).

For RealESRGAN-style Phase 1 degradation, install:

```bash
uv pip install -e ".[degradation]"
```

This adds `basicsr` and `opencv-python`.

## 2. Generate Data (Phase 1)

```bash
zimagesr-data gather \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --out-dir ./zimage_offline_pairs \
  --n 2400 \
  --hr-size 1024 \
  --debug \
  --debug-every 1
```

Resume example:

```bash
zimagesr-data gather --out-dir ./zimage_offline_pairs --start-index 285 --n 1200
```

Generate placeholder LR/LR-up files:

```bash
zimagesr-data degrade --out-dir ./zimage_offline_pairs --n 1200
```

Generate LR/LR-up files with RealESRGAN second-order degradation:

```bash
zimagesr-data degrade \
  --out-dir ./zimage_offline_pairs \
  --n 1200 \
  --degradation realesrgan \
  --seed 1234
```

## 3. Debug and Inspect

Inspect saved samples and metadata:

```bash
zimagesr-data inspect --out-dir ./zimage_offline_pairs --limit 5
```

Main debug artifacts:
- `zimage_offline_pairs/metadata.json`
- `zimage_offline_pairs/debug_trace.jsonl`
- `zimage_offline_pairs/pairs/<sample_id>/eps.pt`
- `zimage_offline_pairs/pairs/<sample_id>/z0.pt`
- `zimage_offline_pairs/pairs/<sample_id>/x0.png`

## 4. VAE Round-Trip Quality Check

Before training, verify that VAE encode/decode preserves image quality.
If decoding `z0.pt` already looks soft compared to `x0.png`, that softness
is your quality ceiling — no amount of LoRA training will produce crisper results.

```bash
# Check first 10 samples
uv run zimagesr-data decode-check \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --limit 10

# Check specific samples
uv run zimagesr-data decode-check \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --ids 000000,000010,000020
```

Each sample gets:
- `z0_decoded.png` — VAE decode of the z0 latent
- `z0_roundtrip_grid.png` — side-by-side comparison: `x0.png (original)` vs `z0 decoded (VAE round-trip)`

## 5. S3 Sync

Upload:

```bash
zimagesr-data s3-upload \
  --out-dir ./zimage_offline_pairs \
  --s3-uri s3://YOUR_BUCKET/zimagesr/run-001
```

Download:

```bash
zimagesr-data s3-download \
  --out-dir ./zimage_offline_pairs \
  --s3-uri s3://YOUR_BUCKET/zimagesr/run-001
```

## 6. FTD Training

FTD (Flow Trajectory Distillation) trains a LoRA adapter on the Z-Image transformer
so it can predict clean latents from degraded ones in a single forward pass.
Implements FluxSR paper Eq. 16/17 (FTD loss) and Eq. 18/21 (pixel reconstruction loss).

### Prepare zL latents

If your pairs directory does not yet contain `zL.pt` files (VAE-encoded LR images),
generate them first:

```bash
zimagesr-data generate-zl \
  --out-dir ./zimage_offline_pairs \
  --model-id Tongyi-MAI/Z-Image-Turbo
```

### Run training

```bash
zimagesr-data train \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --max-steps 750 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 5e-5 \
  --tl 0.25 \
  --lora-rank 16 \
  --save-dir ./zimage_sr_lora_runs/ftd_run \
  --save-every 150
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--tl` | 0.25 | Truncation level TL |
| `--rec-loss-every` | 8 | Pixel recon loss frequency (0 to disable) |
| `--lambda-tvlpips` | 1.0 | Weight for TV-LPIPS recon loss |
| `--lambda-z0` | 0.0 | Weight for latent endpoint loss `SmoothL1(z0_hat, z0)` |
| `--lambda-adl` | 0.0 | ADL regularization weight (set > 0 to enable) |
| `--detach-recon` / `--no-detach-recon` | on | Gradient-free recon (saves VRAM) |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | on | Reduce VRAM at cost of speed |
| `--mixed-precision` | no | `no`, `fp16`, or `bf16` |
| `--seed` | none | Reproducibility seed |
| `--save-dir` | `./zimage_sr_lora_runs/ftd_run_<timestamp>` | Checkpoint output directory |
| `--wandb` / `--no-wandb` | off | Enable native WandB logging for training |
| `--wandb-project` | zimagesr | WandB project name |
| `--wandb-mode` | online | `online` or `offline` |
| `--wandb-log-checkpoints` | on | Log saved LoRA checkpoints as model artifacts |
| `--save-full-state` / `--no-save-full-state` | off | Save optimizer/scheduler state for resume (large files) |
| `--checkpoint-infer-grid` / `--no-checkpoint-infer-grid` | off | Save checkpoint-time inference grids (one-step + optional multi-step sweeps) |
| `--wandb-log-checkpoint-grids` / `--no-wandb-log-checkpoint-grids` | on | Log checkpoint inference grids as WandB images |
| `--checkpoint-eval-ids` | empty | Fixed pair IDs used for checkpoint grids (instead of random batch sample) |
| `--checkpoint-eval-images-dir` | none | Folder of arbitrary images to include in checkpoint grids |
| `--checkpoint-eval-images-limit` | 4 | Max number of images loaded from `--checkpoint-eval-images-dir` |
| `--checkpoint-eval-input-upscale` | 4.0 | Bicubic upscale factor before VAE encode for eval images |
| `--checkpoint-eval-fit-multiple` | 16 | Resize eval images to a multiple before VAE encode |
| `--checkpoint-sr-scales` | 1.3,1.6 | Extra sr_scale values rendered in checkpoint grids (empty to disable) |
| `--checkpoint-refine-steps` | empty | Extra multi-step refinement counts rendered in checkpoint grids (e.g. `4,8,16`) |
| `--resume-from` | none | Resume from a checkpoint directory (auto-detects mode) |

Enable WandB example:

```bash
zimagesr-data train \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --wandb \
  --wandb-project zimagesr \
  --wandb-run-name ftd-run-001
```

Checkpoint inference-grid example:

```bash
zimagesr-data train \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --save-every 500 \
  --checkpoint-infer-grid \
  --checkpoint-eval-ids 000000,000123,000777 \
  --checkpoint-eval-images-dir ./eval_images \
  --checkpoint-sr-scales 1.3,1.6 \
  --checkpoint-refine-steps 4,8 \
  --wandb \
  --wandb-log-checkpoint-grids
```

Each checkpoint grid renders a baseline plus configured sweeps, e.g.:
`LR | Base SR | LoRA SR (1.0, 1-step) | LoRA SR (1.3) | LoRA SR (1.6) | LoRA SR (4-step) | LoRA SR (8-step) | HR`.
`--checkpoint-sr-scales` controls correction-strength sweep and `--checkpoint-refine-steps`
controls multi-step refinement sweep.

`--checkpoint-infer-grid` runs extra forward/decoder passes at each checkpoint step, so it adds runtime and some transient VRAM usage. Keep it off for max throughput.

### Resume training

If a run is interrupted, resume with `--resume-from`.
Resume mode is auto-detected from checkpoint contents:

Important: by default checkpoints are lightweight LoRA-only.
Use `--save-full-state` during training if you need seamless full-state resume.

```bash
# Full resume — restores optimizer state, RNG, and step counter
zimagesr-data train \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --resume-from ./zimage_sr_lora_runs/ftd_run/lora_step_300 \
  --save-dir ./zimage_sr_lora_runs/ftd_run \
  --max-steps 750
```

Two modes are auto-detected from the checkpoint directory contents:

| Mode | Detected when | Behaviour |
|---|---|---|
| **Full** | `training_state.json` + `accelerator_state/` present | Seamless resume: optimizer momentum, RNG, and step counter restored |
| **Weights-only** | Only `adapter_config.json` present | Warm restart: LoRA weights loaded, fresh optimizer at step 0 |

For **full** resume, the trainer now validates key structural settings
(`model_id`, LoRA rank/alpha/dropout, and selected training structure flags)
against the checkpoint config before loading state, and raises a clear error on mismatch.

Weights-only mode is useful for resuming from older checkpoints (before this feature)
or from checkpoints saved by other tools:

```bash
# Weights-only resume — loads LoRA weights, starts fresh optimizer at step 0
zimagesr-data train \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --resume-from ./old_checkpoint_without_state \
  --max-steps 750
```

### Checkpoint directory structure

```
lora_step_300/
    adapter_config.json          # LoRA config (PEFT)
    adapter_model.safetensors    # LoRA weights
    inference_grid_*.png         # (optional) checkpoint inference previews (one per eval sample)
    training_state.json          # (--save-full-state) step counter + serialized config
    accelerator_state/           # (--save-full-state) optimizer state, RNG, scheduler
```

By default only LoRA weights are saved (lightweight). Pass `--save-full-state`
to also save optimizer/scheduler state for seamless resume.
LoRA checkpoints are saved as PEFT adapters and remain directly usable for inference
even without the training state files.

### Dataset layout

Each sample directory under `pairs/` should contain:

```
pairs/000000/
  eps.pt        # noise tensor (1, 16, 128, 128)
  z0.pt         # clean latent  (1, 16, 128, 128)
  zL.pt         # degraded latent (1, 16, 128, 128)
  x0.png        # (optional) HR ground truth for recon loss
  lr_up.png     # (optional) upscaled LR image for zL generation
```

## 7. SR Inference (One-step + Multi-step)

After training, run single-step super-resolution from Python:

```python
import torch
from diffusers import ZImageImg2ImgPipeline, ZImageTransformer2DModel
from zimagesr.training.inference import one_step_sr
from zimagesr.training.lora import load_lora_for_inference
from zimagesr.training.transformer_utils import prepare_cap_feats

device, dtype = "cuda", torch.bfloat16

# Load pipeline (needed for VAE) and base transformer
pipe = ZImageImg2ImgPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", torch_dtype=dtype
).to(device)

base_tr = pipe.transformer
base_tr.requires_grad_(False)

# Load LoRA adapter
lora_tr = load_lora_for_inference(base_tr, "path/to/lora_final", device, dtype)

# Prepare null caption features
cap_feats = prepare_cap_feats(pipe, device, dtype)  # (1, 2560)

# Load a degraded latent
zL = torch.load("path/to/pairs/000000/zL.pt", weights_only=True).to(device=device, dtype=dtype)

# Run one-step SR
pil_image = one_step_sr(
    transformer=lora_tr,
    vae=pipe.vae,
    lr_latent=zL,
    tl=0.25,
    sr_scale=1.0,
    refine_steps=1,
    vae_sf=0.3611,
    cap_feats_2d=cap_feats,
)
pil_image.save("sr_output.png")
```

CLI from existing pair directory (`zL.pt`):

```bash
zimagesr-data infer \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --lora-path ./zimage_sr_lora_runs/ftd_run/lora_final \
  --pair-dir ./zimage_offline_pairs/pairs/000000 \
  --sr-scale 1.0,1.2 \
  --refine-steps 1,8 \
  --compare-grid \
  --output ./sr_from_pair.png
```

CLI from arbitrary input image:

```bash
zimagesr-data infer \
  --model-id Tongyi-MAI/Z-Image-Turbo \
  --lora-path ./zimage_sr_lora_runs/ftd_run/lora_final \
  --input-image ./my_input.png \
  --input-upscale 4.0 \
  --fit-multiple 16 \
  --sr-scale 0.9 \
  --refine-steps 1,8 \
  --compare-grid \
  --output ./sr_from_image.png
```

Notes for `infer`:
- In `--pair-dir` mode, the command loads `zL.pt` directly (paper notation).
- In `--input-image` mode, the image is RGB-converted, optionally bicubic-upscaled (`--input-upscale`), then resized to dimensions divisible by `--fit-multiple` before VAE encoding.
- Set `--input-upscale 1.0` if your input is already in the intended pre-upscaled space.
- `--sr-scale` controls correction strength at inference (`z0_hat = zL - sr_scale * v(TL) * TL`).
- `--refine-steps` controls Euler refinement steps from `t=TL` to `0` (`1` reproduces one-step inference).
- Add `--compare-grid` to save `<output>_grid.png` with `LR (decoded) | Base SR | LoRA SR ...` and optional `HR (ground truth)` if `x0.png` exists in `--pair-dir`.

## 8. ZenML Minimal Pipelines

Run gather pipeline:

```bash
zimagesr-data zenml-run --mode gather --out-dir ./zimage_offline_pairs --n 1200
```

Run gather pipeline and upload to S3:

```bash
zimagesr-data zenml-run \
  --mode gather \
  --out-dir ./zimage_offline_pairs \
  --s3-uri s3://YOUR_BUCKET/zimagesr/run-001
```

Run download pipeline:

```bash
zimagesr-data zenml-run \
  --mode download \
  --s3-uri s3://YOUR_BUCKET/zimagesr/run-001 \
  --out-dir ./zimage_offline_pairs
```

## 9. ZenML Stack Bootstrap

The repository ships with `zenml.yaml`, used by the dedicated bootstrap helper.

1. Edit `zenml.yaml`:
- set `components.artifact_stores.s3.enabled: true` for S3
- set S3 path to your bucket
- set `components.experiment_tracker.wandb.enabled: true` for WandB
- optionally set `entity` for WandB

2. Export optional credentials:

```bash
export WANDB_API_KEY=...
export AWS_PROFILE=...
```

3. Bootstrap ZenML components/stacks:

```bash
zimagesr-zenml-bootstrap --config zenml.yaml
```

Dry-run preview:

```bash
zimagesr-zenml-bootstrap --config zenml.yaml --dry-run
```

Override activated stack:

```bash
zimagesr-zenml-bootstrap --config zenml.yaml --activate-stack zimagesr-s3-stack
```

## Notes

- The gather pipeline requires a Z-Image pipeline variant that accepts `latents=`.
- FTD training requires a GPU with sufficient VRAM (tested on 40 GB A100). Reduce `--batch-size` and increase `--gradient-accumulation-steps` for smaller GPUs.
- `torch`/CUDA installation is environment-specific; install the wheel that matches your CUDA runtime.
- S3 sync uses `boto3` default credential chain.
- `peft` and `lpips` are only needed for training and are not required for data generation or S3 sync.

## 10. Citation

If you use this repository, cite both the implementation and the upstream papers.

```bibtex
@software{zimagesr_2026,
  title  = {ZImageSR: FluxSR-style Super-Resolution Pipeline for Z-Image Turbo},
  author = {Krzysztof Gonia},
  year   = {2026},
  note   = {Local project repository}
}

@article{li2025fluxsr,
  title   = {One Diffusion Step to Real-World Super-Resolution via Flow Trajectory Distillation},
  author  = {Li, Jianze and Cao, Jiezhang and Guo, Yong and Li, Wenbo and Zhang, Yulun},
  journal = {arXiv preprint arXiv:2502.01993},
  year    = {2025}
}

@article{cai2025zimage,
  title   = {Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer},
  author  = {Cai, Huanqia and Cao, Sihan and Du, Ruoyi and Gao, Peng and Hoi, Steven and others},
  journal = {arXiv preprint arXiv:2511.22699},
  year    = {2025}
}
```
