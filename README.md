# ZImageSR

FluxSR-style super-resolution pipeline for Z-Image Turbo:
- **Stage 0** — generate offline `(eps, z0, x0)` pairs and placeholder LR images
- **FTD Training** — Flow Trajectory Distillation with LoRA on the Z-Image transformer
- **One-step SR inference** — single forward pass super-resolution from a degraded latent
- inspect tensor/image/model shapes for debugging
- upload/download dataset bundles with S3
- orchestrate runs with minimal ZenML wrappers

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

## 4. S3 Sync for Phase 2

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

## 5. FTD Training (Phase 2)

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

Enable WandB example:

```bash
zimagesr-data train \
  --pairs-dir ./zimage_offline_pairs/pairs \
  --wandb \
  --wandb-project zimagesr \
  --wandb-run-name ftd-run-001
```

LoRA checkpoints are saved as PEFT adapters (`adapter_config.json` + safetensors).

### Dataset layout

Each sample directory under `pairs/` should contain:

```
pairs/0000/
  eps.pt        # noise tensor (1, 16, 128, 128)
  z0.pt         # clean latent  (1, 16, 128, 128)
  zL.pt         # degraded latent (1, 16, 128, 128)
  x0.png        # (optional) HR ground truth for recon loss
  lr_up.png     # (optional) upscaled LR image for zL generation
```

## 6. One-step SR Inference

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
zL = torch.load("path/to/pairs/0000/zL.pt", weights_only=True).to(device=device, dtype=dtype)

# Run one-step SR
pil_image = one_step_sr(
    transformer=lora_tr,
    vae=pipe.vae,
    lr_latent=zL,
    tl=0.25,
    t_scale=1000.0,
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
  --output ./sr_from_image.png
```

Notes for `infer`:
- In `--pair-dir` mode, the command loads `zL.pt` directly (paper notation).
- In `--input-image` mode, the image is RGB-converted, optionally bicubic-upscaled (`--input-upscale`), then resized to dimensions divisible by `--fit-multiple` before VAE encoding.
- Set `--input-upscale 1.0` if your input is already in the intended pre-upscaled space.

## 7. ZenML Minimal Pipelines

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

## 8. ZenML Stack Bootstrap

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
