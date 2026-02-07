# ZImageSR

Data-gathering utilities for the first stage of FluxSR-style training with Z-Image:
- generate offline `(eps, z0, x0)` pairs
- generate placeholder LR images
- inspect tensor/image/model shapes for debugging
- upload/download dataset bundles with S3
- orchestrate runs with minimal ZenML wrappers

## 1. Setup

```bash
uv sync
uv pip install -e .
```

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

## 5. ZenML Minimal Pipelines

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

## 6. ZenML Stack Bootstrap

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
- `torch`/CUDA installation is environment-specific; install the wheel that matches your CUDA runtime.
- S3 sync uses `boto3` default credential chain.
