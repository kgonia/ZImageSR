# ADR-001: Checkpoint Evaluation Grids, SR-Scale, and Training Resume

**Status:** Accepted
**Date:** 2026-02-13
**Context:** charmed-totem-14 training run, post-session review

## Context

During FTD training runs (usual-donkey-3, rich-voice-11, charmed-totem-14), we had no way to visually assess model quality at checkpoint time without manually running inference. Training loss alone is insufficient -- a decreasing FTD loss does not guarantee perceptually good SR output. We also lacked the ability to resume interrupted long runs (50k+ steps).

## Decisions

### 1. Checkpoint-Time Inference Grids

**Decision:** Render LR | Base SR | LoRA SR (+ HR if available) comparison grids at each checkpoint save, logged to WandB and saved as PNGs.

**Eval sample sources (priority order):**
1. `--checkpoint-eval-ids` — fixed pair IDs from the training dataset (e.g. `000000,000123`)
2. `--checkpoint-eval-images-dir` — arbitrary image folder, VAE-encoded at eval time
3. Fallback: current batch sample

**Rationale:** Fixed eval IDs give consistent visual tracking across runs. Arbitrary images let users test on real-world inputs without creating training pairs. The fallback ensures grids always render even with zero config.

**Trade-off:** Checkpoint saves are slower (~5-10s per sample for transformer + VAE decode). Grid rendering is wrapped in a broad `except Exception` to never crash training.

### 2. `sr_scale` Inference Parameter

**Decision:** Add `sr_scale` (default 1.0) to `one_step_sr`:
```
z0_hat = zL - (sr_scale * v) * TL
```

**Rationale:** Allows post-training control over correction strength. Values < 1.0 produce more conservative (less hallucinated) results; values > 1.0 amplify the learned correction. Useful for finding the sweet spot between sharpness and artifacts without retraining.

**Scope:** Inference-only (CLI `--sr-scale`). Not used during training or checkpoint grids (always 1.0).

### 3. Training Resume (`--resume-from`)

**Decision:** Two resume modes, auto-detected from checkpoint contents:

| Mode | Detection | Behavior |
|------|-----------|----------|
| **Full** | `training_state.json` + `accelerator_state/` present | Restores model, optimizer, LR scheduler, global_step. Validates config compatibility (model_id, lora_rank/alpha/dropout, mixed_precision, gradient_checkpointing). |
| **Weights-only** | `adapter_config.json` present, no training state | Loads LoRA adapter with `is_trainable=True`. Fresh optimizer, step counter resets to 0. |

**Known limitation:** DataLoader position is not restored on full resume. After resume, the dataloader restarts from epoch 0 with the same shuffle seed. The model trains the correct total step count but re-sees early data. Accelerate's `skip_first_batches()` could fix this but adds complexity for minimal benefit with `shuffle=True`.

**Config validation:** Full resume checks `RESUME_COMPAT_KEYS` (model_id, lora_rank, lora_alpha, lora_dropout, mixed_precision, gradient_checkpointing, disable_vae_force_upcast). Intentionally excludes batch_size, learning_rate, max_steps -- these are commonly changed on resume.

## Known Issues

1. **Duplicated helpers:** `_build_comparison_grid` and `_resize_to_multiple` exist in both `train.py` and `cli.py` with slightly different signatures (`_resize_to_multiple` returns `(img, bool)` in cli vs `img` in train). Should be extracted to a shared utility module.

2. **No negative-value validation** for `--checkpoint-eval-input-upscale` or `--checkpoint-eval-fit-multiple`.

## Alternatives Considered

- **FID/LPIPS metrics at checkpoint time:** Too expensive and requires a reference dataset. Visual grids provide faster feedback for iterative development.
- **Saving all intermediate samples:** Storage-heavy. Per-sample grids with fixed eval IDs are a good middle ground.
- **Automatic LR warmup on resume:** Considered but deferred -- constant LR with AdamW state restore is sufficient for current runs.
