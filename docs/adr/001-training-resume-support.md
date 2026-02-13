# ADR-001: Training Resume Support

**Status:** Accepted
**Date:** 2026-02-13

## Context

FTD training runs are interrupted by OOM crashes, manual stops, and GPU quota limits. Previously the only saved artifact was LoRA adapter weights (`adapter_config.json` + `adapter_model.safetensors`). Resuming meant restarting from scratch with cold AdamW momentum, wasting compute on re-converging optimizer state.

## Decision

Add `--resume-from <checkpoint_dir>` with two auto-detected modes based on checkpoint directory contents.

### Resume modes

| Mode | Detection criteria | Behaviour |
|---|---|---|
| **Full** | `training_state.json` + `accelerator_state/` present | Restore optimizer state, RNG, step counter via `accelerator.load_state()` |
| **Weights-only** | Only `adapter_config.json` present | Load LoRA weights via `PeftModel.from_pretrained(is_trainable=True)`, fresh optimizer at step 0 |

Auto-detection keeps the CLI simple (one flag) while supporting both old checkpoints (pre-resume) and new ones transparently.

### Checkpoint directory structure (after)

```
lora_step_300/
    adapter_config.json          # LoRA config (PEFT) — existing
    adapter_model.safetensors    # LoRA weights — existing
    inference_grid.png           # optional — existing
    training_state.json          # NEW: {"global_step": 300, "config": {...}}
    accelerator_state/           # NEW: optimizer.bin, random_states_*.pkl
```

### Key design choices

1. **`is_trainable=True` for weights-only resume.** `PeftModel.from_pretrained()` defaults to `is_trainable=False` (inference mode), which freezes all LoRA parameters. The wrapper `_load_weights_only_resume_model()` passes `is_trainable=True` and asserts trainable param count > 0 as a hard guard.

2. **Config compatibility validation for full resume.** `_validate_full_resume_config()` checks structural keys (`model_id`, `lora_rank`, `lora_alpha`, `lora_dropout`, `mixed_precision`, `gradient_checkpointing`, `disable_vae_force_upcast`) between the current CLI args and the serialized checkpoint config. Mismatches raise a clear `ValueError` before `accelerator.load_state()`, which would otherwise fail with opaque shape/device errors.

3. **Validation happens before `accelerator.load_state()`.** The JSON is read and validated first, then the (expensive) state directory is loaded. This gives fast, clear failure on misconfiguration.

4. **`_save_training_state()` writes on every checkpoint.** Both periodic (`save_every`) and final saves include full state. Cost is negligible compared to the LoRA save itself.

5. **Progress bar `initial=global_step`.** On resume the tqdm bar starts at the restored step for correct ETA estimation.

6. **Backward-compatible.** Old checkpoints without `training_state.json` auto-detect as weights-only. No migration needed.

## Consequences

- Every checkpoint is ~2x larger on disk (optimizer state + RNG snapshots alongside LoRA weights).
- Checkpoints remain directly usable for inference — the training state files are optional extras.
- Full resume requires matching structural config; hyperparameters like `learning_rate`, `max_steps`, `batch_size` are deliberately excluded from validation so users can adjust them on resume.

## Test coverage

- `TestDetectResumeMode`: None/full/weights-only detection, empty dir error, nonexistent dir error (5 tests)
- `TestValidateFullResumeConfig`: matching config accepted, `lora_rank` mismatch caught (2 tests)
- `TestWeightsOnlyResumeModel`: round-trip save/load with trainable param assertion (1 test)
- `TestSaveTrainingState`: JSON + accelerator_state created, non-main process skips JSON (2 tests)
- `TestResumeRoundTrip`: save-then-detect returns FULL, JSON round-trips step counter (2 tests)
- CLI tests: `--resume-from` default None, path parsing, config round-trip (3 tests)
