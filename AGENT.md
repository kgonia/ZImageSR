# AGENT.md — What You Need to Know

## What This Repo Actually Is

This implements **Stage 0 only** of the FluxSR paper (arXiv 2502.01993) — offline (eps, z0) pair generation using the Z-Image model instead of FLUX. It is roughly 15% of the full FluxSR pipeline. There is no training loop, no student model, no LoRA fine-tuning, no losses (FTD/Reconstruction/ADL), and no RealESRGAN degradation yet.

The core math: sample random noise `eps`, run it through the diffusion model to get clean latent `z0`. The teacher velocity is `u = eps - z0` (rectified flow). These pairs are saved to disk for later distillation training.

## Model & Architecture

- **Model**: `Tongyi-MAI/Z-Image-Turbo` via `diffusers.ZImageImg2ImgPipeline`
- **Transformer**: ZImageTransformer2DModel — 6.15B params, 30 layers, dim=3840, single-stream attention (S3-DiT), patch_size=2
- **Text encoder**: ~4B params, cap_feat_dim=2560. Offloaded to CPU after null prompt embedding is cached (saves ~8GB VRAM)
- **VAE**: 84M params, spatial compression 8x, `scaling_factor` applied to latents. At 1024px input, latent shape is `[1, 16, 128, 128]` (note: 16 channels, not 4)
- The pipeline uses `output_type="latent"` to get `z0` directly — no VAE round-trip quantization loss

## Key Design Decisions

1. **Two separate generators per sample**: `eps_gen` (seed `base_seed + i`) for noise sampling, `pipe_gen` (seed `base_seed + n + i`) for pipeline internals. This prevents the pipeline's internal RNG consumption from corrupting eps reproducibility.

2. **Null prompt caching**: Prompt embeddings are encoded once, saved to `null_prompt_embeds.pt`, and reused. The text encoder is then offloaded to CPU. This is critical for fitting the 6B transformer + 84M VAE in VRAM.

3. **Img2Img with zero image**: The pipeline is `ZImageImg2ImgPipeline` but receives `strength=1.0` and a black (all-zeros) init image. This effectively makes it a txt2img call while allowing the `latents=` parameter to inject `eps`.

4. **Pipeline signature introspection**: The code inspects `pipe.__call__` and `pipe.encode_prompt` signatures at runtime to handle different diffusers versions gracefully (`inspect_pipe_support`, `try_encode_prompt`).

## Running Tests

```bash
uv run python -m pytest          # 51 unit tests, ~2s
uv run python -m pytest -m slow  # integration tests (need GPU + RUN_ZIMAGESR_INTEGRATION=1)
```

- **Always use `uv run`** — the project uses `uv` for dependency management, not pip/conda
- pytest is a dev dependency: `uv add --dev pytest` if missing
- Integration tests are gated behind `RUN_ZIMAGESR_INTEGRATION=1` env var AND require CUDA

## Test Architecture

- `tests/conftest.py`: Shared fixtures — `FakePipe` (configurable return_mode, consume_generator, latent_offset), `FakeVAE` (tracks encode/decode calls), `FakeTextEncoder`, `gather_config_factory`
- `tests/test_offline_pairs.py`: 31 unit tests covering the full gather pipeline without GPU
- `tests/test_cli.py`: CLI argument parsing and subcommand dispatch
- `tests/test_s3_io.py`: S3 URI parsing and upload/download logic (mocked boto3)
- `tests/test_zenml_pipeline.py`: Step decorator assertions via regex on source file
- `tests/test_integration.py`: Real GPU end-to-end tests

The FakePipe is the central testing abstraction. It supports `return_mode` ("tensor", "list", "pil"), optional generator consumption for seed independence testing, and configurable `latent_offset` to verify z0 = eps + offset.

## Output Structure

```
out_dir/
  metadata.json           # model info, config, latent shape
  null_prompt_embeds.pt   # cached prompt embeddings
  debug_trace.jsonl       # per-sample tensor stats (if debug=True)
  pairs/
    000000/
      eps.pt              # random noise latent [1, 16, H/8, W/8]
      z0.pt               # clean latent from diffusion model
      x0.png              # decoded HR image (optional, save_x0_png)
      lr.png              # 4x downscaled (optional, generate_lr)
      lr_up.png           # lr upscaled back to HR size (optional)
```

## Known Gaps vs FluxSR Paper

- Paper uses 2,400 pairs; default `GatherConfig.n` is 1,200 (ZenML pipeline defaults to 2,400)
- Degradation is simple bicubic 4x downscale; paper uses RealESRGAN pipeline
- No distillation training (FTD + Reconstruction + ADL losses)
- No student model or LoRA
- `num_inference_steps=30` default is overkill for a Turbo model (typically 1-4 steps suffice)

## Gotchas

- `torch.load` calls must always use `weights_only=True` (security)
- VAE `scaling_factor` must be applied/reversed correctly: encode multiplies, decode divides
- CPU does not support float16/bfloat16 — `resolve_device_dtype` auto-falls back to float32
- `skip_existing` checks for all expected files (eps.pt, z0.pt, and optionally x0.png) — missing any one triggers regeneration
- CLI entry point is `zimagesr-data` (defined in pyproject.toml `[project.scripts]`)
