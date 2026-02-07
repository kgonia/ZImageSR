# ZImageSR Test Plan

## 1. Unit Tests: `offline_pairs.py`

### 1.1 Pure utility functions (no GPU / no pipeline needed)

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| U01 | `_to_str_dtype` | Returns `"float32"` for `torch.float32`, `"bfloat16"` for `torch.bfloat16`, `"none"` for `None` | None |
| U02 | `append_jsonl` | Appends valid JSON lines to a temp file; multiple calls produce multiple lines; file is parseable line-by-line | `tmp_path` |
| U03 | `tensor_info` | Returns correct shape/dtype/device/stats for a known tensor; returns `None` for `None` input; `include_stats=False` omits mean/std/min/max | None |
| U04 | `tensor_info` (edge) | Empty tensor (`numel()==0`) with `include_stats=True` does not crash | None |
| U05 | `module_info` | Returns correct param counts for a small `nn.Linear`; returns `None` for `None` | None |
| U06 | `resolve_device_dtype` | `(None, None)` on CPU returns `("cpu", torch.float32, "float32")`; explicit `"cuda"` without CUDA falls back to CPU; unsupported dtype raises `ValueError` | Patch `torch.cuda.is_available` |
| U07 | `unwrap_first_tensor` | Returns tensor from list, first tensor from mixed list, `None` from empty list, `None` from `None`, passthrough for raw tensor | None |
| U08 | `normalize_prompt_embeds` | `None` returns `None`; 1D tensor returns `None`; 2D tensor wraps in list on correct device/dtype; list of mixed dims keeps only ndim>=2; raises `TypeError` for unsupported type | None |
| U09 | `simple_x4_degrade` | 1024x1024 input produces lr=(256x256) and lr_up=(1024x1024); both are PIL RGB | None |
| U10 | `should_skip_sample` | Returns `True` when eps.pt+z0.pt exist (and x0.png if `save_x0_png`); returns `False` when any required file is missing | `tmp_path` with dummy files |
| U11 | `ensure_pairs_dir` | Creates `out_dir/pairs` and returns it; idempotent on second call | `tmp_path` |

### 1.2 VAE / pipeline helpers (need mocked pipeline)

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| U12 | `vae_encode_latents_safe` | Produces tensor of expected shape for a 256x256 RGB image; scaling_factor is applied; does NOT call `pipe.vae.to()` (issue 11 fix) | Mock `pipe.vae` with a fake encode returning known latents |
| U13 | `decode_latents_to_pil` | Produces a PIL RGB image of expected size from known latents; divides by scaling_factor before decode | Mock `pipe.vae` with a fake decode returning known tensor |
| U14 | `infer_latent_shape` | Returns correct shape from mocked VAE | Mock via U12's mock |
| U15 | `inspect_pipe_support` | Returns correct dict for a mock pipeline with/without `prompt_embeds`, `latents` in `__call__` signature | Mock pipeline class with controlled `__call__` signature |
| U16 | `try_encode_prompt` | Forwards correct kwargs based on `encode_prompt` signature; raises `RuntimeError` if method missing | Mock pipeline |
| U17 | `load_or_encode_null_prompt_embeds` | Loads from cache when file exists (with `weights_only=True`); encodes and saves when not cached; restores text_encoder device after encoding | Mock pipeline + `tmp_path` |

### 1.3 torch.load safety (issue 8 fix)

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| U18 | `load_or_encode_null_prompt_embeds` | Cached path uses `weights_only=True` - verify by saving a non-tensor pickle and confirming it raises | `tmp_path` |
| U19 | `inspect_local_pairs` | `torch.load` calls use `weights_only=True` - save valid tensors, confirm they load; save invalid pickle, confirm rejection | `tmp_path` |

### 1.4 Generator separation (issue 7 fix)

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| U20 | `gather_offline_pairs` (generator isolation) | For a given sample index `i`, verify that `eps` is deterministic regardless of pipeline internals: run twice with different `num_inference_steps` and confirm same `eps.pt` content | Mock pipeline returning dummy latents |
| U21 | `gather_offline_pairs` (pipe generator) | Verify the pipeline receives a generator seeded at `base_seed + n + i`, not the same generator used for eps | Capture `generator` kwarg from mocked pipeline `__call__` |

### 1.5 Latent output path (issue 6 fix)

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| U22 | `gather_offline_pairs` (output_type) | Verify pipeline is called with `output_type="latent"` | Capture kwargs from mocked pipeline `__call__` |
| U23 | `gather_offline_pairs` (z0 from latent) | When pipeline returns a tensor, z0 is saved directly without VAE re-encode | Mock pipeline returning known latent; load saved z0.pt and compare |
| U24 | `gather_offline_pairs` (z0 from list) | When pipeline returns a list of tensors, first element is used as z0 | Mock pipeline returning `[tensor]` |
| U25 | `gather_offline_pairs` (x0 decode) | When `save_x0_png=True`, x0.png is produced via `decode_latents_to_pil`; when `False`, no x0.png exists | Mock pipeline + mock VAE decode |
| U26 | `gather_offline_pairs` (debug trace x0=None) | When `save_x0_png=False`, debug JSONL entry has `"x0": null` | Parse debug_trace.jsonl |

### 1.6 Metadata and debug

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| U27 | `write_metadata` | Produces valid JSON with expected keys; config values round-trip correctly | `tmp_path` + mock pipe |
| U28 | `gather_offline_pairs` (debug trace) | JSONL file is written with expected fields per sample; `debug_every=5` only writes every 5th | Mock pipeline |
| U29 | `gather_offline_pairs` (skip_existing) | Existing complete sample dirs are skipped; incomplete ones are regenerated | `tmp_path` with partial fixtures |

### 1.7 LR generation

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| U30 | `generate_lr_pairs` | Produces lr.png and lr_up.png for each sample with x0.png; skips missing x0; respects `skip_existing` | `tmp_path` with dummy x0.png files |
| U31 | `inspect_local_pairs` | Returns correct structure with metadata, sample info for eps/z0/x0/lr/lr_up | `tmp_path` with fixture data |

---

## 2. Unit Tests: `s3_io.py`

| ID | Function | Test | Mocking |
|----|----------|------|---------|
| S01 | `parse_s3_uri` | `"s3://bucket/prefix/path"` -> `("bucket", "prefix/path")`; `"s3://bucket"` -> `("bucket", "")`; invalid URIs raise `ValueError` | None |
| S02 | `upload_dir_to_s3` | Uploads all files; `include_debug=False` skips `debug*` files; counts match | Mock `boto3.client` |
| S03 | `upload_dir_to_s3` (missing dir) | Raises `FileNotFoundError` for non-existent directory | None |
| S04 | `download_dir_from_s3` | Downloads files to correct local paths; `overwrite=False` skips existing files; counts match | Mock `boto3.client` + paginator |
| S05 | `write_sync_report` | Writes valid JSON; creates parent dirs | `tmp_path` |

---

## 3. Unit Tests: `cli.py`

| ID | Test | Mocking |
|----|------|---------|
| C01 | `_add_gather_args` adds all expected arguments to a parser (verify by parsing known args) | None |
| C02 | `_gather_config_from_args` produces correct `GatherConfig` from namespace | None |
| C03 | `build_parser` accepts `gather` with all gather args | None |
| C04 | `build_parser` accepts `zenml-run` with all gather args plus zenml-specific args | None |
| C05 | `gather` and `zenml-run` share identical defaults for all GatherConfig fields | Compare parsed defaults |
| C06 | `main` with `gather` calls `gather_offline_pairs` with correct config | Patch `gather_offline_pairs` |
| C07 | `main` with `degrade` calls `generate_lr_pairs` with correct args | Patch `generate_lr_pairs` |
| C08 | `main` with `inspect` calls `inspect_local_pairs` and prints JSON | Patch `inspect_local_pairs` |
| C09 | `main` with `s3-upload` calls `upload_dir_to_s3` | Patch `upload_dir_to_s3` |
| C10 | `main` with `s3-download` calls `download_dir_from_s3` | Patch `download_dir_from_s3` |

---

## 4. Unit Tests: `zenml_pipeline.py`

| ID | Test | Mocking |
|----|------|---------|
| Z01 | `inspect_pairs_step` has caching enabled (no `enable_cache=False`) | Inspect step decorator |
| Z02 | `gather_pairs_step` has caching disabled | Inspect step decorator |
| Z03 | `upload_s3_step` has caching disabled | Inspect step decorator |
| Z04 | `download_s3_step` has caching disabled | Inspect step decorator |
| Z05 | `gather_pairs_step` constructs `GatherConfig` correctly and calls `gather_offline_pairs` | Patch `gather_offline_pairs` |

---

## 5. Integration Tests (require GPU or `--slow` marker)

| ID | Test | Notes |
|----|------|-------|
| I01 | End-to-end `gather` with `n=2`, `hr_size=256`, `num_inference_steps=1` | Verify eps.pt, z0.pt, x0.png are created with correct shapes; z0 latent shape matches VAE config |
| I02 | `gather` + `degrade` round-trip | Run gather then degrade; verify lr.png is 4x smaller than x0.png |
| I03 | `gather` restart with `start_index` | Generate n=4, delete sample 2, rerun with `skip_existing=True`; verify sample 2 regenerated, others untouched |
| I04 | `gather` with `save_x0_png=False` | Verify z0.pt and eps.pt exist but no x0.png |
| I05 | `gather` latent consistency | Compare z0 from `output_type="latent"` path against manual VAE encode of decoded x0; latent path should have lower reconstruction error (no uint8 quantization) |
| I06 | CLI `inspect` after `gather` | Run `zimagesr-data inspect` and verify JSON output has correct shapes |

---

## 6. Test Infrastructure

### Fixtures needed
- **`mock_pipeline`**: Fake `ZImageImg2ImgPipeline` with controllable `__call__`, `encode_prompt`, `vae.encode`, `vae.decode`, `text_encoder`
- **`sample_dir`**: Pre-populated `tmp_path` with dummy eps.pt, z0.pt, x0.png for skip/inspect tests
- **`gather_config`**: Factory fixture returning `GatherConfig` with test-friendly defaults (`n=2`, `hr_size=64`, `num_inference_steps=1`)

### Markers
- `@pytest.mark.slow` - integration tests requiring GPU and real model weights
- `@pytest.mark.gpu` - tests that need CUDA (subset of slow)
- Default (unmarked) - pure unit tests, run in CI without GPU

### File structure
```
tests/
    conftest.py              # shared fixtures
    test_offline_pairs.py    # U01-U31
    test_s3_io.py            # S01-S05
    test_cli.py              # C01-C10
    test_zenml_pipeline.py   # Z01-Z05
    test_integration.py      # I01-I06 (marked slow)
```
