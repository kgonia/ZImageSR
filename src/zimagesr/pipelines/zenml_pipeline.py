from __future__ import annotations

from pathlib import Path
from typing import Any

from zenml import pipeline, step

from zimagesr.data.offline_pairs import GatherConfig, gather_offline_pairs, inspect_local_pairs
from zimagesr.data.s3_io import download_dir_from_s3, upload_dir_to_s3


@step(enable_cache=False)  # Side effects: writes files to disk
def gather_pairs_step(
    model_id: str,
    out_dir: str,
    n: int,
    hr_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    base_seed: int,
    start_index: int,
    prompt: str,
    negative_prompt: str,
    save_x0_png: bool,
    generate_lr: bool,
    skip_existing: bool,
    cache_null_prompt: bool,
    offload_text_encoder: bool,
    device: str | None,
    dtype: str | None,
    debug: bool,
    debug_every: int,
) -> str:
    cfg = GatherConfig(
        model_id=model_id,
        out_dir=Path(out_dir),
        n=n,
        hr_size=hr_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        base_seed=base_seed,
        start_index=start_index,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_x0_png=save_x0_png,
        generate_lr=generate_lr,
        skip_existing=skip_existing,
        cache_null_prompt=cache_null_prompt,
        offload_text_encoder=offload_text_encoder,
        device=device,
        dtype=dtype,
        debug=debug,
        debug_every=debug_every,
    )
    gather_offline_pairs(cfg)
    return out_dir


@step
def inspect_pairs_step(out_dir: str, limit: int = 3) -> dict[str, Any]:
    return inspect_local_pairs(out_dir=out_dir, limit=limit)


@step(enable_cache=False)  # Side effects: uploads to S3
def upload_s3_step(out_dir: str, s3_uri: str, include_debug: bool = True) -> dict[str, Any]:
    result = upload_dir_to_s3(local_dir=out_dir, s3_uri=s3_uri, include_debug=include_debug)
    return result.to_dict()


@step(enable_cache=False)  # Side effects: downloads from S3
def download_s3_step(s3_uri: str, out_dir: str, overwrite: bool = False) -> str:
    download_dir_from_s3(s3_uri=s3_uri, local_dir=out_dir, overwrite=overwrite)
    return out_dir


@pipeline(name="zimagesr_gather_pipeline")
def zimagesr_gather_pipeline(
    model_id: str = "Tongyi-MAI/Z-Image-Turbo",
    out_dir: str = "./zimage_offline_pairs",
    n: int = 2400,
    hr_size: int = 1024,
    num_inference_steps: int = 5, # For quick gathering, can be increased for better quality
    guidance_scale: float = 0.0,
    base_seed: int = 12345,
    start_index: int = 0,
    prompt: str = "",
    negative_prompt: str = "",
    save_x0_png: bool = True,
    generate_lr: bool = False,
    skip_existing: bool = True,
    cache_null_prompt: bool = True,
    offload_text_encoder: bool = True,
    device: str | None = None,
    dtype: str | None = None,
    debug: bool = True,
    debug_every: int = 1,
    inspect_limit: int = 3,
    s3_uri: str | None = None,
    s3_include_debug: bool = True,
) -> None:
    generated_dir = gather_pairs_step(
        model_id=model_id,
        out_dir=out_dir,
        n=n,
        hr_size=hr_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        base_seed=base_seed,
        start_index=start_index,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_x0_png=save_x0_png,
        generate_lr=generate_lr,
        skip_existing=skip_existing,
        cache_null_prompt=cache_null_prompt,
        offload_text_encoder=offload_text_encoder,
        device=device,
        dtype=dtype,
        debug=debug,
        debug_every=debug_every,
    )
    inspect_pairs_step(out_dir=generated_dir, limit=inspect_limit)
    if s3_uri:
        upload_s3_step(out_dir=generated_dir, s3_uri=s3_uri, include_debug=s3_include_debug)


@pipeline(name="zimagesr_download_pipeline")
def zimagesr_download_pipeline(
    s3_uri: str,
    out_dir: str = "./zimage_offline_pairs",
    overwrite: bool = False,
    inspect_limit: int = 3,
) -> None:
    local_dir = download_s3_step(s3_uri=s3_uri, out_dir=out_dir, overwrite=overwrite)
    inspect_pairs_step(out_dir=local_dir, limit=inspect_limit)
