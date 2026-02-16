from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import json

from zimagesr.data.offline_pairs import (
    DTYPE_MAP,
    GatherConfig,
    gather_offline_pairs,
    generate_lr_pairs,
    inspect_local_pairs,
)
from zimagesr.data.s3_io import (
    download_dir_from_s3,
    upload_dir_to_s3,
    write_sync_report,
)

_TRAIN_CONFIG_IMPORT_ERROR: Exception | None = None
try:
    from zimagesr.training.config import TrainConfig as _TrainConfig
except ImportError as exc:  # pragma: no cover - exercised on partial installs
    _TrainConfig = None
    _TRAIN_CONFIG_IMPORT_ERROR = exc


def _train_defaults() -> dict[str, object]:
    """Return defaults used for train/generate-zl CLI arguments.

    Falls back to literal values when ``zimagesr.training`` is unavailable,
    which keeps data-only commands (e.g. ``gather``) operational.
    """
    if _TrainConfig is not None:
        defaults_cfg = _TrainConfig(pairs_dir=Path("./pairs"))
        return {
            "model_id": defaults_cfg.model_id,
            "tl": defaults_cfg.tl,
            "batch_size": defaults_cfg.batch_size,
            "gradient_accumulation_steps": defaults_cfg.gradient_accumulation_steps,
            "learning_rate": defaults_cfg.learning_rate,
            "max_steps": defaults_cfg.max_steps,
            "rec_loss_every": defaults_cfg.rec_loss_every,
            "lambda_tvlpips": defaults_cfg.lambda_tvlpips,
            "gamma_tv": defaults_cfg.gamma_tv,
            "detach_recon": defaults_cfg.detach_recon,
            "lambda_adl": defaults_cfg.lambda_adl,
            "lora_rank": defaults_cfg.lora_rank,
            "lora_alpha": defaults_cfg.lora_alpha,
            "lora_dropout": defaults_cfg.lora_dropout,
            "save_dir": defaults_cfg.save_dir,
            "save_every": defaults_cfg.save_every,
            "save_full_state": defaults_cfg.save_full_state,
            "log_every": defaults_cfg.log_every,
            "mixed_precision": defaults_cfg.mixed_precision,
            "gradient_checkpointing": defaults_cfg.gradient_checkpointing,
            "disable_vae_force_upcast": defaults_cfg.disable_vae_force_upcast,
            "num_workers": defaults_cfg.num_workers,
            "seed": defaults_cfg.seed,
            "wandb_enabled": defaults_cfg.wandb_enabled,
            "wandb_project": defaults_cfg.wandb_project,
            "wandb_entity": defaults_cfg.wandb_entity,
            "wandb_run_name": defaults_cfg.wandb_run_name,
            "wandb_mode": defaults_cfg.wandb_mode,
            "wandb_log_checkpoints": defaults_cfg.wandb_log_checkpoints,
            "wandb_log_checkpoint_grids": defaults_cfg.wandb_log_checkpoint_grids,
            "checkpoint_infer_grid": defaults_cfg.checkpoint_infer_grid,
            "checkpoint_eval_ids": defaults_cfg.checkpoint_eval_ids,
            "checkpoint_eval_images_dir": defaults_cfg.checkpoint_eval_images_dir,
            "checkpoint_eval_images_limit": defaults_cfg.checkpoint_eval_images_limit,
            "checkpoint_eval_input_upscale": defaults_cfg.checkpoint_eval_input_upscale,
            "checkpoint_eval_fit_multiple": defaults_cfg.checkpoint_eval_fit_multiple,
            "checkpoint_sr_scales": defaults_cfg.checkpoint_sr_scales,
            "resume_from": defaults_cfg.resume_from,
        }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "tl": 0.25,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "max_steps": 750,
        "rec_loss_every": 8,
        "lambda_tvlpips": 1.0,
        "gamma_tv": 0.5,
        "detach_recon": True,
        "lambda_adl": 0.0,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "save_dir": Path(f"./zimage_sr_lora_runs/ftd_run_{ts}"),
        "save_every": 150,
        "save_full_state": False,
        "log_every": 20,
        "mixed_precision": "no",
        "gradient_checkpointing": True,
        "disable_vae_force_upcast": True,
        "num_workers": 2,
        "seed": None,
        "wandb_enabled": False,
        "wandb_project": "zimagesr",
        "wandb_entity": None,
        "wandb_run_name": None,
        "wandb_mode": "online",
        "wandb_log_checkpoints": True,
        "wandb_log_checkpoint_grids": True,
        "checkpoint_infer_grid": False,
        "checkpoint_eval_ids": (),
        "checkpoint_eval_images_dir": None,
        "checkpoint_eval_images_limit": 4,
        "checkpoint_eval_input_upscale": 4.0,
        "checkpoint_eval_fit_multiple": 16,
        "checkpoint_sr_scales": (1.3, 1.6),
        "resume_from": None,
    }


def _load_train_config():
    """Import and return TrainConfig with a clear error on partial installs."""
    if _TrainConfig is None:
        raise RuntimeError(
            "Training config module is unavailable (`zimagesr.training`). "
            "If you only need data gathering, use `gather/degrade/inspect/s3-*`. "
            "For training commands, ensure repo includes `src/zimagesr/training/`."
        ) from _TRAIN_CONFIG_IMPORT_ERROR
    return _TrainConfig


def _add_gather_args(parser: argparse.ArgumentParser) -> None:
    """Add common GatherConfig arguments shared by gather and zenml-run."""
    parser.add_argument("--model-id", default=GatherConfig.model_id)
    parser.add_argument("--out-dir", type=Path, default=GatherConfig.out_dir)
    parser.add_argument("--n", type=int, default=GatherConfig.n)
    parser.add_argument("--hr-size", type=int, default=GatherConfig.hr_size)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=GatherConfig.num_inference_steps,
    )
    parser.add_argument("--guidance-scale", type=float, default=GatherConfig.guidance_scale)
    parser.add_argument("--base-seed", type=int, default=GatherConfig.base_seed)
    parser.add_argument("--start-index", type=int, default=GatherConfig.start_index)
    parser.add_argument("--prompt", default=GatherConfig.prompt)
    parser.add_argument("--negative-prompt", default=GatherConfig.negative_prompt)
    parser.add_argument(
        "--save-x0-png",
        action=argparse.BooleanOptionalAction,
        default=GatherConfig.save_x0_png,
    )
    parser.add_argument(
        "--generate-lr",
        action=argparse.BooleanOptionalAction,
        default=GatherConfig.generate_lr,
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=GatherConfig.skip_existing,
    )
    parser.add_argument(
        "--cache-null-prompt",
        action=argparse.BooleanOptionalAction,
        default=GatherConfig.cache_null_prompt,
    )
    parser.add_argument(
        "--offload-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=GatherConfig.offload_text_encoder,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default=None)
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=GatherConfig.debug,
    )
    parser.add_argument("--debug-every", type=int, default=GatherConfig.debug_every)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add TrainConfig arguments to the given parser."""
    defaults = _train_defaults()
    parser.add_argument("--pairs-dir", type=Path, required=True, help="Path to pairs/ directory")
    parser.add_argument("--model-id", default=defaults["model_id"])
    parser.add_argument("--tl", type=float, default=defaults["tl"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument("--rec-loss-every", type=int, default=defaults["rec_loss_every"])
    parser.add_argument("--lambda-tvlpips", type=float, default=defaults["lambda_tvlpips"])
    parser.add_argument("--gamma-tv", type=float, default=defaults["gamma_tv"])
    parser.add_argument(
        "--detach-recon",
        action=argparse.BooleanOptionalAction,
        default=defaults["detach_recon"],
    )
    parser.add_argument("--lambda-adl", type=float, default=defaults["lambda_adl"])
    parser.add_argument("--lora-rank", type=int, default=defaults["lora_rank"])
    parser.add_argument("--lora-alpha", type=int, default=defaults["lora_alpha"])
    parser.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument("--save-dir", type=Path, default=defaults["save_dir"])
    parser.add_argument("--save-every", type=int, default=defaults["save_every"])
    parser.add_argument(
        "--save-full-state",
        action=argparse.BooleanOptionalAction,
        default=defaults["save_full_state"],
        help="Save full optimizer/scheduler state for resume (large files). Default: on.",
    )
    parser.add_argument("--log-every", type=int, default=defaults["log_every"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=defaults["mixed_precision"])
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=defaults["gradient_checkpointing"],
    )
    parser.add_argument(
        "--disable-vae-force-upcast",
        action=argparse.BooleanOptionalAction,
        default=defaults["disable_vae_force_upcast"],
    )
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=defaults["wandb_enabled"],
        help="Enable native Weights & Biases logging during Phase 2 training.",
    )
    parser.add_argument("--wandb-project", default=defaults["wandb_project"])
    parser.add_argument("--wandb-entity", default=defaults["wandb_entity"])
    parser.add_argument("--wandb-run-name", default=defaults["wandb_run_name"])
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline"],
        default=defaults["wandb_mode"],
    )
    parser.add_argument(
        "--wandb-log-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=defaults["wandb_log_checkpoints"],
        help="Log LoRA checkpoints as WandB model artifacts.",
    )
    parser.add_argument(
        "--wandb-log-checkpoint-grids",
        action=argparse.BooleanOptionalAction,
        default=defaults["wandb_log_checkpoint_grids"],
        help="Log checkpoint-time inference grids as WandB images.",
    )
    parser.add_argument(
        "--checkpoint-infer-grid",
        action=argparse.BooleanOptionalAction,
        default=defaults["checkpoint_infer_grid"],
        help="Run one-step inference preview at checkpoint steps and save comparison grids.",
    )
    parser.add_argument(
        "--checkpoint-eval-ids",
        default=",".join(defaults["checkpoint_eval_ids"]),
        help="Comma-separated fixed pair IDs for checkpoint grids (e.g. 000000,000123).",
    )
    parser.add_argument(
        "--checkpoint-eval-images-dir",
        type=Path,
        default=defaults["checkpoint_eval_images_dir"],
        help="Folder of arbitrary images used for checkpoint-time eval grids.",
    )
    parser.add_argument(
        "--checkpoint-eval-images-limit",
        type=int,
        default=defaults["checkpoint_eval_images_limit"],
        help="Maximum number of images loaded from --checkpoint-eval-images-dir.",
    )
    parser.add_argument(
        "--checkpoint-eval-input-upscale",
        type=float,
        default=defaults["checkpoint_eval_input_upscale"],
        help="Bicubic upscale factor before VAE encode for checkpoint eval images.",
    )
    parser.add_argument(
        "--checkpoint-eval-fit-multiple",
        type=int,
        default=defaults["checkpoint_eval_fit_multiple"],
        help="Resize checkpoint eval images to dimensions divisible by this value before VAE encode.",
    )
    parser.add_argument(
        "--checkpoint-sr-scales",
        type=_parse_csv_floats,
        default=defaults["checkpoint_sr_scales"],
        help="Comma-separated extra sr_scale values for checkpoint grids (e.g. 1.3,1.6).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=defaults["resume_from"],
        help="Resume training from a checkpoint directory (auto-detects full vs weights-only).",
    )


def _parse_csv_values(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _parse_csv_floats(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ()
    out: list[float] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            scale = float(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid --checkpoint-sr-scales value: {token!r} is not a float."
            ) from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise argparse.ArgumentTypeError(
                f"Invalid --checkpoint-sr-scales value: {token!r} must be a finite number > 0."
            )
        out.append(scale)
    return tuple(out)


def _train_config_from_args(args: argparse.Namespace):
    """Construct a TrainConfig from parsed CLI arguments."""
    TrainConfig = _load_train_config()
    return TrainConfig(
        pairs_dir=args.pairs_dir,
        model_id=args.model_id,
        tl=args.tl,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        rec_loss_every=args.rec_loss_every,
        lambda_tvlpips=args.lambda_tvlpips,
        gamma_tv=args.gamma_tv,
        detach_recon=args.detach_recon,
        lambda_adl=args.lambda_adl,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        save_dir=args.save_dir,
        save_every=args.save_every,
        save_full_state=args.save_full_state,
        log_every=args.log_every,
        device=args.device,
        dtype=args.dtype,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        disable_vae_force_upcast=args.disable_vae_force_upcast,
        num_workers=args.num_workers,
        seed=args.seed,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_log_checkpoints=args.wandb_log_checkpoints,
        wandb_log_checkpoint_grids=args.wandb_log_checkpoint_grids,
        checkpoint_infer_grid=args.checkpoint_infer_grid,
        checkpoint_eval_ids=_parse_csv_values(args.checkpoint_eval_ids),
        checkpoint_eval_images_dir=args.checkpoint_eval_images_dir,
        checkpoint_eval_images_limit=args.checkpoint_eval_images_limit,
        checkpoint_eval_input_upscale=args.checkpoint_eval_input_upscale,
        checkpoint_eval_fit_multiple=args.checkpoint_eval_fit_multiple,
        checkpoint_sr_scales=args.checkpoint_sr_scales,
        resume_from=args.resume_from,
    )


def _gather_config_from_args(args: argparse.Namespace) -> GatherConfig:
    """Construct a GatherConfig from parsed CLI arguments."""
    return GatherConfig(
        model_id=args.model_id,
        out_dir=args.out_dir,
        n=args.n,
        hr_size=args.hr_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        base_seed=args.base_seed,
        start_index=args.start_index,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        save_x0_png=args.save_x0_png,
        generate_lr=args.generate_lr,
        skip_existing=args.skip_existing,
        cache_null_prompt=args.cache_null_prompt,
        offload_text_encoder=args.offload_text_encoder,
        device=args.device,
        dtype=args.dtype,
        debug=args.debug,
        debug_every=args.debug_every,
    )


def _lora_weight_stats(model) -> dict[str, float]:
    """Return diagnostic stats for loaded LoRA adapter weights."""
    a_abs, b_abs, a_count, b_count = 0.0, 0.0, 0, 0
    for name, param in model.named_parameters():
        if "lora_A" in name:
            a_abs += param.detach().abs().mean().item()
            a_count += 1
        elif "lora_B" in name:
            b_abs += param.detach().abs().mean().item()
            b_count += 1
    if a_count == 0:
        return {"lora_layers": 0, "lora_A_mean_abs": 0.0, "lora_B_mean_abs": 0.0}
    return {
        "lora_layers": a_count,
        "lora_A_mean_abs": round(a_abs / a_count, 8),
        "lora_B_mean_abs": round(b_abs / b_count, 8) if b_count else 0.0,
    }


def _build_comparison_grid(images: list, labels: list[str]):
    """Build a labeled side-by-side image grid from PIL images."""
    from PIL import Image, ImageDraw

    W, H = images[0].size
    n = len(images)
    canvas = Image.new("RGB", (W * n, H + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        canvas.paste(img.resize((W, H), resample=Image.BICUBIC), (W * i, 0))
        draw.text((W * i + 4, H + 4), lbl, fill="black")
    return canvas


def _infer_default_output(pair_dir: Path | None, input_image: Path | None) -> Path:
    if pair_dir is not None:
        return pair_dir / "sr.png"
    assert input_image is not None
    return input_image.with_name(f"{input_image.stem}_sr.png")


def _resize_to_multiple(pil_img, multiple: int):
    """Resize image so width/height are divisible by *multiple*."""
    if multiple <= 1:
        return pil_img, False
    w, h = pil_img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    if new_w == w and new_h == h:
        return pil_img, False
    from PIL import Image

    return pil_img.resize((new_w, new_h), resample=Image.BICUBIC), True


def _prepare_input_image(path: Path, upscale: float, fit_multiple: int):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    original_size = [img.width, img.height]
    changed_upscale = False
    if upscale > 0 and abs(upscale - 1.0) > 1e-8:
        w = max(1, int(round(img.width * upscale)))
        h = max(1, int(round(img.height * upscale)))
        img = img.resize((w, h), resample=Image.BICUBIC)
        changed_upscale = True
    img, changed_multiple = _resize_to_multiple(img, fit_multiple)
    meta = {
        "input_original_size": original_size,
        "input_upscale": upscale,
        "input_fit_multiple": fit_multiple,
        "input_resized_by_upscale": changed_upscale,
        "input_resized_by_multiple": changed_multiple,
        "prepared_size": [img.width, img.height],
    }
    return img, meta


def _load_zl_from_pair_dir(pair_dir: Path, device: str, dtype):
    import torch

    zl_path = pair_dir / "zL.pt"
    if not zl_path.exists():
        raise FileNotFoundError(f"Missing zL latent file: {zl_path}")
    zL = torch.load(zl_path, map_location="cpu", weights_only=True)
    if zL.ndim == 3:
        zL = zL.unsqueeze(0)
    if zL.ndim != 4:
        raise ValueError(f"Expected zL to have 4 dims (1,C,H,W), got shape: {tuple(zL.shape)}")
    return zL.to(device=device, dtype=dtype), zl_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimagesr-data",
        description="Generate offline data pairs for Z-Image super-resolution.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gather = sub.add_parser("gather", help="Generate (eps, z0, x0) offline pairs.")
    _add_gather_args(gather)

    degrade = sub.add_parser("degrade", help="Generate LR pairs from existing x0.png.")
    degrade.add_argument("--out-dir", type=Path, default=GatherConfig.out_dir)
    degrade.add_argument("--n", type=int, default=GatherConfig.n)
    degrade.add_argument("--start-index", type=int, default=GatherConfig.start_index)
    degrade.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=GatherConfig.skip_existing,
    )
    degrade.add_argument(
        "--degradation",
        choices=["bicubic", "realesrgan"],
        default="bicubic",
        help="Degradation pipeline: simple bicubic or RealESRGAN second-order.",
    )
    degrade.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducible degradation (per-sample seed = seed + index).",
    )

    inspect_cmd = sub.add_parser(
        "inspect",
        help="Inspect local dataset tensor/image shapes and model metadata.",
    )
    inspect_cmd.add_argument("--out-dir", type=Path, default=GatherConfig.out_dir)
    inspect_cmd.add_argument("--limit", type=int, default=3)
    inspect_cmd.add_argument("--json-out", type=Path, default=None)

    s3_upload = sub.add_parser("s3-upload", help="Upload dataset directory to S3.")
    s3_upload.add_argument("--out-dir", type=Path, default=GatherConfig.out_dir)
    s3_upload.add_argument("--s3-uri", required=True)
    s3_upload.add_argument(
        "--include-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    s3_upload.add_argument("--report-json", type=Path, default=None)

    s3_download = sub.add_parser("s3-download", help="Download dataset directory from S3.")
    s3_download.add_argument("--out-dir", type=Path, default=GatherConfig.out_dir)
    s3_download.add_argument("--s3-uri", required=True)
    s3_download.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    s3_download.add_argument("--report-json", type=Path, default=None)

    train_cmd = sub.add_parser("train", help="Run FTD training with LoRA.")
    _add_train_args(train_cmd)

    gen_zl = sub.add_parser("generate-zl", help="Encode lr_up.png -> zL.pt for each pair.")
    gen_zl.add_argument("--out-dir", type=Path, required=True, help="Path to dataset dir containing pairs/")
    gen_zl.add_argument("--model-id", default=_train_defaults()["model_id"])
    gen_zl.add_argument("--device", default=None)
    gen_zl.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default=None)
    gen_zl.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    infer_cmd = sub.add_parser(
        "infer",
        help="Run one-step SR inference from pair zL.pt or arbitrary input image.",
    )
    infer_cmd.add_argument("--model-id", default=_train_defaults()["model_id"])
    infer_cmd.add_argument("--lora-path", type=Path, required=True, help="Path to LoRA adapter directory.")
    infer_src = infer_cmd.add_mutually_exclusive_group(required=True)
    infer_src.add_argument(
        "--pair-dir",
        type=Path,
        help="Pair sample directory containing zL.pt (paper notation).",
    )
    infer_src.add_argument(
        "--input-image",
        type=Path,
        help="Arbitrary input image path (RGB converted).",
    )
    infer_cmd.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SR image path. Defaults to pair-dir/sr.png or <input>_sr.png.",
    )
    infer_cmd.add_argument(
        "--input-upscale",
        type=float,
        default=4.0,
        help="Input image bicubic upscale factor before VAE encoding (image mode only).",
    )
    infer_cmd.add_argument(
        "--fit-multiple",
        type=int,
        default=16,
        help="Resize input image to dimensions divisible by this value (image mode only).",
    )
    infer_cmd.add_argument("--tl", type=float, default=_train_defaults()["tl"])
    infer_cmd.add_argument(
        "--sr-scale",
        type=_parse_csv_floats,
        default=(1.0,),
        help="Comma-separated sr_scale values (e.g. 0.8,1.0,1.3,1.6). Multiple values produce a comparison grid.",
    )
    infer_cmd.add_argument("--device", default=None)
    infer_cmd.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default=None)
    infer_cmd.add_argument(
        "--compare-grid",
        action="store_true",
        default=False,
        help="Save a comparison grid: LR | Base SR | LoRA SR (+ HR if pair has x0.png).",
    )

    decode_check = sub.add_parser(
        "decode-check",
        help="Decode z0.pt latents via VAE and save alongside x0.png for round-trip quality check.",
    )
    decode_check.add_argument("--pairs-dir", type=Path, required=True, help="Path to pairs/ directory.")
    decode_check.add_argument("--model-id", default=_train_defaults()["model_id"])
    decode_check.add_argument(
        "--ids",
        default=None,
        help="Comma-separated sample IDs to check (e.g. 000000,000010). Defaults to first --limit samples.",
    )
    decode_check.add_argument("--limit", type=int, default=5, help="Number of samples when --ids not given.")
    decode_check.add_argument("--device", default=None)
    decode_check.add_argument("--dtype", choices=sorted(DTYPE_MAP.keys()), default=None)

    zenml_run = sub.add_parser(
        "zenml-run",
        help="Run minimal ZenML wrapper pipelines for gather/download.",
    )
    zenml_run.add_argument("--mode", choices=["gather", "download"], default="gather")
    _add_gather_args(zenml_run)
    zenml_run.add_argument("--inspect-limit", type=int, default=3)
    zenml_run.add_argument("--s3-uri", default=None)
    zenml_run.add_argument(
        "--s3-include-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    zenml_run.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "gather":
        gather_offline_pairs(_gather_config_from_args(args))
        return

    if args.command == "degrade":
        generate_lr_pairs(
            out_dir=args.out_dir,
            n=args.n,
            start_index=args.start_index,
            skip_existing=args.skip_existing,
            degradation=args.degradation,
            seed=args.seed,
        )
        return

    if args.command == "inspect":
        data = inspect_local_pairs(
            out_dir=args.out_dir,
            limit=args.limit,
        )
        rendered = json.dumps(data, indent=2)
        print(rendered)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        return

    if args.command == "s3-upload":
        result = upload_dir_to_s3(
            local_dir=args.out_dir,
            s3_uri=args.s3_uri,
            include_debug=args.include_debug,
        )
        print(json.dumps(result.to_dict(), indent=2))
        if args.report_json is not None:
            write_sync_report(args.report_json, result)
        return

    if args.command == "s3-download":
        result = download_dir_from_s3(
            s3_uri=args.s3_uri,
            local_dir=args.out_dir,
            overwrite=args.overwrite,
        )
        print(json.dumps(result.to_dict(), indent=2))
        if args.report_json is not None:
            write_sync_report(args.report_json, result)
        return

    if args.command == "train":
        try:
            from zimagesr.training.train import ftd_train_loop
        except ImportError as exc:
            raise RuntimeError(
                "Training command requires `zimagesr.training` modules plus "
                "optional deps (`peft`, `lpips`, and optionally `wandb`). "
                "Ensure repo has `src/zimagesr/training/`, then run: "
                "uv sync && uv pip install -e '.[training]'"
            ) from exc
        result = ftd_train_loop(_train_config_from_args(args))
        print(json.dumps(result, indent=2))
        return

    if args.command == "generate-zl":
        try:
            from zimagesr.training.dataset import generate_zl_latents
        except ImportError as exc:
            raise RuntimeError(
                "generate-zl requires `zimagesr.training.dataset`. "
                "Ensure repo has `src/zimagesr/training/`, then run: "
                "uv sync && uv pip install -e '.[training]'"
            ) from exc

        import torch
        from diffusers import ZImageImg2ImgPipeline

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map[args.dtype] if args.dtype else (torch.float32 if device == "cpu" else torch.bfloat16)

        pipe = ZImageImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
        pairs_dir = args.out_dir / "pairs"
        count = generate_zl_latents(pairs_dir, pipe, device, dtype, skip_existing=args.skip_existing)
        print(f"Created {count} zL.pt files")
        return

    if args.command == "decode-check":
        import torch
        from diffusers import ZImageImg2ImgPipeline
        from zimagesr.training.transformer_utils import vae_decode_to_pixels
        import torchvision.transforms.functional as TF

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map[args.dtype] if args.dtype else (torch.float32 if device == "cpu" else torch.bfloat16)

        pipe = ZImageImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
        vae_sf = float(getattr(pipe.vae.config, "scaling_factor", 1.0))

        pairs_dir = args.pairs_dir
        if args.ids:
            sample_ids = [v.strip() for v in args.ids.split(",") if v.strip()]
        else:
            sample_dirs = sorted(
                d for d in pairs_dir.iterdir()
                if d.is_dir() and (d / "z0.pt").exists()
            )
            sample_ids = [d.name for d in sample_dirs[: args.limit]]

        if not sample_ids:
            print("No samples with z0.pt found.")
            return

        for sid in sample_ids:
            sample_dir = pairs_dir / sid
            z0_path = sample_dir / "z0.pt"
            x0_path = sample_dir / "x0.png"
            if not z0_path.exists():
                print(f"  {sid}: z0.pt missing, skipping")
                continue

            z0 = torch.load(z0_path, map_location="cpu", weights_only=True)
            if z0.ndim == 3:
                z0 = z0.unsqueeze(0)
            z0 = z0.to(device=device, dtype=dtype)

            with torch.no_grad():
                autocast_dt = torch.bfloat16 if torch.device(device).type == "cuda" else None
                pixels = vae_decode_to_pixels(pipe.vae, z0, vae_sf, autocast_dtype=autocast_dt)
                decoded_pil = TF.to_pil_image(pixels[0].clamp(0, 1).float().cpu())

            decoded_path = sample_dir / "z0_decoded.png"
            decoded_pil.save(decoded_path)

            if x0_path.exists():
                from PIL import Image
                x0_pil = Image.open(x0_path).convert("RGB")
                grid = _build_comparison_grid(
                    [x0_pil, decoded_pil],
                    ["x0.png (original)", "z0 decoded (VAE round-trip)"],
                )
                grid_path = sample_dir / "z0_roundtrip_grid.png"
                grid.save(grid_path)
                print(f"  {sid}: {grid_path}")
            else:
                print(f"  {sid}: {decoded_path} (no x0.png for comparison)")

        print(f"\nDecoded {len(sample_ids)} samples. Compare x0.png vs z0_decoded.png for VAE quality ceiling.")
        return

    if args.command == "infer":
        try:
            import torch
            from diffusers import ZImageImg2ImgPipeline
            from zimagesr.data.offline_pairs import vae_encode_latents_safe
            from zimagesr.training.inference import one_step_sr
            from zimagesr.training.lora import load_lora_for_inference
            from zimagesr.training.transformer_utils import prepare_cap_feats
        except ImportError as exc:
            raise RuntimeError(
                "infer requires `zimagesr.training` modules plus optional dep "
                "(`peft`). Ensure repo has `src/zimagesr/training/`, then run: "
                "uv sync && uv pip install -e '.[training]'"
            ) from exc

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map[args.dtype] if args.dtype else (torch.float32 if device == "cpu" else torch.bfloat16)

        pipe = ZImageImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
        t_scale = float(getattr(pipe.transformer.config, "t_scale", 1.0))
        vae_sf = float(getattr(pipe.vae.config, "scaling_factor", 1.0))

        cap_feats_2d = prepare_cap_feats(pipe, device, dtype)
        lora_tr = load_lora_for_inference(pipe.transformer, args.lora_path, device, dtype)
        lora_stats = _lora_weight_stats(lora_tr)
        print(f"LoRA loaded: {lora_stats['lora_layers']} layers, "
              f"lora_A mean|w|={lora_stats['lora_A_mean_abs']:.6f}, "
              f"lora_B mean|w|={lora_stats['lora_B_mean_abs']:.6f}")
        if lora_stats["lora_B_mean_abs"] < 1e-9:
            print("WARNING: lora_B weights are near-zero — LoRA may not have trained or loaded correctly.")

        source_info: dict[str, object]
        if args.pair_dir is not None:
            zL, zl_path = _load_zl_from_pair_dir(args.pair_dir, device=device, dtype=dtype)
            source_info = {
                "source_mode": "pair_dir",
                "pair_dir": str(args.pair_dir),
                "zL_path": str(zl_path),
                "prepared_size": None,
            }
        else:
            assert args.input_image is not None
            if not args.input_image.exists():
                raise FileNotFoundError(f"Input image not found: {args.input_image}")
            pil_img, prep_meta = _prepare_input_image(
                args.input_image,
                upscale=args.input_upscale,
                fit_multiple=args.fit_multiple,
            )
            zL = vae_encode_latents_safe(pipe, pil_img, device=device, dtype=dtype)
            source_info = {
                "source_mode": "input_image",
                "input_image": str(args.input_image),
                **prep_meta,
            }

        sr_scales = args.sr_scale  # tuple of floats
        sr_common = dict(
            vae=pipe.vae, lr_latent=zL, tl=args.tl,
            vae_sf=vae_sf, cap_feats_2d=cap_feats_2d,
        )

        # Run inference at each scale
        sr_results: list[tuple[float, object]] = []
        for scale in sr_scales:
            img = one_step_sr(transformer=lora_tr, **sr_common, sr_scale=scale)
            sr_results.append((scale, img))

        # Save the first scale as the primary output
        primary_scale, primary_img = sr_results[0]
        out_path = args.output or _infer_default_output(args.pair_dir, args.input_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        primary_img.save(out_path)

        # Build comparison grid: multi-scale always gets a grid
        multi_scale = len(sr_scales) > 1
        if args.compare_grid or multi_scale:
            from zimagesr.training.transformer_utils import vae_decode_to_pixels
            import torchvision.transforms.functional as TF

            autocast_dt = torch.bfloat16 if torch.device(device).type == "cuda" else None
            lr_pixels = vae_decode_to_pixels(pipe.vae, zL, vae_sf, autocast_dtype=autocast_dt)
            lr_pil = TF.to_pil_image(lr_pixels[0].clamp(0, 1).float().cpu())

            grid_imgs = [lr_pil]
            grid_labels = ["LR (decoded)"]

            # Base model (LoRA disabled)
            if args.compare_grid:
                lora_tr.disable_adapter_layers()
                base_img = one_step_sr(transformer=lora_tr, **sr_common, sr_scale=1.0)
                lora_tr.enable_adapter_layers()
                grid_imgs.append(base_img)
                grid_labels.append("Base SR")

            # All scale results
            for scale, img in sr_results:
                grid_imgs.append(img)
                label = f"LoRA SR ({scale})" if multi_scale else "LoRA SR"
                grid_labels.append(label)

            # HR ground truth if available
            if args.pair_dir is not None:
                hr_path = args.pair_dir / "x0.png"
                if hr_path.exists():
                    from PIL import Image
                    grid_imgs.append(Image.open(hr_path).convert("RGB"))
                    grid_labels.append("HR (ground truth)")

            grid = _build_comparison_grid(grid_imgs, grid_labels)
            grid_path = out_path.with_name(out_path.stem + "_grid.png")
            grid.save(grid_path)
            print(f"Comparison grid saved: {grid_path}")

        print(
            json.dumps(
                {
                    "output_path": str(out_path),
                    "model_id": args.model_id,
                    "lora_path": str(args.lora_path),
                    "device": device,
                    "dtype": str(dtype),
                    "tl": args.tl,
                    "sr_scale": list(sr_scales),
                    "t_scale": t_scale,
                    "vae_sf": vae_sf,
                    **lora_stats,
                    **source_info,
                },
                indent=2,
            )
        )
        return

    if args.command == "zenml-run":
        try:
            from zimagesr.pipelines.zenml_pipeline import (
                zimagesr_download_pipeline,
                zimagesr_gather_pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ZenML pipeline wrapper requires zenml to be installed. "
                "Install dependencies and retry."
            ) from exc

        if args.mode == "gather":
            zimagesr_gather_pipeline(
                model_id=args.model_id,
                out_dir=str(args.out_dir),
                n=args.n,
                hr_size=args.hr_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                base_seed=args.base_seed,
                start_index=args.start_index,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                save_x0_png=args.save_x0_png,
                generate_lr=args.generate_lr,
                skip_existing=args.skip_existing,
                cache_null_prompt=args.cache_null_prompt,
                offload_text_encoder=args.offload_text_encoder,
                device=args.device,
                dtype=args.dtype,
                debug=args.debug,
                debug_every=args.debug_every,
                inspect_limit=args.inspect_limit,
                s3_uri=args.s3_uri,
                s3_include_debug=args.s3_include_debug,
            )
            return

        if not args.s3_uri:
            parser.error("--s3-uri is required when --mode download")
        zimagesr_download_pipeline(
            s3_uri=args.s3_uri,
            out_dir=str(args.out_dir),
            overwrite=args.overwrite,
            inspect_limit=args.inspect_limit,
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
