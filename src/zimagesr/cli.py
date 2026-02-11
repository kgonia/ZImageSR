from __future__ import annotations

import argparse
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
        return {
            "model_id": _TrainConfig.model_id,
            "tl": _TrainConfig.tl,
            "batch_size": _TrainConfig.batch_size,
            "gradient_accumulation_steps": _TrainConfig.gradient_accumulation_steps,
            "learning_rate": _TrainConfig.learning_rate,
            "max_steps": _TrainConfig.max_steps,
            "rec_loss_every": _TrainConfig.rec_loss_every,
            "lambda_tvlpips": _TrainConfig.lambda_tvlpips,
            "gamma_tv": _TrainConfig.gamma_tv,
            "detach_recon": _TrainConfig.detach_recon,
            "lambda_adl": _TrainConfig.lambda_adl,
            "lora_rank": _TrainConfig.lora_rank,
            "lora_alpha": _TrainConfig.lora_alpha,
            "lora_dropout": _TrainConfig.lora_dropout,
            "save_dir": _TrainConfig.save_dir,
            "save_every": _TrainConfig.save_every,
            "log_every": _TrainConfig.log_every,
            "mixed_precision": _TrainConfig.mixed_precision,
            "gradient_checkpointing": _TrainConfig.gradient_checkpointing,
            "disable_vae_force_upcast": _TrainConfig.disable_vae_force_upcast,
            "num_workers": _TrainConfig.num_workers,
            "seed": _TrainConfig.seed,
        }
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
        "save_dir": Path("./zimage_sr_lora_runs/ftd_run"),
        "save_every": 150,
        "log_every": 20,
        "mixed_precision": "no",
        "gradient_checkpointing": True,
        "disable_vae_force_upcast": True,
        "num_workers": 2,
        "seed": None,
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
        log_every=args.log_every,
        device=args.device,
        dtype=args.dtype,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        disable_vae_force_upcast=args.disable_vae_force_upcast,
        num_workers=args.num_workers,
        seed=args.seed,
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
                "optional deps (`peft`, `lpips`). "
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
