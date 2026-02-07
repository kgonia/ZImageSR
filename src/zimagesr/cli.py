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
