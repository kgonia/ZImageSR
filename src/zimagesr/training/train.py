from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

import json as _json

from zimagesr.training.config import TrainConfig
from zimagesr.training.dataset import FTDPairDataset, ftd_collate
from zimagesr.training.inference import one_step_sr
from zimagesr.training.lora import apply_lora, save_lora
from zimagesr.training.losses import TVLPIPSLoss, compute_adl_loss
from zimagesr.training.transformer_utils import (
    ADLHookContext,
    call_transformer,
    prepare_cap_feats,
    vae_decode_to_pixels,
)

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype_str: str | None, device: str) -> torch.dtype:
    """Resolve dtype string to torch.dtype with a safe CPU fallback."""
    if dtype_str:
        return DTYPE_MAP[dtype_str]
    # Default: bfloat16 on CUDA, float32 on CPU
    if device.startswith("cpu"):
        return torch.float32
    return torch.bfloat16


class ResumeMode:
    NONE = "none"
    FULL = "full"
    WEIGHTS_ONLY = "weights_only"


RESUME_COMPAT_KEYS = (
    "model_id",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "mixed_precision",
    "gradient_checkpointing",
    "disable_vae_force_upcast",
)


def _detect_resume_mode(resume_path: Path | None) -> str:
    """Auto-detect resume mode from checkpoint contents.

    Returns:
        ``ResumeMode.NONE`` if *resume_path* is ``None``.
        ``ResumeMode.FULL`` if the directory contains both
        ``training_state.json`` and ``accelerator_state/``.
        ``ResumeMode.WEIGHTS_ONLY`` if the directory contains
        ``adapter_config.json`` but not the full training state.

    Raises:
        FileNotFoundError: If *resume_path* doesn't exist or lacks
            recognisable checkpoint files.
    """
    if resume_path is None:
        return ResumeMode.NONE

    resume_path = Path(resume_path)
    if not resume_path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_path}")

    has_training_state = (resume_path / "training_state.json").is_file()
    has_accelerator_state = (resume_path / "accelerator_state").is_dir()
    has_adapter_config = (resume_path / "adapter_config.json").is_file()

    if has_training_state and has_accelerator_state:
        return ResumeMode.FULL
    if has_adapter_config:
        return ResumeMode.WEIGHTS_ONLY

    raise FileNotFoundError(
        f"Checkpoint directory {resume_path} contains neither "
        f"training_state.json + accelerator_state/ (full resume) nor "
        f"adapter_config.json (weights-only resume). "
        f"Contents: {sorted(p.name for p in resume_path.iterdir())}"
    )


def _count_trainable_params(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_weights_only_resume_model(
    transformer: torch.nn.Module,
    resume_path: Path,
) -> torch.nn.Module:
    """Load LoRA adapter for training continuation from weights-only checkpoint."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        transformer,
        str(resume_path),
        is_trainable=True,
    )
    if _count_trainable_params(model) == 0:
        raise RuntimeError(
            "Loaded weights-only checkpoint has no trainable parameters. "
            "Expected trainable LoRA adapter weights for resumed training."
        )
    return model


def _validate_full_resume_config(
    config: TrainConfig,
    state_json: dict[str, Any],
) -> None:
    """Validate config compatibility before full-state resume."""
    saved_cfg = state_json.get("config")
    if not isinstance(saved_cfg, dict):
        logger.warning(
            "Full resume checkpoint is missing serialized config in training_state.json; "
            "skipping compatibility checks."
        )
        return

    current_cfg = _wandb_config_dict(config)
    mismatches: list[tuple[str, Any, Any]] = []
    for key in RESUME_COMPAT_KEYS:
        if saved_cfg.get(key) != current_cfg.get(key):
            mismatches.append((key, saved_cfg.get(key), current_cfg.get(key)))

    if mismatches:
        details = "; ".join(
            f"{key}: checkpoint={saved!r}, current={current!r}"
            for key, saved, current in mismatches
        )
        raise ValueError(
            "Incompatible full-resume configuration. "
            "Use the same model/LoRA/training-structure settings as the checkpoint. "
            f"Mismatches: {details}"
        )


def _save_training_state(
    checkpoint_dir: Path,
    accelerator: Any,
    global_step: int,
    config: TrainConfig,
) -> None:
    """Save accelerator state and training metadata to *checkpoint_dir*."""
    accelerator.save_state(str(checkpoint_dir / "accelerator_state"))
    if accelerator.is_main_process:
        state = {
            "global_step": global_step,
            "config": _wandb_config_dict(config),
        }
        (checkpoint_dir / "training_state.json").write_text(
            _json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )


def _wandb_config_dict(config: TrainConfig) -> dict[str, Any]:
    """Convert TrainConfig to a WandB-safe config dict."""
    cfg = asdict(config)
    for key, value in list(cfg.items()):
        if isinstance(value, Path):
            cfg[key] = str(value)
    return cfg


def _build_comparison_grid(images: list[Any], labels: list[str]) -> Any:
    """Build a labeled side-by-side image grid from PIL images."""
    from PIL import Image, ImageDraw

    width, height = images[0].size
    canvas = Image.new("RGB", (width * len(images), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img.resize((width, height), resample=Image.BICUBIC), (idx * width, 0))
        draw.text((idx * width + 4, height + 4), label, fill="black")
    return canvas


def _save_checkpoint_inference_grid(
    *,
    transformer: torch.nn.Module,
    vae: torch.nn.Module,
    zL: torch.Tensor,
    x0_pixels: torch.Tensor | None,
    tl: float,
    vae_sf: float,
    cap_feats_2d: torch.Tensor,
    out_path: Path,
    extra_sr_scales: tuple[float, ...] = (),
    extra_refine_steps: tuple[int, ...] = (),
) -> Path:
    """Save an LR|Base SR|LoRA SR( + HR) comparison grid for checkpoint diagnostics.

    When *extra_sr_scales* is non-empty, additional LoRA SR columns are
    rendered at each scale (e.g. 1.3, 1.6) so the user can compare
    correction strength without confusing model quality with scale choice.
    When *extra_refine_steps* is non-empty, additional LoRA SR columns are
    rendered with multi-step refinement counts (e.g. 4, 8).
    """
    import torchvision.transforms.functional as TF

    device = zL.device
    autocast_dt = torch.bfloat16 if device.type == "cuda" else None

    _sr_kwargs = dict(
        transformer=transformer, vae=vae, lr_latent=zL,
        tl=tl, vae_sf=vae_sf, cap_feats_2d=cap_feats_2d,
    )

    was_training = transformer.training
    transformer.eval()
    try:
        with torch.no_grad():
            lr_pixels = vae_decode_to_pixels(vae, zL, vae_sf, autocast_dtype=autocast_dt)
            lr_pil = TF.to_pil_image(lr_pixels[0].clamp(0, 1).float().cpu())

            lora_img = one_step_sr(**_sr_kwargs, refine_steps=1)

            base_img = None
            if hasattr(transformer, "disable_adapter_layers") and hasattr(transformer, "enable_adapter_layers"):
                transformer.disable_adapter_layers()
                try:
                    base_img = one_step_sr(**_sr_kwargs)
                finally:
                    transformer.enable_adapter_layers()

            sweep_imgs = []
            for scale in extra_sr_scales:
                if abs(scale - 1.0) < 1e-8:
                    continue
                sweep_imgs.append((f"LoRA SR ({scale})", one_step_sr(**_sr_kwargs, sr_scale=scale, refine_steps=1)))
            for steps in extra_refine_steps:
                if steps == 1:
                    continue
                sweep_imgs.append((f"LoRA SR ({steps}-step)", one_step_sr(**_sr_kwargs, refine_steps=steps)))

            images = [lr_pil]
            labels = ["LR (decoded)"]
            if base_img is not None:
                images.append(base_img)
                labels.append("Base SR")
            images.append(lora_img)
            labels.append("LoRA SR (1.0, 1-step)")
            for label, img in sweep_imgs:
                images.append(img)
                labels.append(label)
            if x0_pixels is not None:
                hr_pil = TF.to_pil_image(x0_pixels[0].clamp(0, 1).float().cpu())
                images.append(hr_pil)
                labels.append("HR (ground truth)")
            grid = _build_comparison_grid(images, labels)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            grid.save(out_path)
    finally:
        if was_training:
            transformer.train()
    return out_path


def _sanitize_eval_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    safe = safe.strip("_")
    return safe or "sample"


def _resize_to_multiple(pil_img: Any, multiple: int) -> Any:
    if multiple <= 1:
        return pil_img
    width, height = pil_img.size
    new_w = ((width + multiple - 1) // multiple) * multiple
    new_h = ((height + multiple - 1) // multiple) * multiple
    if new_w == width and new_h == height:
        return pil_img
    from PIL import Image

    return pil_img.resize((new_w, new_h), resample=Image.BICUBIC)


def _prepare_eval_input_image(path: Path, upscale: float, fit_multiple: int) -> Any:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if upscale > 0 and abs(upscale - 1.0) > 1e-8:
        w = max(1, int(round(img.width * upscale)))
        h = max(1, int(round(img.height * upscale)))
        img = img.resize((w, h), resample=Image.BICUBIC)
    return _resize_to_multiple(img, fit_multiple)


def _load_png_tensor(path: Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(path).convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _resolve_pair_id_dir(pairs_dir: Path, sample_id: str) -> Path | None:
    sid = sample_id.strip()
    if not sid:
        return None
    candidates = [pairs_dir / sid]
    if sid.isdigit():
        candidates.insert(0, pairs_dir / f"{int(sid):06d}")
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _gather_checkpoint_eval_samples(
    *,
    config: TrainConfig,
    pipe: Any,
    device: torch.device,
    dtype: torch.dtype,
    default_zL: torch.Tensor,
    default_x0_pixels: torch.Tensor | None,
) -> list[tuple[str, torch.Tensor, torch.Tensor | None]]:
    """Collect zL/x0 sample tuples for checkpoint-time grid rendering."""
    samples: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []

    # Fixed pair IDs from pairs_dir.
    for sample_id in config.checkpoint_eval_ids:
        pair_dir = _resolve_pair_id_dir(config.pairs_dir, sample_id)
        if pair_dir is None:
            logger.warning("Checkpoint eval pair ID not found: %s", sample_id)
            continue
        zl_path = pair_dir / "zL.pt"
        if not zl_path.exists():
            logger.warning("Checkpoint eval pair missing zL.pt: %s", pair_dir)
            continue
        zL_eval = torch.load(zl_path, map_location="cpu", weights_only=True)
        if zL_eval.ndim == 3:
            zL_eval = zL_eval.unsqueeze(0)
        if zL_eval.ndim != 4:
            logger.warning("Checkpoint eval pair has invalid zL shape: %s", tuple(zL_eval.shape))
            continue
        x0_pixels = None
        x0_path = pair_dir / "x0.png"
        if x0_path.exists():
            x0_pixels = _load_png_tensor(x0_path).to(device=device, dtype=dtype)
        samples.append((f"pair_{pair_dir.name}", zL_eval.to(device=device, dtype=dtype), x0_pixels))

    # Arbitrary image folder eval.
    if config.checkpoint_eval_images_dir is not None:
        from zimagesr.data.offline_pairs import vae_encode_latents_safe

        image_dir = Path(config.checkpoint_eval_images_dir)
        if not image_dir.is_dir():
            logger.warning("Checkpoint eval image folder not found: %s", image_dir)
        else:
            image_paths = sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
            )
            limit = max(0, int(config.checkpoint_eval_images_limit))
            if limit:
                image_paths = image_paths[:limit]
            for img_path in image_paths:
                pil_img = _prepare_eval_input_image(
                    img_path,
                    upscale=config.checkpoint_eval_input_upscale,
                    fit_multiple=config.checkpoint_eval_fit_multiple,
                )
                zL_eval = vae_encode_latents_safe(
                    pipe,
                    pil_img,
                    device=str(device),
                    dtype=dtype,
                )
                samples.append((f"img_{img_path.stem}", zL_eval.to(device=device, dtype=dtype), None))

    # Fallback: current batch sample.
    if not samples:
        samples.append(("batch", default_zL[:1], default_x0_pixels))

    return samples


def ftd_train_loop(config: TrainConfig) -> dict[str, Any]:
    """Run FTD training following FluxSR Eq. 16/17/18/21.

    Returns:
        Summary dict with final loss values and checkpoint path.
    """
    from accelerate import Accelerator
    from diffusers import ZImageImg2ImgPipeline

    # ── Accelerator (must be created first for correct device assignment) ─
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # ── Device / dtype ──────────────────────────────────────────────────
    # Use accelerator.device so each rank gets its correct device.
    # config.device is only used for the initial pipeline load (before
    # accelerator.prepare), then we switch to accelerator.device.
    load_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(config.dtype, load_device)

    # ── Load pipeline ───────────────────────────────────────────────────
    logger.info("Loading pipeline %s", config.model_id)
    pipe = ZImageImg2ImgPipeline.from_pretrained(config.model_id, torch_dtype=dtype).to(load_device)
    pipe.vae.requires_grad_(False)
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ── Model-intrinsic values ──────────────────────────────────────────
    t_scale = float(getattr(pipe.transformer.config, "t_scale", 1.0))
    vae_sf = float(getattr(pipe.vae.config, "scaling_factor", 1.0))

    # ── Conditioning ────────────────────────────────────────────────────
    cap_feats_2d = prepare_cap_feats(pipe, load_device, dtype)
    logger.info("cap_feats_2d: %s", tuple(cap_feats_2d.shape))

    # Offload text encoder
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    # ── Gradient checkpointing ──────────────────────────────────────────
    if config.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()

    if config.disable_vae_force_upcast:
        pipe.vae.config.force_upcast = False

    # ── Resume detection ──────────────────────────────────────────────
    resume_mode = _detect_resume_mode(config.resume_from)
    logger.info("Resume mode: %s", resume_mode)

    # ── LoRA ────────────────────────────────────────────────────────────
    if resume_mode == ResumeMode.WEIGHTS_ONLY:
        assert config.resume_from is not None
        pipe.transformer = _load_weights_only_resume_model(
            pipe.transformer,
            config.resume_from,
        )
        logger.info("Loaded LoRA weights from %s (weights-only resume)", config.resume_from)
    else:
        # NONE: fresh random LoRA; FULL: structure created here, weights overwritten by accelerator.load_state
        pipe.transformer = apply_lora(
            pipe.transformer,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )

    # ── Dataset ─────────────────────────────────────────────────────────
    ds = FTDPairDataset(config.pairs_dir, load_pixels=(config.rec_loss_every > 0))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=ftd_collate,
        drop_last=True,
    )
    logger.info("Dataset: %d samples", len(ds))

    wandb = None
    wandb_run = None
    if config.wandb_enabled and accelerator.is_main_process:
        try:
            import wandb as _wandb
        except ImportError as exc:
            raise RuntimeError(
                "WandB logging is enabled but `wandb` is not installed. "
                "Install with: uv pip install -e '.[training]'"
            ) from exc
        wandb = _wandb
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            mode=config.wandb_mode,
            config=_wandb_config_dict(config),
        )
        wandb_run.summary["dataset_size"] = len(ds)
        wandb_run.summary["t_scale"] = t_scale
        wandb_run.summary["vae_scaling_factor"] = vae_sf
        wandb_run.summary["cap_feats_shape"] = tuple(int(v) for v in cap_feats_2d.shape)
        if resume_mode != ResumeMode.NONE:
            wandb_run.summary["resumed_from"] = str(config.resume_from)
            wandb_run.summary["resume_mode"] = resume_mode

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    pipe.transformer, optimizer, dl = accelerator.prepare(pipe.transformer, optimizer, dl)

    # From here on, use accelerator.device for all training tensors so
    # that each rank gets its correct device in multi-GPU setups.
    device = accelerator.device

    # Move cap_feats_2d to the correct accelerator device
    cap_feats_2d = cap_feats_2d.to(device=device, dtype=dtype)

    # Move VAE to accelerator device (not wrapped by prepare, but must
    # be on the correct rank device for recon loss decoding).
    pipe.vae.to(device)

    # ── TV-LPIPS loss (initialized once, pinned to training device) ─────
    tv_lpips: TVLPIPSLoss | None = None
    if config.rec_loss_every > 0:
        tv_lpips = TVLPIPSLoss(gamma=config.gamma_tv).eval()
        # Eagerly initialize LPIPS once so .to(device) can move real weights.
        tv_lpips._ensure_lpips()
        tv_lpips.to(device)
        for p in tv_lpips.parameters():
            p.requires_grad_(False)

    # ── TL constants ────────────────────────────────────────────────────
    TL = config.tl
    TL_bc = torch.tensor([TL], device=device, dtype=dtype).view(1, 1, 1, 1)

    # ── Seed ────────────────────────────────────────────────────────────
    if config.seed is not None:
        torch.manual_seed(config.seed)

    # ── Resume: load full state ──────────────────────────────────────────
    global_step = 0
    if resume_mode == ResumeMode.FULL:
        assert config.resume_from is not None
        state_json = _json.loads(
            (config.resume_from / "training_state.json").read_text(encoding="utf-8")
        )
        _validate_full_resume_config(config, state_json)
        if "global_step" not in state_json:
            raise ValueError(
                f"Missing global_step in checkpoint state file: "
                f"{config.resume_from / 'training_state.json'}"
            )
        accelerator.load_state(str(config.resume_from / "accelerator_state"))
        global_step = int(state_json["global_step"])
        logger.info("Resumed full training state from step %d", global_step)

    # ── Training loop ───────────────────────────────────────────────────
    pipe.transformer.train()
    pipe.vae.eval()

    use_adl = config.lambda_adl > 0
    use_z0 = config.lambda_z0 > 0
    loss_log: dict[str, float] = {"ftd": 0.0, "z0": 0.0, "rec": 0.0, "adl": 0.0, "total": 0.0}
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=config.max_steps, initial=global_step, desc="FluxSR-FTD", disable=not accelerator.is_local_main_process)

    try:
        while global_step < config.max_steps:
            for batch in dl:
                if global_step >= config.max_steps:
                    break

                eps = batch["eps"].to(device=device, dtype=dtype)
                z0 = batch["z0"].to(device=device, dtype=dtype)
                zL = batch["zL"].to(device=device, dtype=dtype)
                B = eps.shape[0]

                u_t = eps - z0  # rectified flow teacher velocity

                with accelerator.accumulate(pipe.transformer):
                    # ── FTD loss (Eq. 16/17) ────────────────────────────────
                    t = torch.rand(B, device=device, dtype=dtype) * (1.0 - TL) + TL
                    t_bc = t.view(B, 1, 1, 1)

                    # Eq. 16: student trajectory interpolation
                    x_t = ((1.0 - t_bc) / (1.0 - TL)) * zL + ((t_bc - TL) / (1.0 - TL)) * eps

                    if use_adl:
                        with ADLHookContext(pipe.transformer) as adl_ctx:
                            v_theta = call_transformer(
                                pipe.transformer,
                                latents=x_t,
                                timestep=t,
                                cap_feats_2d=cap_feats_2d,
                            )
                        L_ADL = compute_adl_loss(
                            adl_ctx.features,
                            default_device=device,
                            default_dtype=dtype,
                        )
                        L_ADL = L_ADL.to(device=device, dtype=dtype)
                    else:
                        v_theta = call_transformer(
                            pipe.transformer,
                            latents=x_t,
                            timestep=t,
                            cap_feats_2d=cap_feats_2d,
                        )
                        L_ADL = torch.zeros((), device=device, dtype=dtype)

                    # Eq. 17
                    ftd_pred = u_t - v_theta * TL_bc
                    ftd_target = eps - zL
                    L_FTD = F.mse_loss(ftd_pred, ftd_target)

                    # ── Latent endpoint + recon losses (Eq. 18/21) ───────────
                    L_Z0 = torch.zeros((), device=device, dtype=dtype)
                    L_Rec = torch.tensor(0.0, device=device)
                    do_rec = (
                        config.rec_loss_every > 0
                        and global_step % config.rec_loss_every == 0
                        and "x0_pixels" in batch
                    )
                    do_z0 = use_z0
                    do_tl = do_rec or do_z0

                    if do_tl:
                        TL_t = torch.full((B,), TL, device=device, dtype=dtype)
                        need_grad_tl = do_z0 or not config.detach_recon

                        if need_grad_tl:
                            v_TL = call_transformer(
                                pipe.transformer,
                                latents=zL,
                                timestep=TL_t,
                                cap_feats_2d=cap_feats_2d,
                            )
                            z0_hat = zL - v_TL * TL_bc
                            if do_z0:
                                L_Z0 = F.smooth_l1_loss(z0_hat.float(), z0.float()).to(device=device, dtype=dtype)
                        else:
                            with torch.no_grad():
                                v_TL = call_transformer(
                                    pipe.transformer,
                                    latents=zL,
                                    timestep=TL_t,
                                    cap_feats_2d=cap_feats_2d,
                                )
                                z0_hat = zL - v_TL * TL_bc

                        if do_rec:
                            assert tv_lpips is not None
                            x_HR = batch["x0_pixels"].to(device=device, dtype=dtype)
                            autocast_dt = torch.bfloat16 if device.type != "cpu" else None
                            z_for_rec = z0_hat.detach() if config.detach_recon else z0_hat

                            if config.detach_recon:
                                with torch.no_grad():
                                    x0_hat = vae_decode_to_pixels(pipe.vae, z_for_rec, vae_sf, autocast_dtype=autocast_dt)
                                    L_MSE = F.mse_loss(x0_hat, x_HR)
                                    L_TVLP = tv_lpips(x0_hat.float(), x_HR.float())
                                L_Rec = (L_MSE + config.lambda_tvlpips * L_TVLP).to(device=device, dtype=dtype)
                            else:
                                x0_hat = vae_decode_to_pixels(pipe.vae, z_for_rec, vae_sf, autocast_dtype=autocast_dt)
                                L_MSE = F.mse_loss(x0_hat, x_HR)
                                L_TVLP = tv_lpips(x0_hat.float(), x_HR.float())
                                L_Rec = L_MSE + config.lambda_tvlpips * L_TVLP

                    loss = L_FTD + config.lambda_z0 * L_Z0 + L_Rec + config.lambda_adl * L_ADL

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # ── Logging ─────────────────────────────────────────────────
                global_step += 1
                loss_log["ftd"] += L_FTD.item()
                loss_log["z0"] += L_Z0.item()
                loss_log["rec"] += L_Rec.item()
                loss_log["adl"] += L_ADL.item()
                loss_log["total"] += loss.item()

                pbar.update(1)
                if global_step % config.log_every == 0:
                    n = config.log_every
                    avg_ftd = loss_log["ftd"] / n
                    avg_z0 = loss_log["z0"] / n
                    avg_rec = loss_log["rec"] / n
                    avg_adl = loss_log["adl"] / n
                    avg_total = loss_log["total"] / n
                    postfix = {
                        "ftd": f"{avg_ftd:.4f}",
                        "rec": f"{avg_rec:.4f}",
                        "tot": f"{avg_total:.4f}",
                    }
                    if use_z0:
                        postfix["z0"] = f"{avg_z0:.4f}"
                    if use_adl:
                        postfix["adl"] = f"{avg_adl:.4f}"
                    pbar.set_postfix(postfix)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss_ftd": avg_ftd,
                                "train/loss_z0": avg_z0,
                                "train/loss_rec": avg_rec,
                                "train/loss_adl": avg_adl,
                                "train/loss_total": avg_total,
                                "train/lr": optimizer.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )
                    loss_log = {k: 0.0 for k in loss_log}

                if config.save_every > 0 and global_step % config.save_every == 0:
                    sp = save_dir / f"lora_step_{global_step}"
                    save_lora(pipe.transformer, sp, accelerator=accelerator)
                    if config.save_full_state:
                        _save_training_state(sp, accelerator, global_step, config)
                    ok = (sp / "adapter_config.json").exists()
                    logger.info("Step %d saved: %s (%s)", global_step, sp, "OK" if ok else "MISSING adapter_config!")
                    if config.checkpoint_infer_grid and accelerator.is_main_process:
                        try:
                            x0_default = batch.get("x0_pixels")
                            if x0_default is not None:
                                x0_default = x0_default[:1].to(device=device, dtype=dtype)
                            eval_samples = _gather_checkpoint_eval_samples(
                                config=config,
                                pipe=pipe,
                                device=device,
                                dtype=dtype,
                                default_zL=zL,
                                default_x0_pixels=x0_default,
                            )
                            wandb_payload: dict[str, Any] = {}
                            for sample_name, sample_zL, sample_x0 in eval_samples:
                                safe_name = _sanitize_eval_name(sample_name)
                                grid_path = _save_checkpoint_inference_grid(
                                    transformer=pipe.transformer,
                                    vae=pipe.vae,
                                    zL=sample_zL[:1],
                                    x0_pixels=sample_x0,
                                    tl=TL,
                                    vae_sf=vae_sf,
                                    cap_feats_2d=cap_feats_2d,
                                    out_path=sp / f"inference_grid_{safe_name}.png",
                                    extra_sr_scales=config.checkpoint_sr_scales,
                                    extra_refine_steps=config.checkpoint_refine_steps,
                                )
                                logger.info("Saved checkpoint inference grid [%s]: %s", safe_name, grid_path)
                                if wandb_run is not None and wandb is not None and config.wandb_log_checkpoint_grids:
                                    wandb_payload[f"train/checkpoint_grid/{safe_name}"] = wandb.Image(str(grid_path))
                            if wandb_payload and wandb_run is not None:
                                wandb_run.log(wandb_payload, step=global_step)
                        except Exception:
                            logger.exception("Failed to save checkpoint inference grid at step %d", global_step)
                    if wandb_run is not None and wandb is not None and config.wandb_log_checkpoints and ok:
                        artifact = wandb.Artifact(
                            name=f"zimagesr-lora-{wandb_run.id}-step-{global_step}",
                            type="model",
                            metadata={"step": global_step},
                        )
                        artifact.add_dir(str(sp))
                        wandb_run.log_artifact(artifact)

        # ── Final save ──────────────────────────────────────────────────────
        final = save_dir / "lora_final"
        save_lora(pipe.transformer, final, accelerator=accelerator)
        if config.save_full_state:
            _save_training_state(final, accelerator, global_step, config)
        logger.info("Final LoRA saved: %s", final)

        if wandb_run is not None and wandb is not None and config.wandb_log_checkpoints:
            artifact = wandb.Artifact(
                name=f"zimagesr-lora-{wandb_run.id}-final",
                type="model",
                metadata={"step": global_step},
            )
            artifact.add_dir(str(final))
            wandb_run.log_artifact(artifact)
            wandb_run.summary["final_path"] = str(final)

        return {
            "final_path": str(final),
            "steps": global_step,
            "wandb_enabled": bool(wandb_run is not None),
            "resumed_from": str(config.resume_from) if config.resume_from else None,
            "resume_mode": resume_mode,
        }
    finally:
        pbar.close()
        if wandb_run is not None:
            wandb_run.finish()
