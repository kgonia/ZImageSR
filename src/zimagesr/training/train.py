from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from zimagesr.training.config import TrainConfig
from zimagesr.training.dataset import FTDPairDataset, ftd_collate
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

    # ── LoRA ────────────────────────────────────────────────────────────
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

    # ── TV-LPIPS loss (lazy, moved to device only when needed) ──────────
    tv_lpips: TVLPIPSLoss | None = None
    if config.rec_loss_every > 0:
        tv_lpips = TVLPIPSLoss(gamma=config.gamma_tv).eval()
        for p in tv_lpips.parameters():
            p.requires_grad_(False)

    # ── TL constants ────────────────────────────────────────────────────
    TL = config.tl
    TL_bc = torch.tensor([TL], device=device, dtype=dtype).view(1, 1, 1, 1)

    # ── Seed ────────────────────────────────────────────────────────────
    if config.seed is not None:
        torch.manual_seed(config.seed)

    # ── Training loop ───────────────────────────────────────────────────
    global_step = 0
    pipe.transformer.train()
    pipe.vae.eval()

    use_adl = config.lambda_adl > 0
    loss_log: dict[str, float] = {"ftd": 0.0, "rec": 0.0, "adl": 0.0, "total": 0.0}
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=config.max_steps, desc="FluxSR-FTD", disable=not accelerator.is_local_main_process)

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
                            timestep=t * t_scale,
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
                        timestep=t * t_scale,
                        cap_feats_2d=cap_feats_2d,
                    )
                    L_ADL = torch.zeros((), device=device, dtype=dtype)

                # Eq. 17
                ftd_pred = u_t - v_theta * TL_bc
                ftd_target = eps - zL
                L_FTD = F.mse_loss(ftd_pred, ftd_target)

                # ── Recon loss (Eq. 18/21) ──────────────────────────────
                L_Rec = torch.tensor(0.0, device=device)
                do_rec = (
                    config.rec_loss_every > 0
                    and global_step % config.rec_loss_every == 0
                    and "x0_pixels" in batch
                )

                if do_rec:
                    assert tv_lpips is not None
                    x_HR = batch["x0_pixels"].to(device=device, dtype=dtype)
                    TL_t = torch.full((B,), TL * t_scale, device=device, dtype=dtype)

                    if config.detach_recon:
                        with torch.no_grad():
                            v_TL = call_transformer(
                                pipe.transformer,
                                latents=zL,
                                timestep=TL_t,
                                cap_feats_2d=cap_feats_2d,
                            )
                            z0_hat = zL - v_TL * TL_bc
                            autocast_dt = torch.bfloat16 if device.type != "cpu" else None
                            x0_hat = vae_decode_to_pixels(pipe.vae, z0_hat, vae_sf, autocast_dtype=autocast_dt)

                            L_MSE = F.mse_loss(x0_hat, x_HR)

                            tv_lpips.to(device)
                            L_TVLP = tv_lpips(x0_hat.float(), x_HR.float())
                            tv_lpips.to("cpu")

                        L_Rec = (L_MSE + config.lambda_tvlpips * L_TVLP).to(device=device, dtype=dtype)
                    else:
                        v_TL = call_transformer(
                            pipe.transformer,
                            latents=zL,
                            timestep=TL_t,
                            cap_feats_2d=cap_feats_2d,
                        )
                        z0_hat = zL - v_TL * TL_bc
                        autocast_dt = torch.bfloat16 if device.type != "cpu" else None
                        x0_hat = vae_decode_to_pixels(pipe.vae, z0_hat, vae_sf, autocast_dtype=autocast_dt)

                        L_MSE = F.mse_loss(x0_hat, x_HR)

                        tv_lpips.to(device)
                        L_TVLP = tv_lpips(x0_hat, x_HR)
                        tv_lpips.to("cpu")
                        torch.cuda.empty_cache()

                        L_Rec = L_MSE + config.lambda_tvlpips * L_TVLP

                loss = L_FTD + L_Rec + config.lambda_adl * L_ADL

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # ── Logging ─────────────────────────────────────────────────
            global_step += 1
            loss_log["ftd"] += L_FTD.item()
            loss_log["rec"] += L_Rec.item()
            loss_log["adl"] += L_ADL.item()
            loss_log["total"] += loss.item()

            pbar.update(1)
            if global_step % config.log_every == 0:
                n = config.log_every
                postfix = {
                    "ftd": f"{loss_log['ftd'] / n:.4f}",
                    "rec": f"{loss_log['rec'] / n:.4f}",
                    "tot": f"{loss_log['total'] / n:.4f}",
                }
                if use_adl:
                    postfix["adl"] = f"{loss_log['adl'] / n:.4f}"
                pbar.set_postfix(postfix)
                loss_log = {k: 0.0 for k in loss_log}

            if config.save_every > 0 and global_step % config.save_every == 0:
                sp = save_dir / f"lora_step_{global_step}"
                save_lora(pipe.transformer, sp, accelerator=accelerator)
                ok = (sp / "adapter_config.json").exists()
                logger.info("Step %d saved: %s (%s)", global_step, sp, "OK" if ok else "MISSING adapter_config!")

    pbar.close()

    # ── Final save ──────────────────────────────────────────────────────
    final = save_dir / "lora_final"
    save_lora(pipe.transformer, final, accelerator=accelerator)
    logger.info("Final LoRA saved: %s", final)

    return {
        "final_path": str(final),
        "steps": global_step,
    }
