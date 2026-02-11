from __future__ import annotations

import inspect
import re
from typing import Self

import torch
import torch.nn as nn


_ADL_TARGET_PATTERN = re.compile(
    r"(?:layers|context_refiner)\.\d+\.attention\.to_out\.0$"
)


class ADLHookContext:
    """Context manager to capture attention output features for ADL loss.

    Registers forward hooks on ``attention.to_out.0`` modules inside
    ``layers.*`` and ``context_refiner.*`` blocks.  Works regardless of
    PEFT wrapping depth because it matches the *suffix* of each module's
    fully-qualified name.
    """

    def __init__(self, transformer: nn.Module) -> None:
        self._transformer = transformer
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._features: dict[int, torch.Tensor] = {}

    def __enter__(self) -> Self:
        idx = 0
        for name, module in self._transformer.named_modules():
            # Match suffix regardless of PEFT prefix like base_model.model.*
            parts = name.split(".")
            # Rebuild suffix candidates and check pattern
            for start in range(len(parts)):
                suffix = ".".join(parts[start:])
                if _ADL_TARGET_PATTERN.fullmatch(suffix):
                    capture_idx = idx
                    self._hooks.append(
                        module.register_forward_hook(
                            self._make_hook(capture_idx)
                        )
                    )
                    idx += 1
                    break
        return self

    def __exit__(self, *args: object) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, idx: int):
        def hook(module: nn.Module, input: object, output: torch.Tensor) -> None:
            # output from Linear: (*, D) — reshape to (B_eff, N, D)
            if output.ndim == 2:
                # (N, D) — single sample, add batch dim
                feat = output.unsqueeze(0)
            elif output.ndim == 3:
                feat = output  # already (B, N, D)
            else:
                # Flatten all but last dim, treat as (1, N_total, D)
                feat = output.reshape(1, -1, output.shape[-1])
            self._features[idx] = feat

        return hook

    @property
    def features(self) -> dict[int, torch.Tensor]:
        return self._features


def call_transformer(
    transformer: nn.Module,
    *,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    cap_feats_2d: torch.Tensor,
) -> torch.Tensor:
    """Call the Z-Image transformer in its expected List-of-tensors format.

    Args:
        transformer: Z-Image transformer (or PeftModel wrapping one).
        latents: ``(B, C, H, W)`` float tensor.
        timestep: ``(B,)`` float tensor (already multiplied by ``t_scale``).
        cap_feats_2d: ``(seq_len, cap_dim)`` — single 2-D tensor, replicated per sample.

    Returns:
        ``(B, C, H, W)`` velocity prediction.
    """
    B = latents.shape[0]

    # (B, C, H, W) -> List of (C, 1, H, W)
    all_image = [latents[i].unsqueeze(1) for i in range(B)]

    # One cap_feats per sample
    all_cap_feats = [cap_feats_2d for _ in range(B)]

    out = transformer(
        all_image,
        timestep,
        all_cap_feats,
        return_dict=False,
    )

    # Unwrap output
    result = out[0] if isinstance(out, (tuple, list)) else out

    if isinstance(result, list):
        processed = []
        for r in result:
            if r.ndim == 4 and r.shape[1] == 1:
                r = r.squeeze(1)  # (C, 1, H, W) -> (C, H, W)
            processed.append(r)
        result = torch.stack(processed, dim=0)  # (B, C, H, W)
    elif result.ndim == 5 and result.shape[2] == 1:
        result = result.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)

    return result


def prepare_cap_feats(
    pipe: object,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Extract null caption features from the pipeline.

    Z-Image's ``encode_prompt`` returns no tensor, so this falls back to
    zeros of shape ``(1, cap_feat_dim)``.

    Returns:
        ``(seq_len, cap_dim)`` tensor on *device* with *dtype*.
    """
    # Try extracting from the base transformer config, handle PeftModel wrapping
    transformer = pipe.transformer
    if hasattr(transformer, "config"):
        cfg = transformer.config
    elif hasattr(transformer, "base_model"):
        cfg = transformer.base_model.model.config
    else:
        cfg = None

    cap_dim = int(getattr(cfg, "cap_feat_dim", 2560)) if cfg is not None else 2560

    # Attempt to call encode_prompt
    null_cap_feats = None
    if hasattr(pipe, "encode_prompt"):
        sig = inspect.signature(pipe.encode_prompt)
        kwargs: dict = {}
        if "device" in sig.parameters:
            kwargs["device"] = device
        if "do_classifier_free_guidance" in sig.parameters:
            kwargs["do_classifier_free_guidance"] = False
        if "num_images_per_prompt" in sig.parameters:
            kwargs["num_images_per_prompt"] = 1

        with torch.no_grad():
            pe = pipe.encode_prompt("", **kwargs)

        # Find first tensor in return value
        null_cap_feats = _first_tensor(pe)

    if null_cap_feats is not None:
        cf = null_cap_feats.detach().to(device=device, dtype=dtype)
        while cf.ndim > 2:
            cf = cf[0]
        if cf.ndim == 1:
            cf = cf.unsqueeze(0)
        if cf.shape[-1] != cap_dim:
            cf = torch.zeros(1, cap_dim, device=device, dtype=dtype)
        return cf

    return torch.zeros(1, cap_dim, device=device, dtype=dtype)


def _first_tensor(x: object) -> torch.Tensor | None:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for v in x:
            if torch.is_tensor(v):
                return v
    return None


def vae_decode_to_pixels(
    vae: nn.Module,
    latents: torch.Tensor,
    scaling_factor: float,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Decode VAE latents to pixel-space ``[0, 1]`` tensor.

    Args:
        vae: VAE decoder module.
        latents: ``(B, C, H, W)`` latents (already scaled by ``scaling_factor``).
        scaling_factor: VAE ``scaling_factor`` used during encoding.
        autocast_dtype: If set, wraps decode in ``torch.amp.autocast``.

    Returns:
        ``(B, 3, H', W')`` float tensor in ``[0, 1]``.
    """
    z_raw = latents / scaling_factor
    if autocast_dtype is not None and latents.device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            x = vae.decode(z_raw).sample
    else:
        x = vae.decode(z_raw).sample
    return (x / 2 + 0.5).clamp(0, 1)
