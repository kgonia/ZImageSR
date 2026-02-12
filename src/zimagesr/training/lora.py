from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def find_lora_targets(model: nn.Module) -> list[str]:
    """Find attention ``nn.Linear`` layers suitable for LoRA targeting.

    Includes layers whose names contain attention-related keywords
    (``attn``, ``to_q``, ``to_k``, ``to_v``, ``to_out``, ``proj``, ``mlp``, ``ff``).
    Excludes norms, embeddings, time projections, and PEFT internals.

    Falls back to all ``nn.Linear`` layers (except norm/embed) if no
    attention layers are found.
    """
    exclude_keywords = ("norm", "embed", "time", "t_embed", "pos", "lora_", "base_layer", "peft_")
    attn_keywords = ("attention", "attn", "to_q", "to_k", "to_v", "to_out", "proj", "mlp", "ff")

    names: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(x in name for x in exclude_keywords):
            continue
        if any(x in name for x in attn_keywords):
            names.append(name)

    if not names:
        # Fallback: all Linear except norm/embed
        names = sorted(
            set(
                name
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear) and not any(x in name for x in ("lora_", "base_layer", "norm", "embed"))
            )
        )

    return sorted(set(names))


def apply_lora(
    transformer: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    targets: list[str] | None = None,
) -> nn.Module:
    """Wrap *transformer* with a PEFT LoRA adapter.

    Args:
        transformer: Base transformer module (frozen).
        rank: LoRA rank.
        alpha: LoRA alpha (scaling = alpha / rank).
        dropout: LoRA dropout.
        targets: Target module names. If ``None``, calls :func:`find_lora_targets`.

    Returns:
        ``PeftModel`` wrapping the transformer.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for LoRA training. Install with: pip install peft"
        ) from exc

    if targets is None:
        targets = find_lora_targets(transformer)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
    )
    return get_peft_model(transformer, lora_cfg)


def save_lora(model: nn.Module, path: Path | str, accelerator: object | None = None) -> None:
    """Save LoRA adapter weights to *path*.

    If an ``accelerator`` is provided, unwraps the model first.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        model = accelerator.unwrap_model(model)

    model.save_pretrained(path)


def load_lora_for_inference(
    base_transformer: nn.Module,
    lora_path: Path | str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> nn.Module:
    """Load a saved LoRA adapter onto a base transformer for inference.

    Args:
        base_transformer: Clean (non-PEFT) transformer.
        lora_path: Path to saved LoRA adapter directory.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``PeftModel`` in eval mode on the specified device.
    """
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "peft is required to load LoRA adapters. Install with: pip install peft"
        ) from exc

    lora_path = Path(lora_path).resolve()
    if not lora_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_path}")
    if not (lora_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"adapter_config.json not found in {lora_path}. "
            f"Contents: {[p.name for p in lora_path.iterdir()]}"
        )
    model = PeftModel.from_pretrained(base_transformer, str(lora_path))
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model
