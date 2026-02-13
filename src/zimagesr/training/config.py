from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def _default_save_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"./zimage_sr_lora_runs/ftd_run_{ts}")


@dataclass
class TrainConfig:
    """Configuration for FTD training loop."""

    pairs_dir: Path  # required — path to pairs/ directory

    model_id: str = "Tongyi-MAI/Z-Image-Turbo"

    # FTD
    tl: float = 0.25
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    max_steps: int = 750

    # Loss
    rec_loss_every: int = 8  # 0 to disable
    lambda_tvlpips: float = 1.0
    gamma_tv: float = 0.5
    detach_recon: bool = True  # gradient-free recon (saves VRAM)
    lambda_adl: float = 0.0  # ADL weight (0 = disabled)

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Checkpointing
    save_dir: Path = field(default_factory=_default_save_dir)
    save_every: int = 150
    log_every: int = 20

    # Device
    device: str | None = None
    dtype: str | None = None
    mixed_precision: str = "no"
    gradient_checkpointing: bool = True
    disable_vae_force_upcast: bool = True
    num_workers: int = 2
    seed: int | None = None

    # WandB
    wandb_enabled: bool = False
    wandb_project: str = "zimagesr"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"  # online/offline
    wandb_log_checkpoints: bool = True
    wandb_log_checkpoint_grids: bool = True

    # Checkpoint-time inference preview
    checkpoint_infer_grid: bool = False
    checkpoint_eval_ids: tuple[str, ...] = ()
    checkpoint_eval_images_dir: Path | None = None
    checkpoint_eval_images_limit: int = 4
    checkpoint_eval_input_upscale: float = 4.0
    checkpoint_eval_fit_multiple: int = 16

    # Resume
    resume_from: Path | None = None
