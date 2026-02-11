from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FTDPairDataset(Dataset):
    """Dataset of ``(eps, z0, zL)`` latent pairs for FTD training.

    Optionally loads ``x0.png`` as pixel tensor for reconstruction loss.
    """

    def __init__(self, pairs_dir: Path, load_pixels: bool = False) -> None:
        self.load_pixels = load_pixels
        self.items: list[Path] = []
        for d in sorted(pairs_dir.iterdir()):
            if not d.is_dir():
                continue
            if (d / "eps.pt").exists() and (d / "z0.pt").exists() and (d / "zL.pt").exists():
                if load_pixels and not (d / "x0.png").exists():
                    continue
                self.items.append(d)
        if not self.items:
            raise RuntimeError(f"No valid samples found in {pairs_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = self.items[idx]
        eps = torch.load(d / "eps.pt", map_location="cpu", weights_only=True).squeeze(0)
        z0 = torch.load(d / "z0.pt", map_location="cpu", weights_only=True).squeeze(0)
        zL = torch.load(d / "zL.pt", map_location="cpu", weights_only=True).squeeze(0)
        out: dict[str, torch.Tensor] = {"eps": eps, "z0": z0, "zL": zL}
        if self.load_pixels:
            from PIL import Image

            x0 = Image.open(d / "x0.png").convert("RGB")
            x0_np = np.array(x0).astype(np.float32) / 255.0
            out["x0_pixels"] = torch.from_numpy(x0_np).permute(2, 0, 1)
        return out


def ftd_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sample dicts into batched tensors."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


@torch.no_grad()
def generate_zl_latents(
    pairs_dir: Path,
    pipe: Any,
    device: torch.device | str,
    dtype: torch.dtype,
    skip_existing: bool = True,
) -> int:
    """Encode ``lr_up.png`` → ``zL.pt`` for each sample directory.

    Uses the existing :func:`vae_encode_latents_safe` logic.

    Returns:
        Number of ``zL.pt`` files created.
    """
    from zimagesr.data.offline_pairs import vae_encode_latents_safe

    created = 0
    dirs = sorted([p for p in pairs_dir.iterdir() if p.is_dir()])
    for d in tqdm(dirs, desc="zL.pt"):
        zl_path = d / "zL.pt"
        if skip_existing and zl_path.exists():
            continue
        lr_up = d / "lr_up.png"
        if not lr_up.exists():
            continue
        from PIL import Image

        pil_img = Image.open(lr_up).convert("RGB")
        zL = vae_encode_latents_safe(pipe, pil_img, device=device, dtype=dtype).cpu()
        torch.save(zL, zl_path)
        created += 1
    return created
