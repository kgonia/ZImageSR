from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_adl_loss(
    features_by_layer: dict[int, torch.Tensor],
    *,
    default_device: torch.device | str | None = None,
    default_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Attention Diversification Loss (FluxSR Eq. 22-23).

    Computes mean cosine similarity of each token to the layer mean,
    averaged across all layers.

    Args:
        features_by_layer: ``{layer_idx: (B, N, D)}`` tensors from
            attention output projections.
        default_device: Device used for the zero fallback when no features
            are provided.
        default_dtype: Dtype used for the zero fallback when no features
            are provided.

    Returns:
        Scalar loss (0.0 if no layers provided).
    """
    if not features_by_layer:
        kwargs: dict[str, object] = {}
        if default_device is not None:
            kwargs["device"] = default_device
        if default_dtype is not None:
            kwargs["dtype"] = default_dtype
        return torch.zeros((), **kwargs)

    layer_losses = []
    for feat in features_by_layer.values():
        # feat: (B, N, D)
        mean_token = feat.mean(dim=1, keepdim=True)  # (B, 1, D)
        # Cosine similarity between each token and the mean
        cos_sim = F.cosine_similarity(feat, mean_token, dim=-1)  # (B, N)
        layer_losses.append(cos_sim.mean())

    return torch.stack(layer_losses).mean()


def total_variation_filter(x: torch.Tensor) -> torch.Tensor:
    """Compute TV gradient magnitudes: ``|dh| + |dw|``.

    Args:
        x: ``(B, C, H, W)`` tensor.

    Returns:
        ``(B, C, H-1, W-1)`` tensor of absolute gradient sums.
    """
    dh = torch.abs(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
    dw = torch.abs(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
    return dh + dw


class TVLPIPSLoss(nn.Module):
    """LPIPS + gamma * LPIPS-on-TV-filtered loss (FluxSR Eq. 21).

    The LPIPS network is lazy-initialized on first forward call to avoid
    importing ``lpips`` at module load time.
    """

    def __init__(self, gamma: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self._lpips_fn: nn.Module | None = None

    def _ensure_lpips(self) -> nn.Module:
        if self._lpips_fn is None:
            try:
                import lpips as _lpips
            except ImportError as exc:
                raise RuntimeError(
                    "lpips is required for TVLPIPSLoss. "
                    "Install with: pip install lpips"
                ) from exc
            self._lpips_fn = _lpips.LPIPS(net="vgg")
            for p in self._lpips_fn.parameters():
                p.requires_grad_(False)
        return self._lpips_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute TV-LPIPS loss.

        Args:
            x: Predicted pixels ``(B, 3, H, W)`` in ``[0, 1]``.
            y: Target pixels ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns:
            Scalar loss.
        """
        lpips_fn = self._ensure_lpips()

        # LPIPS expects [-1, 1]
        xn, yn = x * 2 - 1, y * 2 - 1
        loss1 = lpips_fn(xn, yn).mean()

        tvx = total_variation_filter(x)
        tvy = total_variation_filter(y)
        tv_max = max(tvx.detach().max().item(), tvy.detach().max().item(), 1e-6)
        tvx = (tvx / tv_max).clamp(0, 1)
        tvy = (tvy / tv_max).clamp(0, 1)
        loss2 = lpips_fn(tvx * 2 - 1, tvy * 2 - 1).mean()

        return loss1 + self.gamma * loss2
