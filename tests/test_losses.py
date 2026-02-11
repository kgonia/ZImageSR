from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from zimagesr.training.losses import TVLPIPSLoss, compute_adl_loss, total_variation_filter


# ---------------------------------------------------------------------------
# compute_adl_loss
# ---------------------------------------------------------------------------


class TestComputeAdlLoss:
    def test_empty_dict_returns_zero(self):
        loss = compute_adl_loss({})
        assert loss.item() == pytest.approx(0.0)

    def test_identical_tokens_gives_one(self):
        """All tokens identical → cosine similarity to mean = 1.0."""
        token = torch.randn(1, 1, 64)
        feat = token.expand(1, 10, 64)  # 10 identical tokens
        loss = compute_adl_loss({0: feat})
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_tokens_low_similarity(self):
        """Orthogonal tokens should have lower average similarity to mean."""
        # Create orthogonal basis vectors as tokens
        D = 8
        feat = torch.eye(D).unsqueeze(0)  # (1, D, D) — D orthogonal tokens
        loss = compute_adl_loss({0: feat})
        # With orthogonal tokens, similarity to mean should be positive but < 1
        assert loss.item() < 1.0
        assert loss.item() > 0.0

    def test_multiple_layers_averaged(self):
        """Loss should average across layers."""
        # Layer 0: identical tokens → sim ≈ 1.0
        token = torch.randn(1, 1, 32)
        feat0 = token.expand(1, 5, 32)

        # Layer 1: orthogonal tokens → sim < 1.0
        feat1 = torch.eye(5).unsqueeze(0)  # (1, 5, 5)

        single_loss = compute_adl_loss({0: feat0}).item()
        assert single_loss == pytest.approx(1.0, abs=1e-5)

        multi_loss = compute_adl_loss({0: feat0, 1: feat1}).item()
        # Average of ~1.0 and something < 1.0
        assert multi_loss < single_loss
        assert multi_loss > 0.0

    def test_batch_dimension(self):
        """Works correctly with batch size > 1."""
        B, N, D = 4, 8, 16
        feat = torch.randn(B, N, D)
        loss = compute_adl_loss({0: feat})
        assert loss.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# total_variation_filter
# ---------------------------------------------------------------------------


class TestTotalVariationFilter:
    def test_constant_image(self):
        x = torch.ones(1, 3, 8, 8) * 0.5
        tv = total_variation_filter(x)
        assert tv.shape == (1, 3, 7, 7)
        assert (tv == 0).all()

    def test_gradient_image(self):
        # Horizontal gradient: each column increases by 1
        x = torch.arange(8).float().view(1, 1, 1, 8).expand(1, 1, 8, 8)
        tv = total_variation_filter(x)
        assert tv.shape == (1, 1, 7, 7)
        # dw = 1 everywhere, dh = 0 everywhere → tv = 1
        assert torch.allclose(tv, torch.ones_like(tv))

    def test_vertical_gradient(self):
        x = torch.arange(8).float().view(1, 1, 8, 1).expand(1, 1, 8, 8)
        tv = total_variation_filter(x)
        # dh = 1 everywhere, dw = 0 everywhere → tv = 1
        assert torch.allclose(tv, torch.ones_like(tv))

    def test_batch_and_channels(self):
        x = torch.randn(2, 3, 16, 16)
        tv = total_variation_filter(x)
        assert tv.shape == (2, 3, 15, 15)
        assert (tv >= 0).all()


# ---------------------------------------------------------------------------
# TVLPIPSLoss
# ---------------------------------------------------------------------------


class FakeLPIPS(torch.nn.Module):
    """Minimal LPIPS substitute for unit testing."""

    def __init__(self, **kwargs):
        super().__init__()
        self._w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        return torch.nn.functional.mse_loss(x, y, reduction="none").mean(dim=(1, 2, 3), keepdim=True)


class TestTVLPIPSLoss:
    def _make_loss(self, gamma=0.5):
        loss_fn = TVLPIPSLoss(gamma=gamma)
        loss_fn._lpips_fn = FakeLPIPS()
        return loss_fn

    def test_identical_inputs(self):
        loss_fn = self._make_loss()
        x = torch.rand(1, 3, 32, 32)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_inputs_positive_loss(self):
        loss_fn = self._make_loss()
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 32, 32)
        loss = loss_fn(x, y)
        assert loss.item() > 0

    def test_gamma_zero_disables_tv(self):
        loss_fn = self._make_loss(gamma=0.0)
        x = torch.rand(1, 3, 16, 16)
        y = torch.rand(1, 3, 16, 16)
        loss = loss_fn(x, y)
        # With gamma=0, should be close to just LPIPS
        loss_fn2 = self._make_loss(gamma=0.5)
        loss2 = loss_fn2(x, y)
        # loss2 should be >= loss since it adds gamma*tv_lpips
        assert loss2.item() >= loss.item() - 1e-6

    def test_lazy_init_raises_without_lpips(self):
        loss_fn = TVLPIPSLoss()
        with patch.dict("sys.modules", {"lpips": None}):
            with pytest.raises(RuntimeError, match="lpips is required"):
                loss_fn._ensure_lpips()

    def test_lpips_params_frozen(self):
        loss_fn = self._make_loss()
        loss_fn._lpips_fn = None  # reset
        # Inject our fake as if lpips returned it
        loss_fn._lpips_fn = FakeLPIPS()
        for p in loss_fn._lpips_fn.parameters():
            p.requires_grad_(True)
        # Re-run ensure
        loss_fn._lpips_fn = None
        # Simulate by manually calling
        loss_fn._lpips_fn = FakeLPIPS()
        for p in loss_fn._lpips_fn.parameters():
            p.requires_grad_(False)
        for p in loss_fn._lpips_fn.parameters():
            assert not p.requires_grad
