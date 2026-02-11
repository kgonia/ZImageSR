from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from zimagesr.training.config import TrainConfig
from zimagesr.training.losses import compute_adl_loss
from zimagesr.training.train import _resolve_dtype, _wandb_config_dict


# ---------------------------------------------------------------------------
# FTD math tests (Eq. 16/17 from FluxSR paper)
# ---------------------------------------------------------------------------


class TestFTDMath:
    """Verify the FTD loss formulas match the paper exactly."""

    def test_eq16_interpolation_at_t_equals_TL(self):
        """At t=TL, x_t should equal zL."""
        TL = 0.25
        B, C, H, W = 2, 16, 4, 4
        zL = torch.randn(B, C, H, W)
        eps = torch.randn(B, C, H, W)
        t_bc = torch.full((B, 1, 1, 1), TL)

        x_t = ((1.0 - t_bc) / (1.0 - TL)) * zL + ((t_bc - TL) / (1.0 - TL)) * eps

        assert torch.allclose(x_t, zL, atol=1e-6)

    def test_eq16_interpolation_at_t_equals_1(self):
        """At t=1, x_t should equal eps."""
        TL = 0.25
        B, C, H, W = 2, 16, 4, 4
        zL = torch.randn(B, C, H, W)
        eps = torch.randn(B, C, H, W)
        t_bc = torch.ones(B, 1, 1, 1)

        x_t = ((1.0 - t_bc) / (1.0 - TL)) * zL + ((t_bc - TL) / (1.0 - TL)) * eps

        assert torch.allclose(x_t, eps, atol=1e-6)

    def test_eq17_perfect_prediction_gives_zero_loss(self):
        """If v_theta predicts perfectly, FTD loss should be zero."""
        TL = 0.25
        B, C, H, W = 1, 16, 4, 4
        eps = torch.randn(B, C, H, W)
        z0 = torch.randn(B, C, H, W)
        zL = torch.randn(B, C, H, W)

        u_t = eps - z0  # teacher velocity
        ftd_target = eps - zL

        # Perfect prediction: v_theta * TL = u_t - ftd_target = (eps - z0) - (eps - zL) = zL - z0
        # So v_theta = (zL - z0) / TL
        v_theta_perfect = (zL - z0) / TL

        TL_bc = torch.tensor([TL]).view(1, 1, 1, 1)
        ftd_pred = u_t - v_theta_perfect * TL_bc

        loss = F.mse_loss(ftd_pred, ftd_target)
        assert loss.item() < 1e-10

    def test_t_sampling_range(self):
        """Sampled t should be in [TL, 1]."""
        TL = 0.25
        B = 1000
        t = torch.rand(B) * (1.0 - TL) + TL
        assert t.min() >= TL - 1e-6
        assert t.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_defaults(self, tmp_path):
        cfg = TrainConfig(pairs_dir=tmp_path)
        assert cfg.tl == 0.25
        assert cfg.batch_size == 4
        assert cfg.lora_rank == 16
        assert cfg.max_steps == 750
        assert cfg.rec_loss_every == 8
        assert cfg.lambda_adl == 0.0
        assert cfg.wandb_enabled is False
        assert cfg.wandb_project == "zimagesr"

    def test_custom_values(self, tmp_path):
        cfg = TrainConfig(
            pairs_dir=tmp_path,
            tl=0.15,
            batch_size=2,
            learning_rate=1e-4,
            max_steps=100,
        )
        assert cfg.tl == 0.15
        assert cfg.batch_size == 2
        assert cfg.learning_rate == 1e-4
        assert cfg.max_steps == 100

    def test_pairs_dir_required(self):
        with pytest.raises(TypeError):
            TrainConfig()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Recon loss: one-step prediction z0_hat = zL - v * TL
# ---------------------------------------------------------------------------


class TestReconStep:
    def test_one_step_prediction(self):
        """z0_hat = zL - v_theta(zL, TL) * TL."""
        TL = 0.25
        B, C, H, W = 1, 16, 4, 4
        zL = torch.randn(B, C, H, W)

        # Simulate: perfect velocity that recovers z0
        z0_true = torch.randn(B, C, H, W)
        v_at_TL = (zL - z0_true) / TL  # v such that zL - v*TL = z0

        TL_bc = torch.tensor([TL]).view(1, 1, 1, 1)
        z0_hat = zL - v_at_TL * TL_bc

        assert torch.allclose(z0_hat, z0_true, atol=1e-6)


# ---------------------------------------------------------------------------
# _resolve_dtype
# ---------------------------------------------------------------------------


class TestResolveDtype:
    def test_explicit_dtype_used(self):
        assert _resolve_dtype("float16", "cuda") == torch.float16
        assert _resolve_dtype("float32", "cuda") == torch.float32
        assert _resolve_dtype("bfloat16", "cpu") == torch.bfloat16  # explicit override

    def test_cpu_defaults_to_float32(self):
        assert _resolve_dtype(None, "cpu") == torch.float32

    def test_cuda_defaults_to_bfloat16(self):
        assert _resolve_dtype(None, "cuda") == torch.bfloat16
        assert _resolve_dtype(None, "cuda:0") == torch.bfloat16


# ---------------------------------------------------------------------------
# ADL fallback/device behavior
# ---------------------------------------------------------------------------


class TestAdlFallback:
    def test_empty_features_returns_device_aware_zero(self):
        loss = compute_adl_loss(
            {},
            default_device="cpu",
            default_dtype=torch.float32,
        )
        assert loss.item() == pytest.approx(0.0)
        assert loss.device.type == "cpu"
        assert loss.dtype == torch.float32

    def test_non_empty_features_preserves_value(self):
        feat = torch.randn(2, 8, 16)
        expected = compute_adl_loss({0: feat})
        actual = compute_adl_loss(
            {0: feat},
            default_device="cpu",
            default_dtype=torch.float32,
        )
        assert actual.item() == pytest.approx(expected.item(), abs=1e-6)


class TestWandbConfigDict:
    def test_paths_are_serialized(self, tmp_path):
        cfg = TrainConfig(pairs_dir=tmp_path / "pairs", save_dir=tmp_path / "save")
        data = _wandb_config_dict(cfg)
        assert isinstance(data["pairs_dir"], str)
        assert isinstance(data["save_dir"], str)
