from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from zimagesr.training.config import TrainConfig
from zimagesr.training.losses import compute_adl_loss
from zimagesr.training.train import (
    ResumeMode,
    _count_trainable_params,
    _detect_resume_mode,
    _load_weights_only_resume_model,
    _resolve_dtype,
    _save_training_state,
    _validate_full_resume_config,
    _wandb_config_dict,
)


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
        assert cfg.save_dir.name.startswith("ftd_run_")
        assert cfg.wandb_enabled is False
        assert cfg.wandb_project == "zimagesr"
        assert cfg.wandb_log_checkpoint_grids is True
        assert cfg.checkpoint_infer_grid is False
        assert cfg.checkpoint_eval_ids == ()
        assert cfg.checkpoint_eval_images_dir is None
        assert cfg.checkpoint_eval_images_limit == 4
        assert cfg.checkpoint_eval_input_upscale == 4.0
        assert cfg.checkpoint_eval_fit_multiple == 16

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


# ---------------------------------------------------------------------------
# _detect_resume_mode
# ---------------------------------------------------------------------------


class TestDetectResumeMode:
    def test_none_returns_none_mode(self):
        assert _detect_resume_mode(None) == ResumeMode.NONE

    def test_full_checkpoint(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        (tmp_path / "adapter_model.safetensors").write_bytes(b"")
        (tmp_path / "training_state.json").write_text('{"global_step": 100}')
        (tmp_path / "accelerator_state").mkdir()
        assert _detect_resume_mode(tmp_path) == ResumeMode.FULL

    def test_weights_only_checkpoint(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        (tmp_path / "adapter_model.safetensors").write_bytes(b"")
        assert _detect_resume_mode(tmp_path) == ResumeMode.WEIGHTS_ONLY

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="contains neither"):
            _detect_resume_mode(tmp_path)

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _detect_resume_mode(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Resume config compatibility validation
# ---------------------------------------------------------------------------


class TestValidateFullResumeConfig:
    def test_matching_resume_config_is_accepted(self, tmp_path):
        cfg = TrainConfig(pairs_dir=tmp_path / "pairs")
        state_json = {"global_step": 123, "config": _wandb_config_dict(cfg)}
        _validate_full_resume_config(cfg, state_json)

    def test_lora_mismatch_raises_clear_error(self, tmp_path):
        cfg = TrainConfig(pairs_dir=tmp_path / "pairs", lora_rank=16)
        saved_cfg = _wandb_config_dict(cfg)
        saved_cfg["lora_rank"] = 8
        state_json = {"global_step": 123, "config": saved_cfg}
        with pytest.raises(ValueError, match="lora_rank"):
            _validate_full_resume_config(cfg, state_json)


# ---------------------------------------------------------------------------
# Weights-only resume trainability
# ---------------------------------------------------------------------------


class TestWeightsOnlyResumeModel:
    def test_weights_only_resume_loads_trainable_lora(self, tmp_path):
        pytest.importorskip("peft")
        from zimagesr.training.lora import apply_lora, save_lora

        base = torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
        )
        lora_model = apply_lora(
            base,
            rank=4,
            alpha=4,
            targets=["0", "2"],
        )
        save_lora(lora_model, tmp_path)

        resumed = _load_weights_only_resume_model(
            torch.nn.Sequential(
                torch.nn.Linear(8, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 8),
            ),
            tmp_path,
        )
        assert _count_trainable_params(resumed) > 0
        assert any("lora_" in name and p.requires_grad for name, p in resumed.named_parameters())


# ---------------------------------------------------------------------------
# _save_training_state
# ---------------------------------------------------------------------------


class TestSaveTrainingState:
    def _make_accelerator_stub(self, *, is_main: bool = True):
        """Create a minimal accelerator mock."""

        class AccStub:
            is_main_process = is_main

            def save_state(self, output_dir):
                from pathlib import Path

                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / "pytorch_model.bin").write_bytes(b"fake")

        return AccStub()

    def test_creates_json_and_accelerator_dir(self, tmp_path):
        import json

        acc = self._make_accelerator_stub()
        cfg = TrainConfig(pairs_dir=tmp_path / "pairs")
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()

        _save_training_state(ckpt, acc, global_step=42, config=cfg)

        assert (ckpt / "accelerator_state").is_dir()
        assert (ckpt / "training_state.json").is_file()
        state = json.loads((ckpt / "training_state.json").read_text())
        assert state["global_step"] == 42
        assert "config" in state

    def test_non_main_process_skips_json(self, tmp_path):
        acc = self._make_accelerator_stub(is_main=False)
        cfg = TrainConfig(pairs_dir=tmp_path / "pairs")
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()

        _save_training_state(ckpt, acc, global_step=10, config=cfg)

        assert (ckpt / "accelerator_state").is_dir()
        assert not (ckpt / "training_state.json").exists()


# ---------------------------------------------------------------------------
# Resume round-trip
# ---------------------------------------------------------------------------


class TestResumeRoundTrip:
    def test_save_then_detect_full(self, tmp_path):
        """Save state, then detect → should be FULL."""
        import json

        class AccStub:
            is_main_process = True

            def save_state(self, output_dir):
                from pathlib import Path

                Path(output_dir).mkdir(parents=True, exist_ok=True)

        cfg = TrainConfig(pairs_dir=tmp_path / "pairs")
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()

        _save_training_state(ckpt, AccStub(), global_step=300, config=cfg)

        assert _detect_resume_mode(ckpt) == ResumeMode.FULL

    def test_json_round_trips_step_counter(self, tmp_path):
        import json

        class AccStub:
            is_main_process = True

            def save_state(self, output_dir):
                from pathlib import Path

                Path(output_dir).mkdir(parents=True, exist_ok=True)

        cfg = TrainConfig(pairs_dir=tmp_path / "pairs")
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()

        _save_training_state(ckpt, AccStub(), global_step=512, config=cfg)

        state = json.loads((ckpt / "training_state.json").read_text())
        assert state["global_step"] == 512
