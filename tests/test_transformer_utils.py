from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from zimagesr.training.transformer_utils import (
    ADLHookContext,
    _first_tensor,
    call_transformer,
    prepare_cap_feats,
    vae_decode_to_pixels,
)


# ---------------------------------------------------------------------------
# Fake transformer that mimics Z-Image's forward signature
# ---------------------------------------------------------------------------


class FakeZImageTransformer(torch.nn.Module):
    """Mimics Z-Image transformer: accepts List[Tensor(C,1,H,W)]."""

    def __init__(self, *, cap_feat_dim: int = 2560):
        super().__init__()
        self.config = SimpleNamespace(cap_feat_dim=cap_feat_dim, t_scale=1000.0)
        self.forward_calls: list = []

    def forward(self, all_image, t, all_cap_feats, return_dict=True):
        self.forward_calls.append(
            {
                "all_image_len": len(all_image),
                "shapes": [tuple(x.shape) for x in all_image],
                "cap_shapes": [tuple(c.shape) for c in all_cap_feats],
                "t": t,
                "return_dict": return_dict,
            }
        )
        # Return list of per-sample results: same shape as input
        out = [img.clone() * 0.1 for img in all_image]
        if return_dict:
            return SimpleNamespace(sample=out)
        return (out,)


# ---------------------------------------------------------------------------
# Fake transformer with attention.to_out.0 submodules for ADL tests
# ---------------------------------------------------------------------------


class FakeAttentionBlock(torch.nn.Module):
    """Mimics a Z-Image attention block with to_out.0 projection."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention.to_out[0](x)


class FakeTransformerWithAttention(torch.nn.Module):
    """Transformer with layers.N.attention.to_out.0 structure."""

    def __init__(self, num_layers: int = 3, num_cr: int = 1, dim: int = 32):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeAttentionBlock(dim) for _ in range(num_layers)])
        self.context_refiner = torch.nn.ModuleList([FakeAttentionBlock(dim) for _ in range(num_cr)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        for cr in self.context_refiner:
            x = cr(x)
        return x


# ---------------------------------------------------------------------------
# ADLHookContext tests
# ---------------------------------------------------------------------------


class TestADLHookContext:
    def test_hooks_registered_on_correct_modules(self):
        tr = FakeTransformerWithAttention(num_layers=3, num_cr=1)
        with ADLHookContext(tr) as ctx:
            assert len(ctx._hooks) == 4  # 3 layers + 1 context_refiner

    def test_features_captured_after_forward(self):
        tr = FakeTransformerWithAttention(num_layers=2, num_cr=1, dim=16)
        x = torch.randn(2, 8, 16)  # (B, N, D)
        with ADLHookContext(tr) as ctx:
            tr(x)
            feats = ctx.features
            assert len(feats) == 3  # 2 layers + 1 cr
            for feat in feats.values():
                assert feat.shape == (2, 8, 16)

    def test_hooks_removed_on_exit(self):
        tr = FakeTransformerWithAttention(num_layers=2, num_cr=0)
        with ADLHookContext(tr) as ctx:
            hooks = list(ctx._hooks)
        # After exit, hooks list is cleared
        assert len(ctx._hooks) == 0
        # Features from previous forward should still be accessible
        # but no new hooks fire
        tr(torch.randn(1, 4, 32))
        assert len(ctx.features) == 0  # nothing captured since hooks removed

    def test_no_matching_modules(self):
        """Transformer without attention.to_out.0 → no hooks."""
        tr = torch.nn.Linear(16, 16)
        with ADLHookContext(tr) as ctx:
            assert len(ctx._hooks) == 0
            assert len(ctx.features) == 0

    def test_works_with_peft_prefix(self):
        """Hooks should match even when wrapped in base_model.model.*."""
        inner = FakeTransformerWithAttention(num_layers=2, num_cr=0)
        # Simulate PEFT wrapping: base_model.model.layers.*
        wrapper = torch.nn.Module()
        wrapper.base_model = torch.nn.Module()
        wrapper.base_model.model = inner
        with ADLHookContext(wrapper) as ctx:
            assert len(ctx._hooks) == 2

    def test_2d_output_gets_batch_dim(self):
        """Linear output (N, D) should be reshaped to (1, N, D)."""
        tr = FakeTransformerWithAttention(num_layers=1, num_cr=0, dim=8)
        x = torch.randn(4, 8)  # 2D input to linear: (N, D)
        with ADLHookContext(tr) as ctx:
            tr.layers[0](x)
            feat = ctx.features[0]
            assert feat.ndim == 3
            assert feat.shape[0] == 1  # batch dim added

    def test_hooks_removed_when_exception_raised(self):
        tr = FakeTransformerWithAttention(num_layers=2, num_cr=0, dim=8)
        with pytest.raises(RuntimeError, match="boom"):
            with ADLHookContext(tr) as ctx:
                assert len(ctx._hooks) == 2
                raise RuntimeError("boom")
        # New forwards should not populate old context after exception exit.
        tr(torch.randn(1, 4, 8))
        assert len(ctx.features) == 0


# ---------------------------------------------------------------------------
# call_transformer tests
# ---------------------------------------------------------------------------


class TestCallTransformer:
    def test_basic_shapes(self):
        tr = FakeZImageTransformer()
        B, C, H, W = 2, 16, 8, 8
        latents = torch.randn(B, C, H, W)
        ts = torch.tensor([500.0, 500.0])
        cap = torch.zeros(1, 2560)

        out = call_transformer(tr, latents=latents, timestep=ts, cap_feats_2d=cap)

        assert out.shape == (B, C, H, W)
        # Transformer was called once
        assert len(tr.forward_calls) == 1
        call = tr.forward_calls[0]
        assert call["all_image_len"] == B
        assert call["shapes"] == [(C, 1, H, W)] * B
        assert call["return_dict"] is False

    def test_single_batch(self):
        tr = FakeZImageTransformer()
        latents = torch.randn(1, 16, 4, 4)
        ts = torch.tensor([250.0])
        cap = torch.zeros(1, 2560)

        out = call_transformer(tr, latents=latents, timestep=ts, cap_feats_2d=cap)
        assert out.shape == (1, 16, 4, 4)

    def test_output_stacked_tensor(self):
        """Transformer returning a stacked 5D tensor instead of list."""

        class StackedTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(cap_feat_dim=2560)

            def forward(self, all_image, t, all_cap_feats, return_dict=True):
                B = len(all_image)
                C, _, H, W = all_image[0].shape
                out = torch.randn(B, C, 1, H, W)
                return (out,)

        tr = StackedTransformer()
        out = call_transformer(
            tr,
            latents=torch.randn(3, 16, 4, 4),
            timestep=torch.tensor([1.0, 1.0, 1.0]),
            cap_feats_2d=torch.zeros(1, 2560),
        )
        assert out.shape == (3, 16, 4, 4)

    def test_cap_feats_replicated(self):
        tr = FakeZImageTransformer()
        B = 3
        latents = torch.randn(B, 16, 4, 4)
        ts = torch.ones(B)
        cap = torch.ones(2, 2560)

        call_transformer(tr, latents=latents, timestep=ts, cap_feats_2d=cap)

        call = tr.forward_calls[0]
        assert call["cap_shapes"] == [(2, 2560)] * B


# ---------------------------------------------------------------------------
# _first_tensor tests
# ---------------------------------------------------------------------------


class TestFirstTensor:
    def test_tensor_input(self):
        t = torch.tensor([1.0])
        assert _first_tensor(t) is t

    def test_tuple_input(self):
        t = torch.tensor([2.0])
        result = _first_tensor((None, t, "abc"))
        assert result is t

    def test_none_result(self):
        assert _first_tensor("hello") is None
        assert _first_tensor((None, "x")) is None
        assert _first_tensor(42) is None


# ---------------------------------------------------------------------------
# prepare_cap_feats tests
# ---------------------------------------------------------------------------


class TestPrepareCapFeats:
    def test_returns_zeros_when_no_encode_prompt(self):
        pipe = SimpleNamespace(
            transformer=SimpleNamespace(
                config=SimpleNamespace(cap_feat_dim=128),
            ),
        )
        result = prepare_cap_feats(pipe, "cpu", torch.float32)
        assert result.shape == (1, 128)
        assert (result == 0).all()

    def test_returns_zeros_when_encode_prompt_returns_none(self):
        class FakePipe:
            transformer = SimpleNamespace(
                config=SimpleNamespace(cap_feat_dim=64),
            )

            def encode_prompt(self, prompt, *, device=None, num_images_per_prompt=1):
                return (None, None)

        result = prepare_cap_feats(FakePipe(), "cpu", torch.float32)
        assert result.shape == (1, 64)

    def test_returns_tensor_from_encode_prompt(self):
        cap_dim = 32

        class FakePipe:
            transformer = SimpleNamespace(
                config=SimpleNamespace(cap_feat_dim=cap_dim),
            )

            def encode_prompt(self, prompt, *, device=None, num_images_per_prompt=1):
                return (torch.ones(1, 3, cap_dim),)

        result = prepare_cap_feats(FakePipe(), "cpu", torch.float32)
        assert result.shape == (3, cap_dim)
        assert result.dtype == torch.float32

    def test_wrong_dim_falls_back_to_zeros(self):
        class FakePipe:
            transformer = SimpleNamespace(
                config=SimpleNamespace(cap_feat_dim=64),
            )

            def encode_prompt(self, prompt, *, device=None, num_images_per_prompt=1):
                return (torch.ones(1, 3, 999),)

        result = prepare_cap_feats(FakePipe(), "cpu", torch.float32)
        assert result.shape == (1, 64)
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# vae_decode_to_pixels tests
# ---------------------------------------------------------------------------


class TestVaeDecodeToPixels:
    def _make_fake_vae(self):
        class FakeVAE(torch.nn.Module):
            def decode(self, z):
                # Return values in [-1, 1] range
                B, C, H, W = z.shape
                sample = torch.zeros(B, 3, H * 8, W * 8, device=z.device, dtype=z.dtype)
                sample[:] = 0.4  # mid-range
                return SimpleNamespace(sample=sample)

        return FakeVAE()

    def test_basic_decode(self):
        vae = self._make_fake_vae()
        latents = torch.randn(1, 16, 4, 4) * 0.3611
        out = vae_decode_to_pixels(vae, latents, scaling_factor=0.3611)
        assert out.shape == (1, 3, 32, 32)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_scaling_applied(self):
        calls = []

        class TrackVAE(torch.nn.Module):
            def decode(self, z):
                calls.append(z.clone())
                return SimpleNamespace(sample=torch.zeros(1, 3, 8, 8))

        vae = TrackVAE()
        latents = torch.ones(1, 4, 1, 1) * 2.0
        vae_decode_to_pixels(vae, latents, scaling_factor=2.0)
        # z_raw should be latents / 2.0 = 1.0
        assert torch.allclose(calls[0], torch.ones(1, 4, 1, 1))

    def test_autocast_dtype_none(self):
        vae = self._make_fake_vae()
        latents = torch.randn(1, 16, 2, 2) * 0.5
        out = vae_decode_to_pixels(vae, latents, scaling_factor=0.5, autocast_dtype=None)
        assert out.shape[1] == 3
