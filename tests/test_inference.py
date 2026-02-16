from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zimagesr.training import inference as inf


def test_one_step_sr_uses_normalized_timestep(monkeypatch):
    captured: dict[str, torch.Tensor] = {}

    def fake_call_transformer(transformer, *, latents, timestep, cap_feats_2d):
        captured["timestep"] = timestep.detach().cpu()
        return torch.zeros_like(latents)

    def fake_vae_decode_to_pixels(vae, latents, scaling_factor, autocast_dtype=None):
        bsz = latents.shape[0]
        return torch.zeros((bsz, 3, 16, 16), dtype=torch.float32, device=latents.device)

    monkeypatch.setattr(inf, "call_transformer", fake_call_transformer)
    monkeypatch.setattr(inf, "vae_decode_to_pixels", fake_vae_decode_to_pixels)

    _ = inf.one_step_sr(
        transformer=torch.nn.Identity(),
        vae=torch.nn.Identity(),
        lr_latent=torch.zeros((1, 16, 4, 4), dtype=torch.float32),
        tl=0.25,
        vae_sf=0.3611,
        cap_feats_2d=torch.zeros((1, 2560), dtype=torch.float32),
    )

    assert "timestep" in captured
    assert tuple(captured["timestep"].shape) == (1,)
    assert captured["timestep"].item() == pytest.approx(0.25)


def test_one_step_sr_multistep_refinement_calls_decreasing_t(monkeypatch):
    captured_steps: list[float] = []

    def fake_call_transformer(transformer, *, latents, timestep, cap_feats_2d):
        captured_steps.append(float(timestep[0].item()))
        return torch.zeros_like(latents)

    def fake_vae_decode_to_pixels(vae, latents, scaling_factor, autocast_dtype=None):
        bsz = latents.shape[0]
        return torch.zeros((bsz, 3, 16, 16), dtype=torch.float32, device=latents.device)

    monkeypatch.setattr(inf, "call_transformer", fake_call_transformer)
    monkeypatch.setattr(inf, "vae_decode_to_pixels", fake_vae_decode_to_pixels)

    _ = inf.one_step_sr(
        transformer=torch.nn.Identity(),
        vae=torch.nn.Identity(),
        lr_latent=torch.zeros((1, 16, 4, 4), dtype=torch.float32),
        tl=0.25,
        vae_sf=0.3611,
        cap_feats_2d=torch.zeros((1, 2560), dtype=torch.float32),
        refine_steps=4,
    )

    assert len(captured_steps) == 4
    assert captured_steps[0] == pytest.approx(0.25)
    assert captured_steps[-1] == pytest.approx(0.0625)
