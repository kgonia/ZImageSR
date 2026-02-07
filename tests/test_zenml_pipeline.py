from __future__ import annotations

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_FILE = ROOT / "src" / "zimagesr" / "pipelines" / "zenml_pipeline.py"


def _decorator_for(func_name: str) -> str:
    source = PIPELINE_FILE.read_text(encoding="utf-8")
    pattern = rf"(@step[^\n]*)\ndef {re.escape(func_name)}\("
    match = re.search(pattern, source)
    assert match is not None, f"Decorator not found for {func_name}"
    return match.group(1)


def test_z01_inspect_pairs_step_cache_enabled():
    dec = _decorator_for("inspect_pairs_step")
    assert dec.strip() == "@step"


def test_z02_gather_pairs_step_cache_disabled():
    dec = _decorator_for("gather_pairs_step")
    assert "enable_cache=False" in dec


def test_z03_upload_s3_step_cache_disabled():
    dec = _decorator_for("upload_s3_step")
    assert "enable_cache=False" in dec


def test_z04_download_s3_step_cache_disabled():
    dec = _decorator_for("download_s3_step")
    assert "enable_cache=False" in dec


def test_z05_gather_pairs_step_constructs_config_and_calls_gather(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("zenml")

    from zimagesr.pipelines import zenml_pipeline as zp

    captured = {}

    def fake_gather(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(zp, "gather_offline_pairs", fake_gather)

    entrypoint = getattr(zp.gather_pairs_step, "entrypoint", None)
    if entrypoint is None:
        pytest.skip("ZenML step object has no accessible entrypoint")

    out = entrypoint(
        model_id="model-id",
        out_dir="/tmp/out",
        n=9,
        hr_size=128,
        num_inference_steps=2,
        guidance_scale=0.1,
        base_seed=77,
        start_index=3,
        prompt="",
        negative_prompt="neg",
        save_x0_png=True,
        generate_lr=False,
        skip_existing=True,
        cache_null_prompt=False,
        offload_text_encoder=True,
        device="cpu",
        dtype="float32",
        debug=True,
        debug_every=7,
    )
    assert out == "/tmp/out"
    cfg = captured["cfg"]
    assert cfg.model_id == "model-id"
    assert str(cfg.out_dir) == "/tmp/out"
    assert cfg.n == 9
    assert cfg.hr_size == 128
    assert cfg.debug_every == 7
