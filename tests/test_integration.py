from __future__ import annotations

import os
from pathlib import Path
import json

import pytest

pytestmark = [pytest.mark.slow]


if os.getenv("RUN_ZIMAGESR_INTEGRATION") != "1":
    pytest.skip("Set RUN_ZIMAGESR_INTEGRATION=1 to run integration tests.", allow_module_level=True)

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for integration tests.", allow_module_level=True)

from zimagesr.data.offline_pairs import GatherConfig, gather_offline_pairs, generate_lr_pairs, inspect_local_pairs


MODEL_ID = os.getenv("ZIMAGESR_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")


@pytest.mark.gpu
def test_i01_end_to_end_gather(tmp_path):
    cfg = GatherConfig(
        model_id=MODEL_ID,
        out_dir=tmp_path / "run",
        n=2,
        hr_size=256,
        num_inference_steps=1,
        guidance_scale=0.0,
        base_seed=123,
        start_index=0,
        save_x0_png=True,
        generate_lr=False,
        skip_existing=False,
        cache_null_prompt=True,
        offload_text_encoder=True,
        device="cuda",
        dtype="bfloat16",
        debug=True,
        debug_every=1,
    )
    gather_offline_pairs(cfg)
    for i in range(2):
        sample = cfg.out_dir / "pairs" / f"{i:06d}"
        assert (sample / "eps.pt").exists()
        assert (sample / "z0.pt").exists()
        assert (sample / "x0.png").exists()


@pytest.mark.gpu
def test_i02_gather_then_degrade(tmp_path):
    cfg = GatherConfig(
        model_id=MODEL_ID,
        out_dir=tmp_path / "run",
        n=1,
        hr_size=256,
        num_inference_steps=1,
        save_x0_png=True,
        skip_existing=False,
        device="cuda",
        dtype="bfloat16",
    )
    gather_offline_pairs(cfg)
    generate_lr_pairs(cfg.out_dir, n=1, skip_existing=False)
    sample = cfg.out_dir / "pairs" / "000000"
    from PIL import Image

    x0 = Image.open(sample / "x0.png")
    lr = Image.open(sample / "lr.png")
    assert lr.size == (x0.size[0] // 4, x0.size[1] // 4)


@pytest.mark.gpu
def test_i03_restart_with_start_index(tmp_path):
    cfg = GatherConfig(
        model_id=MODEL_ID,
        out_dir=tmp_path / "run",
        n=4,
        hr_size=256,
        num_inference_steps=1,
        save_x0_png=False,
        skip_existing=True,
        device="cuda",
        dtype="bfloat16",
    )
    gather_offline_pairs(cfg)
    sample2 = cfg.out_dir / "pairs" / "000002"
    for p in [sample2 / "eps.pt", sample2 / "z0.pt"]:
        if p.exists():
            p.unlink()
    gather_offline_pairs(cfg)
    assert (sample2 / "eps.pt").exists()
    assert (sample2 / "z0.pt").exists()


@pytest.mark.gpu
def test_i04_gather_without_x0(tmp_path):
    cfg = GatherConfig(
        model_id=MODEL_ID,
        out_dir=tmp_path / "run",
        n=1,
        hr_size=256,
        num_inference_steps=1,
        save_x0_png=False,
        skip_existing=False,
        device="cuda",
        dtype="bfloat16",
    )
    gather_offline_pairs(cfg)
    sample = cfg.out_dir / "pairs" / "000000"
    assert (sample / "eps.pt").exists()
    assert (sample / "z0.pt").exists()
    assert not (sample / "x0.png").exists()


@pytest.mark.gpu
def test_i05_latent_path_outputs(tmp_path):
    cfg = GatherConfig(
        model_id=MODEL_ID,
        out_dir=tmp_path / "run",
        n=1,
        hr_size=256,
        num_inference_steps=1,
        save_x0_png=True,
        skip_existing=False,
        device="cuda",
        dtype="bfloat16",
    )
    gather_offline_pairs(cfg)
    sample = cfg.out_dir / "pairs" / "000000"
    z0 = torch.load(sample / "z0.pt", map_location="cpu", weights_only=True)
    assert z0.ndim == 4
    assert z0.shape[1] == 4


@pytest.mark.gpu
def test_i06_cli_inspect_after_gather(tmp_path):
    cfg = GatherConfig(
        model_id=MODEL_ID,
        out_dir=tmp_path / "run",
        n=1,
        hr_size=256,
        num_inference_steps=1,
        save_x0_png=False,
        skip_existing=False,
        device="cuda",
        dtype="bfloat16",
    )
    gather_offline_pairs(cfg)
    data = inspect_local_pairs(cfg.out_dir, limit=1)
    rendered = json.dumps(data)
    assert '"samples"' in rendered
