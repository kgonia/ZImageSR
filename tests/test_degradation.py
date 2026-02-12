from __future__ import annotations

import importlib.util

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from zimagesr.data.degradation import realesrgan_degrade

HAS_BASICSR = importlib.util.find_spec("basicsr") is not None
requires_basicsr = pytest.mark.skipif(not HAS_BASICSR, reason="basicsr is not installed")


def _make_test_image(size: int = 1024) -> Image.Image:
    arr = np.random.RandomState(0).randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@requires_basicsr
def test_output_sizes_default_scale():
    img = _make_test_image(1024)
    lr, lr_up = realesrgan_degrade(img, scale=4, seed=42)
    assert lr.mode == "RGB"
    assert lr_up.mode == "RGB"
    assert lr.size == (256, 256)
    assert lr_up.size == (1024, 1024)


@requires_basicsr
def test_output_sizes_scale_2():
    img = _make_test_image(512)
    lr, lr_up = realesrgan_degrade(img, scale=2, seed=99)
    assert lr.size == (256, 256)
    assert lr_up.size == (512, 512)


@requires_basicsr
def test_seed_reproducibility():
    img = _make_test_image(256)
    lr1, lr_up1 = realesrgan_degrade(img, scale=4, seed=42)
    lr2, lr_up2 = realesrgan_degrade(img, scale=4, seed=42)
    assert np.array_equal(np.array(lr1), np.array(lr2))
    assert np.array_equal(np.array(lr_up1), np.array(lr_up2))


@requires_basicsr
def test_different_seeds_differ():
    img = _make_test_image(256)
    lr1, _ = realesrgan_degrade(img, scale=4, seed=1)
    lr2, _ = realesrgan_degrade(img, scale=4, seed=2)
    assert not np.array_equal(np.array(lr1), np.array(lr2))


@requires_basicsr
def test_seed_does_not_corrupt_external_rng():
    import random
    import torch

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    py_before = random.random()
    np_before = np.random.random()
    torch_before = torch.rand(1).item()

    # Reset and run degradation in the middle
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    img = _make_test_image(128)
    realesrgan_degrade(img, scale=4, seed=42)

    py_after = random.random()
    np_after = np.random.random()
    torch_after = torch.rand(1).item()

    assert py_before == py_after
    assert np_before == np_after
    assert torch_before == torch_after


@requires_basicsr
def test_no_seed_runs_without_error():
    img = _make_test_image(128)
    lr, lr_up = realesrgan_degrade(img, scale=4, seed=None)
    assert lr.size == (32, 32)
    assert lr_up.size == (128, 128)


@requires_basicsr
def test_small_input_clamped_to_one_pixel():
    img = _make_test_image(3)
    lr, lr_up = realesrgan_degrade(img, scale=4, seed=7)
    assert lr.size == (1, 1)
    assert lr_up.size == (3, 3)


def test_invalid_scale_raises():
    img = _make_test_image(64)
    with pytest.raises(ValueError, match="scale must be >= 1"):
        realesrgan_degrade(img, scale=0, seed=0)


def test_basicsr_import_guard(monkeypatch):
    """If basicsr is not importable, a clear RuntimeError is raised."""
    import builtins

    original_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name.startswith("basicsr"):
            raise ImportError("mocked basicsr unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    img = _make_test_image(64)
    with pytest.raises(RuntimeError, match="basicsr is required"):
        realesrgan_degrade(img, scale=4, seed=0)
