from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from zimagesr.training.dataset import FTDPairDataset, ftd_collate, generate_zl_latents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(d: Path, *, with_pixels: bool = False, with_lr_up: bool = False) -> None:
    """Create a minimal pair directory with eps.pt, z0.pt, zL.pt."""
    d.mkdir(parents=True, exist_ok=True)
    C, H, W = 16, 8, 8
    torch.save(torch.randn(1, C, H, W), d / "eps.pt")
    torch.save(torch.randn(1, C, H, W), d / "z0.pt")
    torch.save(torch.randn(1, C, H, W), d / "zL.pt")
    if with_pixels:
        Image = pytest.importorskip("PIL.Image")
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(d / "x0.png")
    if with_lr_up:
        Image = pytest.importorskip("PIL.Image")
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        img.save(d / "lr_up.png")


# ---------------------------------------------------------------------------
# FTDPairDataset
# ---------------------------------------------------------------------------


class TestFTDPairDataset:
    def test_loads_basic_tensors(self, tmp_path):
        _make_pair(tmp_path / "pairs" / "000")
        _make_pair(tmp_path / "pairs" / "001")
        ds = FTDPairDataset(tmp_path / "pairs", load_pixels=False)
        assert len(ds) == 2

        sample = ds[0]
        assert set(sample.keys()) == {"eps", "z0", "zL"}
        assert sample["eps"].shape == (16, 8, 8)  # squeezed from (1,16,8,8)
        assert sample["z0"].shape == (16, 8, 8)
        assert sample["zL"].shape == (16, 8, 8)

    def test_loads_pixels(self, tmp_path):
        _make_pair(tmp_path / "pairs" / "000", with_pixels=True)
        ds = FTDPairDataset(tmp_path / "pairs", load_pixels=True)
        sample = ds[0]
        assert "x0_pixels" in sample
        assert sample["x0_pixels"].shape == (3, 64, 64)
        assert sample["x0_pixels"].dtype == torch.float32
        assert sample["x0_pixels"].max() <= 1.0

    def test_skips_incomplete_dir(self, tmp_path):
        # dir without zL.pt
        d = tmp_path / "pairs" / "bad"
        d.mkdir(parents=True)
        torch.save(torch.randn(1, 16, 8, 8), d / "eps.pt")
        torch.save(torch.randn(1, 16, 8, 8), d / "z0.pt")

        _make_pair(tmp_path / "pairs" / "good")

        ds = FTDPairDataset(tmp_path / "pairs", load_pixels=False)
        assert len(ds) == 1

    def test_skips_missing_x0_when_load_pixels(self, tmp_path):
        _make_pair(tmp_path / "pairs" / "no_x0")  # no x0.png
        _make_pair(tmp_path / "pairs" / "has_x0", with_pixels=True)

        ds = FTDPairDataset(tmp_path / "pairs", load_pixels=True)
        assert len(ds) == 1

    def test_raises_on_empty_dir(self, tmp_path):
        (tmp_path / "pairs").mkdir()
        with pytest.raises(RuntimeError, match="No valid samples"):
            FTDPairDataset(tmp_path / "pairs")

    def test_ignores_files_in_pairs_dir(self, tmp_path):
        (tmp_path / "pairs").mkdir()
        (tmp_path / "pairs" / "readme.txt").write_text("ignored")
        _make_pair(tmp_path / "pairs" / "000")

        ds = FTDPairDataset(tmp_path / "pairs")
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# ftd_collate
# ---------------------------------------------------------------------------


class TestFtdCollate:
    def test_stacks_batch(self):
        batch = [
            {"eps": torch.randn(16, 4, 4), "z0": torch.randn(16, 4, 4)},
            {"eps": torch.randn(16, 4, 4), "z0": torch.randn(16, 4, 4)},
        ]
        result = ftd_collate(batch)
        assert result["eps"].shape == (2, 16, 4, 4)
        assert result["z0"].shape == (2, 16, 4, 4)


# ---------------------------------------------------------------------------
# generate_zl_latents
# ---------------------------------------------------------------------------


class TestGenerateZlLatents:
    def test_creates_zl_files(self, tmp_path):
        pairs = tmp_path / "pairs"
        d = pairs / "000"
        _make_pair(d, with_lr_up=True)
        # Remove the zL.pt so it needs to be created
        (d / "zL.pt").unlink()

        fake_vae = SimpleNamespace(
            config=SimpleNamespace(scaling_factor=0.5),
            encode=lambda img: SimpleNamespace(
                latent_dist=SimpleNamespace(
                    sample=lambda: torch.ones(1, 16, img.shape[2] // 8, img.shape[3] // 8)
                )
            ),
        )
        fake_pipe = SimpleNamespace(vae=fake_vae)

        count = generate_zl_latents(pairs, fake_pipe, "cpu", torch.float32)
        assert count == 1
        assert (d / "zL.pt").exists()

    def test_skip_existing(self, tmp_path):
        pairs = tmp_path / "pairs"
        d = pairs / "000"
        _make_pair(d, with_lr_up=True)
        # zL.pt already exists

        fake_pipe = SimpleNamespace(vae=SimpleNamespace())
        count = generate_zl_latents(pairs, fake_pipe, "cpu", torch.float32, skip_existing=True)
        assert count == 0

    def test_skips_dirs_without_lr_up(self, tmp_path):
        pairs = tmp_path / "pairs"
        d = pairs / "000"
        _make_pair(d)  # no lr_up.png
        (d / "zL.pt").unlink()

        fake_pipe = SimpleNamespace(vae=SimpleNamespace())
        count = generate_zl_latents(pairs, fake_pipe, "cpu", torch.float32)
        assert count == 0
