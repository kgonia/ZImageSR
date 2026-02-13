from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from zimagesr import cli
from zimagesr.data.offline_pairs import GatherConfig


def test_c01_add_gather_args_parsing():
    parser = argparse.ArgumentParser()
    cli._add_gather_args(parser)
    args = parser.parse_args(
        [
            "--model-id",
            "m",
            "--out-dir",
            "/tmp/out",
            "--n",
            "12",
            "--hr-size",
            "128",
            "--num-inference-steps",
            "3",
            "--guidance-scale",
            "0.5",
            "--base-seed",
            "11",
            "--start-index",
            "2",
            "--prompt",
            "p",
            "--negative-prompt",
            "np",
            "--no-save-x0-png",
            "--generate-lr",
            "--no-skip-existing",
            "--no-cache-null-prompt",
            "--no-offload-text-encoder",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--debug",
            "--debug-every",
            "7",
        ]
    )
    assert args.model_id == "m"
    assert str(args.out_dir) == "/tmp/out"
    assert args.n == 12
    assert args.hr_size == 128
    assert args.num_inference_steps == 3
    assert args.guidance_scale == 0.5
    assert args.base_seed == 11
    assert args.start_index == 2
    assert args.prompt == "p"
    assert args.negative_prompt == "np"
    assert args.save_x0_png is False
    assert args.generate_lr is True
    assert args.skip_existing is False
    assert args.cache_null_prompt is False
    assert args.offload_text_encoder is False
    assert args.device == "cpu"
    assert args.dtype == "float32"
    assert args.debug is True
    assert args.debug_every == 7


def test_c02_gather_config_from_args():
    ns = argparse.Namespace(
        model_id="id",
        out_dir=GatherConfig.out_dir,
        n=9,
        hr_size=64,
        num_inference_steps=2,
        guidance_scale=1.0,
        base_seed=7,
        start_index=1,
        prompt="",
        negative_prompt="n",
        save_x0_png=True,
        generate_lr=False,
        skip_existing=True,
        cache_null_prompt=False,
        offload_text_encoder=True,
        device="cpu",
        dtype="float32",
        debug=True,
        debug_every=4,
    )
    cfg = cli._gather_config_from_args(ns)
    assert isinstance(cfg, GatherConfig)
    assert dataclasses.asdict(cfg)["n"] == 9
    assert cfg.dtype == "float32"
    assert cfg.debug_every == 4


def test_c03_build_parser_accepts_gather():
    parser = cli.build_parser()
    args = parser.parse_args(["gather", "--n", "3", "--hr-size", "64", "--debug-every", "2"])
    assert args.command == "gather"
    assert args.n == 3
    assert args.hr_size == 64
    assert args.debug_every == 2


def test_c04_build_parser_accepts_zenml_run():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "zenml-run",
            "--mode",
            "download",
            "--n",
            "3",
            "--inspect-limit",
            "10",
            "--s3-uri",
            "s3://bucket/path",
            "--overwrite",
        ]
    )
    assert args.command == "zenml-run"
    assert args.mode == "download"
    assert args.n == 3
    assert args.inspect_limit == 10
    assert args.s3_uri == "s3://bucket/path"
    assert args.overwrite is True


def test_c05_gather_and_zenml_defaults_match():
    parser = cli.build_parser()
    g = parser.parse_args(["gather"])
    z = parser.parse_args(["zenml-run"])
    gather_fields = [f.name for f in dataclasses.fields(GatherConfig)]
    for name in gather_fields:
        assert getattr(g, name) == getattr(z, name), f"default mismatch for {name}"


def test_c06_main_gather_dispatch(monkeypatch):
    called = {}

    def fake_gather(cfg):
        called["cfg"] = cfg

    monkeypatch.setattr(cli, "gather_offline_pairs", fake_gather)
    monkeypatch.setattr("sys.argv", ["zimagesr-data", "gather", "--n", "4"])
    cli.main()
    assert called["cfg"].n == 4


def test_c07_main_degrade_dispatch(monkeypatch):
    called = {}

    def fake_degrade(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "generate_lr_pairs", fake_degrade)
    monkeypatch.setattr("sys.argv", ["zimagesr-data", "degrade", "--n", "5", "--start-index", "2"])
    cli.main()
    assert called["n"] == 5
    assert called["start_index"] == 2


def test_c08_main_inspect_dispatch_and_print(monkeypatch, capsys):
    monkeypatch.setattr(cli, "inspect_local_pairs", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr("sys.argv", ["zimagesr-data", "inspect"])
    cli.main()
    out = capsys.readouterr().out
    assert json.loads(out) == {"ok": True}


def test_c09_main_s3_upload_dispatch(monkeypatch, capsys):
    class Result:
        def to_dict(self):
            return {"uploaded_files": 3}

    monkeypatch.setattr(cli, "upload_dir_to_s3", lambda **_kwargs: Result())
    monkeypatch.setattr("sys.argv", ["zimagesr-data", "s3-upload", "--s3-uri", "s3://bucket/path"])
    cli.main()
    out = capsys.readouterr().out
    assert json.loads(out) == {"uploaded_files": 3}


def test_c10_main_s3_download_dispatch(monkeypatch, capsys):
    class Result:
        def to_dict(self):
            return {"downloaded_files": 2}

    monkeypatch.setattr(cli, "download_dir_from_s3", lambda **_kwargs: Result())
    monkeypatch.setattr("sys.argv", ["zimagesr-data", "s3-download", "--s3-uri", "s3://bucket/path"])
    cli.main()
    out = capsys.readouterr().out
    assert json.loads(out) == {"downloaded_files": 2}


def test_c11_parser_still_works_without_training_module(monkeypatch):
    monkeypatch.setattr(cli, "_TrainConfig", None)
    monkeypatch.setattr(cli, "_TRAIN_CONFIG_IMPORT_ERROR", ModuleNotFoundError("zimagesr.training"))

    parser = cli.build_parser()
    gather_args = parser.parse_args(["gather", "--n", "1"])
    assert gather_args.command == "gather"
    train_args = parser.parse_args(["train", "--pairs-dir", "/tmp/pairs"])
    assert train_args.command == "train"

    with pytest.raises(RuntimeError, match="Training config module is unavailable"):
        cli._train_config_from_args(train_args)


def test_c12_build_parser_accepts_infer_pair_dir():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--lora-path",
            "/tmp/lora",
            "--pair-dir",
            "/tmp/pairs/000001",
        ]
    )
    assert args.command == "infer"
    assert args.lora_path.as_posix() == "/tmp/lora"
    assert args.pair_dir.as_posix() == "/tmp/pairs/000001"
    assert args.input_image is None
    assert args.sr_scale == 1.0


def test_c13_build_parser_accepts_infer_input_image():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--lora-path",
            "/tmp/lora",
            "--input-image",
            "/tmp/in.png",
            "--input-upscale",
            "1.0",
            "--fit-multiple",
            "8",
            "--sr-scale",
            "0.85",
            "--output",
            "/tmp/out.png",
        ]
    )
    assert args.command == "infer"
    assert args.input_image.as_posix() == "/tmp/in.png"
    assert args.pair_dir is None
    assert args.input_upscale == 1.0
    assert args.fit_multiple == 8
    assert args.sr_scale == 0.85
    assert args.output.as_posix() == "/tmp/out.png"


def test_c14_infer_requires_exactly_one_source():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["infer", "--lora-path", "/tmp/lora"])


def test_c15_prepare_input_image_resizes_to_multiple(tmp_path):
    from PIL import Image
    import numpy as np

    p = tmp_path / "in.png"
    Image.fromarray(np.zeros((101, 203, 3), dtype=np.uint8), mode="RGB").save(p)

    img, meta = cli._prepare_input_image(p, upscale=1.0, fit_multiple=16)

    assert img.size == (208, 112)
    assert meta["input_original_size"] == [203, 101]
    assert meta["input_resized_by_upscale"] is False
    assert meta["input_resized_by_multiple"] is True


def test_c16_infer_default_output_paths():
    pair_out = cli._infer_default_output(Path("/tmp/pairs/000001"), None)
    assert pair_out.as_posix() == "/tmp/pairs/000001/sr.png"

    img_out = cli._infer_default_output(None, Path("/tmp/in.png"))
    assert img_out.as_posix() == "/tmp/in_sr.png"
