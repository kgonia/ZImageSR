from __future__ import annotations

import inspect
import json
import pickle
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from zimagesr.data import offline_pairs as op


def _has_weights_only_load() -> bool:
    return "weights_only" in inspect.signature(torch.load).parameters


def test_u01_to_str_dtype():
    assert op._to_str_dtype(torch.float32) == "float32"
    assert op._to_str_dtype(torch.bfloat16) == "bfloat16"
    assert op._to_str_dtype(None) == "none"


def test_u02_append_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    op.append_jsonl(path, {"a": 1})
    op.append_jsonl(path, {"b": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}


def test_u03_tensor_info_basic():
    t = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float32)
    info = op.tensor_info(t, include_stats=True)
    assert info is not None
    assert info["shape"] == [2, 2]
    assert info["dtype"] == "float32"
    assert "cpu" in info["device"]
    assert info["mean"] == pytest.approx(4.0)
    assert info["min"] == pytest.approx(1.0)
    assert info["max"] == pytest.approx(7.0)

    info_no_stats = op.tensor_info(t, include_stats=False)
    assert "mean" not in info_no_stats
    assert "std" not in info_no_stats
    assert op.tensor_info(None) is None


def test_u04_tensor_info_empty_tensor_no_crash():
    t = torch.empty((0,), dtype=torch.float32)
    info = op.tensor_info(t, include_stats=True)
    assert info is not None
    assert info["shape"] == [0]
    assert "mean" not in info


def test_u05_module_info():
    module = torch.nn.Linear(2, 3, bias=True)
    info = op.module_info(module)
    assert info is not None
    assert info["class"] == "Linear"
    assert info["total_params"] == 9
    assert info["trainable_params"] == 9
    assert op.module_info(None) is None


def test_u06_resolve_device_dtype(monkeypatch):
    monkeypatch.setattr(op.torch.cuda, "is_available", lambda: False)
    device, dtype, dtype_name = op.resolve_device_dtype(None, None)
    assert device == "cpu"
    assert dtype == torch.float32
    assert dtype_name == "float32"

    device2, dtype2, dtype_name2 = op.resolve_device_dtype("cuda", None)
    assert device2 == "cpu"
    assert dtype2 == torch.float32
    assert dtype_name2 == "float32"

    with pytest.raises(ValueError):
        op.resolve_device_dtype(None, "bad-dtype")


def test_u07_unwrap_first_tensor():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    assert op.unwrap_first_tensor([a, b]) is a
    assert op.unwrap_first_tensor(["x", b, "y"]) is b
    assert op.unwrap_first_tensor([]) is None
    assert op.unwrap_first_tensor(None) is None
    assert op.unwrap_first_tensor(a) is a


def test_u08_normalize_prompt_embeds():
    assert op.normalize_prompt_embeds(None, device="cpu", dtype=torch.float32) is None

    one_d = torch.ones((8,), dtype=torch.float32)
    assert op.normalize_prompt_embeds(one_d, device="cpu", dtype=torch.float32) is None

    two_d = torch.ones((2, 8), dtype=torch.float32)
    out = op.normalize_prompt_embeds(two_d, device="cpu", dtype=torch.float32)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == two_d.shape

    mixed = [torch.ones((8,)), torch.ones((2, 8)), torch.ones((1, 2, 8))]
    out2 = op.normalize_prompt_embeds(mixed, device="cpu", dtype=torch.float32)
    assert len(out2) == 2
    assert out2[0].ndim >= 2
    assert out2[1].ndim >= 2

    with pytest.raises(TypeError):
        op.normalize_prompt_embeds({"bad": "type"}, device="cpu", dtype=torch.float32)


def test_u09_simple_x4_degrade():
    img = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8), mode="RGB")
    lr, lr_up = op.simple_x4_degrade(img)
    assert lr.mode == "RGB"
    assert lr_up.mode == "RGB"
    assert lr.size == (256, 256)
    assert lr_up.size == (1024, 1024)


def test_u10_should_skip_sample(tmp_path):
    sample = tmp_path / "000000"
    sample.mkdir(parents=True)
    (sample / "eps.pt").write_bytes(b"x")
    (sample / "z0.pt").write_bytes(b"y")
    (sample / "x0.png").write_bytes(b"z")

    assert op.should_skip_sample(sample, save_x0_png=True)
    assert op.should_skip_sample(sample, save_x0_png=False)

    (sample / "x0.png").unlink()
    assert not op.should_skip_sample(sample, save_x0_png=True)
    assert op.should_skip_sample(sample, save_x0_png=False)

    (sample / "z0.pt").unlink()
    assert not op.should_skip_sample(sample, save_x0_png=False)


def test_u11_ensure_pairs_dir(tmp_path):
    out_dir = tmp_path / "data"
    pairs = op.ensure_pairs_dir(out_dir)
    assert pairs == out_dir / "pairs"
    assert pairs.exists()
    assert op.ensure_pairs_dir(out_dir) == pairs


def test_u12_vae_encode_latents_safe_no_vae_to_call():
    class FakeVAE:
        def __init__(self):
            self.config = SimpleNamespace(scaling_factor=3.0)
            self.to_called = False

        def to(self, *_args, **_kwargs):
            self.to_called = True
            return self

        def encode(self, img):
            b, _c, h, w = img.shape
            z = torch.ones((b, 4, h // 8, w // 8), dtype=img.dtype, device=img.device)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: z))

    pipe = SimpleNamespace(vae=FakeVAE())
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), mode="RGB")
    z = op.vae_encode_latents_safe(pipe, img, device="cpu", dtype=torch.float32)
    assert z.shape == (1, 4, 32, 32)
    assert torch.allclose(z, torch.full_like(z, 3.0))
    assert pipe.vae.to_called is False


def test_u13_decode_latents_to_pil_divides_by_scaling():
    class FakeVAE:
        def __init__(self):
            self.config = SimpleNamespace(scaling_factor=4.0)
            self.last_decode_input = None

        def decode(self, z):
            self.last_decode_input = z
            out = torch.zeros((1, 3, 16, 16), dtype=z.dtype, device=z.device)
            out[:, 0] = 1.0
            return SimpleNamespace(sample=out)

    pipe = SimpleNamespace(vae=FakeVAE())
    latents = torch.full((1, 4, 2, 2), 8.0)
    img = op.decode_latents_to_pil(pipe, latents)
    assert img.mode == "RGB"
    assert img.size == (16, 16)
    assert torch.allclose(pipe.vae.last_decode_input, torch.full((1, 4, 2, 2), 2.0))


def test_u14_infer_latent_shape():
    class FakeVAE:
        def __init__(self):
            self.config = SimpleNamespace(scaling_factor=1.0)

        def encode(self, img):
            b, _c, h, w = img.shape
            z = torch.zeros((b, 4, h // 8, w // 8), dtype=img.dtype, device=img.device)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: z))

    pipe = SimpleNamespace(vae=FakeVAE())
    shape = op.infer_latent_shape(pipe, hr_size=256, device="cpu", dtype=torch.float32)
    assert shape == torch.Size([1, 4, 32, 32])


def test_u15_inspect_pipe_support():
    class PipeA:
        def __call__(
            self,
            prompt,
            image,
            strength,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            generator,
            latents,
            output_type,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            negative_prompt=None,
        ):
            return None

    class PipeB:
        def __call__(
            self,
            prompt,
            image,
            strength,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            generator,
            output_type,
        ):
            return None

    a = op.inspect_pipe_support(PipeA())
    b = op.inspect_pipe_support(PipeB())
    assert a == {
        "prompt_embeds": True,
        "negative_prompt_embeds": True,
        "latents": True,
        "negative_prompt": True,
    }
    assert b == {
        "prompt_embeds": False,
        "negative_prompt_embeds": False,
        "latents": False,
        "negative_prompt": False,
    }


def test_u16_try_encode_prompt():
    class Pipe:
        def __init__(self):
            self.kwargs = None

        def encode_prompt(
            self,
            prompt,
            device=None,
            num_images_per_prompt=None,
            do_classifier_free_guidance=None,
            negative_prompt=None,
        ):
            self.kwargs = {
                "prompt": prompt,
                "device": device,
                "num_images_per_prompt": num_images_per_prompt,
                "do_classifier_free_guidance": do_classifier_free_guidance,
                "negative_prompt": negative_prompt,
            }
            return "ok"

    p = Pipe()
    out = op.try_encode_prompt(
        p,
        prompt="",
        negative_prompt="neg",
        device="cpu",
        num_images_per_prompt=2,
        do_classifier_free_guidance=True,
    )
    assert out == "ok"
    assert p.kwargs == {
        "prompt": "",
        "device": "cpu",
        "num_images_per_prompt": 2,
        "do_classifier_free_guidance": True,
        "negative_prompt": "neg",
    }

    with pytest.raises(RuntimeError):
        op.try_encode_prompt(object(), prompt="")


def test_u17_load_or_encode_null_prompt_embeds(tmp_path):
    cache = tmp_path / "null_prompt_embeds.pt"
    cached_data = {
        "prompt_embeds": torch.ones((1, 4)),
        "negative_prompt_embeds": torch.zeros((1, 4)),
    }
    torch.save(cached_data, cache)

    pipe = SimpleNamespace()
    pe, ne = op.load_or_encode_null_prompt_embeds(
        pipe,
        out_dir=tmp_path,
        device="cpu",
        prompt="",
        negative_prompt="",
    )
    assert torch.equal(pe, cached_data["prompt_embeds"])
    assert torch.equal(ne, cached_data["negative_prompt_embeds"])

    class TextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return super().to(*args, **kwargs)

    class Pipe2:
        def __init__(self):
            self.text_encoder = TextEncoder()

        def encode_prompt(
            self,
            prompt,
            device=None,
            num_images_per_prompt=None,
            do_classifier_free_guidance=None,
            negative_prompt=None,
        ):
            return torch.ones((1, 4)), torch.zeros((1, 4))

    cache.unlink()
    pipe2 = Pipe2()
    pe2, ne2 = op.load_or_encode_null_prompt_embeds(
        pipe2,
        out_dir=tmp_path,
        device="cpu",
        prompt="",
        negative_prompt="",
    )
    assert cache.exists()
    assert isinstance(pe2, torch.Tensor)
    assert isinstance(ne2, torch.Tensor)
    assert len(pipe2.text_encoder.to_calls) >= 2


@pytest.mark.skipif(not _has_weights_only_load(), reason="torch.load weights_only= not supported")
def test_u18_cached_load_rejects_non_tensor_pickle(tmp_path):
    class Pipe:
        pass

    bad_path = tmp_path / "null_prompt_embeds.pt"
    with bad_path.open("wb") as f:
        pickle.dump({"prompt_embeds": object()}, f)

    with pytest.raises(Exception):
        op.load_or_encode_null_prompt_embeds(
            Pipe(),
            out_dir=tmp_path,
            device="cpu",
            prompt="",
            negative_prompt="",
        )


@pytest.mark.skipif(not _has_weights_only_load(), reason="torch.load weights_only= not supported")
def test_u19_inspect_local_pairs_uses_safe_load(tmp_path):
    out_dir = tmp_path / "dataset"
    sample_dir = out_dir / "pairs" / "000000"
    sample_dir.mkdir(parents=True)
    torch.save(torch.ones((1, 4, 8, 8)), sample_dir / "eps.pt")
    torch.save(torch.zeros((1, 4, 8, 8)), sample_dir / "z0.pt")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(sample_dir / "x0.png")

    inspected = op.inspect_local_pairs(out_dir, limit=1)
    assert inspected["samples"][0]["eps"]["shape"] == [1, 4, 8, 8]

    with (sample_dir / "eps.pt").open("wb") as f:
        pickle.dump({"bad": object()}, f)
    with pytest.raises(Exception):
        op.inspect_local_pairs(out_dir, limit=1)


def test_u20_gather_generator_isolation(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    pipes = [
        mock_pipeline_factory(consume_generator=True),
        mock_pipeline_factory(consume_generator=True),
    ]
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipes.pop(0))
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))

    cfg1 = gather_config_factory(
        out_dir=tmp_path / "run1",
        n=1,
        num_inference_steps=1,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=False,
    )
    cfg2 = gather_config_factory(
        out_dir=tmp_path / "run2",
        n=1,
        num_inference_steps=9,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=False,
    )
    op.gather_offline_pairs(cfg1)
    op.gather_offline_pairs(cfg2)

    eps1 = torch.load(cfg1.out_dir / "pairs" / "000000" / "eps.pt", map_location="cpu")
    eps2 = torch.load(cfg2.out_dir / "pairs" / "000000" / "eps.pt", map_location="cpu")
    assert torch.equal(eps1, eps2)


def test_u21_gather_pipeline_generator_seed(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    pipe = mock_pipeline_factory(consume_generator=True)
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))

    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=3,
        start_index=2,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=False,
    )
    op.gather_offline_pairs(cfg)

    expected_seed = cfg.base_seed + cfg.n + cfg.start_index
    g = torch.Generator(device="cpu").manual_seed(expected_seed)
    expected_probe = torch.randint(0, 2**31 - 1, (1,), generator=g).item()
    assert pipe.gen_probes[0] == expected_probe


def test_u22_gather_uses_output_type_latent(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    pipe = mock_pipeline_factory()
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))

    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=1,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=False,
    )
    op.gather_offline_pairs(cfg)
    assert pipe.calls[0]["output_type"] == "latent"


def test_u23_gather_saves_z0_from_tensor_output(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    pipe = mock_pipeline_factory(return_mode="tensor", latent_offset=5.0)
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))
    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=1,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=False,
        save_x0_png=False,
    )
    op.gather_offline_pairs(cfg)

    eps = torch.load(cfg.out_dir / "pairs" / "000000" / "eps.pt", map_location="cpu")
    z0 = torch.load(cfg.out_dir / "pairs" / "000000" / "z0.pt", map_location="cpu")
    assert torch.allclose(z0, eps + 5.0)


def test_u24_gather_saves_z0_from_list_output(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    pipe = mock_pipeline_factory(return_mode="list", latent_offset=2.0)
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))
    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=1,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=False,
        save_x0_png=False,
    )
    op.gather_offline_pairs(cfg)
    eps = torch.load(cfg.out_dir / "pairs" / "000000" / "eps.pt", map_location="cpu")
    z0 = torch.load(cfg.out_dir / "pairs" / "000000" / "z0.pt", map_location="cpu")
    assert torch.allclose(z0, eps + 2.0)


def test_u25_gather_x0_png_decode_path(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))

    pipe1 = mock_pipeline_factory(return_mode="tensor")
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe1)
    cfg1 = gather_config_factory(
        out_dir=tmp_path / "run1",
        n=1,
        cache_null_prompt=False,
        offload_text_encoder=False,
        save_x0_png=True,
        debug=False,
    )
    op.gather_offline_pairs(cfg1)
    assert (cfg1.out_dir / "pairs" / "000000" / "x0.png").exists()

    pipe2 = mock_pipeline_factory(return_mode="tensor")
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe2)
    cfg2 = gather_config_factory(
        out_dir=tmp_path / "run2",
        n=1,
        cache_null_prompt=False,
        offload_text_encoder=False,
        save_x0_png=False,
        debug=False,
    )
    op.gather_offline_pairs(cfg2)
    assert not (cfg2.out_dir / "pairs" / "000000" / "x0.png").exists()


def test_u26_gather_debug_trace_x0_none_when_not_saved(
    tmp_path,
    gather_config_factory,
    mock_pipeline_factory,
    monkeypatch,
):
    pipe = mock_pipeline_factory()
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))
    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=1,
        save_x0_png=False,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=True,
        debug_every=1,
    )
    op.gather_offline_pairs(cfg)
    trace = (cfg.out_dir / "debug_trace.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(trace[0])
    assert row["x0"] is None


def test_u27_write_metadata(tmp_path, gather_config_factory, mock_pipeline_factory):
    cfg = gather_config_factory(out_dir=tmp_path / "run")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = mock_pipeline_factory()
    op.write_metadata(
        out_dir=cfg.out_dir,
        config=cfg,
        device="cpu",
        dtype_name="float32",
        latent_shape=[1, 4, 8, 8],
        support={"latents": True},
        pipe=pipe,
    )
    data = json.loads((cfg.out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert data["device"] == "cpu"
    assert data["dtype"] == "float32"
    assert data["latent_shape"] == [1, 4, 8, 8]
    assert data["config"]["n"] == cfg.n
    assert data["model"]["pipeline_class"] == pipe.__class__.__name__


def test_u28_gather_debug_every_sampling(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    pipe = mock_pipeline_factory()
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))
    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=11,
        cache_null_prompt=False,
        offload_text_encoder=False,
        debug=True,
        debug_every=5,
    )
    op.gather_offline_pairs(cfg)
    trace_lines = (cfg.out_dir / "debug_trace.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in trace_lines]
    assert [r["sample_index"] for r in rows] == [0, 5, 10]


def test_u29_gather_skip_existing(tmp_path, gather_config_factory, mock_pipeline_factory, monkeypatch):
    out_dir = tmp_path / "run"
    pairs = out_dir / "pairs"
    s0 = pairs / "000000"
    s1 = pairs / "000001"
    s0.mkdir(parents=True)
    s1.mkdir(parents=True)

    eps0 = torch.full((1, 4, 8, 8), 99.0)
    z0_0 = torch.full((1, 4, 8, 8), 77.0)
    torch.save(eps0, s0 / "eps.pt")
    torch.save(z0_0, s0 / "z0.pt")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(s0 / "x0.png")

    torch.save(torch.ones((1, 4, 8, 8)), s1 / "eps.pt")

    pipe = mock_pipeline_factory(return_mode="tensor", latent_offset=3.0)
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))
    cfg = gather_config_factory(
        out_dir=out_dir,
        n=2,
        cache_null_prompt=False,
        offload_text_encoder=False,
        skip_existing=True,
        debug=False,
    )
    op.gather_offline_pairs(cfg)

    eps0_after = torch.load(s0 / "eps.pt", map_location="cpu")
    assert torch.equal(eps0_after, eps0)
    assert (s1 / "z0.pt").exists()
    assert len(pipe.calls) == 1


def test_u30_generate_lr_pairs(tmp_path):
    out_dir = tmp_path / "run"
    s0 = out_dir / "pairs" / "000000"
    s1 = out_dir / "pairs" / "000001"
    s0.mkdir(parents=True)
    s1.mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(s0 / "x0.png")

    op.generate_lr_pairs(out_dir=out_dir, n=2, start_index=0, skip_existing=True)
    lr_path = s0 / "lr.png"
    lr_up_path = s0 / "lr_up.png"
    assert lr_path.exists()
    assert lr_up_path.exists()
    assert not (s1 / "lr.png").exists()

    mtime_before = lr_path.stat().st_mtime_ns
    op.generate_lr_pairs(out_dir=out_dir, n=2, start_index=0, skip_existing=True)
    mtime_after = lr_path.stat().st_mtime_ns
    assert mtime_before == mtime_after

    time.sleep(0.01)
    op.generate_lr_pairs(out_dir=out_dir, n=2, start_index=0, skip_existing=False)
    assert lr_path.stat().st_mtime_ns >= mtime_after


def test_u31_inspect_local_pairs_structure(tmp_path):
    out_dir = tmp_path / "run"
    sample = out_dir / "pairs" / "000000"
    sample.mkdir(parents=True)
    torch.save(torch.ones((1, 4, 8, 8)), sample / "eps.pt")
    torch.save(torch.zeros((1, 4, 8, 8)), sample / "z0.pt")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(sample / "x0.png")
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8), mode="RGB").save(sample / "lr.png")
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB").save(sample / "lr_up.png")
    (out_dir / "metadata.json").write_text(json.dumps({"k": "v"}), encoding="utf-8")

    out = op.inspect_local_pairs(out_dir, limit=1)
    assert out["metadata_present"] is True
    assert out["metadata"] == {"k": "v"}
    assert len(out["samples"]) == 1
    row = out["samples"][0]
    assert row["id"] == "000000"
    assert row["eps"]["shape"] == [1, 4, 8, 8]
    assert row["z0"]["shape"] == [1, 4, 8, 8]
    assert row["x0"]["size"] == [64, 64]
    assert row["lr"]["size"] == [16, 16]
    assert row["lr_up"]["size"] == [64, 64]


def test_u32_gather_prompt_embed_mode_uses_prompt_none_and_detaches_text_encoder(
    tmp_path,
    gather_config_factory,
    mock_pipeline_factory,
    monkeypatch,
):
    pipe = mock_pipeline_factory()
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))

    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=1,
        cache_null_prompt=True,
        offload_text_encoder=True,
        save_x0_png=False,
        debug=False,
    )
    op.gather_offline_pairs(cfg)

    assert pipe.calls[0]["prompt"] is None
    assert pipe.calls[0]["prompt_embeds"] is not None
    assert pipe.text_encoder is None


def test_u33_gather_skips_text_encoder_offload_without_cached_prompt_embeds(
    tmp_path,
    gather_config_factory,
    mock_pipeline_factory,
    monkeypatch,
):
    pipe = mock_pipeline_factory()
    monkeypatch.setattr(op, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(op, "resolve_device_dtype", lambda *_a, **_k: ("cpu", torch.float32, "float32"))

    cfg = gather_config_factory(
        out_dir=tmp_path / "run",
        n=1,
        cache_null_prompt=False,
        offload_text_encoder=True,
        prompt="hello",
        save_x0_png=False,
        debug=False,
    )
    op.gather_offline_pairs(cfg)

    assert pipe.calls[0]["prompt"] == "hello"
    assert pipe.calls[0]["prompt_embeds"] is None
    assert pipe.text_encoder is not None
