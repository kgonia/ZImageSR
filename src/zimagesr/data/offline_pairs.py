from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import inspect
import json
import time
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class GatherConfig:
    model_id: str = "Tongyi-MAI/Z-Image-Turbo"
    out_dir: Path = Path("./zimage_offline_pairs")
    n: int = 1200
    hr_size: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 0.0
    base_seed: int = 12345
    start_index: int = 0
    prompt: str = ""
    negative_prompt: str = ""
    save_x0_png: bool = True
    generate_lr: bool = False
    skip_existing: bool = True
    cache_null_prompt: bool = True
    offload_text_encoder: bool = True
    device: str | None = None
    dtype: str | None = None
    debug: bool = True
    debug_every: int = 1


def _to_str_dtype(x: Any) -> str:
    if x is None:
        return "none"
    return str(x).replace("torch.", "")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def tensor_info(t: torch.Tensor | None, include_stats: bool = True) -> dict[str, Any] | None:
    if t is None:
        return None
    info: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": _to_str_dtype(t.dtype),
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad),
    }
    if include_stats and t.numel() > 0:
        x = t.detach().float()
        info["mean"] = float(x.mean().item())
        info["std"] = float(x.std(unbiased=False).item())
        info["min"] = float(x.min().item())
        info["max"] = float(x.max().item())
    return info


def module_info(m: Any) -> dict[str, Any] | None:
    if m is None:
        return None
    total = 0
    trainable = 0
    first_param = None
    for p in m.parameters():
        if first_param is None:
            first_param = p
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "class": m.__class__.__name__,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "device": str(first_param.device) if first_param is not None else "none",
        "dtype": _to_str_dtype(first_param.dtype) if first_param is not None else "none",
        "training": bool(getattr(m, "training", False)),
    }


def resolve_device_dtype(device: str | None, dtype: str | None) -> tuple[str, torch.dtype, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    if dtype is None:
        dtype = "bfloat16" if device == "cuda" else "float32"
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {sorted(DTYPE_MAP)}.")

    torch_dtype = DTYPE_MAP[dtype]
    if device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print(f"CPU may not support dtype '{dtype}'. Falling back to float32.")
        torch_dtype = torch.float32
        dtype = "float32"
    return device, torch_dtype, dtype


def load_pipeline(model_id: str, device: str, torch_dtype: torch.dtype):
    try:
        from diffusers import ZImageImg2ImgPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers is required to run the data gathering pipeline. "
            "Install project dependencies and try again."
        ) from exc

    pipe = ZImageImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    return pipe


def inspect_pipe_support(pipe) -> dict[str, bool]:
    sig = inspect.signature(pipe.__call__)
    return {
        "prompt_embeds": "prompt_embeds" in sig.parameters,
        "negative_prompt_embeds": "negative_prompt_embeds" in sig.parameters,
        "latents": "latents" in sig.parameters,
        "negative_prompt": "negative_prompt" in sig.parameters,
    }


def _resolve_execution_device(pipe, fallback: str) -> str:
    """Best-effort resolution of diffusers runtime device for intermediates."""
    try:
        dev = getattr(pipe, "_execution_device")
        if dev is None:
            return fallback
        return str(torch.device(dev))
    except Exception:
        return fallback


def _devices_compatible(requested: str, actual: str) -> bool:
    req = torch.device(requested)
    act = torch.device(actual)
    if req.type != act.type:
        return False
    if req.index is not None and act.index is not None and req.index != act.index:
        return False
    return True


def unwrap_first_tensor(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        for v in x:
            if torch.is_tensor(v):
                return v
        return x[0]
    return x


def try_encode_prompt(
    pipe,
    prompt: str,
    negative_prompt: str | None = None,
    device: str | None = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
):
    if not hasattr(pipe, "encode_prompt"):
        raise RuntimeError("Pipeline has no encode_prompt method.")
    enc_sig = inspect.signature(pipe.encode_prompt)
    kwargs: dict[str, Any] = {}
    if "device" in enc_sig.parameters:
        kwargs["device"] = device
    if "num_images_per_prompt" in enc_sig.parameters:
        kwargs["num_images_per_prompt"] = num_images_per_prompt
    if "do_classifier_free_guidance" in enc_sig.parameters:
        kwargs["do_classifier_free_guidance"] = do_classifier_free_guidance
    if "negative_prompt" in enc_sig.parameters:
        kwargs["negative_prompt"] = negative_prompt
    return pipe.encode_prompt(prompt, **kwargs)


def load_or_encode_null_prompt_embeds(
    pipe,
    out_dir: Path,
    device: str,
    prompt: str,
    negative_prompt: str,
) -> tuple[Any, Any]:
    null_emb_path = out_dir / "null_prompt_embeds.pt"
    if null_emb_path.exists():
        data = torch.load(null_emb_path, map_location="cpu", weights_only=True)
        return data.get("prompt_embeds", None), data.get("negative_prompt_embeds", None)

    prev_dev = None
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        prev_dev = next(pipe.text_encoder.parameters()).device
        pipe.text_encoder.to(device)
    try:
        with torch.no_grad():
            enc = try_encode_prompt(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
    finally:
        if prev_dev is not None:
            pipe.text_encoder.to(prev_dev)

    if isinstance(enc, (tuple, list)):
        prompt_embeds = enc[0]
        negative_prompt_embeds = enc[1] if len(enc) > 1 else None
    else:
        prompt_embeds = enc
        negative_prompt_embeds = None

    pe = unwrap_first_tensor(prompt_embeds)
    ne = unwrap_first_tensor(negative_prompt_embeds)
    torch.save(
        {
            "prompt_embeds": pe.detach().cpu() if torch.is_tensor(pe) else pe,
            "negative_prompt_embeds": ne.detach().cpu() if torch.is_tensor(ne) else None,
        },
        null_emb_path,
    )
    return pe, ne


def normalize_prompt_embeds(pe: Any, device: str, dtype: torch.dtype) -> list[torch.Tensor] | None:
    if pe is None:
        return None
    if torch.is_tensor(pe):
        if pe.ndim == 1:
            return None
        return [pe.to(device=device, dtype=dtype)]
    if isinstance(pe, (list, tuple)):
        tensors = [t for t in pe if torch.is_tensor(t) and t.ndim >= 2]
        if len(tensors) == 0:
            return None
        return [t.to(device=device, dtype=dtype) for t in tensors]
    raise TypeError(f"Unsupported prompt_embeds type: {type(pe)}")


@torch.no_grad()
def vae_encode_latents_safe(
    pipe,
    pil_img: Image.Image,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    img = np.array(pil_img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    img = img * 2.0 - 1.0
    img = img.to(device=device, dtype=dtype)
    latents = pipe.vae.encode(img).latent_dist.sample()
    scaling = getattr(pipe.vae.config, "scaling_factor", 1.0)
    return latents * scaling


@torch.no_grad()
def decode_latents_to_pil(pipe, latents: torch.Tensor) -> Image.Image:
    """Decode VAE latents to a PIL image. Caller must ensure VAE is on the correct device."""
    scaling = getattr(pipe.vae.config, "scaling_factor", 1.0)
    decoded = pipe.vae.decode(latents / scaling).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded[0].cpu().permute(1, 2, 0).float().numpy()
    return Image.fromarray((decoded * 255).round().astype(np.uint8))


def infer_latent_shape(pipe, hr_size: int, device: str, dtype: torch.dtype) -> torch.Size:
    dummy = Image.fromarray(np.zeros((hr_size, hr_size, 3), dtype=np.uint8))
    z = vae_encode_latents_safe(pipe, dummy, device=device, dtype=dtype)
    return z.shape


def write_metadata(
    out_dir: Path,
    config: GatherConfig,
    device: str,
    dtype_name: str,
    latent_shape: Iterable[int],
    support: dict[str, bool],
    pipe,
):
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "dtype": dtype_name,
        "latent_shape": list(latent_shape),
        "support": support,
        "model": {
            "pipeline_class": pipe.__class__.__name__,
            "transformer": module_info(getattr(pipe, "transformer", None)),
            "unet": module_info(getattr(pipe, "unet", None)),
            "vae": module_info(getattr(pipe, "vae", None)),
            "text_encoder": module_info(getattr(pipe, "text_encoder", None)),
        },
        "config": {
            **asdict(config),
            "out_dir": str(config.out_dir),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def ensure_pairs_dir(out_dir: Path) -> Path:
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    return pairs_dir


def should_skip_sample(sample_dir: Path, save_x0_png: bool) -> bool:
    eps_path = sample_dir / "eps.pt"
    z0_path = sample_dir / "z0.pt"
    x0_path = sample_dir / "x0.png"
    if not eps_path.exists() or not z0_path.exists():
        return False
    if save_x0_png and not x0_path.exists():
        return False
    return True


def _print_stage(step: int, total: int, message: str) -> None:
    print(f"[{step}/{total}] {message}")


def gather_offline_pairs(config: GatherConfig) -> None:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = ensure_pairs_dir(out_dir)
    debug_path = out_dir / "debug_trace.jsonl"
    if config.debug and debug_path.exists() and config.start_index == 0:
        debug_path.unlink()

    total_steps = 7
    _print_stage(1, total_steps, "Resolving device/dtype and loading pipeline")
    device, torch_dtype, dtype_name = resolve_device_dtype(config.device, config.dtype)
    pipe = load_pipeline(config.model_id, device=device, torch_dtype=torch_dtype)
    support = inspect_pipe_support(pipe)

    if not support["latents"]:
        raise RuntimeError(
            "Pipeline does not accept latents=. "
            "Cannot bind sampled eps to generated images."
        )

    _print_stage(2, total_steps, "Preparing null prompt embeddings")
    null_prompt_embeds = None
    null_negative_prompt_embeds = None
    if support["prompt_embeds"] and config.cache_null_prompt:
        null_prompt_embeds, null_negative_prompt_embeds = load_or_encode_null_prompt_embeds(
            pipe,
            out_dir=out_dir,
            device=device,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
        )

    _print_stage(3, total_steps, "Inferring latent shape and writing metadata")
    latent_shape = infer_latent_shape(pipe, config.hr_size, device=device, dtype=torch_dtype)
    write_metadata(
        out_dir,
        config,
        device,
        dtype_name,
        latent_shape,
        support,
        pipe,
    )

    _print_stage(4, total_steps, "Placing model modules on target device")
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        pipe.transformer.to(device=device, dtype=torch_dtype)
    if hasattr(pipe, "unet") and pipe.unet is not None:
        pipe.unet.to(device=device, dtype=torch_dtype)
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to(device=device, dtype=torch_dtype)

    probe_prompt_embeds = None
    if support["prompt_embeds"] and null_prompt_embeds is not None:
        probe_prompt_embeds = normalize_prompt_embeds(null_prompt_embeds, device=device, dtype=torch_dtype)

    if config.offload_text_encoder and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        if probe_prompt_embeds is not None:
            # Diffusers resolves execution device from module order; dropping text_encoder
            # avoids CPU timesteps while keeping prompt-embed mode active.
            pipe.text_encoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(
                "Skipping text-encoder offload because prompt embeds are unavailable; "
                "offloading would force CPU execution in this pipeline."
            )

    runtime_device = _resolve_execution_device(pipe, fallback=device)
    if not _devices_compatible(device, runtime_device):
        raise RuntimeError(
            f"Pipeline execution device mismatch: requested '{device}', resolved '{runtime_device}'. "
            "Disable --offload-text-encoder or enable --cache-null-prompt."
        )

    cached_prompt_embeds = None
    cached_negative_prompt_embeds = None
    if support["prompt_embeds"] and null_prompt_embeds is not None:
        cached_prompt_embeds = normalize_prompt_embeds(null_prompt_embeds, device=runtime_device, dtype=torch_dtype)
        if support["negative_prompt_embeds"] and null_negative_prompt_embeds is not None:
            cached_negative_prompt_embeds = normalize_prompt_embeds(
                null_negative_prompt_embeds,
                device=runtime_device,
                dtype=torch_dtype,
            )

    call_sig = inspect.signature(pipe.__call__)
    call_params = set(call_sig.parameters)
    required = {
        "image",
        "strength",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "generator",
        "latents",
        "output_type",
    }
    missing = required.difference(call_params)
    if missing:
        raise RuntimeError(f"Pipeline call signature missing expected params: {sorted(missing)}")

    _print_stage(5, total_steps, "Generating (eps, z0, x0) samples")
    loop = range(config.start_index, config.n)
    for i in tqdm(loop, desc="Sampling HR pairs"):
        sample_dir = pairs_dir / f"{i:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        if config.skip_existing and should_skip_sample(sample_dir, config.save_x0_png):
            continue

        eps_gen = torch.Generator(device=runtime_device).manual_seed(config.base_seed + i)
        eps = torch.randn(latent_shape, device=runtime_device, dtype=torch_dtype, generator=eps_gen)
        pipe_gen = torch.Generator(device=runtime_device).manual_seed(config.base_seed + config.n + i)
        init = torch.zeros((1, 3, config.hr_size, config.hr_size), device=runtime_device, dtype=torch_dtype)

        call_kwargs: dict[str, Any] = {
            "image": init,
            "strength": 1.0,
            "height": config.hr_size,
            "width": config.hr_size,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "generator": pipe_gen,
            "latents": eps,
            "output_type": "latent",
        }
        pe_list: list[torch.Tensor] | None = cached_prompt_embeds
        ne_list: list[torch.Tensor] | None = cached_negative_prompt_embeds

        if pe_list is not None:
            call_kwargs["prompt_embeds"] = pe_list
            if ne_list is not None:
                call_kwargs["negative_prompt_embeds"] = ne_list
            call_kwargs["prompt"] = None
        else:
            call_kwargs["prompt"] = config.prompt
            if support["negative_prompt"]:
                call_kwargs["negative_prompt"] = config.negative_prompt

        with torch.no_grad():
            out = pipe(**call_kwargs)
            z0 = out.images
            if isinstance(z0, list):
                z0 = z0[0]

        torch.save(eps.detach().cpu(), sample_dir / "eps.pt")
        torch.save(z0.detach().cpu(), sample_dir / "z0.pt")
        x0 = None
        if config.save_x0_png:
            x0 = decode_latents_to_pil(pipe, z0)
            x0.save(sample_dir / "x0.png")

        if config.debug and (config.debug_every <= 1 or (i % config.debug_every == 0)):
            append_jsonl(
                debug_path,
                {
                    "sample_index": i,
                    "seed": config.base_seed + i,
                    "eps": tensor_info(eps, include_stats=True),
                    "z0": tensor_info(z0, include_stats=True),
                    "init_image": tensor_info(init, include_stats=False),
                    "x0": {
                        "type": "PIL.Image",
                        "mode": x0.mode,
                        "size": [x0.size[0], x0.size[1]],
                    } if x0 is not None else None,
                    "prompt_embeds": [tensor_info(t, include_stats=False) for t in (pe_list or [])],
                    "negative_prompt_embeds": [
                        tensor_info(t, include_stats=False) for t in (ne_list or [])
                    ],
                },
            )

    _print_stage(6, total_steps, "Generating LR placeholders (optional)")
    if config.generate_lr:
        generate_lr_pairs(out_dir, config.n, config.start_index, config.skip_existing)

    _print_stage(7, total_steps, "Done")


def simple_x4_degrade(pil_img: Image.Image) -> tuple[Image.Image, Image.Image]:
    hr = pil_img
    lr_size = (hr.size[0] // 4, hr.size[1] // 4)
    lr = hr.resize(lr_size, resample=Image.BICUBIC)
    lr_up = lr.resize(hr.size, resample=Image.BICUBIC)
    return lr, lr_up


def generate_lr_pairs(
    out_dir: Path,
    n: int,
    start_index: int = 0,
    skip_existing: bool = True,
) -> None:
    pairs_dir = ensure_pairs_dir(out_dir)
    loop = range(start_index, n)
    for i in tqdm(loop, desc="Generating LR pairs"):
        sample_dir = pairs_dir / f"{i:06d}"
        x0_path = sample_dir / "x0.png"
        if not x0_path.exists():
            continue

        lr_path = sample_dir / "lr.png"
        lr_up_path = sample_dir / "lr_up.png"
        if skip_existing and lr_path.exists() and lr_up_path.exists():
            continue

        x0 = Image.open(x0_path).convert("RGB")
        lr, lr_up = simple_x4_degrade(x0)
        lr.save(lr_path)
        lr_up.save(lr_up_path)


def inspect_local_pairs(out_dir: Path, limit: int = 3) -> dict[str, Any]:
    out_dir = Path(out_dir)
    pairs_dir = out_dir / "pairs"
    if not pairs_dir.exists():
        raise FileNotFoundError(f"No pairs directory found: {pairs_dir}")

    metadata_path = out_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    sample_dirs = sorted(p for p in pairs_dir.iterdir() if p.is_dir())[:limit]
    samples = []
    for d in sample_dirs:
        sample: dict[str, Any] = {"id": d.name}
        eps_path = d / "eps.pt"
        z0_path = d / "z0.pt"
        x0_path = d / "x0.png"
        lr_path = d / "lr.png"
        lr_up_path = d / "lr_up.png"

        if eps_path.exists():
            eps = torch.load(eps_path, map_location="cpu", weights_only=True)
            sample["eps"] = tensor_info(eps, include_stats=False)
        if z0_path.exists():
            z0 = torch.load(z0_path, map_location="cpu", weights_only=True)
            sample["z0"] = tensor_info(z0, include_stats=False)
        if x0_path.exists():
            x0 = Image.open(x0_path).convert("RGB")
            sample["x0"] = {"type": "PIL.Image", "mode": x0.mode, "size": [x0.width, x0.height]}
        if lr_path.exists():
            lr = Image.open(lr_path).convert("RGB")
            sample["lr"] = {"type": "PIL.Image", "mode": lr.mode, "size": [lr.width, lr.height]}
        if lr_up_path.exists():
            lr_up = Image.open(lr_up_path).convert("RGB")
            sample["lr_up"] = {
                "type": "PIL.Image",
                "mode": lr_up.mode,
                "size": [lr_up.width, lr_up.height],
            }
        samples.append(sample)

    return {
        "out_dir": str(out_dir),
        "metadata_present": metadata is not None,
        "metadata": metadata,
        "samples": samples,
    }
