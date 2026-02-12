from __future__ import annotations

import io
import math
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image

# Real-ESRGAN default kernel types and probabilities
_KERNEL_LIST = [
    "iso",
    "aniso",
    "generalized_iso",
    "generalized_aniso",
    "plateau_iso",
    "plateau_aniso",
]
_KERNEL_PROB = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]


def _generate_kernel(
    random_mixed_kernels,
    circular_lowpass_kernel,
    *,
    sigma_range: list[float],
    sinc_prob: float,
    kernel_size: int = 21,
) -> np.ndarray:
    """Generate a blur kernel: either sinc or mixed Gaussian/plateau."""
    if np.random.uniform() < sinc_prob:
        ks = random.choice(range(7, 22, 2))
        omega_c = np.random.uniform(np.pi / 3, np.pi) if ks < 13 else np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, ks, pad_to=kernel_size)
    else:
        kernel = random_mixed_kernels(
            _KERNEL_LIST,
            _KERNEL_PROB,
            kernel_size=kernel_size,
            sigma_x_range=sigma_range,
            sigma_y_range=sigma_range,
            rotation_range=[-math.pi, math.pi],
            betag_range=[0.5, 4.0],
            betap_range=[1.0, 2.0],
            noise_range=None,
        )
    return kernel


def _random_resize(img: torch.Tensor, probs: list[float], scale_range: list[float]) -> torch.Tensor:
    """Apply a random resize (up/down/keep) with a random interpolation mode."""
    updown = random.choices(["up", "down", "keep"], probs)[0]
    if updown == "up":
        factor = np.random.uniform(max(1.0, scale_range[0]), scale_range[1])
    elif updown == "down":
        factor = np.random.uniform(scale_range[0], min(1.0, scale_range[1]))
    else:
        factor = 1.0

    mode = random.choice(["area", "bilinear", "bicubic"])
    kwargs: dict = {"scale_factor": factor, "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["antialias"] = True
    return F.interpolate(img, **kwargs)


def _apply_jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    """Apply JPEG compression to a [1,3,H,W] float tensor via PIL round-trip."""
    from PIL import Image as PILImage

    arr = (img[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    arr = np.array(PILImage.open(buf).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def _generate_sinc_kernel(circular_lowpass_kernel, pad_to: int = 21) -> np.ndarray:
    """Generate a random sinc (circular lowpass) filter kernel."""
    ks = random.choice(range(7, 22, 2))
    omega_c = np.random.uniform(np.pi / 3, np.pi) if ks < 13 else np.random.uniform(np.pi / 5, np.pi)
    return circular_lowpass_kernel(omega_c, ks, pad_to=pad_to)


def realesrgan_degrade(
    pil_img: Image.Image,
    scale: int = 4,
    seed: int | None = None,
) -> tuple[Image.Image, Image.Image]:
    """RealESRGAN second-order degradation pipeline.

    Applies two rounds of {blur, resize, noise, JPEG} degradation following
    the Real-ESRGAN paper defaults, then resizes to the target LR resolution
    and bicubic-upscales back to HR size.

    Returns (lr, lr_up) -- same interface as simple_x4_degrade.

    Requires ``basicsr`` and ``opencv-python``:
        uv pip install -e '.[degradation]'
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    try:
        # basicsr 1.4.2 imports from torchvision.transforms.functional_tensor
        # which was removed in torchvision >=0.17. Shim it before importing.
        import sys
        import types

        if "torchvision.transforms.functional_tensor" not in sys.modules:
            from torchvision.transforms.functional import rgb_to_grayscale

            _shim = types.ModuleType("torchvision.transforms.functional_tensor")
            _shim.rgb_to_grayscale = rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = _shim

        from basicsr.data.degradations import (
            circular_lowpass_kernel,
            random_add_gaussian_noise_pt,
            random_add_poisson_noise_pt,
            random_mixed_kernels,
        )
        from basicsr.utils.img_process_util import filter2D
    except ImportError as exc:
        raise RuntimeError(
            "basicsr is required for RealESRGAN degradation. "
            "Install with: uv pip install -e '.[degradation]'"
        ) from exc

    from PIL import Image as PILImage

    # Seed all RNGs for reproducibility (save/restore to avoid side effects)
    cuda_states = None
    if seed is not None:
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    try:
        return _run_pipeline(
            pil_img,
            scale,
            random_mixed_kernels=random_mixed_kernels,
            circular_lowpass_kernel=circular_lowpass_kernel,
            random_add_gaussian_noise_pt=random_add_gaussian_noise_pt,
            random_add_poisson_noise_pt=random_add_poisson_noise_pt,
            filter2D=filter2D,
            PILImage=PILImage,
        )
    finally:
        if seed is not None:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def _run_pipeline(
    pil_img,
    scale,
    *,
    random_mixed_kernels,
    circular_lowpass_kernel,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    filter2D,
    PILImage,
) -> tuple:
    hr_w, hr_h = pil_img.size

    # basicsr's 21x21 reflect-padding blur kernels cannot run on tiny inputs.
    # Fall back to simple bicubic degradation for these edge cases.
    if min(hr_w, hr_h) < 21:
        lr_w, lr_h = max(1, hr_w // scale), max(1, hr_h // scale)
        lr = pil_img.resize((lr_w, lr_h), resample=PILImage.BICUBIC)
        lr_up = lr.resize((hr_w, hr_h), resample=PILImage.BICUBIC)
        return lr, lr_up

    # PIL → tensor [1, 3, H, W] float32 [0, 1]
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()

    # ── Stage 1 ──────────────────────────────────────────────────────────
    # Blur
    kernel1 = _generate_kernel(
        random_mixed_kernels, circular_lowpass_kernel,
        sigma_range=[0.2, 3.0], sinc_prob=0.1,
    )
    img = filter2D(img, torch.FloatTensor(kernel1).unsqueeze(0))

    # Resize
    img = _random_resize(img, probs=[0.2, 0.7, 0.1], scale_range=[0.15, 1.5])

    # Noise
    gray_noise_prob = 0.4
    if np.random.uniform() < 0.5:
        img = random_add_gaussian_noise_pt(
            img, sigma_range=[1, 30], clip=True, rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        img = random_add_poisson_noise_pt(
            img, scale_range=[0.05, 3.0], clip=True, rounds=False,
            gray_prob=gray_noise_prob,
        )

    # JPEG
    img = _apply_jpeg(img, quality=int(np.random.uniform(30, 95)))

    # ── Stage 2 ──────────────────────────────────────────────────────────
    # Blur (applied with probability 0.8)
    if np.random.uniform() < 0.8:
        kernel2 = _generate_kernel(
            random_mixed_kernels, circular_lowpass_kernel,
            sigma_range=[0.2, 1.5], sinc_prob=0.1,
        )
        img = filter2D(img, torch.FloatTensor(kernel2).unsqueeze(0))

    # Resize
    img = _random_resize(img, probs=[0.3, 0.4, 0.3], scale_range=[0.3, 1.2])

    # Noise
    if np.random.uniform() < 0.5:
        img = random_add_gaussian_noise_pt(
            img, sigma_range=[1, 25], clip=True, rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        img = random_add_poisson_noise_pt(
            img, scale_range=[0.05, 2.5], clip=True, rounds=False,
            gray_prob=gray_noise_prob,
        )

    # Final: 50/50 order of JPEG and sinc filter
    final_sinc_prob = 0.8
    if np.random.uniform() < 0.5:
        # JPEG first, then sinc
        img = _apply_jpeg(img, quality=int(np.random.uniform(30, 95)))
        if np.random.uniform() < final_sinc_prob:
            sinc_k = _generate_sinc_kernel(circular_lowpass_kernel)
            img = filter2D(img, torch.FloatTensor(sinc_k).unsqueeze(0))
    else:
        # Sinc first, then JPEG
        if np.random.uniform() < final_sinc_prob:
            sinc_k = _generate_sinc_kernel(circular_lowpass_kernel)
            img = filter2D(img, torch.FloatTensor(sinc_k).unsqueeze(0))
        img = _apply_jpeg(img, quality=int(np.random.uniform(30, 95)))

    # ── Post: resize to LR and upscale back ──────────────────────────────
    lr_h, lr_w = max(1, hr_h // scale), max(1, hr_w // scale)
    img = F.interpolate(img, size=(lr_h, lr_w), mode="bicubic", antialias=True)
    img = img.clamp(0, 1)

    lr_arr = (img[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    lr = PILImage.fromarray(lr_arr)
    lr_up = lr.resize((hr_w, hr_h), resample=PILImage.BICUBIC)
    return lr, lr_up
