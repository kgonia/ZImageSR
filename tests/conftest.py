from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def torch_mod():
    return pytest.importorskip("torch")


@pytest.fixture
def gather_config_factory(tmp_path):
    def _factory(**overrides):
        from zimagesr.data.offline_pairs import GatherConfig

        defaults = {
            "model_id": "dummy/model",
            "out_dir": tmp_path / "out",
            "n": 2,
            "hr_size": 64,
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
            "base_seed": 123,
            "start_index": 0,
            "prompt": "",
            "negative_prompt": "",
            "save_x0_png": True,
            "generate_lr": False,
            "skip_existing": True,
            "cache_null_prompt": True,
            "offload_text_encoder": True,
            "device": "cpu",
            "dtype": "float32",
            "debug": True,
            "debug_every": 1,
        }
        defaults.update(overrides)
        return GatherConfig(**defaults)

    return _factory


@pytest.fixture
def mock_pipeline_factory(torch_mod):
    import numpy as np
    from PIL import Image

    torch = torch_mod

    class FakeVAE(torch.nn.Module):
        def __init__(self, scaling_factor: float = 0.5):
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=scaling_factor)
            self._w = torch.nn.Parameter(torch.zeros(1))
            self.to_called = False
            self.last_decode_input = None
            self.last_encode_input = None

        def to(self, *args, **kwargs):  # pragma: no cover - behavior checked via flags
            self.to_called = True
            return super().to(*args, **kwargs)

        def encode(self, img):
            self.last_encode_input = img
            b, _c, h, w = img.shape
            latent = torch.ones((b, 4, max(1, h // 8), max(1, w // 8)), device=img.device, dtype=img.dtype)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))

        def decode(self, z):
            self.last_decode_input = z
            b, _c, h, w = z.shape
            rgb = torch.zeros((b, 3, h * 8, w * 8), device=z.device, dtype=z.dtype)
            rgb[:, 0] = 0.2
            rgb[:, 1] = 0.4
            rgb[:, 2] = 0.6
            return SimpleNamespace(sample=rgb)

    class FakeTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._w = torch.nn.Parameter(torch.zeros(1))
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append((args, kwargs))
            return super().to(*args, **kwargs)

    class FakePipe:
        def __init__(
            self,
            return_mode: str = "tensor",
            consume_generator: bool = False,
            supports_prompt_embeds: bool = True,
            supports_negative_prompt_embeds: bool = True,
            supports_negative_prompt: bool = True,
            latent_offset: float = 1.0,
        ):
            self.return_mode = return_mode
            self.consume_generator = consume_generator
            self.supports_prompt_embeds = supports_prompt_embeds
            self.supports_negative_prompt_embeds = supports_negative_prompt_embeds
            self.supports_negative_prompt = supports_negative_prompt
            self.latent_offset = latent_offset
            self.calls = []
            self.gen_probes = []
            self.encode_prompt_calls = []

            self.vae = FakeVAE(scaling_factor=0.5)
            self.transformer = torch.nn.Linear(1, 1)
            self.unet = torch.nn.Linear(1, 1)
            self.text_encoder = FakeTextEncoder()

        def to(self, *_args, **_kwargs):
            return self

        def encode_prompt(
            self,
            prompt,
            device=None,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        ):
            self.encode_prompt_calls.append(
                {
                    "prompt": prompt,
                    "device": device,
                    "num_images_per_prompt": num_images_per_prompt,
                    "do_classifier_free_guidance": do_classifier_free_guidance,
                    "negative_prompt": negative_prompt,
                }
            )
            pe = torch.ones((1, 4), dtype=torch.float32)
            ne = torch.zeros((1, 4), dtype=torch.float32)
            return pe, ne

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
            call = {
                "prompt": prompt,
                "strength": strength,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "output_type": output_type,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt": negative_prompt,
                "generator": generator,
                "latents": latents,
            }
            self.calls.append(call)
            if self.consume_generator:
                probe = torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
                self.gen_probes.append(probe)

            z0 = latents + self.latent_offset
            if self.return_mode == "list":
                images = [z0]
            elif self.return_mode == "pil":
                images = [Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))]
            else:
                images = z0
            return SimpleNamespace(images=images)

    return FakePipe
