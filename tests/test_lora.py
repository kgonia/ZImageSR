from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from zimagesr.training.lora import find_lora_targets


# ---------------------------------------------------------------------------
# find_lora_targets
# ---------------------------------------------------------------------------


class TestFindLoraTargets:
    def _make_model(self, layers: dict[str, torch.nn.Module]) -> torch.nn.Module:
        model = torch.nn.Module()
        for name, mod in layers.items():
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    setattr(parent, part, torch.nn.Module())
                parent = getattr(parent, part)
            setattr(parent, parts[-1], mod)
        return model

    def test_finds_attention_linears(self):
        model = self._make_model(
            {
                "layers.0.attention.to_q": torch.nn.Linear(64, 64),
                "layers.0.attention.to_k": torch.nn.Linear(64, 64),
                "layers.0.attention.to_v": torch.nn.Linear(64, 64),
                "layers.0.attention.to_out.0": torch.nn.Linear(64, 64),
                "layers.0.norm": torch.nn.LayerNorm(64),
                "embed.proj": torch.nn.Linear(64, 64),  # excluded: "embed" in name
            }
        )
        targets = find_lora_targets(model)
        assert "layers.0.attention.to_q" in targets
        assert "layers.0.attention.to_k" in targets
        assert "layers.0.attention.to_v" in targets
        assert "layers.0.attention.to_out.0" in targets
        # embed.proj has "embed" keyword → excluded
        assert "embed.proj" not in targets

    def test_excludes_norm_and_embed(self):
        model = self._make_model(
            {
                "norm_layer": torch.nn.Linear(32, 32),
                "embedding.dense": torch.nn.Linear(32, 32),
                "time_proj": torch.nn.Linear(32, 32),
                "attn.to_q": torch.nn.Linear(32, 32),
            }
        )
        targets = find_lora_targets(model)
        assert "norm_layer" not in targets
        assert "embedding.dense" not in targets
        assert "time_proj" not in targets
        assert "attn.to_q" in targets

    def test_fallback_when_no_attn_layers(self):
        model = self._make_model(
            {
                "block.dense1": torch.nn.Linear(32, 32),
                "block.dense2": torch.nn.Linear(32, 32),
            }
        )
        targets = find_lora_targets(model)
        # Fallback: should find the linear layers
        assert "block.dense1" in targets
        assert "block.dense2" in targets

    def test_excludes_peft_internals(self):
        model = self._make_model(
            {
                "attn.to_q": torch.nn.Linear(32, 32),
                "attn.to_q.lora_A": torch.nn.Linear(32, 4),
                "peft_layer.dense": torch.nn.Linear(32, 32),
            }
        )
        targets = find_lora_targets(model)
        assert "attn.to_q" in targets
        assert "attn.to_q.lora_A" not in targets
        assert "peft_layer.dense" not in targets

    def test_deduplicates(self):
        model = self._make_model(
            {
                "attn.to_q": torch.nn.Linear(16, 16),
            }
        )
        targets = find_lora_targets(model)
        assert targets == sorted(set(targets))


# ---------------------------------------------------------------------------
# apply_lora (requires peft, skip if not installed)
# ---------------------------------------------------------------------------


class TestApplyLora:
    def test_raises_without_peft(self):
        from zimagesr.training.lora import apply_lora

        model = torch.nn.Linear(4, 4)
        with patch.dict("sys.modules", {"peft": None}):
            with pytest.raises(RuntimeError, match="peft is required"):
                apply_lora(model, targets=[""])

    @pytest.mark.skipif(
        not pytest.importorskip("peft", reason="peft not installed"),
        reason="peft not installed",
    )
    def test_apply_lora_wraps_model(self):
        from peft import PeftModel

        from zimagesr.training.lora import apply_lora

        model = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.Linear(16, 16),
        )
        model.requires_grad_(False)
        wrapped = apply_lora(model, rank=4, alpha=4, targets=["0", "1"])
        assert isinstance(wrapped, PeftModel)
        trainable = sum(1 for p in wrapped.parameters() if p.requires_grad)
        assert trainable > 0


# ---------------------------------------------------------------------------
# save_lora
# ---------------------------------------------------------------------------


class TestSaveLora:
    def test_save_creates_directory(self, tmp_path):
        from zimagesr.training.lora import save_lora

        save_path = tmp_path / "lora_out"

        class FakeModel:
            def save_pretrained(self, path):
                (path / "saved.txt").write_text("ok")

        save_lora(FakeModel(), save_path)
        assert (save_path / "saved.txt").exists()

    def test_save_with_accelerator_unwrap(self, tmp_path):
        from zimagesr.training.lora import save_lora

        save_path = tmp_path / "lora_out2"
        inner = SimpleNamespace(
            save_pretrained=lambda path: (path / "saved.txt").write_text("ok")
        )

        class FakeAccelerator:
            def unwrap_model(self, m):
                return inner

        save_lora(SimpleNamespace(), save_path, accelerator=FakeAccelerator())
        assert (save_path / "saved.txt").exists()


# ---------------------------------------------------------------------------
# load_lora_for_inference
# ---------------------------------------------------------------------------


class TestLoadLoraForInference:
    def test_raises_without_peft(self):
        from zimagesr.training.lora import load_lora_for_inference

        with patch.dict("sys.modules", {"peft": None}):
            with pytest.raises(RuntimeError, match="peft is required"):
                load_lora_for_inference(torch.nn.Linear(4, 4), "/fake", "cpu", torch.float32)
