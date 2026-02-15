from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from zimagesr import cli
from zimagesr.training.config import TrainConfig


class TestAddTrainArgs:
    def test_parses_required_and_defaults(self, tmp_path):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        args = parser.parse_args(["--pairs-dir", str(tmp_path)])

        assert args.pairs_dir == tmp_path
        assert args.model_id == TrainConfig.model_id
        assert args.tl == TrainConfig.tl
        assert args.batch_size == TrainConfig.batch_size
        assert args.max_steps == TrainConfig.max_steps
        assert args.lora_rank == TrainConfig.lora_rank
        assert args.lambda_adl == TrainConfig.lambda_adl
        assert args.save_dir.name.startswith("ftd_run_")
        assert args.save_full_state == TrainConfig.save_full_state
        assert args.wandb == TrainConfig.wandb_enabled
        assert args.wandb_project == TrainConfig.wandb_project
        assert args.wandb_mode == TrainConfig.wandb_mode
        assert args.wandb_log_checkpoints == TrainConfig.wandb_log_checkpoints
        assert args.wandb_log_checkpoint_grids == TrainConfig.wandb_log_checkpoint_grids
        assert args.checkpoint_infer_grid == TrainConfig.checkpoint_infer_grid
        assert args.checkpoint_eval_ids == ""
        assert args.checkpoint_eval_images_dir is None
        assert args.checkpoint_eval_images_limit == TrainConfig.checkpoint_eval_images_limit
        assert args.checkpoint_eval_input_upscale == TrainConfig.checkpoint_eval_input_upscale
        assert args.checkpoint_eval_fit_multiple == TrainConfig.checkpoint_eval_fit_multiple
        assert args.checkpoint_sr_scales == TrainConfig.checkpoint_sr_scales
        assert args.seed is None
        assert args.resume_from is None

    def test_parses_all_overrides(self, tmp_path):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        args = parser.parse_args(
            [
                "--pairs-dir", str(tmp_path),
                "--model-id", "other/model",
                "--tl", "0.15",
                "--batch-size", "2",
                "--gradient-accumulation-steps", "4",
                "--learning-rate", "1e-4",
                "--max-steps", "100",
                "--rec-loss-every", "0",
                "--lambda-tvlpips", "2.0",
                "--gamma-tv", "0.3",
                "--no-detach-recon",
                "--lambda-adl", "0.01",
                "--lora-rank", "8",
                "--lora-alpha", "8",
                "--lora-dropout", "0.1",
                "--save-dir", str(tmp_path / "saves"),
                "--save-every", "50",
                "--no-save-full-state",
                "--log-every", "10",
                "--device", "cpu",
                "--dtype", "float32",
                "--mixed-precision", "bf16",
                "--no-gradient-checkpointing",
                "--no-disable-vae-force-upcast",
                "--num-workers", "0",
                "--seed", "42",
                "--wandb",
                "--wandb-project", "proj-x",
                "--wandb-entity", "team-y",
                "--wandb-run-name", "run-z",
                "--wandb-mode", "offline",
                "--no-wandb-log-checkpoints",
                "--no-wandb-log-checkpoint-grids",
                "--checkpoint-infer-grid",
                "--checkpoint-eval-ids", "000000,7,custom_id",
                "--checkpoint-eval-images-dir", str(tmp_path / "eval_images"),
                "--checkpoint-eval-images-limit", "3",
                "--checkpoint-eval-input-upscale", "2.0",
                "--checkpoint-eval-fit-multiple", "8",
                "--checkpoint-sr-scales", "1.25,1.5",
            ]
        )

        assert args.model_id == "other/model"
        assert args.tl == 0.15
        assert args.batch_size == 2
        assert args.gradient_accumulation_steps == 4
        assert args.learning_rate == 1e-4
        assert args.max_steps == 100
        assert args.rec_loss_every == 0
        assert args.lambda_tvlpips == 2.0
        assert args.gamma_tv == 0.3
        assert args.detach_recon is False
        assert args.lambda_adl == 0.01
        assert args.lora_rank == 8
        assert args.lora_alpha == 8
        assert args.lora_dropout == 0.1
        assert args.save_full_state is False
        assert args.mixed_precision == "bf16"
        assert args.gradient_checkpointing is False
        assert args.disable_vae_force_upcast is False
        assert args.num_workers == 0
        assert args.seed == 42
        assert args.wandb is True
        assert args.wandb_project == "proj-x"
        assert args.wandb_entity == "team-y"
        assert args.wandb_run_name == "run-z"
        assert args.wandb_mode == "offline"
        assert args.wandb_log_checkpoints is False
        assert args.wandb_log_checkpoint_grids is False
        assert args.checkpoint_infer_grid is True
        assert args.checkpoint_eval_ids == "000000,7,custom_id"
        assert args.checkpoint_eval_images_dir.as_posix() == (tmp_path / "eval_images").as_posix()
        assert args.checkpoint_eval_images_limit == 3
        assert args.checkpoint_eval_input_upscale == 2.0
        assert args.checkpoint_eval_fit_multiple == 8
        assert args.checkpoint_sr_scales == (1.25, 1.5)

    def test_resume_from_parses_path(self, tmp_path):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        ckpt = tmp_path / "lora_step_50"
        args = parser.parse_args(["--pairs-dir", str(tmp_path), "--resume-from", str(ckpt)])
        from pathlib import Path

        assert args.resume_from == Path(str(ckpt))

    def test_pairs_dir_required(self):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_checkpoint_sr_scales_rejects_invalid_values(self, tmp_path):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--pairs-dir", str(tmp_path), "--checkpoint-sr-scales", "-1.0,1.3"]
            )


class TestTrainConfigFromArgs:
    def test_round_trip(self, tmp_path):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        args = parser.parse_args(
            [
                "--pairs-dir", str(tmp_path),
                "--tl", "0.3",
                "--seed", "7",
            ]
        )
        cfg = cli._train_config_from_args(args)
        assert isinstance(cfg, TrainConfig)
        assert cfg.pairs_dir == tmp_path
        assert cfg.tl == 0.3
        assert cfg.seed == 7
        assert cfg.batch_size == TrainConfig.batch_size  # default
        assert cfg.save_full_state == TrainConfig.save_full_state
        assert cfg.wandb_enabled is TrainConfig.wandb_enabled
        assert cfg.wandb_log_checkpoint_grids is TrainConfig.wandb_log_checkpoint_grids
        assert cfg.checkpoint_infer_grid is TrainConfig.checkpoint_infer_grid
        assert cfg.checkpoint_eval_ids == ()
        assert cfg.checkpoint_eval_images_dir is None
        assert cfg.checkpoint_eval_images_limit == TrainConfig.checkpoint_eval_images_limit
        assert cfg.checkpoint_eval_input_upscale == TrainConfig.checkpoint_eval_input_upscale
        assert cfg.checkpoint_eval_fit_multiple == TrainConfig.checkpoint_eval_fit_multiple
        assert cfg.checkpoint_sr_scales == TrainConfig.checkpoint_sr_scales
        assert cfg.resume_from is None

    def test_checkpoint_eval_values_round_trip(self, tmp_path):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        images_dir = tmp_path / "imgs"
        args = parser.parse_args(
            [
                "--pairs-dir", str(tmp_path),
                "--checkpoint-eval-ids", "000000, 42 ,custom",
                "--checkpoint-eval-images-dir", str(images_dir),
                "--checkpoint-eval-images-limit", "5",
                "--checkpoint-eval-input-upscale", "1.5",
                "--checkpoint-eval-fit-multiple", "32",
                "--checkpoint-sr-scales", "1.1,1.4,2.0",
            ]
        )
        cfg = cli._train_config_from_args(args)
        assert cfg.checkpoint_eval_ids == ("000000", "42", "custom")
        assert cfg.checkpoint_eval_images_dir == images_dir
        assert cfg.checkpoint_eval_images_limit == 5
        assert cfg.checkpoint_eval_input_upscale == 1.5
        assert cfg.checkpoint_eval_fit_multiple == 32
        assert cfg.checkpoint_sr_scales == (1.1, 1.4, 2.0)

    def test_resume_from_round_trip(self, tmp_path):
        parser = argparse.ArgumentParser()
        cli._add_train_args(parser)
        ckpt = tmp_path / "lora_step_50"
        args = parser.parse_args(
            ["--pairs-dir", str(tmp_path), "--resume-from", str(ckpt)]
        )
        cfg = cli._train_config_from_args(args)
        from pathlib import Path

        assert cfg.resume_from == Path(str(ckpt))


class TestBuildParserHasTrainAndGenerateZl:
    def test_train_subparser_exists(self):
        parser = cli.build_parser()
        args = parser.parse_args(["train", "--pairs-dir", "/tmp/pairs"])
        assert args.command == "train"
        assert args.pairs_dir.as_posix() == "/tmp/pairs"

    def test_generate_zl_subparser_exists(self):
        parser = cli.build_parser()
        args = parser.parse_args(["generate-zl", "--out-dir", "/tmp/data"])
        assert args.command == "generate-zl"
        assert args.out_dir.as_posix() == "/tmp/data"

    def test_generate_zl_skip_existing_default(self):
        parser = cli.build_parser()
        args = parser.parse_args(["generate-zl", "--out-dir", "/tmp/data"])
        assert args.skip_existing is True

    def test_generate_zl_no_skip_existing(self):
        parser = cli.build_parser()
        args = parser.parse_args(["generate-zl", "--out-dir", "/tmp/data", "--no-skip-existing"])
        assert args.skip_existing is False
