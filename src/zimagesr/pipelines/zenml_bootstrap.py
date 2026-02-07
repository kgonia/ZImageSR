from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at root: {path}")
    return data


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _run_zenml_command(
    args: list[str],
    dry_run: bool = False,
    tolerate_exists: bool = False,
) -> None:
    cmd = ["zenml", *args]
    print(f"$ {_shell_join(cmd)}")
    if dry_run:
        return

    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    if proc.returncode == 0:
        return
    if tolerate_exists and "already" in stderr.lower():
        return
    raise RuntimeError(f"Command failed with code {proc.returncode}: {_shell_join(cmd)}")


def _pick_stack_name(raw: dict[str, Any], override: str | None) -> str | None:
    if override is not None:
        if override in ("none", ""):
            return None
        return override

    stacks = raw.get("stacks", {})
    if not isinstance(stacks, dict):
        return None

    local_cfg = stacks.get("local", {})
    if isinstance(local_cfg, dict) and local_cfg.get("set_active", False):
        return str(local_cfg.get("name", "zimagesr-local-stack"))

    s3_cfg = stacks.get("s3", {})
    if isinstance(s3_cfg, dict) and s3_cfg.get("set_active", False):
        return str(s3_cfg.get("name", "zimagesr-s3-stack"))

    return None


def bootstrap_stack(
    config_path: Path,
    dry_run: bool = False,
    initialize_repo: bool = True,
    activate_stack: str | None = None,
) -> dict[str, Any]:
    raw = _read_yaml(config_path)
    components = raw.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("`components` must be a mapping.")

    orchestrator_cfg = components.get("orchestrator", {})
    if not isinstance(orchestrator_cfg, dict):
        raise ValueError("`components.orchestrator` must be a mapping.")

    stores_cfg = components.get("artifact_stores", {})
    if not isinstance(stores_cfg, dict):
        raise ValueError("`components.artifact_stores` must be a mapping.")

    local_store_cfg = stores_cfg.get("local", {})
    s3_store_cfg = stores_cfg.get("s3", {})
    if not isinstance(local_store_cfg, dict):
        raise ValueError("`components.artifact_stores.local` must be a mapping.")
    if not isinstance(s3_store_cfg, dict):
        raise ValueError("`components.artifact_stores.s3` must be a mapping.")

    tracker_cfg_root = components.get("experiment_tracker", {})
    if not isinstance(tracker_cfg_root, dict):
        raise ValueError("`components.experiment_tracker` must be a mapping.")
    wandb_cfg = tracker_cfg_root.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        raise ValueError("`components.experiment_tracker.wandb` must be a mapping.")

    stacks_cfg = raw.get("stacks", {})
    if not isinstance(stacks_cfg, dict):
        raise ValueError("`stacks` must be a mapping.")
    local_stack_cfg = stacks_cfg.get("local", {})
    s3_stack_cfg = stacks_cfg.get("s3", {})
    if not isinstance(local_stack_cfg, dict):
        raise ValueError("`stacks.local` must be a mapping.")
    if not isinstance(s3_stack_cfg, dict):
        raise ValueError("`stacks.s3` must be a mapping.")

    orchestrator_name = str(orchestrator_cfg.get("name", "zimagesr-local-orchestrator"))
    orchestrator_flavor = str(orchestrator_cfg.get("flavor", "local"))

    local_store_enabled = bool(local_store_cfg.get("enabled", True))
    local_store_name = str(local_store_cfg.get("name", "zimagesr-local-store"))
    local_store_path = Path(_expand(local_store_cfg.get("path", "./.zenml/artifacts"))).resolve()

    s3_store_enabled = bool(s3_store_cfg.get("enabled", False))
    s3_store_name = str(s3_store_cfg.get("name", "zimagesr-s3-store"))
    s3_store_path = str(_expand(s3_store_cfg.get("path", "")))

    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_name = str(wandb_cfg.get("name", "zimagesr-wandb"))
    wandb_project = str(wandb_cfg.get("project_name", "zimagesr"))
    wandb_entity = wandb_cfg.get("entity", None)
    wandb_api_key_env = str(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))

    local_stack_enabled = bool(local_stack_cfg.get("enabled", True))
    local_stack_name = str(local_stack_cfg.get("name", "zimagesr-local-stack"))
    local_stack_tracker = local_stack_cfg.get("experiment_tracker", None)
    if local_stack_tracker in ("", "null"):
        local_stack_tracker = None

    s3_stack_enabled = bool(s3_stack_cfg.get("enabled", False))
    s3_stack_name = str(s3_stack_cfg.get("name", "zimagesr-s3-stack"))
    s3_stack_tracker = s3_stack_cfg.get("experiment_tracker", None)
    if s3_stack_tracker in ("", "null"):
        s3_stack_tracker = None

    created: list[str] = []

    if initialize_repo:
        _run_zenml_command(["init"], dry_run=dry_run)

    _run_zenml_command(
        ["orchestrator", "register", orchestrator_name, "--flavor", orchestrator_flavor],
        dry_run=dry_run,
        tolerate_exists=True,
    )
    created.append(f"orchestrator:{orchestrator_name}")

    if local_store_enabled:
        local_store_path.mkdir(parents=True, exist_ok=True)
        _run_zenml_command(
            [
                "artifact-store",
                "register",
                local_store_name,
                "--flavor",
                "local",
                "--path",
                str(local_store_path),
            ],
            dry_run=dry_run,
            tolerate_exists=True,
        )
        created.append(f"artifact-store:{local_store_name}")

    if s3_store_enabled:
        if not s3_store_path.startswith("s3://"):
            raise ValueError(
                "S3 artifact store is enabled but path is invalid. "
                "Expected an s3:// URI."
            )
        _run_zenml_command(
            [
                "artifact-store",
                "register",
                s3_store_name,
                "--flavor",
                "s3",
                "--path",
                s3_store_path,
            ],
            dry_run=dry_run,
            tolerate_exists=True,
        )
        created.append(f"artifact-store:{s3_store_name}")

    tracker_name: str | None = None
    if wandb_enabled:
        tracker_cmd = [
            "experiment-tracker",
            "register",
            wandb_name,
            "--flavor",
            "wandb",
            "--project_name",
            wandb_project,
        ]
        if wandb_entity is not None and str(wandb_entity).strip():
            tracker_cmd.extend(["--entity", str(wandb_entity)])
        _run_zenml_command(
            tracker_cmd,
            dry_run=dry_run,
            tolerate_exists=True,
        )
        created.append(f"experiment-tracker:{wandb_name}")
        tracker_name = wandb_name
        if not os.getenv(wandb_api_key_env):
            print(
                f"Warning: {wandb_api_key_env} is not set. "
                "WandB auth may fail until the environment variable is provided."
            )

    if local_stack_enabled:
        local_tracker_name = str(local_stack_tracker) if local_stack_tracker else tracker_name
        stack_cmd = [
            "stack",
            "register",
            local_stack_name,
            "-o",
            orchestrator_name,
            "-a",
            local_store_name,
        ]
        if local_tracker_name:
            stack_cmd.extend(["-e", local_tracker_name])
        _run_zenml_command(stack_cmd, dry_run=dry_run, tolerate_exists=True)
        created.append(f"stack:{local_stack_name}")

    if s3_stack_enabled:
        if not s3_store_enabled:
            raise ValueError(
                "S3 stack is enabled but S3 artifact store is disabled. "
                "Enable `components.artifact_stores.s3.enabled` first."
            )
        s3_tracker_name = str(s3_stack_tracker) if s3_stack_tracker else tracker_name
        stack_cmd = [
            "stack",
            "register",
            s3_stack_name,
            "-o",
            orchestrator_name,
            "-a",
            s3_store_name,
        ]
        if s3_tracker_name:
            stack_cmd.extend(["-e", s3_tracker_name])
        _run_zenml_command(stack_cmd, dry_run=dry_run, tolerate_exists=True)
        created.append(f"stack:{s3_stack_name}")

    stack_to_activate = _pick_stack_name(raw, activate_stack)
    if stack_to_activate:
        _run_zenml_command(
            ["stack", "set", stack_to_activate],
            dry_run=dry_run,
            tolerate_exists=False,
        )

    return {
        "config_path": str(config_path),
        "dry_run": dry_run,
        "activated_stack": stack_to_activate,
        "registered": created,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimagesr-zenml-bootstrap",
        description="Bootstrap ZenML stacks/components from zenml.yaml.",
    )
    parser.add_argument("--config", type=Path, default=Path("zenml.yaml"))
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run `zenml init` before registering components.",
    )
    parser.add_argument(
        "--activate-stack",
        default=None,
        help="Override stack activation target (e.g. zimagesr-local-stack, zimagesr-s3-stack, none).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = bootstrap_stack(
        config_path=args.config,
        dry_run=args.dry_run,
        initialize_repo=args.init,
        activate_stack=args.activate_stack,
    )
    print(result)


if __name__ == "__main__":
    main()
