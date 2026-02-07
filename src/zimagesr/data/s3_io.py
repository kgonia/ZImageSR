from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


def _load_boto3():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for S3 sync operations. Install dependencies and retry."
        ) from exc
    return boto3


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI '{s3_uri}'. Expected format: s3://bucket/prefix")
    path = s3_uri[len("s3://") :]
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if bucket == "":
        raise ValueError(f"Invalid S3 URI '{s3_uri}'. Bucket is empty.")
    return bucket, prefix.strip("/")


@dataclass
class S3SyncResult:
    operation: str
    bucket: str
    prefix: str
    uploaded_files: int = 0
    downloaded_files: int = 0
    skipped_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "bucket": self.bucket,
            "prefix": self.prefix,
            "uploaded_files": self.uploaded_files,
            "downloaded_files": self.downloaded_files,
            "skipped_files": self.skipped_files,
        }


def upload_dir_to_s3(
    local_dir: Path,
    s3_uri: str,
    include_debug: bool = True,
) -> S3SyncResult:
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    bucket, prefix = parse_s3_uri(s3_uri)
    boto3 = _load_boto3()
    s3 = boto3.client("s3")
    result = S3SyncResult(operation="upload", bucket=bucket, prefix=prefix)

    files = [p for p in sorted(local_dir.rglob("*")) if p.is_file()]
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        if not include_debug and rel.startswith("debug"):
            result.skipped_files += 1
            continue
        key = f"{prefix}/{rel}" if prefix else rel
        s3.upload_file(str(path), bucket, key)
        result.uploaded_files += 1

    return result


def download_dir_from_s3(
    s3_uri: str,
    local_dir: Path,
    overwrite: bool = False,
) -> S3SyncResult:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    bucket, prefix = parse_s3_uri(s3_uri)
    boto3 = _load_boto3()
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    result = S3SyncResult(operation="download", bucket=bucket, prefix=prefix)

    pagination_args = {"Bucket": bucket}
    if prefix:
        pagination_args["Prefix"] = prefix

    for page in paginator.paginate(**pagination_args):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key[len(prefix) :].lstrip("/") if prefix else key
            target = local_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists() and not overwrite:
                result.skipped_files += 1
                continue

            s3.download_file(bucket, key, str(target))
            result.downloaded_files += 1

    return result


def write_sync_report(out_path: Path, result: S3SyncResult) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
