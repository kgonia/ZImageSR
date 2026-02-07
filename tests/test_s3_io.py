from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zimagesr.data import s3_io


def test_s01_parse_s3_uri():
    assert s3_io.parse_s3_uri("s3://bucket/prefix/path") == ("bucket", "prefix/path")
    assert s3_io.parse_s3_uri("s3://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        s3_io.parse_s3_uri("http://bucket/prefix")
    with pytest.raises(ValueError):
        s3_io.parse_s3_uri("s3:///prefix")


def test_s02_upload_dir_to_s3(tmp_path, monkeypatch):
    local_dir = tmp_path / "dataset"
    (local_dir / "pairs").mkdir(parents=True)
    (local_dir / "pairs" / "000000").mkdir(parents=True)
    (local_dir / "pairs" / "000000" / "eps.pt").write_bytes(b"eps")
    (local_dir / "debug_trace.jsonl").write_text("{}", encoding="utf-8")
    (local_dir / "metadata.json").write_text("{}", encoding="utf-8")

    uploads = []

    class FakeS3:
        def upload_file(self, src, bucket, key):
            uploads.append((Path(src).name, bucket, key))

    fake_boto3 = SimpleNamespace(client=lambda service: FakeS3())
    monkeypatch.setattr(s3_io, "_load_boto3", lambda: fake_boto3)

    out = s3_io.upload_dir_to_s3(local_dir=local_dir, s3_uri="s3://my-bucket/root", include_debug=False)
    assert out.operation == "upload"
    assert out.bucket == "my-bucket"
    assert out.prefix == "root"
    assert out.uploaded_files == 2
    assert out.skipped_files == 1
    assert all(bucket == "my-bucket" for _, bucket, _ in uploads)
    assert all(key.startswith("root/") for _, _, key in uploads)


def test_s03_upload_dir_to_s3_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        s3_io.upload_dir_to_s3(local_dir=tmp_path / "missing", s3_uri="s3://bucket/path")


def test_s04_download_dir_from_s3(tmp_path, monkeypatch):
    keys = [
        "root/pairs/000000/eps.pt",
        "root/pairs/000000/z0.pt",
        "root/debug_trace.jsonl",
    ]
    download_calls = []

    class FakePaginator:
        def paginate(self, **_kwargs):
            return [{"Contents": [{"Key": k} for k in keys]}]

    class FakeS3:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

        def download_file(self, bucket, key, dst):
            download_calls.append((bucket, key, dst))
            Path(dst).write_text(f"from {bucket}/{key}", encoding="utf-8")

    fake_boto3 = SimpleNamespace(client=lambda service: FakeS3())
    monkeypatch.setattr(s3_io, "_load_boto3", lambda: fake_boto3)

    out_dir = tmp_path / "downloaded"
    out = s3_io.download_dir_from_s3("s3://bucket/root", out_dir, overwrite=False)
    assert out.downloaded_files == 3
    assert out.skipped_files == 0
    assert (out_dir / "pairs" / "000000" / "eps.pt").exists()

    out2 = s3_io.download_dir_from_s3("s3://bucket/root", out_dir, overwrite=False)
    assert out2.downloaded_files == 0
    assert out2.skipped_files == 3

    out3 = s3_io.download_dir_from_s3("s3://bucket/root", out_dir, overwrite=True)
    assert out3.downloaded_files == 3
    assert len(download_calls) == 6


def test_s05_write_sync_report(tmp_path):
    result = s3_io.S3SyncResult(
        operation="upload",
        bucket="b",
        prefix="p",
        uploaded_files=2,
        downloaded_files=0,
        skipped_files=1,
    )
    out_path = tmp_path / "reports" / "sync.json"
    s3_io.write_sync_report(out_path, result)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["operation"] == "upload"
    assert data["uploaded_files"] == 2
