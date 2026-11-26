"""Download and verify a frozen BindingDB TSV release.

The downloader deliberately keeps the upstream zip archive. Preparation is a
separate, reproducible step implemented in ``prepare_bindingdb.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_RELEASE = "202611"
URL_TEMPLATE = (
    "https://www.bindingdb.org/rwd/bind/downloads/"
    "BindingDB_All_{release}_tsv.zip"
)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/bindingdb"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(path: Path) -> dict:
    """Return archive metadata or raise if the download is not a usable TSV zip."""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"BindingDB download is not a zip archive: {path}")
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"Corrupt member in BindingDB archive: {corrupt}")
        members = [
            info for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith((".tsv", ".txt"))
        ]
        if not members:
            raise ValueError("BindingDB archive contains no TSV/TXT data member")
        return {
            "members": [m.filename for m in members],
            "uncompressed_bytes": sum(m.file_size for m in members),
        }


def download_bindingdb(
    release: str = DEFAULT_RELEASE,
    *,
    url: str | None = None,
    output: Path | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Download one date-stamped BindingDB archive atomically and write a manifest."""
    url = url or URL_TEMPLATE.format(release=release)
    output = output or RAW_DIR / f"BindingDB_All_{release}_tsv.zip"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")

    if force or not output.exists():
        if partial.exists():
            partial.unlink()
        print(f"Downloading frozen BindingDB release {release}")
        print(f"Source: {url}")
        print(f"Destination: {output}")
        started = time.time()
        request = urllib.request.Request(url, headers={"User-Agent": "DataSAIL-scaling/1.0"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
            content_length = response.headers.get("Content-Length")
        os.replace(partial, output)
        elapsed = time.time() - started
    else:
        print(f"Using existing BindingDB archive: {output}")
        content_length = None
        elapsed = 0.0

    archive_info = validate_archive(output)
    digest = sha256_file(output)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(
            f"BindingDB SHA-256 mismatch: expected {expected_sha256}, observed {digest}"
        )
    manifest = {
        "release": release,
        "url": url,
        "archive": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        "archive_bytes": output.stat().st_size,
        "http_content_length": int(content_length) if content_length else None,
        "sha256": digest,
        "download_seconds": elapsed,
        **archive_info,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="BindingDB YYYYMM release")
    parser.add_argument("--url", help="Override the release download URL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sha256", help="Expected archive SHA-256")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_bindingdb(
        args.release,
        url=args.url,
        output=args.output,
        expected_sha256=args.sha256,
        force=args.force,
    )


if __name__ == "__main__":
    main()
