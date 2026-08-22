#!/usr/bin/env python3
"""Download all matching files from a Zenodo record with checksum validation."""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("record_id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--concurrent-files", type=int, default=3)
    parser.add_argument("--workers-per-file", type=int, default=1)
    parser.add_argument("--retries", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://zenodo.org/api/records/{args.record_id}"
    request = urllib.request.Request(
        api_url, headers={"User-Agent": "energy-nonnormal-reproducibility/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        record = json.load(response)

    selected = [
        item for item in record["files"] if fnmatch.fnmatch(item["key"], args.pattern)
    ]
    if not selected:
        raise RuntimeError(f"record {args.record_id} has no files matching {args.pattern!r}")

    downloader = Path(__file__).with_name("download_zenodo_parallel.py")

    def ensure(item: dict) -> tuple[str, str]:
        name = item["key"]
        destination = args.output_dir / name
        expected_size = int(item["size"])
        expected_md5 = item["checksum"].split(":", 1)[-1].lower()
        if destination.exists():
            if destination.stat().st_size != expected_size:
                raise RuntimeError(
                    f"existing {destination} has size {destination.stat().st_size}, "
                    f"expected {expected_size}"
                )
            observed = md5sum(destination)
            if observed != expected_md5:
                raise RuntimeError(
                    f"existing {destination} has md5 {observed}, expected {expected_md5}"
                )
            return name, "already verified"

        command = [
            sys.executable,
            str(downloader),
            item["links"].get("content", item["links"]["self"]),
            str(destination),
            "--size",
            str(expected_size),
            "--md5",
            expected_md5,
            "--workers",
            str(args.workers_per_file),
            "--retries",
            str(args.retries),
        ]
        subprocess.run(command, check=True)
        return name, "downloaded and verified"

    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrent_files
    ) as pool:
        futures = {pool.submit(ensure, item): item["key"] for item in selected}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                _, status = future.result()
                print(f"{name}: {status}", flush=True)
            except Exception as error:
                failures.append((name, repr(error)))
                print(f"{name}: FAILED: {error}", file=sys.stderr, flush=True)

    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 1
    print(f"verified {len(selected)} files from Zenodo record {args.record_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
