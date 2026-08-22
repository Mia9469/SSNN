#!/usr/bin/env python3
"""Fetch and verify the exact FlyWire v783 inputs used by the Letter.

The connectivity table is pinned to the Shiu et al. reproduction repository.
The complete 139,255-neuron universe is read from the public Codex v783 neuron
table and reduced to the two-column format expected by flywire_gain_routing.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
from pathlib import Path
import urllib.request


SOURCE_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
GRAPH_URL = (
    "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/"
    f"{SOURCE_COMMIT}/Connectivity_783.parquet"
)
CODEX_NODES_URL = (
    "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/"
    "neurons.csv.gz"
)

GRAPH_SHA256 = "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"
CODEX_NODES_GZ_SHA256 = "6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719"
DERIVED_NODES_SHA256 = "d144a7a4d5b25c13d452c23ad04bf90ccbbded2c03b492b8eb024d13ac6a987d"
EXPECTED_NEURONS = 139_255


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def download(url: str, destination: Path, expected: str, force: bool) -> None:
    if destination.exists() and not force:
        require_hash(destination, expected)
        print(f"verified existing {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "flywire-v783-reproduction"}
    )
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
        "wb"
    ) as out:
        while block := response.read(1024 * 1024):
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for downloaded {url}: "
            f"{digest.hexdigest()} != {expected}"
        )
    os.replace(temporary, destination)
    print(f"downloaded and verified {destination}")


def materialize_node_table(source_gz: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        require_hash(destination, DERIVED_NODES_SHA256)
        print(f"verified existing {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    count = 0
    seen: set[int] = set()
    with gzip.open(source_gz, "rt", newline="") as source, temporary.open(
        "w", newline="\n"
    ) as out:
        reader = csv.DictReader(source)
        if "root_id" not in (reader.fieldnames or []):
            raise RuntimeError("Codex neuron table lacks a root_id column")
        out.write(",Completed\n")
        for row in reader:
            root_id = int(row["root_id"])
            if root_id in seen:
                raise RuntimeError(f"duplicate Codex root ID: {root_id}")
            seen.add(root_id)
            out.write(f"{root_id},True\n")
            count += 1
    if count != EXPECTED_NEURONS:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"expected {EXPECTED_NEURONS} neurons, found {count}")
    require_hash(temporary, DERIVED_NODES_SHA256)
    os.replace(temporary, destination)
    print(f"materialized and verified {destination} ({count:,} neurons)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--force", action="store_true", help="replace even valid existing files"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph = args.output_dir / "Connectivity_783.parquet"
    nodes_gz = args.output_dir / "Codex_fafb_v783_neurons.csv.gz"
    nodes = args.output_dir / "Completeness_783_all_139255.csv"
    download(GRAPH_URL, graph, GRAPH_SHA256, args.force)
    download(CODEX_NODES_URL, nodes_gz, CODEX_NODES_GZ_SHA256, args.force)
    materialize_node_table(nodes_gz, nodes, args.force)
    print("PASS: exact FlyWire v783 analysis inputs are ready")


if __name__ == "__main__":
    main()
