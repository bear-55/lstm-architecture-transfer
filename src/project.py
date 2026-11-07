"""Shared project paths, configuration and provenance helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/pilot.yaml").read_text(encoding="utf-8-sig"))


def paths() -> dict:
    return json.loads((ROOT / "configs/paths.local.json").read_text(encoding="utf-8-sig"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def split_ids() -> dict[str, list[str]]:
    manifest_path = ROOT / "splits/selection_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Freeze the basin selection before reading any Q")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {}
    for role in ("train", "val", "test"):
        path = ROOT / f"splits/random_pub_{role}.txt"
        if digest(path) != manifest["split_hashes"][role]:
            raise RuntimeError(f"Frozen {role} basin list changed")
        result[role] = path.read_text(encoding="utf-8").splitlines()
        if len(result[role]) != {"train": 40, "val": 10, "test": 10}[role]:
            raise ValueError("Pilot basin counts differ from the frozen plan")
    if len(set(sum(result.values(), []))) != 60:
        raise ValueError("Split IDs overlap or contain duplicates")
    return result
