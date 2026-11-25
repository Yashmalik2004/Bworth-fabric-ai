"""Frozen Phase 5 protocol (experiments/phase5/protocol.json) and shared constants."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_FILE = PROJECT_ROOT / "experiments" / "phase5" / "protocol.json"
PHASE5_DIR = PROJECT_ROOT / "experiments" / "phase5"
LABEL_SCHEME = "phase5_band4"
CLASSES = ["low", "medium", "high", "very_high"]
SYNTHETIC_BANNER = "PIPELINE TEST - synthetic data - NOT model performance"


def protocol_sha256(path: Path = PROTOCOL_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(path: Path = PROTOCOL_FILE) -> dict:
    return _load(str(path))


def protocol_version(path: Path = PROTOCOL_FILE) -> str:
    return str(load_protocol(path)["protocol_version"])


def gates(path: Path = PROTOCOL_FILE) -> dict:
    return load_protocol(path)["readiness_gates"]


def bands(path: Path = PROTOCOL_FILE) -> dict[str, tuple[float, float]]:
    return {k: (float(v[0]), float(v[1])) for k, v in load_protocol(path)["task"]["bands"].items()}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
