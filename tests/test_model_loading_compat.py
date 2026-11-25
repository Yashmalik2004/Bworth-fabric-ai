"""Every historical checkpoint must load and predict exactly as before the Phase 5 loader change.

Reference logits were captured BEFORE training/composition/model.py was extended
(tests/fixtures/checkpoint_reference_outputs.json, fixed random input).
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from training.composition.model import load_model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "tests" / "fixtures" / "checkpoint_reference_outputs.json").read_text())


@pytest.mark.parametrize("path", sorted(REF["checkpoints"]))
def test_historical_checkpoint_outputs_unchanged(path):
    r = REF["checkpoints"][path]
    f = ROOT / path
    if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
        pytest.skip("checkpoint not present locally (git-ignored) or different file")
    ck = torch.load(f, map_location="cpu", weights_only=False)
    assert ck.get("architecture", "resnet50") == "resnet50"
    m = load_model_from_checkpoint(ck, torch.device("cpu"))
    x = torch.randn(2, 3, 320, 320, generator=torch.Generator().manual_seed(1234))
    with torch.no_grad():
        out = m(x)
    assert torch.equal(out, torch.tensor(r["logits"], dtype=out.dtype)) or torch.allclose(out, torch.tensor(r["logits"]), atol=1e-5)


def test_checkpoint_without_architecture_key_defaults_to_resnet50():
    from training.composition.model import build_model

    m = build_model(3, pretrained=False)
    ck = {"model_state_dict": m.state_dict(), "class_to_idx": {"a": 0, "b": 1, "c": 2}, "config": {}}
    loaded = load_model_from_checkpoint(ck, torch.device("cpu"))
    assert type(loaded.fc[1]).__name__ == "Linear" and loaded.fc[1].out_features == 3
