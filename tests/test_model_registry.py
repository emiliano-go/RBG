import json

import pytest

from harness import model_registry


def test_known_models_are_listed():
    names = {m["name"] for m in model_registry.list_models()}
    assert {"potion-code", "code-rank", "gte-reranker"} <= names


def test_unknown_model_is_an_error():
    with pytest.raises(KeyError):
        model_registry.model_dir("nope")


def test_unpulled_model_is_not_resolved(monkeypatch, tmp_path):
    monkeypatch.setenv("RBG_MODEL_DIR", str(tmp_path))
    assert model_registry.resolve("potion-code") is None
    assert model_registry.verify("potion-code")["ok"] is False


def test_manifest_verifies_and_detects_tampering(monkeypatch, tmp_path):
    monkeypatch.setenv("RBG_MODEL_DIR", str(tmp_path))
    path = model_registry.model_dir("potion-code")
    path.mkdir(parents=True)
    (path / "weights.bin").write_bytes(b"abc")
    (path / ".rbg-manifest.json").write_text(
        json.dumps({"files": {"weights.bin": model_registry.hashlib.sha256(b"abc").hexdigest()}}))
    assert model_registry.verify("potion-code")["ok"] is True
    (path / "weights.bin").write_bytes(b"tampered")
    assert model_registry.verify("potion-code")["ok"] is False
