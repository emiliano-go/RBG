"""Pinned, offline-capable model artifacts.

Nothing is downloaded silently.  ``rbg models pull NAME`` materialises a
checkpoint into the shared cache and records a manifest of file digests;
``rbg models verify NAME`` re-checks it.  The runtime only reads the cache, so
an air-gapped install works after one explicit pull.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# revision pins the checkpoint for reproducibility; update deliberately.
SPECS: dict[str, dict] = {
    "potion-code": {
        "repo": "minishlab/potion-code-16M-v2",
        "revision": "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b",
        "kind": "static",
        "dim": 256,
        "note": "16M static code encoder, CPU-only, distilled from CodeRankEmbed",
    },
    "code-rank": {
        "repo": "nomic-ai/CodeRankEmbed",
        "revision": "3c4b60807d71f79b43f3c4363786d9493691f8b1",
        "kind": "sentence-transformer",
        "dim": 768,
        "note": "137M code bi-encoder, 8192 context; needs the semantic extra",
    },
    "gte-reranker": {
        "repo": "Alibaba-NLP/gte-reranker-modernbert-base",
        "revision": "f7481e6055501a30fb19d090657df9ec1f79ab2c",
        "kind": "cross-encoder",
        "dim": 0,
        "note": "149M code reranker, 8192 context; needs the semantic extra",
    },
}

_ENV_DIR = "RBG_MODEL_DIR"


def cache_root() -> Path:
    return Path(os.environ.get(_ENV_DIR) or
                (Path(os.path.expanduser("~")) / ".local" / "share" / "rbg" / "models"))


def model_dir(name: str) -> Path:
    spec = SPECS.get(name)
    if spec is None:
        raise KeyError(f"unknown model: {name} (known: {', '.join(SPECS)})")
    return cache_root() / name


def resolve(name: str) -> str | None:
    """Local path for a pulled model, or ``None`` when it is not present."""
    path = model_dir(name)
    return str(path) if path.is_dir() and any(path.iterdir()) else None


def _manifest(path: Path) -> dict:
    files = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.name != ".rbg-manifest.json":
            files[str(file.relative_to(path))] = hashlib.sha256(
                file.read_bytes()).hexdigest()
    return {"version": 1, "files": files}


def pull(name: str, revision: str | None = None) -> str:
    spec = SPECS.get(name)
    if spec is None:
        raise KeyError(f"unknown model: {name}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface-hub to pull models") from exc
    if revision and revision != spec["revision"]:
        raise ValueError(f"{name} is pinned to {spec['revision']}; refusing {revision}")
    dest = model_dir(name)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec["repo"], revision=spec["revision"],
                      local_dir=str(dest))
    manifest = _manifest(dest)
    manifest.update({"repo": spec["repo"], "revision": spec["revision"]})
    (dest / ".rbg-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return str(dest)


def verify(name: str) -> dict:
    path = model_dir(name)
    manifest = path / ".rbg-manifest.json"
    if not manifest.exists():
        return {"model": name, "ok": False, "reason": "not pulled"}
    doc = json.loads(manifest.read_text())
    spec = SPECS[name]
    if doc.get("repo") != spec["repo"] or doc.get("revision") != spec["revision"]:
        return {"model": name, "ok": False, "reason": "manifest does not match pinned spec",
                "path": str(path)}
    expected = doc.get("files", {})
    actual = _manifest(path)["files"]
    bad = [f for f in expected if expected[f] != actual.get(f)]
    missing = [f for f in expected if f not in actual]
    unexpected = [f for f in actual if f not in expected]
    return {"model": name, "ok": not (bad or missing or unexpected),
            "bad": bad[:20], "missing": missing[:20],
            "unexpected": unexpected[:20], "path": str(path)}


def list_models() -> list[dict]:
    out = []
    for name, spec in SPECS.items():
        path = model_dir(name)
        out.append({"name": name, **spec, "path": str(path),
                    "pulled": resolve(name) is not None})
    return out
