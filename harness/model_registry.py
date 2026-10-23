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
        "revision": "main",
        "kind": "static",
        "dim": 256,
        "note": "16M static code encoder, CPU-only, distilled from CodeRankEmbed",
    },
    "code-rank": {
        "repo": "nomic-ai/CodeRankEmbed",
        "revision": "main",
        "kind": "sentence-transformer",
        "dim": 768,
        "note": "137M code bi-encoder, 8192 context; needs the semantic extra",
    },
    "gte-reranker": {
        "repo": "Alibaba-NLP/gte-reranker-modernbert-base",
        "revision": "main",
        "kind": "cross-encoder",
        "dim": 0,
        "note": "149M code reranker, 8192 context; needs the semantic extra",
    },
}

_ENV_DIR = "RBG_MODEL_DIR"


def cache_root() -> Path:
    return Path(os.environ.get(_ENV_DIR) or
                (Path(os.path.expanduser("~")) / ".local" / "share" / "harness" / "models"))


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
    return {"files": files}


def pull(name: str, revision: str | None = None) -> str:
    spec = SPECS.get(name)
    if spec is None:
        raise KeyError(f"unknown model: {name}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface-hub to pull models") from exc
    dest = model_dir(name)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec["repo"], revision=revision or spec["revision"],
                      local_dir=str(dest))
    (dest / ".rbg-manifest.json").write_text(
        json.dumps(_manifest(dest), indent=2, sort_keys=True) + "\n")
    return str(dest)


def verify(name: str) -> dict:
    path = model_dir(name)
    manifest = path / ".rbg-manifest.json"
    if not manifest.exists():
        return {"model": name, "ok": False, "reason": "not pulled"}
    expected = json.loads(manifest.read_text()).get("files", {})
    actual = _manifest(path)["files"]
    bad = [f for f in expected if expected[f] != actual.get(f)]
    missing = [f for f in expected if f not in actual]
    return {"model": name, "ok": not bad and not missing,
            "bad": bad[:20], "missing": missing[:20],
            "path": str(path)}


def list_models() -> list[dict]:
    out = []
    for name, spec in SPECS.items():
        path = model_dir(name)
        out.append({"name": name, **spec, "path": str(path),
                    "pulled": resolve(name) is not None})
    return out
