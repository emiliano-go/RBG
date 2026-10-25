"""Safe artifact persistence.

Model and encoder artifacts used to be ``pickle.loads``-ed straight from the
database.  A database is attacker-influenced (``RBG_DB_IN_REPO=1`` puts it
inside the checkout; a remote URL is another trust boundary), so that was
arbitrary code execution on load.

This module keeps the compact pickle encoding but wraps it in two controls:

* a whitelist unpickler that can only reconstruct harness model classes,
  numpy arrays, and inert builtins; and
* a SHA-256 digest stored alongside the payload, so a tampered artifact is
  rejected instead of silently trusted.

Legacy artifacts (no digest) are refused: callers treat them as a miss and
re-index/re-train, which is cheaper than trusting an unsigned blob.
"""

from __future__ import annotations

import hashlib
import io
import pickle

_ALLOWED_BUILTINS = {
    "set", "frozenset", "slice", "complex", "bytearray", "bytes",
    "str", "int", "float", "bool", "tuple", "list", "dict", "object",
}
_ALLOWED_PREFIXES = (
    "rbg.models.",
    "rbg.retrieval.",
    "rbg.training.",
    "numpy.",
    "numpy.core.",
    "numpy._core.",
    "numpy.dtypes.",
)


class UnsafeArtifact(ValueError):
    """An artifact referenced a class outside the allowlist."""


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "builtins" and name in _ALLOWED_BUILTINS:
            return super().find_class(module, name)
        if any(module == prefix.rstrip(".") or module.startswith(prefix)
               for prefix in _ALLOWED_PREFIXES):
            return super().find_class(module, name)
        raise UnsafeArtifact(f"artifact references forbidden class {module}.{name}")


def artifact_digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def dump(obj) -> tuple[bytes, str]:
    """Serialize ``obj`` and return ``(payload, sha256)``."""
    blob = pickle.dumps(obj, protocol=4)
    return blob, artifact_digest(blob)


def load(blob, expected_digest: str = ""):
    """Deserialize an artifact after verifying its digest.

    ``expected_digest`` is required: an unsigned payload is refused.  A digest
    mismatch and a forbidden class both raise :class:`UnsafeArtifact`.
    """
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if not isinstance(blob, (bytes, bytearray)):
        raise UnsafeArtifact("artifact payload is not bytes")
    blob = bytes(blob)
    if not expected_digest:
        raise UnsafeArtifact("artifact has no digest (legacy unsigned payload)")
    if artifact_digest(blob) != expected_digest:
        raise UnsafeArtifact("artifact digest mismatch")
    try:
        return _SafeUnpickler(io.BytesIO(blob)).load()
    except UnsafeArtifact:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is a miss, not a crash
        raise UnsafeArtifact(f"artifact cannot be decoded: {type(exc).__name__}") from exc
