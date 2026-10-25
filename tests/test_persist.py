"""Artifact loading must not execute code from an untrusted database."""

import pickle

import numpy as np
import pytest

from rbg import persist


def test_roundtrip_and_digest():
    obj = {"a": np.arange(4), "b": [1, 2]}
    blob, digest = persist.dump(obj)
    got = persist.load(blob, digest)
    assert (got["a"] == obj["a"]).all() and got["b"] == [1, 2]


def test_digest_mismatch_is_rejected():
    blob, digest = persist.dump({"x": 1})
    with pytest.raises(persist.UnsafeArtifact):
        persist.load(blob, "0" * 64)


def test_unsigned_legacy_payload_is_rejected():
    with pytest.raises(persist.UnsafeArtifact, match="no digest"):
        persist.load(pickle.dumps({"x": 1}), "")


def test_forbidden_class_never_executes(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    blob = pickle.dumps(Evil())
    with pytest.raises(persist.UnsafeArtifact):
        persist.load(blob, persist.artifact_digest(blob))
    assert not marker.exists()


def test_db_refuses_legacy_artifact(tmp_path):
    from rbg.db import Store

    store = Store(str(tmp_path))
    store.conn.execute(
        "INSERT INTO models(name, revision, version, payload, metrics, updated_at) "
        "VALUES('learned','r','v',?,?,0)", (pickle.dumps({"x": 1}), "{}"))
    assert store.load_model("learned") is None
    store.close()
