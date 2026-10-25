"""Property-based checks for paths, plans, and artifact validation."""

import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbg import persist  # noqa: E402
from rbg.paths import resolve_within  # noqa: E402
from rbg.plan import parse_plan, scope_drift  # noqa: E402

_SEGMENT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-", min_size=1, max_size=12)
_SAFE_PATH = st.lists(_SEGMENT, min_size=1, max_size=5).map(lambda parts: os.path.join(*parts))


@settings(max_examples=100)
@given(_SAFE_PATH)
def test_contained_relative_paths_stay_under_root(rel):
    root = os.path.abspath(os.sep + "rbg-root")
    got = resolve_within(root, rel)
    assert os.path.realpath(got) == os.path.realpath(os.path.join(root, rel))
    assert os.path.commonpath([os.path.realpath(got), os.path.realpath(root)]) == os.path.realpath(root)


@settings(max_examples=50)
@given(st.lists(st.just(".."), min_size=1, max_size=4), _SAFE_PATH)
def test_parent_traversal_is_always_rejected(ups, tail):
    root = os.path.abspath(os.sep + "rbg-root")
    with pytest.raises(PermissionError):
        resolve_within(root, os.path.join(*ups, tail))


@settings(max_examples=100)
@given(st.sets(_SAFE_PATH), st.sets(_SAFE_PATH))
def test_scope_drift_is_the_set_difference(planned, changed):
    known = planned | changed
    text = "\n".join(f"1. Edit {p}." for p in sorted(planned))
    plan = parse_plan(text, known)
    drift = scope_drift(plan, sorted(changed))
    assert set(drift.missing) == set(plan.edit_files()) - set(changed)
    assert set(drift.extra) == set(changed) - set(plan.edit_files())


@settings(max_examples=100)
@given(st.binary(max_size=512))
def test_artifact_roundtrip_and_tamper_rejection(payload):
    marker = {"payload": payload}
    blob, digest = persist.dump(marker)
    assert persist.load(blob, digest) == marker
    if not blob:
        return
    tampered = bytes([blob[0] ^ 0xFF]) + blob[1:]
    with pytest.raises(persist.UnsafeArtifact):
        persist.load(tampered, digest)
    with pytest.raises(persist.UnsafeArtifact):
        persist.load(blob, "")
