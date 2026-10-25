"""Terminal task status, derived only from observed evidence (spec 3.2, 6.11).

``verified`` means every *required* gate actually ran and passed, and the
sandbox that ran them was isolated.  The empty-``all()`` trap is avoided: a
gate that never ran is ``None`` (unverified), not a pass.  A type checker that
is not installed is unverified but non-blocking; tests that collected nothing
(pytest exit 5) and an unisolated sandbox both block ``verified``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TerminalState(str, Enum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Gates:
    compile: bool | None = None
    typecheck: bool | None = None
    tests: bool | None = None
    isolated: bool = True


def gate_state(checks: list[dict], name: str) -> bool | None:
    """True (all passed) / False (some failed) / None (never ran)."""
    ran = [c for c in checks if c.get("name") == name]
    if not ran:
        return None
    return all(c.get("exit_code") == 0 for c in ran)


def derive_status(gates: Gates) -> TerminalState:
    states = (gates.compile, gates.typecheck, gates.tests)
    if all(s is None for s in states):
        return TerminalState.SKIPPED
    if any(s is False for s in states):
        return TerminalState.FAILED
    if not gates.isolated:
        return TerminalState.PARTIAL  # unisolated evidence is never verified
    if gates.compile is not True or gates.tests is not True:
        return TerminalState.PARTIAL  # a required gate never ran
    return TerminalState.VERIFIED
