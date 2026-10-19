"""Runnable checks for the evidence-derived terminal status."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.status import Gates, TerminalState, derive_status, gate_state  # noqa: E402


def test_all_required_gates_pass_and_isolated_is_verified():
    assert derive_status(Gates(compile=True, typecheck=True, tests=True)) is TerminalState.VERIFIED


def test_absent_typechecker_is_unverified_but_not_blocking():
    assert derive_status(Gates(compile=True, typecheck=None, tests=True)) is TerminalState.VERIFIED


def test_no_tests_collected_is_partial_not_verified():
    assert derive_status(Gates(compile=True, typecheck=True, tests=None)) is TerminalState.PARTIAL


def test_unisolated_is_never_verified():
    assert derive_status(Gates(compile=True, typecheck=True, tests=True,
                               isolated=False)) is TerminalState.PARTIAL


def test_any_failed_gate_is_failed():
    assert derive_status(Gates(compile=False, tests=True)) is TerminalState.FAILED


def test_nothing_ran_is_skipped():
    assert derive_status(Gates()) is TerminalState.SKIPPED


def test_gate_state_ran_all_or_some():
    assert gate_state([], "compile") is None
    assert gate_state([{"name": "compile", "exit_code": 0}], "compile") is True
    assert gate_state([{"name": "compile", "exit_code": 0},
                       {"name": "compile", "exit_code": 1}], "compile") is False
