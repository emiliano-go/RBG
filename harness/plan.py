"""Structured plans and step-level drift checks.

The main LLM still returns a free-text plan, but here that text is parsed into
a :class:`Plan` of concrete steps so the harness can verify, after every step,
that the agent did what it said it would.  Everything is deterministic (no new
model, no extra LLM call): a planned file that was never touched is a *missing*
step, a changed file nobody planned is *unplanned scope*, and a file that calls
a changed symbol but was never touched or retrieved is a completeness gap.

Three checks, all advisory except ``missing`` (which the orchestrator feeds
back into the repair loop):

* scope drift        - plan vs the working tree
* retrieval coverage - does the plan cover the retrieved source
* completeness audit - callers of changed symbols that were not touched
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# A leading list marker: "- ", "* ", "1. ", "1) ", "Step 3: ".
_MARKER = re.compile(r"^\s*(?:[-*]|\d+[.)]|step\s+\d+\s*[:.)]?)\s+", re.I)
_VERIFY = re.compile(r"\b(run|test|verify|check|validate|confirm|ensure|assert|re-?run)\b", re.I)
_EDIT = re.compile(r"\b(edit|add|update|modify|change|create|write|implement|fix|remove|delete|rename|refactor)\b", re.I)
# Backticked spans, or a bare token that looks like a filename/path.
_TOKEN = re.compile(r"`([^`]+)`|([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9_]+)")


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _file_candidates(text: str):
    for m in _TOKEN.finditer(text):
        yield (m.group(1) or m.group(2)).strip().strip(".,;:()'\"")


@dataclass
class PlanStep:
    index: int
    description: str
    action: str = "edit"          # "edit" | "verify"
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"index": self.index, "action": self.action,
                "description": self.description, "files": self.files}


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    raw: str = ""

    def edit_files(self) -> list[str]:
        """Files the agent committed to editing (verify steps excluded)."""
        out: list[str] = []
        for s in self.steps:
            if s.action == "edit":
                out.extend(s.files)
        return sorted(set(_norm(f) for f in out))

    def all_files(self) -> list[str]:
        return sorted({_norm(f) for s in self.steps for f in s.files})

    def has_edits(self) -> bool:
        return any(s.action == "edit" for s in self.steps)

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps],
                "edit_files": self.edit_files()}


def parse_plan(text: str, known_files) -> Plan:
    """Parse plan text into steps.  Only file tokens that exist in
    ``known_files`` are recorded, so prose can never invent a planned file."""
    known = {_norm(f) for f in known_files}
    steps: list[PlanStep] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        desc = _MARKER.sub("", line).strip()
        files = sorted({c for c in _file_candidates(desc) if _norm(c) in known})
        if _VERIFY.search(desc) and not _EDIT.search(desc):
            action = "verify"
        else:
            action = "edit"
        steps.append(PlanStep(len(steps) + 1, desc, action,
                              [_norm(f) for f in files]))
    return Plan(steps, raw=text or "")


@dataclass
class Drift:
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict:
        return {"missing": self.missing, "extra": self.extra, "ok": self.ok()}


def scope_drift(plan: Plan, changed_files) -> Drift:
    """Planned-but-unchanged files (missing) and changed-but-unplanned (extra)."""
    planned = set(plan.edit_files())
    changed = {_norm(p) for p in changed_files}
    return Drift(sorted(planned - changed), sorted(changed - planned))


def format_guidance(plan: Plan, drift: Drift) -> str:
    """Repair-loop guidance for missing planned files."""
    if not drift.missing:
        return ""
    return ("Plan drift: these planned files were not edited yet: "
            + ", ".join(drift.missing) + ". Finish them or say why they are out of scope.")


# Names that carry no task-specific routing signal: dunders and ubiquitous
# method/entry names.  ``changed_symbols`` holds bare method names, so a
# method like ``run`` collides with every unrelated ``run`` in the tree; these
# only add noise.  A symbol reachable from more files than the cap is likewise
# structural fan-in, not a task-specific completeness gap.
_MAX_CALLER_FILES = 6
_GENERIC_SYMBOLS = frozenset({
    "main", "run", "handle", "edit", "apply", "call", "invoke", "execute",
    "to_dict", "from_dict", "to_json", "from_json", "serialize", "deserialize",
    "dumps", "loads", "ok", "success", "size", "get", "set", "add", "remove",
    "update", "close", "open", "read", "write",
})


def _is_generic(sym: str) -> bool:
    # Leading underscore (incl. dunders) means module-private by convention:
    # its cross-module "callers" are name collisions, not real dependents.
    return sym.startswith("_") or sym in _GENERIC_SYMBOLS


def audit_completion(index, changed_symbols, changed_files, context_files) -> dict:
    """Completeness audit (advisory): files that reference a changed symbol but
    were neither edited nor part of the retrieved context.  Dunder, generic and
    high-fan-in symbols are skipped, since their reach says nothing about
    whether *this* task is complete."""
    changed = {_norm(p) for p in changed_files}
    ctx = {_norm(p) for p in context_files}
    affected: set[str] = set()
    skipped: list[str] = []
    for sym in changed_symbols:
        if _is_generic(sym):
            skipped.append(sym)
            continue
        callers = {_norm(p) for p in index.callers_of(sym) if p}
        if len(callers) > _MAX_CALLER_FILES:
            skipped.append(sym)
            continue
        affected.update(callers)
    return {
        "advisory": True,
        "changed_symbols": sorted(changed_symbols),
        "skipped_generic_symbols": sorted(set(skipped)),
        "affected_files": sorted(affected),
        "affected_not_edited": sorted(affected - changed - ctx),
    }
