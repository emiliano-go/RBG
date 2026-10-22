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
import time
import uuid
from dataclasses import dataclass, field

# A leading list marker: "- ", "* ", "1. ", "1) ", "Step 3: ".
_MARKER = re.compile(r"^\s*(?:[-*]|\d+[.)]|step\s+\d+\s*[:.)]?)\s+", re.I)
_VERIFY = re.compile(r"\b(run|test|verify|check|validate|confirm|ensure|assert|re-?run)\b", re.I)
_EDIT = re.compile(r"\b(edit|add|update|modify|change|create|write|implement|fix|remove|delete|rename|refactor)\b", re.I)
# Read-only steps name files they merely look at; those are not commitments to
# change the file, so they must not become "missing" drift.
_READ = re.compile(r"\b(inspect|read|review|examine|locate|find|search|trace|understand)\b", re.I)
# Backticked spans, or a bare token that looks like a filename/path.
_TOKEN = re.compile(r"`([^`]+)`|([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9_]+)")


def _norm(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _bounded_list(value, limit: int) -> list:
    """A list field from untrusted JSON, bounded and type-filtered."""

    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value[:limit] if isinstance(item, str)]


def _bounded_objects(value, limit: int) -> list:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value[:limit] if isinstance(item, dict)]


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
        if (_VERIFY.search(desc) or _READ.search(desc)) and not _EDIT.search(desc):
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


# --------------------------------------------------------------------------- #
# JSON plan document + editor (Workstream H).  The plan is server-side state
# addressed by (root, task_id); it is never a file an assistant rewrites.
# --------------------------------------------------------------------------- #
DEPTH_CAP = 5
STATUS = ("open", "done", "blocked", "dropped")
MAX_SECTIONS = 5000
MAX_CHECKS = 200
MAX_NOTES = 500


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class PlanSection:
    id: str
    title: str
    body: str = ""
    status: str = "open"
    notes: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    validation: dict | None = None
    # Machine-checkable scope. A section without these fields is still
    # readable, but its drift cannot be proven and the audit says so.
    files: list = field(default_factory=list)
    allowed_paths: list = field(default_factory=list)
    symbols: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    acceptance: list = field(default_factory=list)
    intent: str = ""
    non_goals: list = field(default_factory=list)
    completed_changes: list = field(default_factory=list)
    completed_digest: str = ""
    children: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanSection":
        sec = cls(
            id=str(d.get("id") or uuid.uuid4().hex[:12]),
            title=str(d.get("title") or "")[:200],
            body=str(d.get("body") or "")[:20000],
            status=d.get("status") or "open",
            notes=_bounded_objects(d.get("notes"), MAX_NOTES),
            checks=_bounded_objects(d.get("checks"), MAX_CHECKS),
            validation=d.get("validation"),
            files=_bounded_list(d.get("files"), 1000),
            allowed_paths=_bounded_list(d.get("allowed_paths"), 1000),
            symbols=_bounded_list(d.get("symbols"), 5000),
            artifacts=_bounded_list(d.get("artifacts"), 1000),
            acceptance=_bounded_objects(d.get("acceptance"), 50),
            intent=str(d.get("intent") or "")[:4000],
            non_goals=_bounded_list(d.get("non_goals"), 200),
            completed_changes=_bounded_list(d.get("completed_changes"), 20000),
            children=[cls.from_dict(c) for c in (d.get("children") or [])],
        )
        if sec.status not in STATUS:
            sec.status = "open"
        return sec

    def declared_files(self) -> list[str]:
        out = set()
        for value in self.files:
            path = _norm(str(value).strip())
            if (not path or path.startswith("/") or path == ".."
                    or path.startswith("../")):
                continue
            out.add(path)
        return sorted(out)

    def declared_symbols(self) -> set[str]:
        return {str(s) for s in self.symbols if str(s).strip()}

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "body": self.body,
            "status": self.status, "notes": list(self.notes),
            "checks": list(self.checks), "validation": self.validation,
            "files": list(self.files), "allowed_paths": list(self.allowed_paths),
            "symbols": list(self.symbols), "artifacts": list(self.artifacts),
            "acceptance": list(self.acceptance), "intent": self.intent,
            "non_goals": list(self.non_goals),
            "completed_changes": list(self.completed_changes),
            "completed_digest": self.completed_digest,
            "children": [c.to_dict() for c in self.children],
        }

    def _find_in(self, root: "PlanSection", section_id: str):
        stack = [(root, 0, None)]
        while stack:
            node, depth, parent = stack.pop()
            if node.id == section_id:
                return node, depth + 1, parent
            for c in node.children:
                stack.append((c, depth + 1, node.children))
        return None

    def add_note(self, text: str) -> None:
        self.notes.append({"ts": _now(), "text": str(text)[:4000]})
        del self.notes[:-MAX_NOTES]


@dataclass
class PlanDocument:
    id: str
    root: str
    task_id: str
    version: int = 1
    title: str = ""
    base_revision: str = ""
    baseline_changes: list = field(default_factory=list)
    scope_version: int = 0
    sections: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanDocument":
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex[:12]),
            root=str(d.get("root") or ""), task_id=str(d.get("task_id") or ""),
            version=int(d.get("version") or 1), title=str(d.get("title") or ""),
            base_revision=str(d.get("base_revision") or ""),
            baseline_changes=_bounded_list(d.get("baseline_changes"), 20000),
            scope_version=int(d.get("scope_version") or 0),
            sections=[PlanSection.from_dict(s) for s in (d.get("sections") or [])],
        )

    def to_dict(self) -> dict:
        return {"id": self.id, "root": self.root, "task_id": self.task_id,
                "version": self.version, "title": self.title,
                "base_revision": self.base_revision,
                "baseline_changes": list(self.baseline_changes),
                "scope_version": self.scope_version,
                "sections": [s.to_dict() for s in self.sections]}

    def iter_sections(self):
        def walk(sec):
            yield sec
            for child in sec.children:
                yield from walk(child)
        for section in self.sections:
            yield from walk(section)

    def find(self, section_id: str):
        for s in self.sections:
            got = s._find_in(s, section_id)
            if got:
                return got
        return None


def effective_status(sec: PlanSection) -> str:
    """A parent is ``done`` only when every descendant is ``done``."""
    if sec.status != "done":
        return sec.status
    if not sec.children:
        return "done"
    effs = [effective_status(c) for c in sec.children]
    if all(e == "done" for e in effs):
        return "done"
    if any(e == "blocked" for e in effs):
        return "blocked"
    return "open"


class PlanEditor:
    """Bounded, versioned mutations over a :class:`PlanDocument`.

    ``validator`` is an optional callable ``(section) -> {verdict, evidence}``
    used by :meth:`complete`; when it is absent a caller must pass an explicit
    ``validation`` object.  Either way ``done`` is refused without a passing
    verdict -- the gate is a constraint, not advice.
    """

    def __init__(self, doc: PlanDocument, validator=None) -> None:
        self.doc = doc
        self.validator = validator

    @classmethod
    def create(cls, root: str, task_id: str, title: str, sections=(), validator=None):
        doc = PlanDocument(uuid.uuid4().hex[:12], root, task_id, title=str(title))
        ed = cls(doc, validator)
        for s in sections:
            ed.append(s)
        return ed

    def _bump(self, section_id: str, what: str) -> None:
        self.doc.version += 1
        got = self.doc.find(section_id)
        if got:
            got[0].add_note(f"v{self.doc.version}: {what}")

    def _section_count(self) -> int:
        return sum(1 for _ in self.doc.iter_sections())

    def append(self, section: dict) -> PlanSection:
        if self._section_count() >= MAX_SECTIONS:
            raise ValueError(f"section cap {MAX_SECTIONS} reached")
        sec = PlanSection.from_dict(section)
        sec.status = "open"
        self.doc.sections.append(sec)
        sec.add_note(f"v{self.doc.version + 1}: added")
        self.doc.version += 1
        return sec

    def attach(self, parent_id: str, section: dict) -> PlanSection:
        got = self.doc.find(parent_id)
        if not got:
            raise KeyError(f"unknown section: {parent_id}")
        _, depth, _ = got
        if depth + 1 > DEPTH_CAP:
            raise ValueError(f"depth cap {DEPTH_CAP} exceeded at section {parent_id}")
        if self._section_count() >= MAX_SECTIONS:
            raise ValueError(f"section cap {MAX_SECTIONS} reached")
        sec = PlanSection.from_dict(section)
        sec.status = "open"
        got[0].children.append(sec)
        sec.add_note(f"v{self.doc.version + 1}: attached")
        self.doc.version += 1
        return sec

    def complete(self, section_id: str, validation: dict | None = None) -> dict:
        got = self.doc.find(section_id)
        if not got:
            raise KeyError(f"unknown section: {section_id}")
        sec = got[0]
        verdict = self._validate(sec, validation)
        sec.validation = verdict
        if verdict.get("verdict") == "pass":
            sec.status = "done"
            self._bump(section_id, "completed (validated)")
            return {"id": section_id, "status": "done", "done": True, "validation": verdict}
        if sec.status not in ("blocked", "dropped"):
            sec.status = "open"
        self._bump(section_id, "complete rejected (no passing validation)")
        return {"id": section_id, "status": sec.status, "done": False, "validation": verdict}

    def _validate(self, sec: PlanSection, validation: dict | None) -> dict:
        if validation is None and self.validator is not None:
            validation = self.validator(sec)
        if validation is None:
            return {"by": "none", "verdict": "fail",
                    "evidence": "no model validation supplied"}
        v = dict(validation)
        v.setdefault("by", "model")
        v.setdefault("model", "")
        v.setdefault("verdict", "fail")
        v.setdefault("evidence", "")
        return v

    def modify(self, section_id: str, fields: dict) -> PlanSection:
        got = self.doc.find(section_id)
        if not got:
            raise KeyError(f"unknown section: {section_id}")
        sec = got[0]
        if "title" in fields:
            sec.title = str(fields["title"])
        if "body" in fields:
            sec.body = str(fields["body"])
        if "status" in fields and fields["status"] in STATUS:
            if fields["status"] == "done":
                raise ValueError("use plan_complete; status cannot be set to done directly")
            sec.status = fields["status"]
        if "files" in fields:
            sec.files = _bounded_list(fields["files"], 1000)
        if "allowed_paths" in fields:
            sec.allowed_paths = _bounded_list(fields["allowed_paths"], 1000)
        if "symbols" in fields:
            sec.symbols = _bounded_list(fields["symbols"], 5000)
        if "artifacts" in fields:
            sec.artifacts = _bounded_list(fields["artifacts"], 1000)
        if "acceptance" in fields:
            sec.acceptance = _bounded_objects(fields["acceptance"], 50)
        if "intent" in fields:
            sec.intent = str(fields["intent"] or "")[:4000]
        if "non_goals" in fields:
            sec.non_goals = _bounded_list(fields["non_goals"], 200)
        self._bump(section_id, "modified")
        return sec

    def note(self, section_id: str, text: str) -> PlanSection:
        got = self.doc.find(section_id)
        if not got:
            raise KeyError(f"unknown section: {section_id}")
        got[0].add_note(text)
        self.doc.version += 1
        return got[0]

    def check(self, section_id: str, name: str, status: str, evidence: str = "") -> dict:
        got = self.doc.find(section_id)
        if not got:
            raise KeyError(f"unknown section: {section_id}")
        entry = {"name": str(name)[:200], "status": str(status)[:32],
                 "evidence": str(evidence)[:4000]}
        got[0].checks.append(entry)
        del got[0].checks[:-MAX_CHECKS]
        self._bump(section_id, f"check {name}: {status}")
        return entry

    def get(self, section_id: str | None = None) -> dict:
        if section_id is None:
            return self.doc.to_dict()
        got = self.doc.find(section_id)
        if not got:
            raise KeyError(f"unknown section: {section_id}")
        return got[0].to_dict()

    def tree(self) -> dict:
        def node(sec: PlanSection, depth: int) -> dict:
            return {"id": sec.id, "title": sec.title, "status": sec.status,
                    "effective": effective_status(sec), "depth": depth,
                    "children": [node(c, depth + 1) for c in sec.children]}
        return {"id": self.doc.id, "version": self.doc.version, "title": self.doc.title,
                "sections": [node(s, 1) for s in self.doc.sections]}

    def audit(self) -> dict:
        rows = []

        def visit(sec: PlanSection, depth: int):
            rows.append({"id": sec.id, "title": sec.title, "status": sec.status,
                         "effective": effective_status(sec), "depth": depth})
            for c in sec.children:
                visit(c, depth + 1)
        for s in self.doc.sections:
            visit(s, 1)
        return {"version": self.doc.version, "sections": rows}
