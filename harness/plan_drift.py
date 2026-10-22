"""Deterministic plan drift: what the plan declared versus what changed.

This is the hard gate. The semantic model in :mod:`harness.models` is advisory
and runs after these checks; it never overrides a proven scope error. Every
finding carries a code and the section that owns it so the repair loop can
feed back a concrete instruction rather than a score.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field

from .diffing import ChangedSet, _norm, content_digest

#: Paths the harness writes itself. They are never plan drift.
_INTERNAL_PREFIXES = (".harness/", "__pycache__/", ".pytest_cache/",
                      ".git/", ".venv/", "node_modules/")

#: Bounds: a plan audit is not a whole-disk scan. Exceeding either cap blocks
#: rather than truncating silently, because a truncated scope is not evidence.
MAX_CHANGED_FILES = 20000
MAX_PLAN_SECTIONS = 5000


def _internal(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _INTERNAL_PREFIXES)


def _matches(path: str, patterns) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


@dataclass
class Reason:
    code: str
    detail: str
    section_id: str = ""

    def to_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail,
                "section_id": self.section_id}


@dataclass
class SectionAudit:
    section_id: str
    title: str
    status: str
    declared_files: list[str] = field(default_factory=list)
    declared_symbols: list[str] = field(default_factory=list)
    acceptance: list = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    outside_scope: list[str] = field(default_factory=list)
    acceptance_failures: list[str] = field(default_factory=list)
    undeclared_symbols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id, "title": self.title,
            "status": self.status, "declared_files": self.declared_files,
            "declared_symbols": self.declared_symbols,
            "acceptance": self.acceptance,
            "missing": self.missing, "outside_scope": self.outside_scope,
            "acceptance_failures": self.acceptance_failures,
            "undeclared_symbols": self.undeclared_symbols,
            "warnings": self.warnings,
        }


def _check_passed(section, item: dict) -> tuple[bool, str]:
    """True when an acceptance item has passing evidence.

    An item may not certify itself with a ``status`` field: the verdict comes
    from a recorded check, which the audit can attribute and the ledger keeps.
    """

    name = str(item.get("name") or item.get("id") or "")
    for check in section.checks or ():
        if not isinstance(check, dict):
            continue
        if str(check.get("name")) == name:
            status = str(check.get("status") or "")
            return status == "pass", status or "unrecorded"
    return False, "not run"


def _artifact_exists(root: str, path: str) -> bool:
    """True for a regular file or directory that resolves inside *root*."""

    if not path or os.path.isabs(path):
        return False
    normalized = os.path.normpath(path)
    if normalized == ".." or normalized.startswith(".." + os.sep):
        return False
    try:
        base = os.path.realpath(root)
        target = os.path.realpath(os.path.join(base, path))
    except (OSError, ValueError):
        return False
    if target != base and not target.startswith(base + os.sep):
        return False
    return os.path.exists(target)


def _acceptance_failures(section, root: str) -> list[str]:
    failures = []
    for item in section.acceptance or ():
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if not kind:
            kind = "artifact" if item.get("path") else (
                "command" if item.get("command") else "check")
        label = str(item.get("id") or item.get("name") or item.get("path")
                    or item.get("command") or kind)
        if kind == "artifact":
            path = item.get("path")
            if not path:
                failures.append(f"{label}: artifact has no path")
            elif not _artifact_exists(root, str(path)):
                failures.append(f"{label}: artifact missing")
            continue
        if kind == "command":
            name = str(item.get("name") or f"cmd:{item.get('command', '')}")
            passed, status = _check_passed(section, {**item, "name": name})
            if not passed:
                failures.append(f"{label}: command {status}")
            continue
        passed, status = _check_passed(section, item)
        if not passed:
            failures.append(f"{label}: check {status}")
    return failures


def audit_document(doc, changed: ChangedSet, root: str,
                   symbols: list[str] | None = None) -> dict:
    """Audit every section and the document-level scope."""

    baseline = {_norm(str(p)) for p in (doc.baseline_changes or ())}
    observed = [p for p in changed.files() if not _internal(p)]
    deleted = [p for p in changed.deleted if not _internal(p)]
    new_files = [p for p in observed if p not in baseline]
    new_deleted = [p for p in deleted if p not in baseline]
    declared_all: set[str] = set()
    allowed: list[str] = []
    all_symbols: set[str] = set()
    section_rows: list[SectionAudit] = []
    blockers: list[Reason] = []
    warnings: list[Reason] = []

    digest = content_digest(root, changed) if any(
        s.completed_digest for s in doc.iter_sections()
    ) else ""
    sections = list(doc.iter_sections())
    if len(sections) > MAX_PLAN_SECTIONS:
        blockers.append(Reason(
            "plan_too_large",
            f"{len(sections)} sections exceeds the audit cap "
            f"({MAX_PLAN_SECTIONS})",
        ))
        sections = sections[:MAX_PLAN_SECTIONS]
    for section in sections:
        declared = [p for p in section.declared_files() if not _internal(p)]
        allowed.extend(str(p) for p in section.allowed_paths or ())
        declared_all.update(declared)
        all_symbols.update(section.declared_symbols())
        row = SectionAudit(section.id, section.title, section.status, declared)
        row.declared_symbols = sorted(section.declared_symbols())
        row.acceptance = list(section.acceptance or ())
        row.missing = sorted(set(declared) - set(observed))
        row.acceptance_failures = _acceptance_failures(section, root)
        if not declared and not section.acceptance and not section.artifacts:
            row.warnings.append("no_declared_scope")
            warnings.append(Reason(
                "no_declared_scope",
                f"{section.title or section.id} declares no files, artifacts, "
                "or acceptance; drift cannot be proven",
                section.id,
            ))
        done = section.status == "done"
        for path in row.missing:
            reason = Reason(
                "planned_file_unchanged" if done else "planned_file_pending",
                f"{path} was planned but is unchanged",
                section.id,
            )
            (blockers if done else warnings).append(reason)
        for failure in row.acceptance_failures:
            reason = Reason(
                "acceptance_failed" if done else "acceptance_pending",
                failure,
                section.id,
            )
            (blockers if done else warnings).append(reason)
        if done:
            previous = set(section.completed_changes or ())
            after = sorted(set(observed) - previous)
            if previous and after:
                warnings.append(Reason(
                    "changed_after_completion",
                    "files changed after the section was completed: "
                    + ", ".join(after[:10]),
                    section.id,
                ))
            elif section.completed_digest and digest != section.completed_digest:
                warnings.append(Reason(
                    "changed_after_completion",
                    "the recorded content changed after the section was completed",
                    section.id,
                ))
            verdict = (section.validation or {}).get("verdict")
            if verdict != "pass":
                blockers.append(Reason(
                    "done_without_validation",
                    f"{section.title or section.id} is done without a passing validation",
                    section.id,
                ))
        section_rows.append(row)

    undeclared_symbols = sorted(set(symbols or ()) - all_symbols)
    if undeclared_symbols:
        warnings.append(Reason(
            "undeclared_symbol",
            "symbols changed but are not declared: " + ", ".join(undeclared_symbols),
        ))
    if len(observed) > MAX_CHANGED_FILES:
        blockers.append(Reason(
            "changed_too_large",
            f"{len(observed)} changed files exceeds the audit cap "
            f"({MAX_CHANGED_FILES})",
        ))
    if getattr(doc, "scope_version", 0):
        warnings.append(Reason(
            "plan_scope_changed",
            f"the plan scope was edited {doc.scope_version} time(s) after creation",
        ))
    if changed.error:
        if changed.error == "no commit to diff against":
            warnings.append(Reason("change_set_incomplete", changed.error))
        else:
            blockers.append(Reason("change_set_unavailable", changed.error))
    extra = sorted(
        p for p in new_files
        if p not in declared_all and not _matches(p, allowed)
    )
    for path in extra:
        blockers.append(Reason("unplanned_file", f"{path} changed but is not planned"))
    for path in new_deleted:
        if path not in declared_all:
            blockers.append(Reason("unplanned_deletion",
                                   f"{path} was deleted but is not planned"))
    if not changed.git:
        warnings.append(Reason(
            "not_a_git_checkout",
            "the root is not a git checkout; the changed set is empty",
        ))
    elif not observed and not deleted:
        warnings.append(Reason(
            "no_changes_seen",
            "the working tree matches the base revision",
        ))

    verdict = "block" if blockers else ("warn" if warnings else "ok")
    return {
        "verdict": verdict,
        "blockers": [r.to_dict() for r in blockers],
        "warnings": [r.to_dict() for r in warnings],
        "missing": sorted({p for row in section_rows for p in row.missing}),
        "extra": extra,
        "unplanned_deletions": [p for p in new_deleted if p not in declared_all],
        "undeclared_symbols": undeclared_symbols,
        "changed": changed.to_dict(),
        "sections": [row.to_dict() for row in section_rows],
    }


def section_validation(doc, section_id: str, changed: ChangedSet, root: str,
                       symbols: list[str] | None = None) -> dict:
    """Validation object for ``PlanEditor.complete``.

    A document-level blocker fails every section: an unplanned write is not
    excused because the section that owns it is not the one being completed.
    """

    report = audit_document(doc, changed, root, symbols=symbols)
    compact = {k: v for k, v in report.items() if k not in ("sections", "changed")}
    section = next((s for s in doc.iter_sections() if s.id == section_id), None)
    if section is None:
        return {"verdict": "fail", "by": "plan-drift-v1",
                "model": "plan-drift-v1",
                "evidence": f"unknown section {section_id}"}
    row = next((r for r in report["sections"] if r["section_id"] == section_id), {})
    pending = list(row.get("missing") or ()) + list(row.get("acceptance_failures") or ())
    if report["verdict"] == "block" or pending:
        details = "; ".join(
            r["detail"] for r in report["blockers"]
            if not r.get("section_id") or r.get("section_id") == section_id
        ) or "; ".join(
            [f"{path} was planned but is unchanged" for path in row.get("missing") or ()]
            + list(row.get("acceptance_failures") or ())
        ) or "; ".join(r["detail"] for r in report["blockers"])
        return {"verdict": "fail", "by": "plan-drift-v1",
                "model": "plan-drift-v1", "evidence": details,
                "drift": compact}
    evidence = "scope and acceptance passed"
    if report["verdict"] == "warn":
        evidence += "; warnings: " + "; ".join(
            r["detail"] for r in report["warnings"]
        )
    return {"verdict": "pass", "by": "plan-drift-v1",
            "model": "plan-drift-v1", "evidence": evidence, "drift": compact}
