"""MCP plan-document tools: create, edit, audit, and read server-owned plans."""

from __future__ import annotations

import json
import os

from .diffing import (
    ChangedSet,
    content_digest,
    git_changed,
    symbols_in,
    workspace_changed,
)
from .mcp_service import (
    _PLAN_DRIFT_LOGGED,
    _PLAN_DRIFT_LOGGED_MAX,
    _authorize_root,
    _index,
    _learned,
    _store,
)
from .models import drift_features
from .plan import PlanDocument, PlanEditor, audit_completion, parse_plan, scope_drift
from .plan_drift import audit_document, section_validation


# --------------------------------------------------------------------------- #
# Plan document tools (Workstream H).  The store is the source of truth; every
# mutation is root-authorized, containment-checked, depth-capped and versioned.
# --------------------------------------------------------------------------- #
def _plan_doc(root: str, task_id: str):
    root = _authorize_root(root)
    store = _store(root)
    raw = store.load_plan(root, task_id)
    if raw is None:
        raise KeyError(f"no plan for task {task_id!r}")
    return root, store, PlanDocument.from_dict(raw)


def _changed_set(root: str, changed_files: list[str] | None = None) -> ChangedSet:
    # Git is the source of truth when it is available. A caller-supplied list
    # is only a fallback for a non-git snapshot, never a way to hide changes
    # from the audit.
    try:
        git = git_changed(root)
        if git.git:
            return git
    except (OSError, ValueError):
        pass
    # Not a checkout itself: a workspace root with one repo per child still
    # yields a real changed set, so cross-repo drift needs no caller list.
    try:
        nested = workspace_changed(root)
        if nested.git:
            return nested
    except (OSError, ValueError):
        pass
    if changed_files is not None:
        return ChangedSet(modified=sorted(changed_files))
    return ChangedSet(git=False)


def _log_plan_drift(store, root: str, task_id: str, action: str,
                    version: int, drift: dict) -> None:
    """Append one bounded drift row, skipping an identical repeat."""

    payload = _ledger_drift(drift)
    key = (os.path.abspath(root), task_id, action, version,
           json.dumps(payload, sort_keys=True))
    if key in _PLAN_DRIFT_LOGGED:
        _PLAN_DRIFT_LOGGED.move_to_end(key)
        return
    _PLAN_DRIFT_LOGGED[key] = True
    while len(_PLAN_DRIFT_LOGGED) > _PLAN_DRIFT_LOGGED_MAX:
        _PLAN_DRIFT_LOGGED.popitem(last=False)
    store.append_ledger({
        "task_id": task_id, "component": "plan_drift",
        "revision": store.repo_revision(), "model_name": "plan-drift-v1",
        "reason_codes": [r["code"] for r in drift.get("blockers", [])],
        "result": {"action": action, "version": version, "drift": payload},
    })


def _ledger_drift(drift: dict | None) -> dict:
    """A bounded copy of an audit for the append-only ledger.

    The MCP response keeps the full report; the ledger keeps the verdict, the
    first few reasons, and counts. A 20k-file change set must not turn every
    audit into a megabyte row.
    """

    if not isinstance(drift, dict):
        return {}

    def trim(reason):
        if isinstance(reason, dict):
            return {**reason, "detail": str(reason.get("detail", ""))[:200]}
        return reason
    changed = drift.get("changed") or {}
    counts = {
        key: len(changed.get(key) or ())
        for key in ("added", "modified", "deleted", "renamed", "untracked")
    }
    return {
        "verdict": drift.get("verdict"),
        "blockers": [trim(r) for r in (drift.get("blockers") or [])[:50]],
        "warnings": [trim(r) for r in (drift.get("warnings") or [])[:50]],
        "missing": (drift.get("missing") or [])[:200],
        "extra": (drift.get("extra") or [])[:200],
        "unplanned_deletions": (drift.get("unplanned_deletions") or [])[:200],
        "undeclared_symbols": (drift.get("undeclared_symbols") or [])[:200],
        "changed": {"counts": counts, "git": changed.get("git"),
                    "base": changed.get("base"),
                    "error": changed.get("error", "")},
        "sections": (drift.get("sections") or [])[:200],
        "model": drift.get("model"),
    }


def _plan_symbols(root: str, doc, changed: ChangedSet) -> list[str] | None:
    if not any(section.symbols for section in doc.iter_sections()):
        return None
    try:
        return symbols_in(changed, _index(root))
    except (OSError, ValueError):
        return None


def _plan_ledger(store, task_id: str, revision: str, result: dict) -> None:
    store.append_ledger({"task_id": task_id, "component": "plan",
                         "revision": revision, "model_name": "",
                         "reason_codes": [], "result": result})


def _mutate_plan(root: str, task_id: str, fn, action: str = "mutate"):
    root, store, doc = _plan_doc(root, task_id)
    out = fn(PlanEditor(doc))
    store.save_plan(root, task_id, doc.to_dict())
    _plan_ledger(store, task_id, store.repo_revision(), {"action": action, "version": doc.version})
    return out


def plan_create(root: str, task_id: str, title: str, sections: list | None = None) -> dict:
    root = _authorize_root(root)
    editor = PlanEditor.create(root, task_id, title, sections or [])
    store = _store(root)
    changed = _changed_set(root)
    editor.doc.base_revision = store.repo_revision() or changed.base
    editor.doc.baseline_changes = changed.files()
    store.save_plan(root, task_id, editor.doc.to_dict())
    _plan_ledger(store, task_id, store.repo_revision(), {"action": "create"})
    return {"root": root, "task_id": task_id, "id": editor.doc.id,
            "version": editor.doc.version, "tree": editor.tree()}


def plan_append(root: str, task_id: str, section: dict) -> dict:
    out = _mutate_plan(root, task_id, lambda ed: ed.append(section), "append")
    return {"version": _plan_doc(root, task_id)[2].version, "section": out.to_dict()}


def plan_attach(root: str, task_id: str, parent_id: str, section: dict) -> dict:
    out = _mutate_plan(root, task_id, lambda ed: ed.attach(parent_id, section), "attach")
    return {"version": _plan_doc(root, task_id)[2].version, "section": out.to_dict()}


def plan_complete(root: str, task_id: str, section_id: str,
                  validation: dict | None = None) -> dict:
    root, store, doc = _plan_doc(root, task_id)
    changed = _changed_set(root)
    symbols = _plan_symbols(root, doc, changed)
    auto = section_validation(doc, section_id, changed, root, symbols=symbols)
    if isinstance(validation, dict) and str(validation.get("verdict")) != "pass":
        auto = {**auto, "verdict": "fail",
                "evidence": str(validation.get("evidence") or auto.get("evidence") or "")[:1000],
                "caller": {
                    "verdict": str(validation.get("verdict") or "")[:32],
                    "evidence": str(validation.get("evidence") or "")[:500],
                    "model": str(validation.get("model") or "")[:100],
                }}
    out = PlanEditor(doc).complete(section_id, auto)
    if out.get("done"):
        section = doc.find(section_id)[0]
        section.completed_changes = sorted(changed.files())
        section.completed_digest = content_digest(root, changed)
    store.save_plan(root, task_id, doc.to_dict())
    store.append_ledger({
        "task_id": task_id, "component": "plan",
        "revision": store.repo_revision(), "model_name": "plan-drift-v1",
        "reason_codes": [r["code"] for r in (auto.get("drift") or {}).get("blockers", [])],
        "result": {"action": "complete", "version": doc.version,
                   "drift": _ledger_drift(auto.get("drift")),
                   "validation": auto},
    })
    out["version"] = doc.version
    out["drift"] = auto.get("drift")
    return out


def plan_modify(root: str, task_id: str, section_id: str, fields: dict) -> dict:
    root, store, doc = _plan_doc(root, task_id)
    before = PlanEditor(doc).get(section_id)
    out = PlanEditor(doc).modify(section_id, fields)
    scope_keys = ("files", "allowed_paths", "symbols", "acceptance")
    after = out.to_dict()
    changed = [key for key in scope_keys if before.get(key) != after.get(key)]
    if changed:
        doc.scope_version += 1
    store.save_plan(root, task_id, doc.to_dict())
    store.append_ledger({
        "task_id": task_id, "component": "plan",
        "revision": store.repo_revision(), "model_name": "plan-drift-v1",
        "reason_codes": ["plan_scope_change"] if changed else [],
        "result": {"action": "modify", "version": doc.version,
                   "scope_fields": changed},
    })
    return {"version": doc.version, "section": after,
            "scope_fields": changed}


def plan_note(root: str, task_id: str, section_id: str, text: str) -> dict:
    out = _mutate_plan(root, task_id, lambda ed: ed.note(section_id, text), "note")
    return {"version": _plan_doc(root, task_id)[2].version, "section": out.to_dict()}


def plan_check(root: str, task_id: str, section_id: str, name: str,
               status: str, evidence: str = "") -> dict:
    out = _mutate_plan(root, task_id,
                       lambda ed: ed.check(section_id, name, status, evidence), "check")
    root2, store, doc = _plan_doc(root, task_id)
    changed = _changed_set(root2)
    drift = audit_document(doc, changed, root2,
                           symbols=_plan_symbols(root2, doc, changed))
    _log_plan_drift(store, root2, task_id, "check", doc.version, drift)
    return {"version": doc.version, "check": out, "drift": drift}


def plan_get(root: str, task_id: str, section_id: str | None = None) -> dict:
    _, _, doc = _plan_doc(root, task_id)
    return {"root": _authorize_root(root), "task_id": task_id,
            "version": doc.version, "plan": PlanEditor(doc).get(section_id)}


def plan_tree(root: str, task_id: str) -> dict:
    _, _, doc = _plan_doc(root, task_id)
    return {"root": _authorize_root(root), "task_id": task_id, "tree": PlanEditor(doc).tree()}


def plan_audit(root: str, plan_text: str = "", task: str = "",
               changed_files: list[str] | None = None,
               task_id: str | None = None) -> dict:
    """Plan audit.  With ``task_id`` it consumes the stored JSON document and
    reports its status rollup (no index, no retriever).  Without it, it falls
    back to parsing free-text plan text against the repo's real files."""
    if task_id:
        root, store, doc = _plan_doc(root, task_id)
        editor = PlanEditor(doc)
        changed = _changed_set(root, changed_files)
        symbols = _plan_symbols(root, doc, changed)
        drift = audit_document(doc, changed, root, symbols=symbols)
        try:
            if _store(root).load_model("learned") is None:
                raise LookupError("no learned model")
            learned = _learned(root)
            head = getattr(learned, "plan_drift", None)
            if head is not None:
                scores = [head.predict(drift_features(section, drift, doc.version))
                          for section in drift["sections"]]
                drift["model"] = {
                    "version": head.version,
                    "trained": bool(head.trained),
                    "max_score": round(max(scores), 4) if scores else 0.0,
                }
        except Exception:  # advisory only; never fail an audit on the head
            drift["model"] = {"version": "unavailable", "trained": False,
                              "max_score": 0.0}
        _log_plan_drift(store, root, task_id, "audit", doc.version, drift)
        response_drift = dict(drift)
        if len(response_drift.get("sections") or ()) > 500:
            response_drift["sections"] = response_drift["sections"][:500]
            response_drift["sections_truncated"] = True
        return {"root": root, "task_id": task_id, "advisory": True,
                "plan": editor.get(), "audit": editor.audit(),
                "gate": True, "drift": response_drift}
    idx = _index(root)
    changed = list(changed_files or [])
    plan = parse_plan(plan_text, set(idx.files))
    drift = scope_drift(plan, changed)
    changed_norm = {p.replace("\\", "/").lstrip("./") for p in changed}
    syms = sorted({c.symbol for c in idx.chunks
                   if c.path.replace("\\", "/").lstrip("./") in changed_norm and c.symbol})
    comp = audit_completion(idx, syms, changed, sorted(set(plan.all_files())))
    return {
        "root": idx.root, "revision": idx.revision, "advisory": True,
        "plan": plan.to_dict(), "scope_drift": drift.to_dict(),
        "completeness": comp,
    }


