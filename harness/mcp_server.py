"""RBG Harness MCP server facade (spec 3.3B, Phase 0 read-only surface).

A dependency-free, stateless JSON-RPC 2.0 server over stdio.  It exposes the
same internal service calls the native CLI uses -- retrieval, symbols, impact,
test ranking, diagnostics -- without adding protocol overhead to the harness's
own hot path (spec 3.3D).  Mutation tools (apply_patch) are intentionally not
implemented yet: Phase 0 defers write capability until the permission model is
proven (spec 12, Phase 0).

Every tool takes an explicit ``root`` (the authorized repository snapshot), so
a task survives a process restart by passing the same identifiers again
(spec 3.3E).  All reads are re-authorized against the filesystem each call.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
import time
import traceback

from . import db
from .index import RepoIndex, _tree_hash, _tree_signature, chunk_key
from .models import TestSelector
from .paths import resolve_within
from .plan import audit_completion, parse_plan, scope_drift
from .retrieval import HybridRetriever, tokenize
from .training import LearnedModels, read_ledger, train_from_ledger
from .verify import Sandbox

SERVER_NAME = "rbg-mcp"
SERVER_VERSION = "0.2.0"
DEFAULT_PROTOCOL = "2025-06-18"

_INDEX_CACHE: dict[str, RepoIndex] = {}
_INDEX_SIGNATURE: dict[str, str] = {}
_LEARNED_CACHE: dict[str, LearnedModels] = {}
_RETRIEVER_CACHE: dict[str, HybridRetriever] = {}

# Advisory read-path budget.  A retrieval result is still returned when the soft
# budget is exceeded, but it is reported as degraded (and the optional reranker
# is skipped) so a caller learns the answer was cheap rather than complete.
_READ_BUDGET_MS = int(os.environ.get("HARNESS_READ_BUDGET_MS", "1500"))


def _read_budget(timeout_ms: int | None = None) -> float:
    return float(_READ_BUDGET_MS if timeout_ms is None else max(0, int(timeout_ms)))


def _timings(t0: float, t_i: float, t_l: float, t_r: float, t_q: float,
             t_x: float | None = None) -> dict:
    def ms(a: float, b: float) -> float:
        return round((b - a) * 1000, 1)
    out = {"index_ms": ms(t0, t_i), "learned_ms": ms(t_i, t_l),
           "retriever_ms": ms(t_l, t_r), "retrieve_ms": ms(t_r, t_q)}
    if t_x is not None:
        out["expand_ms"] = ms(t_q, t_x)
    out["total_ms"] = ms(t0, t_x if t_x is not None else t_q)
    return out
_STORE_CACHE: dict[str, "db.Store | None"] = {}


def _store(root: str) -> "db.Store":
    """The repo's libSQL store (cached per repo)."""
    if root not in _STORE_CACHE:
        _STORE_CACHE[root] = db.Store(root)
    return _STORE_CACHE[root]


def _allowed_roots() -> list[str]:
    """The authorization allowlist from ``HARNESS_ALLOWED_ROOTS`` (os.pathsep
    separated).  Empty means nothing is authorized."""
    raw = os.environ.get("HARNESS_ALLOWED_ROOTS", "")
    return [os.path.realpath(os.path.expanduser(p))
            for p in raw.split(os.pathsep) if p.strip()]


def _authorize_root(root: str) -> str:
    """Resolve a caller-supplied root and require it under the allowlist.

    Deny by default: with no allowlist configured, no repository is authorized,
    so a tool call can only ever touch a path an operator explicitly named.
    """
    resolved = os.path.realpath(os.path.expanduser(root))
    if not os.path.isdir(resolved):
        raise ValueError(f"not a directory: {root}")
    allowed = _allowed_roots()
    if not allowed:
        raise PermissionError(
            "no repository is authorized: set HARNESS_ALLOWED_ROOTS to an "
            "os.pathsep-separated allowlist of repository roots")
    if not any(resolved == a or resolved.startswith(a + os.sep) for a in allowed):
        raise PermissionError(f"repository not allowed: {root}")
    return resolved


def _evict_dependents(root: str) -> None:
    """Drop per-repo derivatives keyed to a revision that is no longer current."""
    _RETRIEVER_CACHE.pop(root, None)
    for key in [k for k in _LEARNED_CACHE if k == root or k.startswith(root + ":")]:
        _LEARNED_CACHE.pop(key, None)


def _index(root: str) -> RepoIndex:
    root = _authorize_root(root)
    signature = _tree_signature(root)
    cached = _INDEX_CACHE.get(root)
    if cached is not None and _INDEX_SIGNATURE.get(root) == signature:
        return cached
    fingerprint = _tree_hash(root)
    store = _store(root)
    try:
        idx = store.load_index(fingerprint)  # cache hit: no re-parse
    except Exception:
        idx = None
    if idx is None:
        idx = RepoIndex.build(root, revision=fingerprint)
        store.save_index(idx)
    if cached is not None and cached.revision != idx.revision:
        _evict_dependents(root)  # stale index/retriever/models must not be served
    _INDEX_CACHE[root] = idx
    _INDEX_SIGNATURE[root] = signature
    return idx


def _retriever(root: str) -> HybridRetriever:
    """Cached per repo: building the LSA projector is not free.  The persisted
    encoder is reloaded from the DB (no re-fit); a fresh one is stored back."""
    idx = _index(root)
    if idx.root not in _RETRIEVER_CACHE:
        store = _store(idx.root)
        try:
            encoder = store.load_encoder(idx.revision, len(idx.chunks))
        except Exception:
            encoder = None
        if encoder is not None:
            retr = HybridRetriever(idx, encoder=encoder)
        else:
            retr = HybridRetriever(idx)
            store.save_encoder(idx.revision, retr.encoder.to_state(),
                               retr.encoder.document_vectors(idx.chunks))
        _RETRIEVER_CACHE[idx.root] = retr
    return _RETRIEVER_CACHE[idx.root]


def _learned(root: str, retrain: bool = False) -> LearnedModels:
    """Trained models for a repo.

    The read path never trains: it loads a model persisted for the current
    revision *and* ledger size (a cheap ``COUNT(*)``), or returns an empty set.
    Training happens only through the explicit ``train_models`` tool, so no
    retrieval or impact query ever pays to refit.
    """
    idx = _index(root)
    store = _store(idx.root)
    key = f"{idx.root}:{idx.revision}:{store.ledger_count()}"
    if retrain:
        models = train_from_ledger(idx.root, index=idx, store=store)
        for k in [k for k in _LEARNED_CACHE if k.startswith(idx.root + ":")]:
            _LEARNED_CACHE.pop(k, None)
        _LEARNED_CACHE[key] = models
        return models
    if key in _LEARNED_CACHE:
        return _LEARNED_CACHE[key]
    models = None
    try:
        got = store.load_model_at("learned", idx.revision)
        if got is not None and got[2].get("ledger_n") == store.ledger_count():
            models = got[0]
    except Exception:
        models = None
    _LEARNED_CACHE[key] = models or LearnedModels()
    return _LEARNED_CACHE[key]


# --------------------------------------------------------------------------- #
# Tool handlers.  Each returns a JSON-serialisable dict.
# --------------------------------------------------------------------------- #
def search_code(root: str, query: str, limit: int = 10, dense: bool = True,
                include: list[str] | None = None, exclude: list[str] | None = None,
                timeout_ms: int | None = None) -> dict:
    t0 = time.perf_counter()
    budget = _read_budget(timeout_ms)
    idx = _index(root)
    t_i = time.perf_counter()
    learned = _learned(root)
    t_l = time.perf_counter()
    retr = _retriever(root)
    t_r = time.perf_counter()
    reranker = (learned.reranker if learned and learned.reranker
                and (t_r - t0) * 1000 <= budget else None)
    retr.reranker = reranker
    hits = retr.retrieve(query, limit=int(limit), include=include, exclude=exclude)
    t_q = time.perf_counter()
    latency_ms = round((t_q - t0) * 1000, 1)
    return {
        "root": idx.root, "revision": idx.revision, "query": query,
        "index_fresh": _index_fresh(idx),
        "encoder": type(retr.encoder).__name__,
        "reranker": "trained" if reranker else ("off" if not (learned and learned.reranker) else "skipped_over_budget"),
        "latency_ms": latency_ms, "budget_ms": budget,
        "degraded": latency_ms > budget,
        "timings": _timings(t0, t_i, t_l, t_r, t_q),
        "results": [h.to_dict() for h in hits],
    }


def retrieve_context(root: str, query: str, limit: int = 8, expand: bool = True,
                     include: list[str] | None = None, exclude: list[str] | None = None,
                     timeout_ms: int | None = None) -> dict:
    t0 = time.perf_counter()
    budget = _read_budget(timeout_ms)
    idx = _index(root)
    t_i = time.perf_counter()
    learned = _learned(root)
    t_l = time.perf_counter()
    retr = _retriever(root)
    t_r = time.perf_counter()
    reranker = (learned.reranker if learned and learned.reranker
                and (t_r - t0) * 1000 <= budget else None)
    retr.reranker = reranker
    hits = retr.retrieve(query, limit=int(limit), include=include, exclude=exclude)
    t_q = time.perf_counter()
    if expand:
        hits = retr.expand(hits)[: int(limit)]
    t_x = time.perf_counter()
    tokens = sum(len(h.content) for h in hits) // 4
    latency_ms = round((t_x - t0) * 1000, 1)
    return {
        "root": idx.root, "revision": idx.revision, "query": query,
        "index_fresh": _index_fresh(idx),
        "context_token_estimate": tokens,
        "latency_ms": latency_ms, "budget_ms": budget,
        "degraded": latency_ms > budget,
        "timings": _timings(t0, t_i, t_l, t_r, t_q, t_x),
        "chunks": [h.to_dict() for h in hits],
    }


def read_chunk(root: str, chunk_id: str) -> dict:
    """Full text of one chunk by id (``path:start-end``), so discovery results
    can stay as capped snippets while the detail is one call away."""
    idx = _index(root)
    for c in idx.chunks:
        if chunk_key(c.path, c.start_line, c.end_line) == chunk_id:
            return {"root": idx.root, "revision": idx.revision, "chunk_id": chunk_id,
                    "path": c.path, "symbol": c.symbol, "kind": c.kind,
                    "start_line": c.start_line, "end_line": c.end_line,
                    "content": c.content, "content_hash": c.content_hash}
    raise ValueError(f"no such chunk: {chunk_id!r}")


def find_symbol(root: str, name: str, exact: bool = False,
                path: str | None = None, kind: str | None = None) -> dict:
    idx = _index(root)
    key = name.lower()
    defs = [
        {"path": s.path, "kind": s.kind, "start_line": s.start_line,
         "end_line": s.end_line,
         "chunk_id": chunk_key(s.path, s.start_line, s.end_line),
         "content_hash": next((c.content_hash for c in idx.chunks
                               if c.path == s.path and c.start_line == s.start_line), "")}
        for s in idx.symbols
        if (s.name == name if exact else key in s.name.lower())
        and (path is None or fnmatch.fnmatch(s.path, path))
        and (kind is None or s.kind == kind)
    ]
    return {
        "root": idx.root, "revision": idx.revision, "name": name,
        "definitions": defs, "referenced_in_files": idx.callers_of(name),
    }


def get_change_impact(root: str, symbols: list[str]) -> dict:
    idx = _index(root)
    learned = _learned(root)
    impact: dict[str, dict] = {}
    for sym in symbols:
        callers = idx.callers_of(sym)
        tests = [c for c in idx.chunks
                 if "test" in c.path and sym.lower() in set(tokenize(c.content))]
        entry = {
            "referenced_in_files": callers,
            "likely_tests": sorted({c.path for c in tests}),
            "fan_out": len(callers),
        }
        if learned.impact_trained:
            scored = [(learned.impact.predict(sym, c.path, set(tokenize(c.content)), len(callers)), c.path)
                      for c in tests]
            entry["predicted_affected"] = [
                {"test": p, "score": round(s, 4)}
                for s, p in sorted(scored, key=lambda kv: -kv[0])[:10]]
            entry["model"] = learned.impact.version
        impact[sym] = entry
    return {"root": idx.root, "revision": idx.revision, "advisory": True, "impact": impact}


def plan_audit(root: str, plan_text: str, task: str = "",
               changed_files: list[str] | None = None) -> dict:
    """Deterministic plan + step-level drift audit (advisory).

    Parses a free-text plan against the repo's real files, then reports the
    scope drift against ``changed_files`` and the completeness gap (callers of
    the changed symbols that were neither edited nor named in the plan).  No
    LLM, and deliberately no retriever: fitting the LSA projector here bought
    nothing the plan's own file list does not already supply.
    """
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


def train_models(root: str) -> dict:
    idx = _index(root)
    learned = _learned(root, retrain=True)
    return {
        "root": idx.root, "revision": idx.revision, "trained": True,
        "risk_trained": learned.risk_trained,
        "selector_trained": learned.selector_trained,
        "impact_trained": learned.impact_trained,
        "reranker_trained": learned.reranker is not None,
        "metrics": learned.metrics,
    }


def get_calibration(root: str) -> dict:
    idx = _index(root)
    learned = _learned(root)
    conf = learned.conformal
    return {
        "root": idx.root, "revision": idx.revision,
        "risk_trained": learned.risk_trained,
        "alpha": conf.alpha, "qhat": conf.qhat, "fitted": conf.fitted,
        "version": conf.version,
        "metrics": learned.metrics,
        "drift": learned.drift,
    }


def check_drift(root: str) -> dict:
    """Time-ordered drift/OOD check over the repo's execution ledger (spec 5.8).

    Retrains from the current ledger so the reference window is up to date, then
    reports feature PSI, calibration delta and OOD rate with a verdict."""
    idx = _index(root)
    learned = _learned(root, retrain=True)
    return {
        "root": idx.root, "revision": idx.revision, "advisory": True,
        "drift": learned.drift,
        "reference": learned.reference,
        "ood_threshold": learned.ood_threshold,
    }


def rank_tests(root: str, changed_symbols: list[str], limit: int = 20) -> dict:
    idx = _index(root)
    learned = _learned(root)
    selector = learned.selector if learned.selector_trained else TestSelector()
    defined = {s.name.lower() for s in idx.symbols}
    from .index import is_test_chunk
    tests = []
    for c in idx.chunks:
        if is_test_chunk(c):
            tests.append({"test_id": f"{c.path}::{c.symbol or c.path}", "duration": 1.0,
                          "history": [], "symbols": sorted(set(tokenize(c.content)) & defined)})
    ranked = selector.rank(tests, changed_symbols)[: int(limit)]
    return {
        "root": idx.root, "revision": idx.revision, "advisory": True,
        "ordering": "relevance desc; mandatory suites flagged, never skipped",
        "selector": selector.version,
        "tests": [r.to_dict() for r in ranked], "n_candidates": len(tests),
    }


def get_diagnostics(root: str, task_id: str = "") -> dict:
    idx = _index(root)
    entries = read_ledger(idx.root)
    traces = [e for e in entries if not task_id or e.get("task_id") == task_id]
    return {"root": idx.root, "task_id": task_id, "traces": traces}


def get_task_status(root: str, task_id: str) -> dict:
    diag = get_diagnostics(root, task_id)
    checks = []
    status = "unknown"
    for entry in diag["traces"]:
        if entry.get("component") == "orchestrator":
            checks = entry.get("result", {}).get("checks_run", [])
            status = entry.get("result", {}).get("status", "unknown")
    return {"root": diag["root"], "task_id": task_id, "status": status, "checks": checks}


_PROFILES = {
    "compile": ["-m", "py_compile", "."],
    "collect_tests": ["-m", "pytest", "--collect-only", "-q"],
    "tests": ["-m", "pytest", "-q", "-rA", "--no-header"],
}


def run_checks(root: str, profile: str = "compile", timeout_s: int = 120) -> dict:
    idx = _index(root)
    if profile not in _PROFILES:
        raise ValueError(f"unknown check profile: {profile!r}; allowed: {sorted(_PROFILES)}")
    sandbox = Sandbox(idx.root, timeout_s=min(int(timeout_s), 300))
    result = sandbox.run([sandbox.python, *_PROFILES[profile]])
    return {"root": idx.root, "revision": idx.revision, "profile": profile,
            "interpreter": sandbox.python, **result.to_dict()}


def db_stats(root: str) -> dict:
    """libSQL store statistics: what is persisted for this repo."""
    idx = _index(root)
    return {"root": idx.root, **_store(idx.root).stats()}


def vector_search(root: str, query: str, limit: int = 10) -> dict:
    """On-disk ANN search through the libsql vector index (query encoded with the
    repo's persisted encoder, scored by ``vector_distance_cos``)."""
    idx = _index(root)
    retr = _retriever(root)
    store = _store(idx.root)
    qv = retr.encoder.encode(query)
    rows = store.vector_search(qv, int(limit))
    return {"root": idx.root, "revision": idx.revision, "query": query,
            "index_fresh": _index_fresh(idx),
            "backend": "libsql_vector_idx", "encoder": type(retr.encoder).__name__,
            "results": [{"chunk_id": cid, "distance": round(d, 6),
                         "path": idx.chunks[cid - 1].path,
                         "start_line": idx.chunks[cid - 1].start_line}
                        for cid, d in rows if 1 <= cid <= len(idx.chunks)]}


def propose_patch(root: str, path: str, content: str) -> dict:
    """Validates a proposed patch but never writes it (spec 3.3B: proposal is
    not approval; apply_patch is deferred)."""
    idx = _index(root)
    full = resolve_within(idx.root, path)
    before = ""
    if os.path.exists(full):
        with open(full, encoding="utf-8", errors="replace") as fh:
            before = fh.read()
    import difflib
    diff = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), content.splitlines(keepends=True),
        fromfile=path, tofile=path))
    return {"root": idx.root, "path": path, "would_write": True,
            "applied": False, "diff": diff}


TOOLS = {
    "search_code": {
        "fn": search_code,
        "description": "Lexical/semantic/symbol search in an authorized repository snapshot.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string", "description": "repository path"},
            "query": {"type": "string"}, "limit": {"type": "integer", "default": 10},
            "include": {"type": "array", "items": {"type": "string"}, "description": "glob filters (paths must match any)"},
            "exclude": {"type": "array", "items": {"type": "string"}, "description": "glob filters to drop"},
            "timeout_ms": {"type": "integer", "description": "per-call soft latency budget (ms); over budget marks the result degraded"}},
            "required": ["root", "query"]},
    },
    "retrieve_context": {
        "fn": retrieve_context,
        "description": "Return a bounded, source-grounded context bundle with provenance.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "query": {"type": "string"},
            "limit": {"type": "integer", "default": 8},
            "expand": {"type": "boolean", "default": True},
            "include": {"type": "array", "items": {"type": "string"}, "description": "glob filters (paths must match any)"},
            "exclude": {"type": "array", "items": {"type": "string"}, "description": "glob filters to drop"},
            "timeout_ms": {"type": "integer", "description": "per-call soft latency budget (ms); over budget marks the result degraded"}},
            "required": ["root", "query"]},
    },
    "find_symbol": {
        "fn": find_symbol,
        "description": "Resolve a symbol and return its definition, references, and path.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "name": {"type": "string"},
            "exact": {"type": "boolean", "default": False},
            "path": {"type": "string", "description": "glob filter for definition paths"},
            "kind": {"type": "string", "description": "symbol kind (function/class/constant)"}},
            "required": ["root", "name"]},
    },
    "read_chunk": {
        "fn": read_chunk,
        "description": "Return the full text of one chunk by id (path:start-end).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"},
            "chunk_id": {"type": "string", "description": "chunk id, path:start-end"}},
            "required": ["root", "chunk_id"]},
    },
    "get_change_impact": {
        "fn": get_change_impact,
        "description": "Return predicted and graph-derived impact scope (advisory).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"},
            "symbols": {"type": "array", "items": {"type": "string"}}},
            "required": ["root", "symbols"]},
    },
    "rank_tests": {
        "fn": rank_tests,
        "description": "Rank candidate tests by regression-detection value, with evidence (advisory).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"},
            "changed_symbols": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "default": 20}},
            "required": ["root", "changed_symbols"]},
    },
    "run_checks": {
        "fn": run_checks,
        "description": "Run an approved check profile inside the sandbox (compile/collect_tests/tests).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "profile": {"type": "string", "default": "compile"},
            "timeout_s": {"type": "integer", "default": 120}},
            "required": ["root"]},
    },
    "get_diagnostics": {
        "fn": get_diagnostics,
        "description": "Return recorded compiler/test evidence for a task handle.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string", "default": ""}},
            "required": ["root"]},
    },
    "get_task_status": {
        "fn": get_task_status,
        "description": "Return status, verified gates, and recorded checks for a task handle.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"}},
            "required": ["root", "task_id"]},
    },
    "propose_patch": {
        "fn": propose_patch,
        "description": "Submit a patch proposal for validation (does NOT apply it).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["root", "path", "content"]},
    },
    "train_models": {
        "fn": train_models,
        "description": "Train the harness models (risk, selector, reranker, impact) from the ledger.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}}, "required": ["root"]},
    },
    "plan_audit": {
        "fn": plan_audit,
        "description": "Audit a free-text plan against the repo: scope drift + completeness (advisory).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "plan_text": {"type": "string"},
            "task": {"type": "string"},
            "changed_files": {"type": "array", "items": {"type": "string"}}},
            "required": ["root", "plan_text"]},
    },
    "get_calibration": {
        "fn": get_calibration,
        "description": "Return conformal calibration state and training metrics (advisory).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}}, "required": ["root"]},
    },
    "check_drift": {
        "fn": check_drift,
        "description": "Time-ordered drift/OOD check over the execution ledger (advisory).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}}, "required": ["root"]},
    },
    "db_stats": {
        "fn": db_stats,
        "description": "libSQL store statistics: persisted chunks, vectors, ledger and models.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}}, "required": ["root"]},
    },
    "vector_search": {
        "fn": vector_search,
        "description": "On-disk ANN search through the libSQL vector index.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}},
            "required": ["root", "query"]},
    },
}

MAX_OUTPUT_BYTES = 512 * 1024

# MCP resource surface (spec 3.3C): read-only views over a repo's store.  The
# repo is passed as a query parameter so one server serves every snapshot.
RESOURCE_TEMPLATES = [
    {"uriTemplate": "harness://repo/ledger{?root}", "name": "Execution ledger",
     "description": "Append-only task/trace ledger for a repository.",
     "mimeType": "application/json"},
    {"uriTemplate": "harness://repo/models{?root}", "name": "Learned models",
     "description": "Training metrics and calibration state for the repo's models.",
     "mimeType": "application/json"},
    {"uriTemplate": "harness://repo/db{?root}", "name": "Store statistics",
     "description": "libSQL store counts: chunks, vectors, symbols, ledger, models.",
     "mimeType": "application/json"},
    {"uriTemplate": "harness://repo/index{?root}", "name": "Repository index",
     "description": "Files, chunks and symbols for the current revision.",
     "mimeType": "application/json"},
]

PROMPTS = {
    "grounded_change": {
        "name": "grounded_change",
        "description": "Plan a repo change using retrieved, source-grounded context.",
        "arguments": [{"name": "root", "required": True},
                      {"name": "task", "required": True}],
    },
}

_RESOURCE_FNS = {
    "/ledger": lambda root: get_diagnostics(root),
    "/models": lambda root: get_calibration(root),
    "/db": lambda root: db_stats(root),
    "/index": lambda root: _index_summary(root),
}


def _index_fresh(idx: RepoIndex) -> bool:
    """Whether the cached index still matches the on-disk tree (spec 5.2: never
    present stale content as current)."""
    try:
        return _tree_signature(idx.root) == _INDEX_SIGNATURE.get(idx.root)
    except OSError:
        return False


def _index_summary(root: str) -> dict:
    idx = _index(root)
    return {"root": idx.root, "revision": idx.revision, "files": idx.files,
            "chunks": len(idx.chunks), "symbols": [s.name for s in idx.symbols]}


def _read_resource(uri: str) -> dict:
    from urllib.parse import parse_qs, urlparse
    parsed = urlparse(uri)
    root = (parse_qs(parsed.query).get("root") or [""])[0]
    fn = _RESOURCE_FNS.get(parsed.path)
    if fn is None:
        raise ValueError(f"unknown resource: {uri}")
    if not root:
        raise ValueError("resource requires a ?root= repository path")
    return {"uri": uri, "mimeType": "application/json",
            "text": json.dumps(fn(root), default=str)[:MAX_OUTPUT_BYTES]}



# --------------------------------------------------------------------------- #
# JSON-RPC 2.0 over stdio
# --------------------------------------------------------------------------- #
def _result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    msg_id = msg.get("id")
    if msg_id is None and method and method.startswith("notifications/"):
        return None
    if method == "initialize":
        params = msg.get("params") or {}
        proto = params.get("protocolVersion") or DEFAULT_PROTOCOL
        return _result(msg_id, {
            "protocolVersion": proto,
            "capabilities": {"tools": {"listChanged": False},
                             "resources": {"listChanged": False},
                             "prompts": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    if method == "ping":
        return _result(msg_id, {})
    if method in ("tools/list", "list_tools"):
        return _result(msg_id, {"tools": [
            {"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]}
            for name, spec in TOOLS.items()]})
    if method in ("tools/call", "call_tool"):
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        spec = TOOLS.get(name)
        if not spec:
            return _error(msg_id, -32602, f"unknown tool: {name}")
        try:
            out = spec["fn"](**args)
            text = json.dumps(out, default=str)[:MAX_OUTPUT_BYTES]
            return _result(msg_id, {"content": [{"type": "text", "text": text}],
                                    "structuredContent": out, "isError": False})
        except Exception as exc:  # noqa: BLE001 - reported as a tool error, never a crash
            tb = traceback.format_exc() if os.environ.get("HARNESS_MCP_DEBUG") else ""
            return _result(msg_id, {"content": [{"type": "text",
                            "text": f"{type(exc).__name__}: {exc}\n{tb}"}], "isError": True})
    if method in ("resources/list", "list_resources"):
        return _result(msg_id, {"resources": []})
    if method in ("resources/templates/list", "list_resource_templates"):
        return _result(msg_id, {"resourceTemplates": RESOURCE_TEMPLATES})
    if method in ("resources/read", "read_resource"):
        params = msg.get("params") or {}
        uri = params.get("uri", "")
        try:
            return _result(msg_id, {"contents": [_read_resource(uri)]})
        except Exception as exc:  # noqa: BLE001
            return _error(msg_id, -32602, f"{type(exc).__name__}: {exc}")
    if method in ("prompts/list", "list_prompts"):
        return _result(msg_id, {"prompts": list(PROMPTS.values())})
    if method in ("prompts/get", "get_prompt"):
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name not in PROMPTS:
            return _error(msg_id, -32602, f"unknown prompt: {name}")
        root = args.get("root", "")
        task = args.get("task", "")
        text = (f"Use the rbg-mcp tools for repository {root!r}. "
                f"First call retrieve_context for the task below, then edit only "
                f"the retrieved files, then run_checks('compile') and "
                f"run_checks('tests').\n\nTask: {task}")
        return _result(msg_id, {"description": PROMPTS[name]["description"],
                                "messages": [{"role": "user",
                                              "content": {"type": "text", "text": text}}]})
    return _error(msg_id, -32601, f"method not found: {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
