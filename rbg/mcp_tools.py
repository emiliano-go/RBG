"""MCP tool implementations (retrieval, diagnostics, checks, patches)."""

from __future__ import annotations

import fnmatch
import os
import time

from .checks import compile_command, detect, runner_argv
from .index import chunk_key
from .mcp_service import (
    _LEARNED_CACHE,
    _RETRIEVER_CACHE,
    _build_retriever,
    _chunk_tokens,
    _index,
    _index_fresh,
    _learned,
    _read_budget,
    _result_cached,
    _retriever,
    _store,
    _timings,
)
from .models import TestSelector
from .paths import resolve_within
from .training import PROMOTION_KEY, read_ledger, validate_and_promote
from .verify import Sandbox


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
    key = (idx.root, idx.revision, "search", query, int(limit),
           tuple(include or ()), tuple(exclude or ()), bool(reranker))
    results = _result_cached(key, lambda: [h.to_dict() for h in
                                           retr.retrieve(query, limit=int(limit),
                                                         include=include, exclude=exclude)])
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
        "results": [dict(d) for d in results],
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

    def _build_context() -> list[dict]:
        hs = retr.retrieve(query, limit=int(limit), include=include, exclude=exclude)
        if expand:
            hs = retr.expand(hs)[: int(limit)]
        return [h.to_dict() for h in hs]

    key = (idx.root, idx.revision, "context", query, int(limit),
           tuple(include or ()), tuple(exclude or ()), bool(reranker), bool(expand))
    chunks = _result_cached(key, _build_context)
    t_x = time.perf_counter()
    tokens = sum(int(d.get("content_chars", 0)) for d in chunks) // 4
    latency_ms = round((t_x - t0) * 1000, 1)
    return {
        "root": idx.root, "revision": idx.revision, "query": query,
        "index_fresh": _index_fresh(idx),
        "context_token_estimate": tokens,
        "latency_ms": latency_ms, "budget_ms": budget,
        "degraded": latency_ms > budget,
        "timings": _timings(t0, t_i, t_l, t_r, t_x),
        "chunks": [dict(d) for d in chunks],
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
    # Tokenize each test chunk once, not once per requested symbol: this was the
    # per-symbol O(chunks x tokens) rescan that dominated the tool's latency.
    test_chunks = [(c, set(_chunk_tokens(c)))
                   for c in idx.chunks if "test" in c.path]
    impact: dict[str, dict] = {}
    for sym in symbols:
        callers = idx.callers_of(sym)
        low = sym.lower()
        tests = [(c, toks) for c, toks in test_chunks if low in toks]
        entry = {
            "referenced_in_files": callers,
            "likely_tests": sorted({c.path for c, _ in tests}),
            "fan_out": len(callers),
        }
        if learned.impact_trained:
            scored = [(learned.impact.predict(sym, c.path, toks, len(callers)), c.path)
                      for c, toks in tests]
            entry["predicted_affected"] = [
                {"test": p, "score": round(s, 4)}
                for s, p in sorted(scored, key=lambda kv: -kv[0])[:10]]
            entry["model"] = learned.impact.version
        impact[sym] = entry
    return {"root": idx.root, "revision": idx.revision, "advisory": True, "impact": impact}


def train_models(root: str, validate: bool = False) -> dict:
    idx = _index(root)
    verdict: dict | None = None
    if validate:
        verdict = validate_and_promote(idx.root)
        for k in [k for k in _LEARNED_CACHE if k.startswith(idx.root + ":")]:
            _LEARNED_CACHE.pop(k, None)
        learned = _learned(root)
    else:
        learned = _learned(root, retrain=True)
    return {
        "root": idx.root, "revision": idx.revision, "trained": True,
        "validated": learned.validated,
        "risk_trained": learned.risk_trained,
        "selector_trained": learned.selector_trained,
        "impact_trained": learned.impact_trained,
        "reranker_trained": learned.reranker is not None,
        "metrics": learned.metrics,
        **({"validation": verdict} if verdict is not None else {}),
    }


def get_calibration(root: str) -> dict:
    idx = _index(root)
    learned = _learned(root)
    conf = learned.conformal
    return {
        "root": idx.root, "revision": idx.revision,
        "risk_trained": learned.risk_trained,
        "validated": learned.validated,
        "validation": _store(idx.root).get_meta(PROMOTION_KEY),
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
                          "history": [], "symbols": sorted(set(_chunk_tokens(c)) & defined)})
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


_PROFILES = ("compile", "collect_tests", "tests")


def run_checks(root: str, profile: str = "compile", timeout_s: int = 120) -> dict:
    idx = _index(root)
    if profile not in _PROFILES:
        raise ValueError(f"unknown check profile: {profile!r}; allowed: {list(_PROFILES)}")
    sandbox = Sandbox(idx.root, timeout_s=min(int(timeout_s), 300))
    tc = detect(idx.root, python=sandbox.python)
    if profile == "compile":
        argv = compile_command(tc, sandbox.python, None)
    elif profile == "collect_tests":
        argv = ([sandbox.python, "-m", "pytest", "--collect-only", "-q"]
                if tc.is_python else [])
    else:
        argv = runner_argv(tc, sandbox.python, None)
    base = {"root": idx.root, "revision": idx.revision, "profile": profile,
            "interpreter": sandbox.python, "language": tc.language}
    if not argv:
        return {**base, "skipped": True,
                "reason": f"no {profile} command for {tc.language} repository"}
    result = sandbox.run(argv)
    return {**base, **result.to_dict()}


def db_stats(root: str) -> dict:
    """libSQL store statistics: what is persisted for this repo."""
    idx = _index(root)
    return {"root": idx.root, **_store(idx.root).stats()}


def vector_search(root: str, query: str, limit: int = 10) -> dict:
    """On-disk ANN search through the libsql vector index (query encoded with the
    repo's persisted encoder, scored by ``vector_distance_cos``)."""
    idx = _index(root)
    # Vector search needs the real encoder; unlike search_code it may pay the
    # fit on a cold call rather than answer from the lexical fallback.
    retr = _RETRIEVER_CACHE.get(idx.root) or _build_retriever(idx)
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


