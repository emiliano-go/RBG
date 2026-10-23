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
import hashlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX; DB busy-timeout still guards writes
    fcntl = None

from . import db
from .checks import compile_command, detect, runner_argv
from .diffing import (ChangedSet, content_digest, git_changed, symbols_in,
                      workspace_changed)
from .index import RepoIndex, _tree_hash, _tree_signature, chunk_key
from .models import TestSelector, drift_features
from .paths import resolve_within
from .plan import PlanDocument, PlanEditor, audit_completion, parse_plan, scope_drift
from .plan_drift import audit_document, section_validation
from .retrieval import (DenseEncoder, FTSBackend, HybridRetriever, LSIEncoder,
                        LSI_MAX_CORPUS, LexicalOnlyEncoder, heavy_encoder, tokenize)
from .training import (LearnedModels, PROMOTION_KEY, read_ledger,
                       train_from_ledger, validate_and_promote)
from .verify import Sandbox

SERVER_NAME = "rbg-mcp"
SERVER_VERSION = "0.2.0"
DEFAULT_PROTOCOL = "2025-06-18"

_INDEX_CACHE: dict[str, RepoIndex] = {}
_INDEX_SIGNATURE: dict[str, str] = {}
_LEARNED_CACHE: dict[str, LearnedModels] = {}
_PLAN_DRIFT_LOGGED: "OrderedDict[tuple, bool]" = OrderedDict()
_PLAN_DRIFT_LOGGED_MAX = 512
_RETRIEVER_CACHE: dict[str, HybridRetriever] = {}

# Bounded LRU over retrieval payloads, keyed by repo/revision/query/filters.
# Only the result list is cached; latency/timing fields are recomputed per call.
_RESULT_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_RESULT_CACHE_MAX = 256


def _result_cached(key: tuple, build):
    hit = _RESULT_CACHE.get(key)
    if hit is not None:
        _RESULT_CACHE.move_to_end(key)
        return hit
    val = build()
    _RESULT_CACHE[key] = val
    if len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
        _RESULT_CACHE.popitem(last=False)
    return val

# Tokenized chunk contents, keyed by content hash.  Tokenizing every test chunk
# on each call dominated get_change_impact/rank_tests on large repos (~0.5 s per
# call on a 6.5k-chunk repo); the hash makes a stale entry impossible.
_TOKENS_CACHE: dict[str, frozenset] = {}


def _chunk_tokens(chunk) -> frozenset:
    toks = _TOKENS_CACHE.get(chunk.content_hash)
    if toks is None:
        if len(_TOKENS_CACHE) >= 200_000:  # ponytail: crude cap; LRU if it ever matters
            _TOKENS_CACHE.clear()
        toks = frozenset(tokenize(chunk.content))
        _TOKENS_CACHE[chunk.content_hash] = toks
    return toks


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
    if not any(a == os.sep or resolved == a or resolved.startswith(a + os.sep)
               for a in allowed):
        raise PermissionError(f"repository not allowed: {root}")
    return resolved


def _evict_dependents(root: str) -> None:
    """Drop per-repo derivatives keyed to a revision that is no longer current."""
    _RETRIEVER_CACHE.pop(root, None)
    _FAST_RETRIEVER_CACHE.pop(root, None)
    _RETRIEVER_BUILD.pop(root, None)
    for key in [k for k in _LEARNED_CACHE if k == root or k.startswith(root + ":")]:
        _LEARNED_CACHE.pop(key, None)
    for key in [k for k in _RESULT_CACHE if k[0] == root]:
        _RESULT_CACHE.pop(key, None)


@contextmanager
def _index_lock(path: str):
    """Cross-process lock so N agents do not all parse and persist the same tree.

    The DB serialises the write, but a 47k-chunk parse is ~5 s of duplicated CPU
    per agent; the first process builds, the rest wait and then read its index.
    """
    if fcntl is None:  # pragma: no cover
        yield
        return
    fd = os.open(path + ".index.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class IndexBuilding(RuntimeError):
    """The index is being built in the background; retry this tool shortly."""

    def __init__(self, root: str, started: float) -> None:
        self.root = root
        self.started = started
        super().__init__("index build in progress")

    def payload(self) -> dict:
        elapsed_ms = round((time.monotonic() - self.started) * 1000, 1)
        return {
            "root": self.root,
            "status": "building",
            "degraded": True,
            "index_build_ms": elapsed_ms,
            "message": "index build in progress; retry this tool in a few seconds",
        }


_BUILD_STATE: dict[str, dict] = {}
_BUILD_LOCK = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _start_index_build(root: str, signature: str) -> None:
    with _BUILD_LOCK:
        state = _BUILD_STATE.get(root)
        if state and state["signature"] == signature and state["status"] != "error":
            return
        _BUILD_STATE[root] = {"signature": signature, "status": "building",
                              "started": time.monotonic(), "error": None}
        started = _BUILD_STATE[root]["started"]

    def _worker() -> None:
        try:
            # Own Store/connection: the main loop may serve other tools on the
            # cached connection while this thread writes the index.
            _index_sync(root, signature, isolated=True)
        except Exception as exc:  # noqa: BLE001 - surfaced on the next call
            with _BUILD_LOCK:
                _BUILD_STATE[root] = {"signature": signature, "status": "error",
                                      "started": started,
                                      "error": f"{type(exc).__name__}: {exc}"}
            return
        with _BUILD_LOCK:
            _BUILD_STATE[root] = {"signature": signature, "status": "ready",
                                  "started": started, "error": None}

    threading.Thread(target=_worker, name=f"index-build:{root}", daemon=True).start()


def _index(root: str) -> RepoIndex:
    root = _authorize_root(root)
    signature = _tree_signature(root)
    cached = _INDEX_CACHE.get(root)
    if cached is not None and _INDEX_SIGNATURE.get(root) == signature:
        return cached
    # Async is the default.  A cold parse of a large repo (plus its tree hash)
    # can otherwise exceed the client's whole request timeout; the caller gets
    # a retryable "building" result while the parse runs in the background.
    explicit_async = "HARNESS_INDEX_ASYNC" in os.environ
    if _env_flag("HARNESS_INDEX_ASYNC", "1"):
        _start_index_build(root, signature)
        # Small repos finish in milliseconds; wait briefly so the common case
        # is still a single call.  An explicitly configured async caller gets
        # the old immediate "building" behaviour.
        grace = 0.0 if explicit_async else max(
            0, int(os.environ.get("HARNESS_INDEX_GRACE_MS", "1000"))) / 1000.0
        deadline = time.monotonic() + grace
        while True:
            with _BUILD_LOCK:
                state = dict(_BUILD_STATE.get(root) or {})
            if state.get("signature") != signature:
                _start_index_build(root, signature)  # tree changed again
                raise IndexBuilding(root, time.monotonic())
            if state.get("status") == "error":
                raise RuntimeError(state.get("error") or "index build failed")
            if state.get("status") == "ready" and root in _INDEX_CACHE:
                return _INDEX_CACHE[root]
            if time.monotonic() >= deadline:
                raise IndexBuilding(root, state.get("started", time.monotonic()))
            time.sleep(0.02)
    return _index_sync(root, signature, cached)


def _index_sync(root: str, signature: str, cached: RepoIndex | None = None,
                isolated: bool = False) -> RepoIndex:
    store = db.Store(root) if isolated else _store(root)
    try:
        # Cheap freshness fast path: a warm DB index carries the metadata
        # signature it was built from, so the whole-tree content hash (123 MB
        # read here) is only paid when files actually changed.
        try:
            if store.get_meta("tree_signature") == signature:
                idx = store.load_index(None)
            else:
                idx = None
        except Exception:
            idx = None
        if idx is None:
            fingerprint = _tree_hash(root)
            try:
                idx = store.load_index(fingerprint)  # cache hit: no re-parse
            except Exception:
                idx = None
            if idx is None:
                with _index_lock(store.path):
                    try:  # another process may have built it while we waited
                        idx = store.load_index(fingerprint)
                    except Exception:
                        idx = None
                    if idx is None:
                        idx = RepoIndex.build(root, revision=fingerprint)
                        store.save_index(idx)
        try:
            if store.get_meta("tree_signature") != signature:
                store.set_meta("tree_signature", signature)
        except Exception:
            pass
    finally:
        if isolated:
            store.close()
    if cached is not None and cached.revision != idx.revision:
        _evict_dependents(root)  # stale index/retriever/models must not be served
    _INDEX_CACHE[root] = idx
    _INDEX_SIGNATURE[root] = signature
    return idx


def _warm_lsi(store, idx: RepoIndex):
    """Re-embed a changed corpus with the previously persisted LSI projector
    instead of refitting the SVD.  Returns None when there is nothing reusable
    (no prior encoder, or a non-LSI encoder)."""
    try:
        prev = store.load_prev_encoder()
    except Exception:
        return None
    if not prev or prev[1].get("kind") != "lsi":
        return None
    if len(idx.chunks) > LSI_MAX_CORPUS:
        return None
    enc = LSIEncoder.from_state(prev[1], [])
    return enc.refresh([c.content for c in idx.chunks])


_FAST_RETRIEVER_CACHE: dict[str, HybridRetriever] = {}
_RETRIEVER_BUILD: dict[str, dict] = {}
_RETRIEVER_BUILD_LOCK = threading.Lock()


def _fast_retriever(idx: RepoIndex) -> HybridRetriever:
    """A lexical-only retriever that is cheap to build on the read path.

    Used while the semantic retriever is warming: FTS/BM25 answers immediately
    instead of making the client wait for an SVD/embedding pass.
    """
    retr = _FAST_RETRIEVER_CACHE.get(idx.root)
    if retr is None or retr.index.revision != idx.revision:
        lexical = FTSBackend(idx.chunks, _store(idx.root).fts_search)
        retr = HybridRetriever(idx, encoder=LexicalOnlyEncoder(), lexical=lexical)
        _FAST_RETRIEVER_CACHE[idx.root] = retr
    return retr


def _start_retriever_build(idx: RepoIndex) -> None:
    with _RETRIEVER_BUILD_LOCK:
        state = _RETRIEVER_BUILD.get(idx.root)
        if state and state["revision"] == idx.revision and state["status"] != "error":
            return
        _RETRIEVER_BUILD[idx.root] = {"revision": idx.revision,
                                      "status": "building", "error": None}

    def _worker() -> None:
        try:
            # Fit only: no DB access from this thread.  A second libSQL
            # connection writing while the main connection reads pansics the
            # binding; the encoder cache is swapped in memory instead.
            heavy = heavy_encoder()
            if heavy is not None:
                encoder = heavy
            elif len(idx.chunks) <= LSI_MAX_CORPUS:
                encoder = LSIEncoder(k=128).fit([c.content for c in idx.chunks])
            else:
                encoder = LexicalOnlyEncoder()
            lexical = FTSBackend(idx.chunks, _store(idx.root).fts_search)
            retr = HybridRetriever(idx, encoder=encoder, lexical=lexical)
            status, error = "ready", None
        except Exception as exc:  # noqa: BLE001 - retryable on the next call
            status, error = "error", f"{type(exc).__name__}: {exc}"
            retr = None
        if retr is not None:
            _RETRIEVER_CACHE[idx.root] = retr
        with _RETRIEVER_BUILD_LOCK:
            _RETRIEVER_BUILD[idx.root] = {"revision": idx.revision,
                                          "status": status, "error": error}

    threading.Thread(target=_worker, name=f"retriever-build:{idx.root}",
                     daemon=True).start()


def _retriever(root: str) -> HybridRetriever:
    """Full retriever when warm, lexical-only while it builds in the background."""
    idx = _index(root)
    retr = _RETRIEVER_CACHE.get(idx.root)
    if retr is not None:
        return retr
    # A persisted encoder is a cheap main-thread read and skips the fit.
    try:
        encoder = _store(idx.root).load_encoder(idx.revision, len(idx.chunks))
    except Exception:
        encoder = None
    if encoder is not None:
        lexical = FTSBackend(idx.chunks, _store(idx.root).fts_search)
        retr = HybridRetriever(idx, encoder=encoder, lexical=lexical)
        _RETRIEVER_CACHE[idx.root] = retr
        return retr
    _start_retriever_build(idx)
    return _fast_retriever(idx)


def _build_retriever(idx: RepoIndex, store=None) -> HybridRetriever:
    store = store or _store(idx.root)
    if idx.root not in _RETRIEVER_CACHE:
        lexical = FTSBackend(idx.chunks, store.fts_search)
        # An opt-in semantic encoder wins over the persisted lexical one, and is
        # persisted itself so vector_search queries vectors from the same model.
        heavy = heavy_encoder()
        if heavy is not None:
            retr = HybridRetriever(idx, encoder=heavy, lexical=lexical)
            store.save_encoder(idx.revision, retr.encoder.to_state(),
                               retr.encoder.document_vectors(idx.chunks))
        else:
            encoder = None
            try:
                encoder = store.load_encoder(idx.revision, len(idx.chunks))
            except Exception:
                encoder = None
            warmed = False
            if encoder is None:
                encoder = _warm_lsi(store, idx)
                warmed = encoder is not None
            if encoder is None and len(idx.chunks) <= LSI_MAX_CORPUS:
                # Reuse a persisted dense encoder's idf across revisions: the
                # document matrix is rebuilt lazily, but the O(corpus) fit is
                # skipped.  (Above the cap HybridRetriever stays lexical-only.)
                try:
                    prev = store.load_prev_encoder()
                except Exception:
                    prev = None
                if prev and prev[1].get("kind") == "dense":
                    enc = DenseEncoder(dim=int(prev[1].get("dim", 512)))
                    enc.idf = dict(prev[1].get("idf", {}))
                    encoder = enc
                    warmed = True
            if encoder is not None:
                retr = HybridRetriever(idx, encoder=encoder, lexical=lexical)
            else:
                retr = HybridRetriever(idx, lexical=lexical)
            if (encoder is None or warmed) and not isinstance(retr.encoder, LexicalOnlyEncoder):
                store.save_encoder(idx.revision, retr.encoder.to_state(),
                                   retr.encoder.document_vectors(idx.chunks))
        _RETRIEVER_CACHE[idx.root] = retr
    return _RETRIEVER_CACHE[idx.root]


def _apply_validation(models: LearnedModels | None, store, revision: str) -> LearnedModels | None:
    """Restore the promotion verdict only for the artifact it was earned on."""
    if models is None:
        return None
    record = store.get_meta(PROMOTION_KEY) or {}
    models.validated = bool(
        record.get("promoted")
        and record.get("revision") == revision
        and int(record.get("ledger_n", -1)) == store.ledger_count())
    return models


def _learned(root: str, retrain: bool = False) -> LearnedModels:
    """Trained models for a repo.

    The read path never trains: it loads a model persisted for the current
    revision *and* ledger size (a cheap ``COUNT(*)``), or returns an empty set.
    Training happens only through the explicit ``train_models`` tool, so no
    retrieval or impact query ever pays to refit.
    """
    if retrain:
        idx = _index_sync(root, _tree_signature(root))
    else:
        try:
            idx = _index(root)
        except IndexBuilding:
            # A model-dependent read degrades to "no model yet" rather than
            # blocking the client on the same cold index build.
            return LearnedModels()
    store = _store(idx.root)
    key = f"{idx.root}:{idx.revision}:{store.ledger_count()}"
    if retrain:
        models = _apply_validation(
            train_from_ledger(idx.root, index=idx, store=store), store, idx.revision)
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
            models = _apply_validation(got[0], store, idx.revision)
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
        "description": "Train the harness models (risk, selector, reranker, impact) from the ledger. With validate=true, also gate the learned risk head on a held-out Brier/coverage comparison and record the verdict.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"},
            "validate": {"type": "boolean", "default": False}}, "required": ["root"]},
    },
    "plan_audit": {
        "fn": plan_audit,
        "description": "Audit a plan: with task_id, run the stored document's deterministic drift gate (scope, acceptance, unplanned changes) and return the plan, rollup and drift verdict; otherwise compute free-text scope drift and completeness.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "plan_text": {"type": "string"},
            "task": {"type": "string"},
            "changed_files": {"type": "array", "items": {"type": "string"}}},
            "required": ["root"]},
    },
    "plan_create": {
        "fn": plan_create,
        "description": "Create a JSON plan document (the server owns the plan; no file rewrite).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "title": {"type": "string"},
            "sections": {"type": "array", "items": {"type": "object"}}},
            "required": ["root", "task_id", "title"]},
    },
    "plan_append": {
        "fn": plan_append,
        "description": "Append a top-level section to a stored plan.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "section": {"type": "object"}}, "required": ["root", "task_id", "section"]},
    },
    "plan_attach": {
        "fn": plan_attach,
        "description": "Attach a subsection under a parent (depth cap 5).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "parent_id": {"type": "string"}, "section": {"type": "object"}},
            "required": ["root", "task_id", "parent_id", "section"]},
    },
    "plan_complete": {
        "fn": plan_complete,
        "description": "Mark a section done only after the automatic drift audit passes. A caller-supplied validation cannot override a deterministic blocker.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "section_id": {"type": "string"},
            "validation": {"type": "object", "description": "{verdict: pass|fail, evidence: str, model: str}"}},
            "required": ["root", "task_id", "section_id"]},
    },
    "plan_modify": {
        "fn": plan_modify,
        "description": "Edit a section's title/body/status (open|blocked|dropped).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "section_id": {"type": "string"}, "fields": {"type": "object"}},
            "required": ["root", "task_id", "section_id", "fields"]},
    },
    "plan_note": {
        "fn": plan_note,
        "description": "Append a timestamped note to a section.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "section_id": {"type": "string"}, "text": {"type": "string"}},
            "required": ["root", "task_id", "section_id", "text"]},
    },
    "plan_check": {
        "fn": plan_check,
        "description": "Record a check result on a section.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "section_id": {"type": "string"}, "name": {"type": "string"},
            "status": {"type": "string"}, "evidence": {"type": "string"}},
            "required": ["root", "task_id", "section_id", "name", "status"]},
    },
    "plan_get": {
        "fn": plan_get,
        "description": "Read a stored plan document (or one section).",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"},
            "section_id": {"type": "string"}}, "required": ["root", "task_id"]},
    },
    "plan_tree": {
        "fn": plan_tree,
        "description": "Return the plan as a depth/status tree with effective rollup.",
        "inputSchema": {"type": "object", "properties": {
            "root": {"type": "string"}, "task_id": {"type": "string"}},
            "required": ["root", "task_id"]},
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
MAX_LINE_BYTES = int(os.environ.get("HARNESS_MCP_MAX_LINE_BYTES", str(1024 * 1024)))

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


_RATE_LOCK = threading.Lock()
_RATE_STATE: dict[str, list[float]] = {}


def _rate_ok(tool: str) -> bool:
    """Token-bucket limiter (MCP requires servers to rate-limit invocations)."""
    try:
        rate = float(os.environ.get("HARNESS_MCP_RATE_PER_MIN", "120")) / 60.0
        burst = float(os.environ.get("HARNESS_MCP_BURST", "30"))
    except ValueError:
        return True
    if rate <= 0:
        return True
    now = time.monotonic()
    with _RATE_LOCK:
        tokens, last = _RATE_STATE.get(tool, [burst, now])
        tokens = min(burst, tokens + (now - last) * rate)
        if tokens < 1.0:
            _RATE_STATE[tool] = [tokens, now]
            return False
        _RATE_STATE[tool] = [tokens - 1.0, now]
        return True


def _validate_args(spec: dict, args) -> str:
    """Minimal JSON-Schema check for the tool surface (no new dependency)."""
    schema = spec.get("inputSchema") or {}
    if not isinstance(args, dict):
        return "arguments must be an object"
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument: {name}"
    for name, value in args.items():
        rule = (schema.get("properties") or {}).get(name)
        if not rule:
            continue
        kind = rule.get("type")
        if kind == "string":
            if not isinstance(value, str):
                return f"{name} must be a string"
            limit = 200_000 if name in ("content",) else 4096
            if len(value) > limit:
                return f"{name} exceeds {limit} characters"
        elif kind == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"{name} must be an integer"
        elif kind == "boolean":
            if not isinstance(value, bool):
                return f"{name} must be a boolean"
        elif kind == "array":
            if not isinstance(value, list):
                return f"{name} must be an array"
            if len(value) > 512:
                return f"{name} exceeds 512 items"
    return ""


def _tool_manifest_hash() -> str:
    canon = json.dumps({name: {"description": spec["description"],
                               "inputSchema": spec["inputSchema"]}
                        for name, spec in sorted(TOOLS.items())},
                       sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _tool_result(msg_id, out, *, is_error: bool = False):
    """Serialize a tool result once, bounding both representations.

    The text copy used to be sliced while ``structuredContent`` kept the whole
    object, so a large bundle defeated ``MAX_OUTPUT_BYTES``.  When the payload
    is over budget, return a compact structured summary plus a bounded preview
    so the caller still learns what happened.
    """
    payload = json.dumps(out, default=str)
    if len(payload) <= MAX_OUTPUT_BYTES:
        return _result(msg_id, {"content": [{"type": "text", "text": payload}],
                                "structuredContent": out, "isError": is_error})
    marker = "\n...[truncated]"
    text = payload[:MAX_OUTPUT_BYTES - len(marker)] + marker
    summary = {"status": "truncated", "truncated": True, "bytes": len(payload),
               "limit": MAX_OUTPUT_BYTES, "preview": text[:8192]}
    return _result(msg_id, {"content": [{"type": "text", "text": text}],
                            "structuredContent": summary, "isError": is_error})


class ToolTimeout(RuntimeError):
    pass


def _on_alarm(signum, frame):  # noqa: ARG001 - signal handler signature
    raise ToolTimeout("tool exceeded its wall-clock budget")


def _call_with_deadline(fn, args: dict):
    """Run a tool under a hard wall-clock cap.

    The read budget only decided whether to skip the reranker; it could not stop
    a cold index/encoder pass from hanging the single-threaded stdio loop.  A
    SIGALRM interrupts that work (it runs on the main thread) and the call
    returns a structured timeout instead of blocking every later request.
    """
    default_ms = int(os.environ.get("HARNESS_TOOL_TIMEOUT_MS", "60000"))
    requested = int(args.get("timeout_ms") or 0)
    timeout_ms = max(default_ms, requested)
    if (timeout_ms <= 0 or not hasattr(signal, "SIGALRM")
            or threading.current_thread() is not threading.main_thread()):
        return fn(**args)
    old = signal.signal(signal.SIGALRM, _on_alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000)
        return fn(**args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


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
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION,
                           "tool_manifest_sha256": _tool_manifest_hash()},
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
        problem = _validate_args(spec, args)
        if problem:
            return _error(msg_id, -32602, f"invalid arguments: {problem}")
        if not _rate_ok(name):
            return _error(msg_id, -32000, f"rate limit exceeded for {name}")
        try:
            out = _call_with_deadline(spec["fn"], args)
            return _tool_result(msg_id, out)
        except IndexBuilding as exc:
            return _tool_result(msg_id, exc.payload())
        except ToolTimeout as exc:
            out = {"status": "timeout", "degraded": True, "tool": name,
                   "message": str(exc)}
            return _tool_result(msg_id, out, is_error=True)
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


def _send(response) -> None:
    if response is not None:
        sys.stdout.write(json.dumps(response, default=str) + "\n")
        sys.stdout.flush()


def _drain_line(stream) -> None:
    """Discard the rest of an over-long request line."""
    while True:
        chunk = stream.readline(MAX_LINE_BYTES + 1)
        if not chunk or chunk.endswith(b"\n"):
            return


def main() -> int:
    stream = sys.stdin.buffer
    while True:
        raw = stream.readline(MAX_LINE_BYTES + 1)
        if not raw:
            break
        if len(raw) > MAX_LINE_BYTES and not raw.endswith(b"\n"):
            _drain_line(stream)
            _send(_error(None, -32600,
                         f"request exceeds {MAX_LINE_BYTES} bytes"))
            continue
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _send(_error(None, -32700, "parse error"))
            continue
        if not isinstance(msg, dict):
            _send(_error(None, -32600, "invalid request: expected a JSON object"))
            continue
        try:
            response = handle(msg)
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill stdio
            tb = traceback.format_exc() if os.environ.get("HARNESS_MCP_DEBUG") else ""
            response = _error(msg.get("id"), -32603,
                              f"internal error: {type(exc).__name__}: {exc}\n{tb}")
        _send(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
