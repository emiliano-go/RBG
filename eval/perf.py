"""Performance benchmark for every harness model and MCP tool.

Measures fit/train time and inference latency for each learned component, and
per-call latency for each MCP tool (through the real JSON-RPC handler) against
a repository.  Pure stdlib + numpy; no external benchmark framework.

Usage:  python3 -m eval.perf <repo> [--repeat N]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from rbg.index import RepoIndex  # noqa: E402
from rbg.models import (  # noqa: E402
    ChangeImpactModel, ConformalCalibrator, FlakyModel, GBDT, PlattScaler,
    Policy, TestSelector,
)
from rbg.retrieval import (  # noqa: E402
    BM25, DenseEncoder, HybridRetriever, LSIEncoder, Reranker, tokenize,
)
from rbg import mcp_server  # noqa: E402


def _bench(fn, repeat: int) -> dict:
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": round(statistics.median(ts), 4),
            "p50_ms": round(float(np.percentile(ts, 50)), 4),
            "p95_ms": round(float(np.percentile(ts, 95)), 4),
            "min_ms": round(min(ts), 4), "repeat": repeat}


def _synthetic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8))
    y = ((X[:, 0] + X[:, 1] ** 2 - 0.5) > 0).astype(float)
    return X, y


def bench_models(index: RepoIndex, repeat: int) -> dict:
    X, y = _synthetic()
    texts = [c.content for c in index.chunks]
    chunks = index.chunks
    query = "compute metric total value helper"
    sel_examples = []
    for i in range(300):
        syms = set(tokenize(query))
        t = chunks[i % len(chunks)]
        sel_examples.append((syms, set(tokenize(t.content)) | set(tokenize(t.path)),
                             t.path, 1.0, [bool(i % 3)], 0 if i % 2 else 1))
    imp_examples = [(f"sym_{i}", f"tests/test_{i % 40}.py",
                     set(tokenize(query)) | {f"t{i % 7}"}, i % 9, 0.0 if i % 2 else 1.0)
                    for i in range(300)]
    rer_examples = [(query, chunks[i % len(chunks)], 1.0 if i % 2 else 0.0)
                    for i in range(200)]

    gbdt = GBDT(n_estimators=60, learning_rate=0.2, max_depth=3, loss="logloss") \
        .fit(X[:320], y[:320])
    P = gbdt.predict(X[:320])
    platt = PlattScaler().fit(P, y[:320])
    conf = ConformalCalibrator().fit(P, y[:320], matrix=X[:320])
    policy = Policy().train_risk(X[:320].tolist(), y[:320].tolist())
    selector = TestSelector().train(sel_examples)
    impact = ChangeImpactModel().train(imp_examples)
    reranker = Reranker.train(rer_examples)
    flaky = FlakyModel()
    bm25 = BM25(chunks)
    dense = DenseEncoder(512).fit(texts)
    lsi = LSIEncoder().fit(texts)
    hybrid = HybridRetriever(index)

    return {
        "GBDT.fit": _bench(lambda: GBDT(n_estimators=60, learning_rate=0.2, max_depth=3,
                                        loss="logloss").fit(X[:320], y[:320]), repeat),
        "GBDT.predict": _bench(lambda: gbdt.predict(X), repeat),
        "Policy.decide": _bench(lambda: policy.decide(
            __import__("rbg.models", fromlist=["extract_features"]).extract_features(query)),
            repeat),
        "PlattScaler.fit": _bench(lambda: PlattScaler().fit(P, y[:320]), repeat),
        "Platt.__call__": _bench(lambda: [platt(float(r)) for r in P], repeat),
        "Conformal.fit": _bench(lambda: ConformalCalibrator().fit(P, y[:320], matrix=X[:320]), repeat),
        "Conformal.predict_set+ood": _bench(
            lambda: [conf.predict_set(float(r)) for r in P] +
                    [conf.ood_score(X[i]) for i in range(50)], repeat),
        "TestSelector.train": _bench(lambda: TestSelector().train(sel_examples), repeat),
        "TestSelector.rank": _bench(lambda: selector.rank(
            [{"test_id": t.path, "duration": 1.0, "history": []} for t in chunks], ["compute"]), repeat),
        "Impact.train": _bench(lambda: ChangeImpactModel().train(imp_examples), repeat),
        "Impact.predict": _bench(lambda: [impact.predict(f"sym_{i}", f"tests/test_{i%40}.py",
                                                          set(tokenize(query))) for i in range(200)], repeat),
        "Reranker.train": _bench(lambda: Reranker.train(rer_examples), repeat),
        "Reranker.rerank": _bench(lambda: reranker.rerank(query, hybrid.retrieve(query, 30), chunks), repeat),
        "Flaky.predict": _bench(lambda: [flaky.predict([bool(i % 2), bool((i + 1) % 2)]) for i in range(100)], repeat),
        "BM25.rank": _bench(lambda: bm25.rank(query, 30), repeat),
        "DenseEncoder.encode": _bench(lambda: dense.encode(query), repeat),
        "DenseEncoder.rank": _bench(lambda: dense.rank(query, chunks, 30), repeat),
        "LSIEncoder.fit": _bench(lambda: LSIEncoder().fit(texts), max(1, repeat // 2)),
        "LSIEncoder.encode": _bench(lambda: lsi.encode(query), repeat),
        "LSIEncoder.rank": _bench(lambda: lsi.rank(query, chunks, 30), repeat),
        "HybridRetriever.retrieve": _bench(lambda: hybrid.retrieve(query, 30), repeat),
    }


def _call(name: str, arguments: dict):
    return mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": name, "arguments": arguments}})


def bench_tools(root: str, index: RepoIndex, repeat: int) -> dict:
    syms = [s.name for s in index.symbols[:3]] or ["x"]
    long_query = ("please compute the metric total value for the helper that the "
                  "helper the total the value should compute and recompute the "
                  "metric helper value total")
    calls = {
        "search_code": lambda: _call("search_code", {"root": root, "query": "compute metric", "limit": 10}),
        "search_code(long query)": lambda: _call("search_code", {"root": root, "query": long_query, "limit": 10}),
        "retrieve_context": lambda: _call("retrieve_context", {"root": root, "query": "compute metric", "limit": 8}),
        "find_symbol": lambda: _call("find_symbol", {"root": root, "name": syms[0]}),
        "get_change_impact": lambda: _call("get_change_impact", {"root": root, "symbols": syms}),
        "rank_tests": lambda: _call("rank_tests", {"root": root, "changed_symbols": syms}),
        "run_checks(compile)": lambda: _call("run_checks", {"root": root, "profile": "compile"}),
        "get_diagnostics": lambda: _call("get_diagnostics", {"root": root}),
        "db_stats": lambda: _call("db_stats", {"root": root}),
        "vector_search": lambda: _call("vector_search", {"root": root, "query": "compute metric"}),
        "get_calibration": lambda: _call("get_calibration", {"root": root}),
    }
    # warm the caches (index/retriever/models) so the timed calls are steady-state
    for name, fn in calls.items():
        res = fn()
        if res.get("result", {}).get("isError"):
            text = res["result"]["content"][0]["text"]
            raise RuntimeError(f"tool {name} errored during warm-up: {text.strip()}")
    out = {}
    for name, fn in calls.items():
        rep = repeat
        if name in ("run_checks(compile)",):
            rep = 1
        out[name] = _bench(fn, rep)
    return out


def budget_verdict(tools: dict, budget_ms: float | None) -> dict:
    """Grade tool p95 latencies against a budget, or abstain when none is set.

    ``ok=None`` means the question was never asked: reporting ``True`` for an
    unset budget would read as a passed check when nothing was measured.
    """
    worst = max((m["p95_ms"] for m in tools.values()), default=0.0)
    if budget_ms is None:
        return {"budget_ms": None, "worst_p95_ms": worst, "over_budget": [], "ok": None}
    over = sorted(name for name, m in tools.items() if m["p95_ms"] > budget_ms)
    return {"budget_ms": budget_ms, "worst_p95_ms": worst, "over_budget": over, "ok": not over}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--budget-ms", type=float, default=None,
                    help="fail (exit 2) if any tool's p95 latency exceeds this; "
                         "omit to report latencies without grading them")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = os.path.abspath(args.repo)
    # The MCP tools deny-by-default (RBG_ALLOWED_ROOTS); without this the
    # timed calls all return instant permission errors and every "latency"
    # reads as a suspicious 0.02 ms.  Authorize just the repo under test.
    os.environ.setdefault("RBG_ALLOWED_ROOTS", root)
    idx = RepoIndex.build(root)
    mcp_server._INDEX_CACHE.clear()
    mcp_server._STORE_CACHE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    report = {
        "repo": root, "revision": idx.revision,
        "files": len(idx.files), "chunks": len(idx.chunks), "symbols": len(idx.symbols),
        "models": bench_models(idx, args.repeat),
        "tools": bench_tools(root, idx, args.repeat),
    }
    report["latency_budget"] = budget_verdict(report["tools"], args.budget_ms)
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 2 if report["latency_budget"]["ok"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
