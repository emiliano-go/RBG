"""Effectiveness evaluation for the RBG Harness, run against a real repo.

Measures exactly the metrics the spec asks for (spec 5.2, 5.4, 9.2):

1. Retrieval: Recall@k and MRR for symbol-name queries (exact-identifier
   lookup) and docstring queries (semantic lookup), comparing a fixed-context
   baseline (B0), BM25 only, dense only, and the hybrid retriever.
2. Test selection: how many change-revealing tests the selector surfaces in
   the top-k versus alphabetical and random ordering.

Usage:  python3 -m eval.effectiveness <repo> [--sample N]
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.index import RepoIndex, is_test_chunk
from harness.models import TestSelector
from harness.retrieval import BM25, DenseEncoder, HybridRetriever, LSIEncoder, tokenize


def symbol_queries(index: RepoIndex) -> list[tuple[str, tuple[str, int], str]]:
    """(query, ground_truth_chunk_key, kind) one per function/class definition."""
    out = []
    for c in index.chunks:
        if c.symbol:
            out.append((c.symbol, (c.path, c.start_line), "symbol"))
    return out


def docstring_queries(index: RepoIndex, root: str) -> list[tuple[str, tuple[str, int], str]]:
    out = []
    for c in index.chunks:
        if not c.symbol:
            continue
        try:
            tree = ast.parse(c.content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc and len(doc) > 20:
                    first = doc.strip().splitlines()[0]
                    out.append((first, (c.path, c.start_line), "docstring"))
                break
    return out


def nl_queries(index: RepoIndex, min_terms: int = 3) -> list[tuple[str, tuple[str, int], str]]:
    """Non-leaky semantic queries.

    The docstring benchmark leaks: its query text sits verbatim inside the
    target chunk, so a substring tool (grep) or a bag-of-words ranker (BM25)
    copies the answer.  Here the query is the *prose of another chunk that
    references the symbol* (its docstring and comments), with every term that
    also occurs in the target chunk removed.  The target provably does not
    contain the query string, so only genuine semantic / co-occurrence matching
    can recover it -- which is exactly what LSA claims to add.
    """
    defs: dict[str, int] = {}
    chunk_tokens = [set(tokenize(c.content)) for c in index.chunks]
    for i, c in enumerate(index.chunks):
        if c.symbol:
            defs.setdefault(c.symbol.lower(), i)
    defsyms = set(defs)
    comment_re = re.compile(r"#\s*(.+)")
    seen: set[str] = set()
    out = []
    for ci, c in enumerate(index.chunks):
        mentioned = chunk_tokens[ci] & defsyms
        if not mentioned:
            continue
        texts: list[str] = []
        try:
            tree = ast.parse(c.content)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    d = ast.get_docstring(node)
                    if d:
                        texts.append(d.splitlines()[0])
                    break
        texts += [m.group(1) for m in comment_re.finditer(c.content)]
        if not texts:
            continue
        prose_tokens = tokenize(" ".join(texts))
        for sym in sorted(mentioned):
            ti = defs[sym]
            if ti == ci:
                continue
            tgt = chunk_tokens[ti]
            terms = [w for w in prose_tokens if len(w) >= 4 and w not in tgt]
            query = " ".join(sorted(set(terms)))
            if len(set(terms)) >= min_terms and query not in seen:
                seen.add(query)
                tc = index.chunks[ti]
                out.append((query, (tc.path, tc.start_line), "nl"))
    random.Random(0).shuffle(out)  # sample shouldn't be dominated by early files
    return out


def _rank_of(order: list[int], target: int, k: int) -> int | None:
    for pos, idx in enumerate(order[:k]):
        if idx == target:
            return pos + 1
    return None


def _ndcg(order: list[int], target: int, k: int) -> float:
    """nDCG@k with one binary-relevant doc: ideal DCG is the target at rank 1."""
    rank = _rank_of(order, target, k)
    return (1.0 / math.log2(rank + 1)) if rank else 0.0


def grep_literal_order(query: str, chunks, k: int = 30) -> list[int]:
    """What ``rg "the whole query"`` gives you: case-insensitive literal substring,
    ranked by occurrence count.  Useless unless you already know the exact string."""
    q = query.lower()
    hits = [(i, c.content.lower().count(q)) for i, c in enumerate(chunks) if q in c.content.lower()]
    hits.sort(key=lambda ic: -ic[1])
    return [i for i, _ in hits[:k]]


def grep_terms_order(query: str, chunks, k: int = 30) -> list[int]:
    """Generous grep: OR of query tokens, ranked by how many tokens a chunk
    contains.  No IDF, no fusion, no semantics -- the lexical floor."""
    toks = set(tokenize(query)) or {query.lower()}
    scored = []
    for i, c in enumerate(chunks):
        lc = c.content.lower()
        n = sum(1 for t in toks if t in lc)
        if n:
            scored.append((n, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in scored[:k]]


def eval_retrieval(index: RepoIndex, queries, sample: int) -> dict:
    chunk_pos = {(c.path, c.start_line): i for i, c in enumerate(index.chunks)}
    fixed_order = sorted(range(len(index.chunks)), key=lambda i: (index.chunks[i].path, index.chunks[i].start_line))
    bm25 = BM25(index.chunks)
    texts = [c.content for c in index.chunks]
    enc = DenseEncoder(512).fit(texts)
    lsi = LSIEncoder().fit(texts)
    hybrid = HybridRetriever(index)
    hybrid_hash = HybridRetriever(index, semantic=False)
    hybrid_prf = HybridRetriever(index, expansion=True)
    hybrid_mmr = HybridRetriever(index, diversify=True)

    systems = {
        "fixed_B0": lambda q: fixed_order[:30],
        "grep(literal)": lambda q: grep_literal_order(q, index.chunks, 30),
        "grep(terms)": lambda q: grep_terms_order(q, index.chunks, 30),
        "bm25": lambda q: [i for i, _ in bm25.rank(q, 30)],
        "dense": lambda q: [i for i, _ in enc.rank(q, index.chunks, 30)],
        "lsi": lambda q: [i for i, _ in lsi.rank(q, index.chunks, 30)],
        "hybrid": lambda q: [chunk_pos[(h.path, h.start_line)] for h in hybrid.retrieve(q, 30)],
        "hybrid+hash": lambda q: [chunk_pos[(h.path, h.start_line)] for h in hybrid_hash.retrieve(q, 30)],
        "hybrid+prf": lambda q: [chunk_pos[(h.path, h.start_line)] for h in hybrid_prf.retrieve(q, 30)],
        "hybrid+mmr": lambda q: [chunk_pos[(h.path, h.start_line)] for h in hybrid_mmr.retrieve(q, 30)],
    }
    results = {name: {"r1": 0, "r5": 0, "r10": 0, "mrr": 0.0, "ndcg": 0.0, "n": 0, "lat": []} for name in systems}
    for query, key, _kind in queries[:sample]:
        target = chunk_pos.get(key)
        if target is None:
            continue
        for name, fn in systems.items():
            t0 = time.perf_counter()
            order = fn(query)
            dt = (time.perf_counter() - t0) * 1000
            r = results[name]
            results[name]["n"] += 1
            results[name]["lat"].append(dt)
            if _rank_of(order, target, 1):
                r["r1"] += 1
            if _rank_of(order, target, 5):
                r["r5"] += 1
            if _rank_of(order, target, 10):
                r["r10"] += 1
            r["ndcg"] += _ndcg(order, target, 10)
            rank = _rank_of(order, target, 30)
            r["mrr"] += (1.0 / rank) if rank else 0.0
    for name, r in results.items():
        n = r["n"] or 1
        r["recall@1"] = round(r["r1"] / n, 4)
        r["recall@5"] = round(r["r5"] / n, 4)
        r["recall@10"] = round(r["r10"] / n, 4)
        r["mrr"] = round(r["mrr"] / n, 4)
        r["ndcg@10"] = round(r["ndcg"] / n, 4)
        r["latency_ms"] = round(statistics.mean(r["lat"]), 3) if r["lat"] else 0.0
        for k in ("r1", "r5", "r10", "ndcg", "lat"):
            r.pop(k)
    return results


def test_to_symbols(index: RepoIndex) -> tuple[list[dict], dict[str, set[str]]]:
    defined = {s.name.lower() for s in index.symbols}
    tests = []
    for c in index.chunks:
        if is_test_chunk(c):
            name = c.symbol or os.path.basename(c.path)
            tests.append({"test_id": f"{c.path}::{name}", "duration": 1.0,
                          "history": [], "symbols": sorted(set(tokenize(c.content)) & defined)})
    # map symbol -> revealing test ids
    rev: dict[str, set[str]] = {}
    for t in tests:
        for s in t["symbols"]:
            rev.setdefault(s, set()).add(t["test_id"])
    return tests, rev


def eval_test_selection(index: RepoIndex, sample: int, seed: int = 0) -> dict:
    tests, rev = test_to_symbols(index)
    selector = TestSelector()
    n_tests = len(tests) or 1
    # Discriminating symbols only: those whose revealing set is a small slice of
    # the suite. A symbol used by 90% of tests makes recall@k uninformative.
    cap = max(3, n_tests // 5)
    candidates = [s for s, ts in rev.items() if 1 <= len(ts) <= cap]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates = candidates[:sample]
    agg = {name: {"r5": 0.0, "r10": 0.0, "third": 0.0, "mrr": 0.0, "prec10": 0.0, "cover": 0.0, "n": 0}
           for name in ("selector", "alpha", "random")}
    alpha_order = [t["test_id"] for t in sorted(tests, key=lambda t: t["test_id"])]
    # Predictive test selection operating point (Facebook PTS): how many
    # failure-revealing tests land in the top third of the suite, the budget at
    # which PTS reports >95% of failures.
    budget = max(1, len(alpha_order) // 3)
    for sym in candidates:
        revealing = rev[sym]
        orders = {
            "selector": [r.test_id for r in selector.rank(tests, [sym])],
            "alpha": alpha_order,
            "random": rng.sample(alpha_order, len(alpha_order)),
        }
        for name, order in orders.items():
            a = agg[name]
            a["n"] += 1
            a["r5"] += sum(1 for t in order[:5] if t in revealing) / len(revealing)
            a["r10"] += sum(1 for t in order[:10] if t in revealing) / len(revealing)
            a["third"] += sum(1 for t in order[:budget] if t in revealing) / len(revealing)
            a["prec10"] += sum(1 for t in order[:10] if t in revealing) / 10
            first = next((i for i, t in enumerate(order) if t in revealing), None)
            a["mrr"] += 1.0 / (first + 1) if first is not None else 0.0
            last = max((i for i, t in enumerate(order) if t in revealing), default=len(order) - 1)
            a["cover"] += (last + 1) / len(order)  # suite fraction needed to reach all revealing tests
    out = {}
    for name, a in agg.items():
        n = a["n"] or 1
        out[name] = {
            "recall@5": round(min(a["r5"] / n, 1.0), 4),
            "recall@10": round(min(a["r10"] / n, 1.0), 4),
            "recall_at_third": round(min(a["third"] / n, 1.0), 4),
            "precision@10": round(a["prec10"] / n, 4),
            "mrr": round(a["mrr"] / n, 4),
            "suite_fraction_to_cover_all": round(a["cover"] / n, 4),
            "n": a["n"],
        }
    return {"n_tests": len(tests), "n_changed_symbols": len(candidates),
            "budget_tests": budget, "scores": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = os.path.abspath(args.repo)

    idx = RepoIndex.build(root)
    print(f"indexed {root}: {len(idx.files)} files, {len(idx.chunks)} chunks, {len(idx.symbols)} symbols")

    sym_q = symbol_queries(idx)
    doc_q = docstring_queries(idx, root)
    nl_q = nl_queries(idx)
    report = {
        "repo": root,
        "revision": idx.revision,
        "files": len(idx.files), "chunks": len(idx.chunks), "symbols": len(idx.symbols),
        "symbol_retrieval": eval_retrieval(idx, sym_q, args.sample),
        "docstring_retrieval": eval_retrieval(idx, doc_q, args.sample),
        "nl_retrieval": eval_retrieval(idx, nl_q, args.sample),
        "test_selection": eval_test_selection(idx, min(args.sample, 200)),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
