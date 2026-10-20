"""Recall ablation for the retriever: lexical vs dense vs hybrid.

Retrieval only -- no orchestrator, no sandbox -- so it is fast and
deterministic.  Point it at the distractor-injected hard suite to see the
reciprocal-rank fusion actually earning its keep:

    python3 -m eval.retrieval_ablation eval/suites/hard
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

from harness.index import RepoIndex
from harness.retrieval import HybridRetriever

from eval.benchmark import _mean, load_suite

MODES = ("lexical", "dense", "hybrid")


class _NullLexical:
    def rank(self, query, limit):
        return []


class _NullEncoder:
    def encode(self, query):
        return np.zeros(1, dtype=float)

    def document_vectors(self, chunks):
        return np.zeros((len(chunks), 1), dtype=float)

    def rank(self, query, chunks, limit):
        return []


def _retriever(mode: str, idx: RepoIndex, reranker=None) -> HybridRetriever:
    if mode == "lexical":
        return HybridRetriever(idx, semantic=True, encoder=_NullEncoder())
    if mode == "dense":
        return HybridRetriever(idx, semantic=True, lexical=_NullLexical())
    return HybridRetriever(idx, semantic=True, reranker=reranker)


def _quality(hits: list, expected: list, k: int) -> tuple[float, float, float]:
    seen: list[str] = []
    for h in hits[:k]:
        if h.path not in seen:
            seen.append(h.path)
    if not expected:
        return 1.0, 1.0, 1.0
    recall = sum(1 for p in expected if p in seen) / len(expected)
    recall_at_1 = 1.0 if seen and seen[0] in expected else 0.0
    mrr = 0.0
    for p in expected:
        if p in seen:
            mrr = max(mrr, 1.0 / (seen.index(p) + 1))
    return recall, recall_at_1, mrr


def _write(root: str, files: dict) -> None:
    for rel, content in files.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full) or root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)


def run(suite_dir: str, k: int = 5, reranker: str = "") -> dict:
    suite = load_suite(suite_dir)
    modes = list(MODES)
    ce = None
    if reranker == "heavy":
        from harness.retrieval_heavy import CrossEncoderReranker
        ce = CrossEncoderReranker()
        modes.append("hybrid+ce")
    workdir = tempfile.mkdtemp(prefix="rbg-retr-")
    rows = {mode: [] for mode in modes}
    try:
        for task in suite["tasks"]:
            root = os.path.join(workdir, task["id"])
            os.makedirs(root, exist_ok=True)
            _write(root, task["files"])
            idx = RepoIndex.build(root)
            expected = task.get("expected_files", [])
            for mode in modes:
                rr = ce if mode == "hybrid+ce" else None
                hits = _retriever(mode, idx, rr).retrieve(task["task"], limit=k)
                rows[mode].append(_quality(hits, expected, k))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {
        "suite": suite.get("name", os.path.basename(suite_dir)),
        "n": len(suite["tasks"]),
        "k": k,
        "modes": {
            mode: {
                "recall_at_1": _mean([r[1] for r in rs]),
                "recall_at_k": _mean([r[0] for r in rs]),
                "mrr": _mean([r[2] for r in rs]),
            }
            for mode, rs in rows.items()
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="retrieval recall ablation")
    ap.add_argument("suite_dir", nargs="?", default="eval/suites/hard")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--reranker", choices=["none", "heavy"], default="none",
                    help="add a hybrid+cross-encoder mode (needs the semantic extra)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--device", default="",
                    help="semantic encoder device (cuda/cpu); empty = auto (CUDA if present)")
    args = ap.parse_args(argv)
    if args.device:
        os.environ["RBG_EMBEDDING_DEVICE"] = args.device
    report = run(args.suite_dir, args.k, args.reranker)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"{report['suite']} ({report['n']} tasks, k={report['k']})")
    for mode in report["modes"]:
        m = report["modes"][mode]
        print(f"  {mode:<8} recall@1={m['recall_at_1']:.3f} "
              f"recall@{report['k']}={m['recall_at_k']:.3f} mrr={m['mrr']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
