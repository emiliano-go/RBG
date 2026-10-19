"""Runnable checks for every model and the end-to-end state machine.

Run: ``python3 -m pytest tests -q``  (numpy only).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.index import RepoIndex  # noqa: E402
from harness.models import (  # noqa: E402
    ACTIONS, FEATURE_NAMES, ChangeImpactModel, ConformalCalibrator, FlakyModel, GBDT,
    PlattScaler, Policy, TestSelector, extract_features, select_action,
)
from harness.retrieval import BM25, DenseEncoder, HybridRetriever  # noqa: E402
from harness.verify import Sandbox, Verifier, parse_pytest  # noqa: E402
from harness.__main__ import (  # noqa: E402
    BUGGY_SHOP, FIXED_SHOP, TEST_SHOP, ScriptedLLM, _make_demo_repo,
)
from harness.orchestrator import EditOp, Orchestrator  # noqa: E402
from harness.training import LearnedModels, train_from_ledger  # noqa: E402
from harness.drift import drift_report, psi  # noqa: E402
from harness import mcp_server  # noqa: E402


def _repo(tmp_path):
    (tmp_path / "a.py").write_text("def alpha():\n    return 'alpha value'\n")
    (tmp_path / "b.py").write_text("def beta():\n    return beta_helper()\n\ndef beta_helper():\n    return 42\n")
    return RepoIndex.build(str(tmp_path))


# -- retrieval -------------------------------------------------------------- #
def test_bm25_ranks_relevant_first():
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    idx = _repo(d)
    bm = BM25(idx.chunks)
    ranked = bm.rank("beta_helper")
    assert ranked, "expected at least one hit"
    assert idx.chunks[ranked[0][0]].path == "b.py"


def test_dense_encoder_similarity():
    enc = DenseEncoder(dim=256).fit(["def parse_config(path)", "def parse_ini_file(path)", "class DatabaseConnection"])
    a = enc.encode("parse config file")
    b = enc.encode("def parse_ini_file(path)")
    c = enc.encode("class DatabaseConnection")
    assert float(a @ b) > float(a @ c)


def test_hybrid_fuses_and_reranker_reorders():
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    idx = _repo(d)
    hits = HybridRetriever(idx).retrieve("beta_helper", limit=5)
    assert hits and hits[0].path == "b.py"
    from harness.retrieval import Reranker
    rr = Reranker().rerank("beta_helper", hits, idx.chunks)
    assert rr[0].path == "b.py"


def test_dense_encoder_is_deterministic_across_processes():
    """The old encoder used salted ``hash()``; two processes must now agree."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys; sys.path.insert(0, __ROOT__);"
        "from harness.retrieval import DenseEncoder;"
        "e=DenseEncoder(256).fit(['def alpha_beta gamma', 'class DeltaEpsilon']);"
        "print(' '.join('%.9f' % x for x in e.encode('alpha beta gamma delta')))"
    ).replace("__ROOT__", repr(root))
    outs = []
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True,
                                   text=True, env=env, check=True).stdout.strip())
    assert outs[0] == outs[1] and outs[0]


def test_query_expansion_adds_related_terms():
    import tempfile
    from pathlib import Path
    from harness.retrieval import BM25, expand_query
    d = Path(tempfile.mkdtemp())
    (d / "m.py").write_text(
        "def beta_helper():\n    return 42\n\n"
        "def gamma_wrapper():\n    return beta_helper()\n"
    )
    idx = RepoIndex.build(str(d))
    bm = BM25(idx.chunks)
    expanded = expand_query("beta_helper gamma", bm, idx.chunks)
    assert expanded.startswith("beta_helper gamma")
    assert len(expanded) >= len("beta_helper gamma")


def test_lsi_encoder_ranks_and_is_normalised():
    from harness.retrieval import LSIEncoder
    corpus = [
        "def compute_invoice_total(items):  # sum line item prices and tax",
        "def billing_report(account):  # uses compute_invoice_total to bill",
        "def render_widget(color):  # draw a widget with a theme colour",
        "def theme_palette(name):  # select colours for the widget theme",
    ]
    enc = LSIEncoder(k=4).fit(corpus)
    assert (enc.doc_vecs ** 2).sum(1) == pytest.approx([1.0] * 4)
    assert (enc.encode(corpus[1]) == enc.encode(corpus[1])).all()
    order = [i for i, _ in enc.rank("widget theme", [type("C", (), {"content": t})() for t in corpus], 4)]
    assert order and order[0] in {2, 3}


def test_mmr_prefers_a_distinct_document():
    from harness.retrieval import mmr
    q = np.array([1.0, 0.0])
    docs = np.array([[0.99, 0.0], [0.98, 0.0], [0.5, 0.86]])  # 0,1 near-duplicate
    chosen = mmr(q, docs, [0, 1, 2], k=2, lam=0.4)
    assert set(chosen) == {0, 2}  # relevance picks 0, diversity drops the duplicate 1


# -- models ----------------------------------------------------------------- #
def test_gbdt_learns_separable_function():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    y = ((X[:, 0] + X[:, 1] ** 2) > 0).astype(float)
    m = GBDT(n_estimators=60, learning_rate=0.3, max_depth=3, loss="logloss").fit(X[:400], y[:400])
    acc = float(((m.predict(X[400:]) > 0.5).astype(float) == y[400:]).mean())
    assert acc > 0.85, acc
    imp = m.feature_importance()
    assert imp[0] > 0 and imp[1] > 0


def test_platt_calibration_improves_ece():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=600) * 1.5
    y = (rng.random(600) < 1 / (1 + np.exp(-raw))).astype(float)
    cal = PlattScaler().fit(raw, y)
    ece = PlattScaler.ece([cal(float(s)) for s in raw], y)
    assert 0.0 <= ece <= 0.25


def test_extract_features_classification():
    f = extract_features("fix the auth token bug")
    assert f.task_class == "security_sensitive"
    assert f.values["security_sensitive"] == 1.0
    g = extract_features("refactor the parser")
    assert g.task_class == "refactor"


def test_policy_risk_and_gates():
    pol = Policy()
    low = pol.decide(extract_features("fix a typo in docstring"))
    high = pol.decide(extract_features("change the public API auth signature"))
    assert high.risk > low.risk
    assert high.verification["require_full_suite"] is True


def test_test_selector_flags_mandatory_and_ranks_by_relevance():
    tests = [
        {"test_id": "tests/test_security.py::test_x", "duration": 2.0, "history": []},
        {"test_id": "tests/test_shop.py::test_apply_discount", "duration": 1.0, "history": [True, False, True]},
        {"test_id": "tests/test_unrelated.py::test_z", "duration": 1.0, "history": []},
    ]
    ranked = TestSelector().rank(tests, ["apply_discount"])
    # relevant test is first; mandatory suite is flagged but not pinned to top
    assert ranked[0].test_id.startswith("tests/test_shop")
    sec = [r for r in ranked if "security" in r.test_id][0]
    assert sec.mandatory and "mandatory_suite" in sec.evidence
    assert not ranked[0].mandatory
    shop = [r for r in ranked if "apply_discount" in r.test_id][0]
    assert shop.predicted_relevance > 0.3


def test_flaky_model():
    assert FlakyModel().predict([True, False, True, False]) > 0.5
    assert FlakyModel().predict([True, True, True, True]) == 0.0


# -- verification sandbox --------------------------------------------------- #
def test_sandbox_blocks_escape_and_runs(tmp_path):
    sb = Sandbox(str(tmp_path))
    out = sb.run(["python3", "-c", "print('hi')"])
    assert out.ok and "hi" in out.stdout
    with pytest.raises(PermissionError):
        sb.run(["ls"], cwd="/")


def test_parse_pytest_extracts_outcomes():
    text = "tests/test_a.py::test_one PASSED\ntests/test_a.py::test_two FAILED\n1 passed 1 failed"
    outcomes = parse_pytest(text)
    assert {o.test_id for o in outcomes} == {"tests/test_a.py::test_one", "tests/test_a.py::test_two"}
    assert any(o.status == "failed" for o in outcomes)


# -- end to end ------------------------------------------------------------- #
def test_end_to_end_demo_verifies(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    orch = Orchestrator(root, ScriptedLLM())
    report = orch.run("fix the discount calculation", task_id="t1")
    assert report.success, report.to_dict()
    assert "shop.py" in report.files_changed
    assert orch.ledger.store.read_ledger(), "ledger must be persisted to libSQL"
    comps = {e["component"] for e in orch.ledger.entries}
    assert {"policy", "retriever", "verifier", "orchestrator"} <= comps
    assert os.path.exists(os.path.join(root, "test_shop.py"))


def test_repair_loop_uses_diagnostics(tmp_path):
    from harness.orchestrator import EditOp

    class TwoStepLLM:
        def __init__(self):
            self.calls = 0

        def plan(self, task, context):
            return "attempt"

        def edit(self, task, plan, context, diagnostics=""):
            self.calls += 1
            if diagnostics:  # second pass sees the compile error
                return [EditOp("shop.py", "def ok():\n    return 1\n")]
            return [EditOp("shop.py", "def broken(:\n    pass\n")]

    root = str(tmp_path / "repo2")
    _make_demo_repo(root)
    llm = TwoStepLLM()
    report = Orchestrator(root, llm, max_iterations=3).run("fix it", task_id="t2")
    assert llm.calls == 2
    assert report.iterations == 2


def test_runs_get_unique_task_ids(tmp_path):
    root = str(tmp_path / "unique")
    _make_demo_repo(root)
    r1 = Orchestrator(root, ScriptedLLM()).run("fix it")
    r2 = Orchestrator(root, ScriptedLLM()).run("fix it")
    assert r1.task_id and r2.task_id and r1.task_id != r2.task_id


def test_loop_unions_edits_across_iterations(tmp_path):
    from harness.orchestrator import EditOp

    class TwoFileLLM:
        def __init__(self):
            self.calls = 0

        def plan(self, task, context):
            return "attempt"

        def edit(self, task, plan, context, diagnostics=""):
            self.calls += 1
            if self.calls == 1:
                return [EditOp("a.py", "def broken(:\n    pass\n")]
            return [EditOp("b.py", "def ok():\n    return 1\n")]

    root = str(tmp_path / "repo_union")
    _make_demo_repo(root)
    report = Orchestrator(root, TwoFileLLM(), max_iterations=2).run("fix it", task_id="union")
    assert set(report.files_changed) == {"a.py", "b.py"}


def test_edits_run_in_disposable_worktree(tmp_path):
    import subprocess
    root = str(tmp_path / "wtrepo")
    _make_demo_repo(root)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "init"], cwd=root, check=True)
    before = open(os.path.join(root, "shop.py")).read()
    report = Orchestrator(root, ScriptedLLM()).run("fix the discount calculation", task_id="wt1")
    assert report.success, report.to_dict()
    assert "shop.py" in report.files_changed
    # the caller's checkout is untouched ...
    assert open(os.path.join(root, "shop.py")).read() == before
    # ... and no worktree is left behind
    wl = subprocess.run(["git", "-C", root, "worktree", "list"],
                        capture_output=True, text=True).stdout.strip().splitlines()
    assert len(wl) == 1, wl



# -- MCP server facade ------------------------------------------------------ #
def _call(name, arguments):
    return mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": name, "arguments": arguments}})


def test_mcp_initialize_and_tools_list():
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                              "params": {"protocolVersion": "2026-07-28"}})
    assert init["result"]["protocolVersion"] == "2026-07-28"
    assert init["result"]["serverInfo"]["name"] == "rbg-mcp"
    listed = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in listed["result"]["tools"]}
    assert {"search_code", "retrieve_context", "find_symbol", "rank_tests",
            "run_checks", "get_diagnostics", "propose_patch", "read_chunk",
            "train_models", "get_calibration"} <= names


def test_mcp_tools_return_structured_content(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    out = _call("retrieve_context", {"root": root, "query": "discount", "limit": 3})
    assert out["result"]["isError"] is False
    sc = out["result"]["structuredContent"]
    assert sc["chunks"] and all(c["snippet"].strip() for c in sc["chunks"])
    assert sc["context_token_estimate"] > 0
    sym = _call("find_symbol", {"root": root, "name": "apply_discount"})
    assert any(d["path"] == "shop.py" for d in sym["result"]["structuredContent"]["definitions"])


def test_mcp_discovery_is_snippet_and_read_chunk_returns_full(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    out = _call("retrieve_context", {"root": root, "query": "discount", "limit": 3})
    chunk = out["result"]["structuredContent"]["chunks"][0]
    assert "chunk_id" in chunk and "content" not in chunk
    full = _call("read_chunk", {
        "root": root, "chunk_id": chunk["chunk_id"]})["result"]["structuredContent"]
    assert full["chunk_id"] == chunk["chunk_id"]
    assert full["content_hash"] == chunk["content_hash"]
    assert full["content"].strip() and len(full["content"]) >= len(chunk["snippet"])


def test_mcp_errors_are_contained():
    assert mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"] == -32601
    bad = _call("search_code", {"root": "/nonexistent/xyz", "query": "a"})
    assert bad["result"]["isError"] is True
    unknown = _call("no_such_tool", {})
    assert "error" in unknown


def test_mcp_notifications_and_propose_patch_never_write(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    assert mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    before = open(os.path.join(root, "shop.py")).read()
    out = _call("propose_patch", {"root": root, "path": "shop.py", "content": "def x():\n    pass\n"})
    assert out["result"]["structuredContent"]["applied"] is False
    assert open(os.path.join(root, "shop.py")).read() == before


# -- root authorization (deny by default) ----------------------------------- #
def test_mcp_root_denied_without_allowlist(tmp_path, monkeypatch):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    monkeypatch.delenv("HARNESS_ALLOWED_ROOTS", raising=False)
    out = _call("retrieve_context", {"root": root, "query": "discount", "limit": 1})
    assert out["result"]["isError"] is True
    assert "authoriz" in out["result"]["content"][0]["text"].lower()


def test_mcp_root_outside_allowlist_is_denied(tmp_path, monkeypatch):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    monkeypatch.setenv("HARNESS_ALLOWED_ROOTS", str(tmp_path / "elsewhere"))
    out = _call("retrieve_context", {"root": root, "query": "discount", "limit": 1})
    assert out["result"]["isError"] is True
    assert "not allowed" in out["result"]["content"][0]["text"].lower()


def test_mcp_root_within_allowlist_is_allowed(tmp_path, monkeypatch):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    monkeypatch.setenv("HARNESS_ALLOWED_ROOTS", str(tmp_path))
    out = _call("retrieve_context", {"root": root, "query": "discount", "limit": 1})
    assert out["result"]["isError"] is False


def test_authorize_root_rejects_sibling_prefix(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "ab").mkdir()
    monkeypatch.setenv("HARNESS_ALLOWED_ROOTS", str(tmp_path / "a"))
    with pytest.raises(PermissionError):
        mcp_server._authorize_root(str(tmp_path / "ab"))


# -- index hardening (gitignore, constants, interpreter) -------------------- #
def test_index_honors_gitignore(tmp_path):
    import subprocess
    repo = tmp_path / "g"
    (repo / "src").mkdir(parents=True)
    (repo / "vendor").mkdir()
    (repo / "src" / "real.py").write_text("def real_symbol():\n    return 1\n")
    (repo / "vendor" / "copied.py").write_text("def vendored_copy():\n    return 1\n")
    (repo / ".gitignore").write_text("vendor/\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    idx = RepoIndex.build(str(repo))
    assert any("real.py" in f for f in idx.files)
    assert not any("vendor" in f for f in idx.files)
    assert all("vendor" not in c.path for c in idx.chunks)


def test_module_level_constants_are_symbols(tmp_path):
    repo = tmp_path / "c"
    repo.mkdir()
    (repo / "rules.py").write_text(
        "import re\n\nMARKER_RE = re.compile(r'x')\nSKIP = frozenset({'a', 'b'})\n\n"
        "def use():\n    return MARKER_RE.search('x')\n")
    idx = RepoIndex.build(str(repo))
    names = {s.name: s.kind for s in idx.symbols}
    assert names.get("MARKER_RE") == "constant"
    assert names.get("SKIP") == "constant"
    assert any(c.symbol == "MARKER_RE" and c.content for c in idx.chunks)


def test_sandbox_prefers_project_venv(tmp_path):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    assert Sandbox(str(tmp_path)).python.endswith(".venv/bin/python")


def test_retrieve_returns_content(tmp_path):
    idx = _repo(tmp_path) if False else RepoIndex.build(str(tmp_path))
    (tmp_path / "mod.py").write_text("def helper():\n    return 'needle value'\n")
    idx = RepoIndex.build(str(tmp_path))
    from harness.retrieval import HybridRetriever
    hits = HybridRetriever(idx).retrieve("helper needle", limit=3)
    assert hits and all(h.content for h in hits)


# -- learned components (wired MLs) ----------------------------------------- #
def _synthetic_ledger(root, n=24):
    """Write a ledger rich enough to train every model, return (root, index)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "src.py").write_text("def compute(x):\n    return x + 1\n\n\ndef other(y):\n    return y * 2\n")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_src.py").write_text(
        "from src import compute\n\n\ndef test_compute():\n    assert compute(1) == 2\n\n\n"
        "def test_other():\n    assert True\n")
    idx = RepoIndex.build(str(root))
    src = [c for c in idx.chunks if c.path == "src.py" and c.kind == "function"][0]
    test = [c for c in idx.chunks if c.path.endswith("test_src.py") and c.kind == "function"][0]
    entries = []
    for i in range(n):
        tid = f"t{i}"
        vec = [0.1 if i % 2 == 0 else 0.9] + [0.0] * (len(FEATURE_NAMES) - 1)
        ok = i % 2 == 0
        entries += [
            {"component": "task_features", "task_id": tid, "result": {"feature_vector": vec}},
            {"component": "orchestrator", "task_id": tid,
             "result": {"success": ok, "files_changed": ["src.py"]}},
            {"component": "test_selector", "task_id": tid,
             "result": {"changed_symbols": ["compute"]}},
            {"component": "verifier", "task_id": tid, "result": {"outcomes": [
                {"test_id": "tests/test_src.py::test_compute",
                 "status": "passed" if ok else "failed", "duration_s": 1.0},
                {"test_id": "tests/test_src.py::test_other",
                 "status": "failed" if ok else "passed", "duration_s": 1.0}]}},
            {"component": "retriever", "task_id": tid, "result": {
                "query": "compute value", "candidates": [
                    {"path": "src.py", "start_line": src.start_line},
                    {"path": "tests/test_src.py", "start_line": test.start_line}]}},
        ]
    store = hdb.Store(str(root))
    store.save_index(idx)
    for e in entries:
        e.setdefault("revision", idx.revision)
        store.append_ledger(e)
    store.close()
    return str(root), idx


def test_conformal_calibrator_coverage_and_ood():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=400) * 2
    y = (rng.random(400) < 1 / (1 + np.exp(-raw))).astype(float)
    cal = ConformalCalibrator(alpha=0.1).fit(raw, y)
    assert 0.75 <= cal.coverage(raw, y) <= 1.0
    train = rng.normal(size=(400, 4))
    cal2 = ConformalCalibrator().fit(raw, y, matrix=train)
    assert cal2.ood_score(np.zeros(4)) < cal2.ood_score(np.ones(4) * 50)


def test_change_impact_model_learns_token_hit():
    ex = []
    for _ in range(60):
        ex.append(("compute_total", "tests/test_compute_total.py", {"compute", "total", "test"}, 3, 1.0))
        ex.append(("compute_total", "tests/test_unrelated.py", {"unrelated", "test"}, 3, 0.0))
    m = ChangeImpactModel().train(ex)
    hi = m.predict("compute_total", "tests/test_compute_total.py", {"compute", "total"})
    lo = m.predict("compute_total", "tests/test_unrelated.py", {"unrelated"})
    assert hi > lo


def test_train_from_ledger_trains_all_models(tmp_path):
    root, idx = _synthetic_ledger(tmp_path / "lr")
    models = train_from_ledger(root, index=idx)
    assert models.risk_trained and models.selector_trained and models.impact_trained
    assert models.reranker is not None
    assert models.metrics["risk"]["conformal_coverage"] >= 0.7
    assert models.metrics["risk"]["positive_rate"] == 0.5


def test_orchestrator_uses_learned_risk_in_shadow(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(float)
    learned = LearnedModels(policy=Policy().train_risk(X.tolist(), y.tolist()))
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    rep = Orchestrator(root, ScriptedLLM(), learned=learned).run("fix discount", task_id="lr1")
    assert rep.decision["risk"]["source"] == "gbdt"


def test_flaky_rerun_recovers_instead_of_failing(tmp_path):
    root = tmp_path / "flaky"
    root.mkdir()
    (root / "shop.py").write_text(BUGGY_SHOP)
    (root / "test_shop.py").write_text(TEST_SHOP)
    (root / "tests").mkdir()
    (root / "tests" / "test_flaky.py").write_text(
        "import os\n\n\n"
        "def test_flaky():\n"
        "    p = os.path.join(os.path.dirname(__file__), '..', 'flaky_counter')\n"
        "    n = int(open(p).read()) if os.path.exists(p) else 0\n"
        "    open(p, 'w').write(str(n + 1))\n"
        "    assert n > 0\n")
    orch = Orchestrator(str(root), ScriptedLLM())
    orch._history = {"tests/test_flaky.py::test_flaky": [True, False]}
    rep = orch.run("fix the discount calculation", task_id="fl1")
    assert any(s.startswith("flaky_rerun_recovered") for s in rep.scope_reductions), rep.to_dict()
    assert rep.success, rep.to_dict()


def test_mcp_train_models_and_calibration(tmp_path):
    root, _ = _synthetic_ledger(tmp_path / "mcp")
    mcp_server._LEARNED_CACHE.clear()
    sc = _call("train_models", {"root": root})["result"]["structuredContent"]
    assert sc["trained"] and sc["risk_trained"] and sc["impact_trained"]
    cal = _call("get_calibration", {"root": root})["result"]["structuredContent"]
    assert cal["fitted"] and cal["qhat"] is not None
    imp = _call("get_change_impact", {"root": root, "symbols": ["compute"]})["result"]["structuredContent"]
    assert "predicted_affected" in imp["impact"]["compute"]


# -- drift monitoring (spec 5.8) -------------------------------------------- #
def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    assert psi(a, a) < 1e-6
    assert psi(a, b) < 0.25
    assert psi(a, b + 5.0) > 0.25


def _ledger_entries(shifts: list[float]) -> list[dict]:
    entries = []
    for i, s in enumerate(shifts):
        vec = [s] + [0.0] * (len(FEATURE_NAMES) - 1)
        entries.append({"component": "task_features", "task_id": f"t{i}",
                        "result": {"feature_vector": vec}})
        entries.append({"component": "orchestrator", "task_id": f"t{i}",
                        "result": {"success": i % 2 == 0, "files_changed": ["src.py"]}})
    return entries


def test_drift_report_flags_feature_shift():
    entries = _ledger_entries([0.0] * 20 + [1.0] * 20)
    rep = drift_report(None, entries, feature_names=FEATURE_NAMES)
    assert rep.n_reference >= 8 and rep.n_recent >= 8
    assert rep.verdict in ("warning", "drifted")
    assert any(c.startswith("feature_") for c in rep.reason_codes)


def test_drift_report_stable_when_unchanged():
    entries = _ledger_entries([0.0] * 40)
    rep = drift_report(None, entries, feature_names=FEATURE_NAMES)
    assert rep.verdict == "stable"


def test_mcp_check_drift_reports_verdict(tmp_path):
    root, _ = _synthetic_ledger(tmp_path / "drift")
    mcp_server._LEARNED_CACHE.clear()
    out = _call("check_drift", {"root": root})["result"]["structuredContent"]
    assert out["advisory"] is True
    assert out["drift"]["verdict"] in ("stable", "warning", "drifted")


def test_nl_queries_are_non_leaky(tmp_path):
    from eval.effectiveness import nl_queries
    repo = tmp_path / "nl"
    repo.mkdir()
    (repo / "lib.py").write_text(
        "def locate_marker(path):\n"
        "    'return the first zero index'\n"
        "    return path.index(0)\n\n\n"
        "def scan_wrapper(path):\n"
        "    # discover sentinel byte position while streaming\n"
        "    return locate_marker(path)\n")
    idx = RepoIndex.build(str(repo))
    qs = nl_queries(idx, min_terms=3)
    assert qs, "expected at least one non-leaky query"
    for query, key, _ in qs:
        target = next(c for c in idx.chunks if (c.path, c.start_line) == key)
        assert not any(t in target.content.lower() for t in query.split()), query


# -- libSQL persistence layer (mandatory: Turso is the only store) ---------- #
from harness import db as hdb  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_global_db(tmp_path_factory, monkeypatch):
    """Keep the shared/cross-repo tier and per-repo stores out of the operator's
    real data directory."""
    base = tmp_path_factory.mktemp("hdb")
    monkeypatch.setenv("HARNESS_GLOBAL_DB", str(base / "global.db"))
    monkeypatch.setenv("HARNESS_DB_ROOT", str(base / "repos"))


def _db_repo(tmp_path):
    repo = tmp_path / "dbrepo"
    repo.mkdir()
    (repo / "core.py").write_text(
        "def compute_metric(samples):\n"
        "    'aggregate samples into a single metric'\n"
        "    return sum(samples)\n\n\n"
        "def compute_total(samples):\n"
        "    'aggregate samples into a running total'\n"
        "    return compute_metric(samples) * 2\n")
    (repo / "test_core.py").write_text(
        "from core import compute_metric\n\n\n"
        "def test_compute_metric():\n"
        "    assert compute_metric([1, 2, 3]) == 6\n")
    return repo


def test_db_index_roundtrip(tmp_path):
    repo = _db_repo(tmp_path)
    idx = RepoIndex.build(str(repo))
    store = hdb.Store(str(repo))
    store.save_index(idx)
    loaded = store.load_index(idx.revision)
    assert loaded is not None and loaded.revision == idx.revision
    assert [c.content_hash for c in loaded.chunks] == [c.content_hash for c in idx.chunks]
    assert sorted(s.name for s in loaded.symbols) == sorted(s.name for s in idx.symbols)
    assert loaded.files == idx.files
    assert loaded.callers_of("compute_metric") == idx.callers_of("compute_metric")
    assert store.load_index("stale-revision") is None
    store.close()


def test_db_encoder_reload_matches_fit(tmp_path):
    repo = _db_repo(tmp_path)
    idx = RepoIndex.build(str(repo))
    store = hdb.Store(str(repo))
    store.save_index(idx)
    retr = HybridRetriever(idx)
    store.save_encoder(idx.revision, retr.encoder.to_state(), retr.encoder.document_vectors(idx.chunks))
    reloaded = store.load_encoder(idx.revision, len(idx.chunks))
    assert reloaded is not None
    assert np.allclose(reloaded.document_vectors(idx.chunks), retr.encoder.document_vectors(idx.chunks))
    a = retr.retrieve("compute metric total", limit=3)
    b = HybridRetriever(idx, encoder=reloaded).retrieve("compute metric total", limit=3)
    assert [h.path for h in a] == [h.path for h in b]
    store.close()


def test_db_vector_search_returns_chunks(tmp_path):
    repo = _db_repo(tmp_path)
    idx = RepoIndex.build(str(repo))
    store = hdb.Store(str(repo))
    store.save_index(idx)
    retr = HybridRetriever(idx)
    store.save_encoder(idx.revision, retr.encoder.to_state(), retr.encoder.document_vectors(idx.chunks))
    rows = store.vector_search(retr.encoder.encode("compute metric"), 3)
    assert rows and all(1 <= cid <= len(idx.chunks) for cid, _ in rows)
    assert store.vector_search(retr.encoder.encode(""), 3) == []
    store.close()


def test_db_vectors_survive_reindex_then_resave(tmp_path):
    """A DELETE never clears libsql's vector-index shadow rows; re-saving after
    a re-index must recreate the table instead of failing."""
    repo = _db_repo(tmp_path)
    idx = RepoIndex.build(str(repo))
    store = hdb.Store(str(repo))
    store.save_index(idx)
    retr = HybridRetriever(idx)
    store.save_encoder(idx.revision, retr.encoder.to_state(), retr.encoder.document_vectors(idx.chunks))
    store.save_index(idx)  # re-index drops the old vector table
    store.save_encoder(idx.revision, retr.encoder.to_state(), retr.encoder.document_vectors(idx.chunks))
    assert store.vector_search(retr.encoder.encode("compute metric"), 3)
    store.close()


def _mixed_repo(tmp_path):
    """Repo with source, docs, and test chunks that all mention the same symbol."""
    repo = tmp_path / "mixed"
    repo.mkdir()
    (repo / "core.py").write_text(
        "def compute_metric(samples):\n"
        "    'aggregate samples into a single metric'\n"
        "    return sum(samples)\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text(
        "# Guide\n\nUse compute_metric to aggregate samples.\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_core.py").write_text(
        "from core import compute_metric\n\n\n"
        "def test_compute_metric():\n"
        "    assert compute_metric([1, 2, 3]) == 6\n\n\n"
        "def _helper():\n"
        "    return compute_metric([1])\n")
    return repo


def test_path_class_weight_prefers_source(tmp_path):
    from harness.retrieval import _path_class, PATH_WEIGHT
    assert _path_class("core.py") == "source"
    assert _path_class("docs/guide.md") == "docs"
    assert _path_class("tests/test_core.py") == "tests"
    assert PATH_WEIGHT["source"] > PATH_WEIGHT["docs"] > PATH_WEIGHT["tests"]
    repo = _mixed_repo(tmp_path)
    idx = RepoIndex.build(str(repo))
    hits = HybridRetriever(idx).retrieve("compute_metric", limit=5)
    assert hits, "expected hits"
    assert hits[0].path == "core.py", f"source should rank first, got {hits[0].path}"


def test_retrieve_include_exclude_globs(tmp_path):
    repo = _mixed_repo(tmp_path)
    idx = RepoIndex.build(str(repo))
    retr = HybridRetriever(idx)
    hits = retr.retrieve("compute_metric", limit=5, include=["core.py"])
    assert hits and all(h.path == "core.py" for h in hits)
    hits = retr.retrieve("compute_metric", limit=5, exclude=["tests/**", "docs/**"])
    assert hits and all(h.path == "core.py" for h in hits)


def test_rank_tests_ignores_helpers(tmp_path):
    from harness.index import is_test_chunk
    repo = _mixed_repo(tmp_path)
    idx = RepoIndex.build(str(repo))
    helper = next(c for c in idx.chunks if c.symbol == "_helper")
    assert not is_test_chunk(helper)
    test_fn = next(c for c in idx.chunks if c.symbol == "test_compute_metric")
    assert is_test_chunk(test_fn)
    out = mcp_server.rank_tests(str(repo), ["compute_metric"])
    ids = [t["test_id"] for t in out["tests"]]
    assert not any("_helper" in i for i in ids)
    assert any("test_compute_metric" in i for i in ids)


def test_find_symbol_path_filter(tmp_path):
    repo = _mixed_repo(tmp_path)
    out = mcp_server.find_symbol(str(repo), "compute_metric")
    assert len(out["definitions"]) >= 2
    out = mcp_server.find_symbol(str(repo), "compute_metric", path="core.py")
    assert out["definitions"] and all(d["path"] == "core.py" for d in out["definitions"])
    out = mcp_server.find_symbol(str(repo), "compute_metric", kind="function")
    assert out["definitions"] and all(d["kind"] == "function" for d in out["definitions"])


def test_ledger_persists_to_store_and_mcp_reads_it(tmp_path):
    from harness.verify import Ledger
    repo = _db_repo(tmp_path)
    store = hdb.Store(str(repo))
    store.save_index(RepoIndex.build(str(repo)))
    ledger = Ledger(store)
    ledger.record(component="orchestrator", task_id="t-db", repo_id="r", revision="rev",
                  model_name="policy", result={"status": "ok"}, reason_codes=["a"])
    store.close()
    out = _call("get_diagnostics", {"root": str(repo), "task_id": "t-db"})
    traces = out["result"]["structuredContent"]["traces"]
    assert traces and traces[0]["task_id"] == "t-db"
    mcp_server._INDEX_CACHE.clear()
    mcp_server._STORE_CACHE.clear()
    assert not os.path.exists(os.path.join(str(repo), ".harness", "traces.jsonl"))


def test_partial_pooling_borrows_from_other_repos(tmp_path):
    """A repo with too few local rows trains on the pooled cross-repo examples
    instead of overfitting its own handful."""
    rich_root, rich_idx = _synthetic_ledger(tmp_path / "rich", n=40)
    rich = train_from_ledger(rich_root, index=rich_idx)
    assert rich.risk_trained
    assert rich.metrics["risk"]["pooling"]["source"] == "local"

    thin_root, thin_idx = _synthetic_ledger(tmp_path / "thin", n=6)
    thin = train_from_ledger(thin_root, index=thin_idx)
    info = thin.metrics["risk"]["pooling"]
    assert thin.risk_trained
    assert info["source"] == "pooled"
    assert info["n_prior"] >= 8
    assert info["n_local"] == 6


def test_mcp_db_tools_and_resources(tmp_path):
    repo = _db_repo(tmp_path)
    mcp_server._INDEX_CACHE.clear()
    mcp_server._STORE_CACHE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    stats = _call("db_stats", {"root": str(repo)})["result"]["structuredContent"]
    assert stats["chunks"] > 0 and stats["attached_global"] is True
    vs = _call("vector_search", {"root": str(repo), "query": "compute metric"})["result"]["structuredContent"]
    assert vs["results"] and vs["backend"] == "libsql_vector_idx"
    templates = mcp_server.handle({"jsonrpc": "2.0", "id": 1,
                                   "method": "resources/templates/list"})["result"]["resourceTemplates"]
    assert any("ledger" in t["uriTemplate"] for t in templates)
    read = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "resources/read",
                              "params": {"uri": f"harness://repo/db?root={repo}"}})
    assert "contents" in read["result"]
    prompt = mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "prompts/get",
                                "params": {"name": "grounded_change",
                                           "arguments": {"root": str(repo), "task": "fix"}}})
    assert prompt["result"]["messages"][0]["content"]["type"] == "text"


# -- spec gap closure: real decision features, action menu, retriever contract #
def test_select_action_is_drawn_from_the_menu():
    cases = {
        ("run_full_suite",): select_action(has_failures=True, full_suite_required=True,
                                           full_suite_ran=False),
        ("ask_main_llm_to_revise",): select_action(has_failures=True, full_suite_required=False,
                                                   full_suite_ran=False),
        ("run_full_suite",): select_action(has_failures=False, full_suite_required=False,
                                           full_suite_ran=False, uncertainty=True),
        ("stop_with_verified_result",): select_action(has_failures=False, full_suite_required=False,
                                                      full_suite_ran=False),
    }
    for (expected,), (action, reasons) in cases.items():
        assert action == expected and action in ACTIONS
        assert reasons


def test_decision_uses_real_diff_features(tmp_path):
    """The risk decision must see changed_files/changed_lines, not the intake
    zeros (spec 5.3 candidate features)."""
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    orch = Orchestrator(root, ScriptedLLM())
    report = orch.run("fix the discount calculation", task_id="feat1")
    feats = [e for e in orch.ledger.store.read_ledger() if e["component"] == "task_features"]
    last = feats[-1]["result"]["values"]
    assert last["changed_files"] >= 1
    assert last["changed_lines"] > 0
    assert report.actions, "an action must be recorded"


def test_typecheck_unverified_only_when_checker_absent(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    orch = Orchestrator(root, ScriptedLLM())
    report = orch.run("fix the discount calculation", task_id="tc1")
    ran = any(c["name"] == "typecheck" for c in report.checks_run)
    assert ran == orch.verifier.typecheck_available()
    assert ("typecheck" in report.unverified) == (not ran)


def test_no_tests_collected_is_not_verified(tmp_path):
    from harness.__main__ import BUGGY_SHOP
    root = tmp_path / "no_tests"
    root.mkdir()
    (root / "shop.py").write_text(BUGGY_SHOP)
    report = Orchestrator(str(root), ScriptedLLM()).run("fix the discount calculation",
                                                        task_id="nt1")
    assert report.status == "partial", report.to_dict()
    assert report.success is False
    assert "tests" in report.unverified


def test_hit_and_mcp_outputs_carry_revision(tmp_path):
    idx = _repo(tmp_path)
    hits = HybridRetriever(idx).retrieve("beta_helper", limit=3)
    assert hits and all(h.commit_hash == idx.revision for h in hits)
    assert all(h.to_dict()["commit_hash"] == idx.revision for h in hits)
    mcp_server._INDEX_CACHE.clear()
    mcp_server._STORE_CACHE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    out = _call("search_code", {"root": str(tmp_path), "query": "beta_helper"})
    sc = out["result"]["structuredContent"]
    assert sc["index_fresh"] is True
    assert sc["results"][0]["commit_hash"] == sc["revision"]


def test_tree_revision_is_content_based_not_metadata(tmp_path):
    from harness.index import _tree_hash, _tree_signature
    root = str(tmp_path / "rev")
    _make_demo_repo(root)
    path = os.path.join(root, "shop.py")
    text = open(path).read()
    fingerprint, signature = _tree_hash(root), _tree_signature(root)
    st = os.stat(path)
    edited = ("x" if text[0] != "x" else "y") + text[1:]   # same length
    with open(path, "w") as fh:
        fh.write(edited)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))    # same mtime
    assert _tree_signature(root) == signature              # metadata unchanged
    assert _tree_hash(root) != fingerprint                 # content still caught


def test_index_cache_invalidates_after_edit(tmp_path):
    root = str(tmp_path / "cache")
    _make_demo_repo(root)
    mcp_server._INDEX_CACHE.clear()
    mcp_server._INDEX_SIGNATURE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    first = _call("search_code", {"root": root, "query": "discount"})["result"]["structuredContent"]
    with open(os.path.join(root, "shop.py"), "w") as fh:
        fh.write(FIXED_SHOP)
    second = _call("search_code", {"root": root, "query": "discount"})["result"]["structuredContent"]
    assert second["revision"] != first["revision"]
    assert second["index_fresh"] is True


# -- plan + step-level drift checks ----------------------------------------- #
def test_parse_plan_extracts_steps_and_files():
    from harness.plan import parse_plan
    known = {"shop.py", "test_shop.py", "src/a.py"}
    p = parse_plan("1. Edit shop.py.\n2. Run test_shop.py.\n3. Mention nonexistent.py.", known)
    assert [s.action for s in p.steps] == ["edit", "verify", "edit"]
    assert p.edit_files() == ["shop.py"]
    assert p.all_files() == ["shop.py", "test_shop.py"]


def test_scope_drift_flags_missing_and_extra():
    from harness.plan import parse_plan, scope_drift
    p = parse_plan("1. Edit shop.py.\n2. Edit test_shop.py.", {"shop.py", "test_shop.py"})
    d = scope_drift(p, ["shop.py", "extra.py"])
    assert d.missing == ["test_shop.py"]
    assert d.extra == ["extra.py"]
    assert not d.ok()


def test_audit_completion_flags_unedited_callers(tmp_path):
    from harness.plan import audit_completion
    idx = _repo(tmp_path)
    comp = audit_completion(idx, ["beta_helper"], ["a.py"], [])
    assert "b.py" in comp["affected_files"]
    assert comp["affected_not_edited"] == ["b.py"]


class _GreedyPlanLLM:
    """Plans two edits but only performs one — the missed step must surface."""
    def plan(self, task, context):
        return "1. Edit shop.py.\n2. Edit test_shop.py."

    def edit(self, task, plan, context, diagnostics=""):
        return [EditOp("shop.py", FIXED_SHOP)]


def test_orchestrator_records_plan_and_drift(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    orch = Orchestrator(root, ScriptedLLM())
    report = orch.run("fix the discount calculation", task_id="drift1")
    assert report.plan["edit_files"] == ["shop.py"]
    stage = {d["stage"] for d in report.drift}
    assert {"plan", "step"} <= stage
    assert report.completeness.get("advisory") is True


def test_orchestrator_feeds_missing_step_back(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    orch = Orchestrator(root, _GreedyPlanLLM())
    report = orch.run("fix the discount calculation", task_id="drift2")
    assert any(d.get("stage") == "step" and d["missing"] for d in report.drift)
    assert any(s.startswith("unmet_plan_steps:") and "test_shop.py" in s
               for s in report.scope_reductions)


def test_plan_audit_mcp_tool(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    out = _call("plan_audit", {
        "root": root, "task": "discount",
        "plan_text": "1. Edit shop.py.\n2. Edit test_shop.py.",
        "changed_files": ["shop.py"]})
    sc = out["result"]["structuredContent"]
    assert sc["plan"]["edit_files"] == ["shop.py", "test_shop.py"]
    assert sc["scope_drift"]["missing"] == ["test_shop.py"]
    assert sc["scope_drift"]["extra"] == []
    assert sc["completeness"]["advisory"] is True


def test_audit_completion_skips_generic_and_high_fanin_symbols(tmp_path):
    from harness.plan import audit_completion
    (tmp_path / "common.py").write_text("def shared():\n    return 1\n")
    (tmp_path / "leaf.py").write_text("def rare():\n    return 2\n")
    (tmp_path / "caller.py").write_text("from leaf import rare\n\ndef c():\n    return rare()\n")
    for i in range(8):
        (tmp_path / f"m{i}.py").write_text(
            f"from common import shared\n\ndef use{i}():\n    return shared()\n")
    idx = RepoIndex.build(str(tmp_path))
    comp = audit_completion(idx, ["shared", "__init__", "to_dict", "rare"],
                            ["common.py"], [])
    assert "shared" in comp["skipped_generic_symbols"], comp        # high fan-in
    assert "__init__" in comp["skipped_generic_symbols"]            # dunder
    assert "to_dict" in comp["skipped_generic_symbols"]             # generic name
    assert comp["affected_not_edited"] == ["caller.py"], comp


# -- real LLM provider (OpenAI-compatible) ---------------------------------- #
def test_parse_edits_accepts_fences_lists_and_bare_objects():
    from harness.providers import parse_edits
    fenced = 'here you go\n```json\n{"edits": [{"path": "a.py", "content": "x = 1\\n"}]}\n```\ndone'
    assert [(o.path, o.content) for o in parse_edits(fenced)] == [("a.py", "x = 1\n")]
    assert parse_edits('{"edits": []}') == []
    assert [o.path for o in parse_edits('[{"path": "b.py", "content": ""}]')] == ["b.py"]
    with pytest.raises(ValueError):
        parse_edits("the model rambled with no json at all")


def test_openai_compatible_llm_maps_reply_to_edit_ops(monkeypatch):
    from harness.providers import OpenAICompatLLM
    llm = OpenAICompatLLM(model="m", base_url="http://localhost:1/v1", api_key="sk-test")
    monkeypatch.setattr(llm, "_chat",
                        lambda system, user: '{"edits": [{"path": "shop.py", "content": "y"}]}')
    ops = llm.edit("fix", "plan", [])
    assert [(o.path, o.content) for o in ops] == [("shop.py", "y")]
