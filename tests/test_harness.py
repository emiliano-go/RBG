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


# -- experience memory (SWE-Exp) -------------------------------------------- #
def test_experience_encoder_is_unfit_and_deterministic():
    from harness.retrieval import EXPERIENCE_DIM, experience_encoder
    enc = experience_encoder()
    assert enc.dim == EXPERIENCE_DIM and enc.idf == {}  # unfit: no corpus stats
    v = enc.encode("fix the discount calculation")
    assert v.shape == (EXPERIENCE_DIM,)
    assert np.array_equal(v, enc.encode("fix the discount calculation"))


def test_experience_search_returns_nearest_and_caps_at_one(tmp_path):
    from harness import db
    from harness.retrieval import experience_encoder
    store = db.Store(str(tmp_path / "exp"))
    enc = experience_encoder()
    for i, task in enumerate(["fix the discount calculation",
                              "repair the invoice tax rounding"]):
        store.save_experience(task_id=f"t{i}", repo_id="r", revision="rev",
                              query=task, outcome="verified",
                              summary=f"plan: {task}", vector=enc.encode(task))
    hits = store.search_experiences(enc.encode("fix the discount calculation"), limit=1)
    assert len(hits) == 1 and hits[0]["task_id"] == "t0"
    store.close()


def test_orchestrator_injects_prior_experience(tmp_path):
    root = str(tmp_path / "exp-repo")
    _make_demo_repo(root)
    assert Orchestrator(root, ScriptedLLM()).run("fix the discount calculation",
                                                 task_id="first").success
    seen = {}

    class Spy(ScriptedLLM):
        def plan(self, task, context):
            seen["paths"] = [h.path for h in context]
            return super().plan(task, context)

    Orchestrator(root, Spy()).run("fix the discount calculation", task_id="second")
    assert Orchestrator.EXPERIENCE_PATH in seen["paths"]


# -- patch correctness ------------------------------------------------------ #
def test_patch_features_flag_test_edits_and_heuristic():
    from harness.models import PATCH_FEATURE_NAMES, PatchCorrectnessModel, patch_features
    clean = {"shop.py": ("def f():\n    return 1\n", "def f():\n    return 2\n")}
    testy = {"tests/test_shop.py": ("def t():\n    assert f() == 1\n",
                                    "def t():\n    assert f() == 2\n")}
    i = PATCH_FEATURE_NAMES.index("touches_tests")
    fc, ft = patch_features(clean, ["f"]), patch_features(testy, ["f"])
    assert fc[i] == 0.0 and ft[i] == 1.0
    assert PatchCorrectnessModel().predict(ft) > PatchCorrectnessModel().predict(fc)


def test_patch_examples_label_from_regression_evidence():
    from harness.training import patch_examples
    entries = [
        {"task_id": "A", "component": "patch_features",
         "result": {"feature_vector": [0.0] * 11}},
        {"task_id": "A", "component": "test_selector", "result": {"changed_symbols": ["f"]}},
        {"task_id": "A", "component": "verifier",
         "result": {"outcomes": [{"test_id": "t1", "status": "passed"}]}},
        # B: a test that passed under A now fails -> regression evidence (label 1)
        {"task_id": "B", "component": "patch_features",
         "result": {"feature_vector": [1.0] * 11}},
        {"task_id": "B", "component": "test_selector", "result": {"changed_symbols": ["g"]}},
        {"task_id": "B", "component": "verifier",
         "result": {"outcomes": [{"test_id": "t1", "status": "failed"}]}},
    ]
    X, y = patch_examples(entries)
    assert list(y) == [0.0, 1.0]
    assert X.shape == (2, 11)


def test_test_selection_beats_alphabetical_at_third_budget(tmp_path):
    from eval.effectiveness import eval_test_selection
    fns = ["alpha_fn", "beta_fn", "gamma_fn", "delta_fn", "epsilon_fn"]
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "shop.py").write_text(
        "\n".join(f"def {f}():\n    return '{f}'\n" for f in fns))
    for i, f in enumerate(fns, 1):
        (tmp_path / f"test_{i}.py").write_text(
            f"from src.shop import {f}\n\ndef test_{f}():\n    assert {f}()\n")
    idx = RepoIndex.build(str(tmp_path))
    res = eval_test_selection(idx, sample=100)
    assert res["budget_tests"] >= 1
    sel, alpha = res["scores"]["selector"], res["scores"]["alpha"]
    assert sel["recall_at_third"] >= alpha["recall_at_third"]
    assert sel["recall_at_third"] >= 0.5


def test_retrieval_heavy_falls_back_without_a_model(tmp_path, monkeypatch):
    from harness import retrieval_heavy
    monkeypatch.setenv("RBG_CROSS_ENCODER", "")
    assert isinstance(retrieval_heavy.available(), bool)
    idx = _repo(tmp_path)
    hits = HybridRetriever(idx).retrieve("beta_helper", limit=5)
    rr = retrieval_heavy.CrossEncoderReranker()
    assert rr.model is None  # no runtime or no model -> linear fallback
    out = rr.rerank("beta_helper", hits, idx.chunks)
    assert [h.path for h in out] == [h.path for h in hits] and out


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


def test_mcp_root_allowlist_of_filesystem_root_allows_any(tmp_path, monkeypatch):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    monkeypatch.setenv("HARNESS_ALLOWED_ROOTS", os.sep)
    assert mcp_server._authorize_root(root) == os.path.realpath(root)


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
    learned.validated = True  # benchmark-validated heads may drive the decision
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    rep = Orchestrator(root, ScriptedLLM(), learned=learned).run("fix discount", task_id="lr1")
    assert rep.decision["risk"]["source"] == "gbdt"


def test_learned_risk_stays_in_shadow_until_validated(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(float)
    learned = LearnedModels(policy=Policy().train_risk(X.tolist(), y.tolist()))
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    rep = Orchestrator(root, ScriptedLLM(), learned=learned).run("fix discount", task_id="sh1")
    assert rep.decision["risk"]["source"] == "rule"
    assert "learned_risk_in_shadow" in rep.decision["reason_codes"]


def test_validate_models_promotes_only_with_enough_held_out_evidence(tmp_path):
    from harness.training import validate_and_promote
    thin, _ = _synthetic_ledger(tmp_path / "thin", n=24)
    refused = validate_and_promote(thin, min_test=20)
    assert refused["promoted"] is False
    assert any("test_n_below" in r for r in refused["reasons"])

    root, _ = _synthetic_ledger(tmp_path / "rich", n=200)
    verdict = validate_and_promote(root, min_test=20)
    assert verdict["promoted"] is True, verdict
    assert verdict["test_brier"] < verdict["rule_brier"]

    mcp_server._LEARNED_CACHE.clear()
    mcp_server._INDEX_CACHE.clear()
    learned = mcp_server._learned(root)
    assert learned.validated is True
    assert mcp_server.get_calibration(root)["validated"] is True
    # the verdict belongs to this artifact only
    mcp_server._LEARNED_CACHE.clear()
    store = hdb.Store(root)
    store.append_ledger({"component": "orchestrator", "task_id": "later",
                         "result": {"success": True}})
    store.close()
    mcp_server._LEARNED_CACHE.clear()
    mcp_server._INDEX_CACHE.clear()
    assert mcp_server._learned(root).validated is False


def test_train_models_cli_and_tool_can_validate(tmp_path):
    from harness.__main__ import main
    root, _ = _synthetic_ledger(tmp_path / "cli", n=200)
    assert main(["validate-models", root, "--min-test", "20"]) == 0
    mcp_server._LEARNED_CACHE.clear()
    mcp_server._INDEX_CACHE.clear()
    out = mcp_server.train_models(root, validate=True)
    assert out["validated"] is True
    assert out["validation"]["promoted"] is True


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


def test_read_only_plan_step_is_not_an_edit_commitment():
    from harness.plan import parse_plan
    p = parse_plan("1. Inspect shop.py and test_shop.py.\n2. Edit shop.py.",
                   {"shop.py", "test_shop.py"})
    assert p.edit_files() == ["shop.py"]


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


# -- temporal split + repo holdout ------------------------------------------ #
def test_risk_split_is_temporal_with_held_out_calibration(tmp_path, monkeypatch):
    base = tmp_path / "g"
    base.mkdir()
    monkeypatch.setenv("HARNESS_GLOBAL_DB", str(base / "global.db"))
    root, idx = _synthetic_ledger(tmp_path / "split", n=60)
    m = train_from_ledger(root, index=idx)
    r = m.metrics["risk"]
    assert r["split"] == "temporal"
    assert r["n"] == 60
    assert r["n_train"] + r["n_cal"] + r["n_test"] == 60
    assert r["n_cal"] > 0 and r["n_test"] > 0
    assert "test_brier" in r and 0.0 <= r["test_brier"] <= 1.0


def test_thin_ledger_marks_in_sample_split(tmp_path, monkeypatch):
    base = tmp_path / "g"
    base.mkdir()
    monkeypatch.setenv("HARNESS_GLOBAL_DB", str(base / "global.db"))
    root, idx = _synthetic_ledger(tmp_path / "tiny", n=10)
    m = train_from_ledger(root, index=idx)
    assert m.metrics["risk"]["split"] == "in_sample"


# -- metric intervals ------------------------------------------------------- #
def test_wilson_ci_and_metric_intervals(tmp_path, monkeypatch):
    from harness.training import wilson_ci
    lo, hi = wilson_ci(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert wilson_ci(0, 0) == [0.0, 1.0]
    base = tmp_path / "g"
    base.mkdir()
    monkeypatch.setenv("HARNESS_GLOBAL_DB", str(base / "global.db"))
    root, idx = _synthetic_ledger(tmp_path / "ci", n=40)
    r = train_from_ledger(root, index=idx).metrics["risk"]
    assert len(r["conformal_coverage_ci"]) == 2
    assert len(r["positive_rate_ci"]) == 2


# -- selector labels are regression evidence, not historical failure -------- #
def test_selector_labels_mark_only_new_regressions():
    from harness.training import selector_examples
    entries = [
        {"component": "test_selector", "task_id": "t0",
         "result": {"changed_symbols": ["a"]}},
        {"component": "verifier", "task_id": "t0", "result": {"outcomes": [
            {"test_id": "x", "status": "passed"},
            {"test_id": "y", "status": "failed"}]}},
        {"component": "test_selector", "task_id": "t1",
         "result": {"changed_symbols": ["a"]}},
        {"component": "verifier", "task_id": "t1", "result": {"outcomes": [
            {"test_id": "x", "status": "failed"},
            {"test_id": "y", "status": "failed"}]}},
    ]
    pairs = [(e[2], e[-1]) for e in selector_examples(entries)]
    assert pairs == [("x", 0), ("y", 1), ("x", 1), ("y", 0)]  # y already red at t1


# -- deterministic benchmark harness ---------------------------------------- #
def test_benchmark_smoke_suite_and_compare():
    import os
    from eval.benchmark import compare, run_suite
    suite = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "eval", "suites", "smoke")
    base = run_suite(suite, "baseline")
    full = run_suite(suite, "full")
    assert base["n"] == 3 and base["metrics"]["resolution_rate"] == 1.0
    assert base["metrics"]["regression_rate"] == 0.0
    assert base["metrics"]["recall_at_k"] == 0.0        # retrieval off
    assert full["metrics"]["recall_at_k"] > 0.0         # retrieval on
    assert compare(base, full)["metrics"]["recall_at_k"]["delta"] > 0


def test_cli_eval_and_compare_forward_their_flags(tmp_path):
    import json
    from harness.__main__ import main
    suite = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "eval", "suites", "smoke")
    out = tmp_path / "baseline.json"
    assert main(["eval", "--suite", suite, "--config", "baseline",
                 "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["n"] == 3 and result["config"] == "baseline"
    assert main(["compare", str(out), str(out)]) == 0


def test_mcp_run_checks_compile_passes_under_the_sandbox(tmp_path):
    root = str(tmp_path / "repo")
    _make_demo_repo(root)
    out = _call("run_checks", {"root": root, "profile": "compile", "timeout_s": 60})
    assert out["result"]["isError"] is False, out["result"]["content"][0]["text"]
    sc = out["result"]["structuredContent"]
    assert sc["language"] == "python"
    assert sc["exit_code"] == 0, sc.get("stdout_tail") or sc.get("stderr_tail")


# -- read path never trains ------------------------------------------------ #
def test_read_path_does_not_train_models(tmp_path):
    from harness import db as hdb
    root, _ = _synthetic_ledger(tmp_path / "noretrain")
    mcp_server._LEARNED_CACHE.clear()
    mcp_server._INDEX_CACHE.clear()
    _call("get_change_impact", {"root": root, "symbols": ["compute"]})
    store = hdb.Store(root)
    assert store.load_model("learned") is None        # a read tool did not train
    store.close()
    sc = _call("train_models", {"root": root})["result"]["structuredContent"]
    assert sc["trained"] and sc["risk_trained"]


def test_plan_audit_does_not_build_retriever(tmp_path):
    root, _ = _synthetic_ledger(tmp_path / "pa")
    mcp_server._RETRIEVER_CACHE.clear()
    mcp_server._INDEX_CACHE.clear()
    out = _call("plan_audit", {"root": root,
                               "plan_text": "1. Edit src.py.\n2. Run test_src.py.",
                               "changed_files": ["src.py"]})
    sc = out["result"]["structuredContent"]
    assert "scope_drift" in sc and "completeness" in sc
    assert mcp_server._RETRIEVER_CACHE == {}


def test_search_code_reports_latency_and_degrades_over_budget(tmp_path, monkeypatch):
    root, _ = _synthetic_ledger(tmp_path / "budget")
    mcp_server._INDEX_CACHE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    mcp_server._LEARNED_CACHE.clear()
    _call("train_models", {"root": root})
    monkeypatch.setattr(mcp_server, "_READ_BUDGET_MS", -1)
    sc = _call("search_code", {"root": root, "query": "compute"})["result"]["structuredContent"]
    assert sc["degraded"] is True and sc["reranker"] == "skipped_over_budget"
    assert sc["latency_ms"] >= 0 and sc["budget_ms"] == -1
    monkeypatch.setattr(mcp_server, "_READ_BUDGET_MS", 10 ** 9)
    sc2 = _call("search_code", {"root": root, "query": "compute"})["result"]["structuredContent"]
    assert sc2["degraded"] is False and sc2["reranker"] == "trained"


def test_search_code_per_call_timeout_override(tmp_path):
    root, _ = _synthetic_ledger(tmp_path / "tmo")
    mcp_server._INDEX_CACHE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    mcp_server._LEARNED_CACHE.clear()
    sc = _call("search_code", {"root": root, "query": "compute",
                               "timeout_ms": 0})["result"]["structuredContent"]
    assert sc["budget_ms"] == 0 and sc["degraded"] is True


def test_perf_bench_reports_p95_and_budget(tmp_path):
    import importlib
    perf = importlib.import_module("eval.perf")
    rep = perf._bench(lambda: sum(range(100)), 5)
    assert {"p50_ms", "p95_ms", "median_ms", "repeat"} <= set(rep)
    assert rep["p95_ms"] >= rep["p50_ms"]

    tools = {"search_code": {"p95_ms": 12.0}, "db_stats": {"p95_ms": 3.0}}
    assert perf.budget_verdict(tools, None)["ok"] is None
    assert perf.budget_verdict(tools, 1.0)["ok"] is False
    assert perf.budget_verdict(tools, 1.0)["over_budget"] == ["db_stats", "search_code"]
    graded = perf.budget_verdict(tools, 20.0)
    assert graded["ok"] is True and graded["over_budget"] == []


def test_clean_query_dedupes_and_caps():
    from harness.retrieval import _QUERY_TERM_CAP, _clean_query
    terms = _clean_query("compute compute the total total value").split()
    assert len(terms) == len(set(terms))
    long = " ".join(f"w{i}" for i in range(_QUERY_TERM_CAP + 50))
    assert len(_clean_query(long).split()) <= _QUERY_TERM_CAP


def test_expand_adds_same_symbol_sibling(tmp_path):
    from harness.index import RepoIndex
    from harness.retrieval import Hit, HybridRetriever
    (tmp_path / "a.py").write_text("def shared():\n    return 1\n")
    (tmp_path / "b.py").write_text("def shared():\n    return 2\n")
    idx = RepoIndex.build(str(tmp_path))
    retr = HybridRetriever(idx)
    hit = Hit("a.py", "shared", 1, 2, 1.0, "rrf", "bm25", "h")
    out = retr.expand([hit], budget=5)
    assert any(h.path == "b.py" and h.symbol == "shared" for h in out)


def test_search_code_result_cache_isolated_from_callers(tmp_path):
    root, _ = _synthetic_ledger(tmp_path / "rc")
    mcp_server._INDEX_CACHE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    mcp_server._LEARNED_CACHE.clear()
    mcp_server._RESULT_CACHE.clear()
    a = _call("search_code", {"root": root, "query": "compute"})["result"]["structuredContent"]
    assert mcp_server._RESULT_CACHE and a["results"]
    b = _call("search_code", {"root": root, "query": "compute"})["result"]["structuredContent"]
    assert [r["chunk_id"] for r in a["results"]] == [r["chunk_id"] for r in b["results"]]
    a["results"][0]["snippet"] = "MUTATED"
    c = _call("search_code", {"root": root, "query": "compute"})["result"]["structuredContent"]
    assert c["results"][0]["snippet"] != "MUTATED"


def test_fts_backend_serves_candidates_and_falls_back(tmp_path):
    from harness import db as hdb
    from harness.index import RepoIndex
    from harness.retrieval import BM25, FTSBackend
    (tmp_path / "m.py").write_text(
        "def compute_metric():\n    return 42\n\n\ndef other_thing():\n    return 1\n")
    idx = RepoIndex.build(str(tmp_path))
    store = hdb.Store(str(tmp_path))
    store.save_index(idx)
    fb = FTSBackend(idx.chunks, store.fts_search)
    got = fb.rank("compute metric", 5)
    assert got and "compute_metric" in idx.chunks[got[0][0]].content
    assert fb.rank("!!!", 5) == BM25(idx.chunks).rank("!!!", 5)
    store.close()


def test_lsi_refresh_reuses_projector():
    from harness.retrieval import LSIEncoder
    texts = ["alpha beta gamma", "delta epsilon zeta", "alpha delta theta"]
    enc = LSIEncoder(k=2).fit(texts)
    vocab, proj = dict(enc.vocab), enc.proj.copy()
    enc.refresh(texts + ["alpha beta new"])
    assert enc.vocab == vocab and enc.proj.shape == proj.shape
    assert enc.doc_vecs.shape[0] == 4


def test_warm_start_after_revision_change(tmp_path):
    from harness import mcp_server
    root = str(tmp_path)
    (tmp_path / "m.py").write_text("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n")
    mcp_server._INDEX_CACHE.clear()
    mcp_server._RETRIEVER_CACHE.clear()
    mcp_server._RESULT_CACHE.clear()
    _call("search_code", {"root": root, "query": "alpha"})
    (tmp_path / "m.py").write_text("def alpha():\n    return 1\n\n\ndef gamma():\n    return 3\n")
    sc = _call("search_code", {"root": root, "query": "gamma"})["result"]["structuredContent"]
    assert sc["results"]


# -- JSON plan document (Workstream H) ------------------------------------- #
def test_plan_editor_create_append_attach_and_depth_cap():
    from harness.plan import PlanEditor
    ed = PlanEditor.create("r", "t", "T", [{"title": "a"}])
    assert len(ed.doc.sections) == 1
    cur = ed.append({"title": "b"})
    for _ in range(4):                      # root 1 -> deepest 5
        cur = ed.attach(cur.id, {"title": "x"})
    assert ed.doc.find(cur.id)[1] == 5
    with pytest.raises(ValueError):
        ed.attach(cur.id, {"title": "too deep"})


def test_plan_complete_requires_passing_validation():
    from harness.plan import PlanEditor
    ed = PlanEditor.create("r", "t", "T", [{"title": "a"}])
    sid = ed.doc.sections[0].id
    out = ed.complete(sid)
    assert out["done"] is False and ed.get(sid)["status"] == "open"
    out = ed.complete(sid, {"verdict": "pass", "evidence": "tests green", "model": "m"})
    assert out["done"] is True and ed.get(sid)["status"] == "done"
    assert ed.get(sid)["validation"]["model"] == "m"


def test_plan_complete_uses_validator_callable():
    from harness.plan import PlanEditor
    ed = PlanEditor.create("r", "t", "T", [{"title": "a"}],
                           validator=lambda sec: {"verdict": "pass", "evidence": "auto"})
    ed.complete(ed.doc.sections[0].id)
    assert ed.get(ed.doc.sections[0].id)["status"] == "done"


def test_plan_parent_rollup_and_mutation_roundtrip():
    from harness.plan import PlanEditor
    ed = PlanEditor.create("r", "t", "T", [{"title": "parent"}])
    parent = ed.doc.sections[0]
    child = ed.attach(parent.id, {"title": "child"})
    ed.complete(parent.id, {"verdict": "pass"})
    ed.complete(child.id, {"verdict": "pass"})
    assert ed.tree()["sections"][0]["effective"] == "done"
    ed.modify(child.id, {"status": "dropped"})
    assert ed.tree()["sections"][0]["effective"] == "open"
    ed.modify(parent.id, {"title": "renamed", "body": "do x", "status": "blocked"})
    ed.note(parent.id, "note!")
    ed.check(parent.id, "pytest", "pass", "3 passed")
    d = ed.get(parent.id)
    assert d["title"] == "renamed" and d["status"] == "blocked"
    assert any(n["text"] == "note!" for n in d["notes"])
    assert d["checks"][0] == {"name": "pytest", "status": "pass", "evidence": "3 passed"}


def test_mcp_plan_tools_roundtrip_and_persist(tmp_path):
    root, _ = _synthetic_ledger(tmp_path / "plans")
    mcp_server._STORE_CACHE.clear()
    out = _call("plan_create", {"root": root, "task_id": "t1", "title": "Fix"})
    assert out["result"]["structuredContent"]["version"] == 1
    sec = _call("plan_append", {"root": root, "task_id": "t1",
                                "section": {"title": "step1"}})["result"]["structuredContent"]
    sid = sec["section"]["id"]
    _call("plan_note", {"root": root, "task_id": "t1", "section_id": sid, "text": "hi"})
    _call("plan_check", {"root": root, "task_id": "t1", "section_id": sid,
                         "name": "tests", "status": "pass", "evidence": "ok"})
    comp = _call("plan_complete", {"root": root, "task_id": "t1", "section_id": sid,
                                   "validation": {"verdict": "pass", "evidence": "ok"}})
    assert comp["result"]["structuredContent"]["done"] is True
    got = _call("plan_get", {"root": root, "task_id": "t1"})["result"]["structuredContent"]
    assert got["plan"]["sections"][0]["status"] == "done"
    aud = _call("plan_audit", {"root": root, "task_id": "t1"})["result"]["structuredContent"]
    assert aud["audit"]["sections"][0]["effective"] == "done"
    mcp_server._STORE_CACHE.clear()          # force a fresh read from disk
    got2 = _call("plan_get", {"root": root, "task_id": "t1"})["result"]["structuredContent"]
    assert got2["plan"]["version"] == got["plan"]["version"]
    from harness import db as hdb
    actions = {e["result"]["action"] for e in hdb.Store(root).read_ledger()
               if e["component"] == "plan"}
    assert {"create", "append", "complete"} <= actions


def test_curated_suite_is_well_formed():
    import os
    from eval.benchmark import load_suite
    suite = load_suite(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "eval", "suites", "curated"))
    tasks = suite["tasks"]
    assert len(tasks) >= 50
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    for t in tasks:
        assert t["files"] and t["fix"] and t["expect_pass"]
        for rel in t["expected_files"]:
            assert rel in t["files"] and rel in t["fix"]
            assert t["files"][rel] != t["fix"][rel]
    assert any(len(t["expected_files"]) > 1 for t in tasks), "expected multi-file tasks"


def test_hard_suite_makes_retrieval_non_trivial():
    import os
    from eval.benchmark import load_suite
    from eval.retrieval_ablation import run
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hard = os.path.join(here, "eval", "suites", "hard")
    suite = load_suite(hard)
    assert len(suite["tasks"]) >= 50
    for t in suite["tasks"]:
        assert any(name.startswith("distractor_") for name in t["files"]), t["id"]
        assert not any(word in t["task"] for word in t["expected_files"]), t["id"]
    modes = run(hard, k=5)["modes"]
    assert modes["lexical"]["recall_at_1"] < 0.9, "suite should not be trivially retrievable"
    assert modes["hybrid"]["recall_at_1"] >= modes["lexical"]["recall_at_1"] - 0.02
    assert modes["hybrid"]["recall_at_1"] >= modes["dense"]["recall_at_1"] - 0.02
    assert modes["hybrid"]["recall_at_k"] >= 0.35


def test_replay_llm_uses_the_edit_contract():
    from harness.providers import ReplayLLM
    llm = ReplayLLM({"plan": "p", "edits": [{"path": "a.py", "content": "x = 1\n"}]})
    assert llm.plan("t", []) == "p"
    edits = llm.edit("t", "p", [])
    assert [(e.path, e.content) for e in edits] == [("a.py", "x = 1\n")]


def test_parse_edits_takes_first_object_from_repeated_truncated_reply():
    # Live reasoning models sometimes echo the object and get cut off mid-copy
    # (finish_reason=length); the first complete object must still parse.
    from harness.providers import parse_edits
    text = ('{"edits": [{"path": "a.py", "content": "x = 1\\n"}]}\n\n'
            '{"edits": [{"path": "a.py", "content": "x = 1')
    edits = parse_edits(text)
    assert [(e.path, e.content) for e in edits] == [("a.py", "x = 1\n")]


def test_parse_edits_rejects_a_reply_with_no_json():
    from harness.providers import parse_edits
    with pytest.raises(ValueError):
        parse_edits("I could not complete this task.")


def test_replay_example_end_to_end(tmp_path):
    import shutil
    from harness.__main__ import cmd_replay
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = tmp_path / "repo"
    shutil.copytree(os.path.join(here, "examples", "replay"), root)
    assert cmd_replay(str(root), str(root / "transcript.json"),
                      "fix median for even-length lists") == 0


def test_blind_ablation_separates_retrieval():
    from eval.benchmark import run_suite
    base = run_suite("eval/suites/smoke", "blind-baseline")["metrics"]
    full = run_suite("eval/suites/smoke", "blind-full")["metrics"]
    assert base["resolution_rate"] == 0.0
    assert base["regression_rate"] == 0.0
    assert full["resolution_rate"] == 1.0
    assert base["recall_at_k"] == 0.0
    assert full["recall_at_k"] == 1.0


def test_plan_audit_finishes_multifile_fixes():
    from eval.benchmark import run_suite
    off = run_suite("eval/suites/multifile", "partial-noaudit")["metrics"]
    on = run_suite("eval/suites/multifile", "partial-audit")["metrics"]
    assert off["resolution_rate"] == 0.0
    assert on["resolution_rate"] == 1.0
    assert off["mean_iterations"] < on["mean_iterations"]


def test_swebench_retrieval_scores_gold_files(tmp_path):
    import subprocess
    from eval.swebench_retrieval import parse_patch_files, run_report
    patch = ("diff --git a/target.py b/target.py\nindex 1..2 100644\n"
             "--- a/target.py\n+++ b/target.py\n")
    assert parse_patch_files(patch) == ["target.py"]
    assert parse_patch_files("") == []

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("def first_n(xs, n):\n    return xs[:n - 1]\n")
    (repo / "unrelated.py").write_text("def other():\n    return 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-qm", "base"], cwd=repo, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True).stdout.strip()

    report = run_report([{
        "instance_id": "local-1", "repo": str(repo), "base_commit": sha,
        "problem_statement": "fix first_n so it returns exactly the first n items",
        "patch": patch,
    }], str(tmp_path / "cache"), k=5)
    assert report["scored"] == 1
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["results"][0]["gold_files"] == ["target.py"]


def test_swebench_index_cache_is_keyed_by_commit(tmp_path, monkeypatch):
    from eval.swebench_retrieval import load_or_build_index
    from harness.index import RepoIndex

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def a():\n    return 1\n")
    calls = {"n": 0}
    build = RepoIndex.build

    def counted(*args, **kwargs):
        calls["n"] += 1
        return build(*args, **kwargs)

    monkeypatch.setattr(RepoIndex, "build", counted)
    cache = str(tmp_path / "cache")
    first = load_or_build_index(str(repo), "org/repo", "abc123", cache)
    again = load_or_build_index(str(repo), "org/repo", "abc123", cache)
    assert calls["n"] == 1
    assert [c.path for c in again.chunks] == [c.path for c in first.chunks]
    load_or_build_index(str(repo), "org/repo", "def456", cache)
    assert calls["n"] == 2


# -- dynamic PTS ground truth (F2) ------------------------------------------ #
def test_pts_mutation_injects_probe_before_body():
    from eval.pts_dynamic import mutate_source
    src = "def alpha():\n    return 1\n"
    out = mutate_source(src, "alpha", 1)
    assert out is not None and "raise RuntimeError(" in out and "rbg-probe" in out
    assert out.index("raise RuntimeError") < out.index("return 1")


def test_pts_mutation_respects_docstring():
    from eval.pts_dynamic import mutate_source
    src = 'def alpha():\n    "doc"\n    return 1\n'
    out = mutate_source(src, "alpha", 1)
    assert out.index('"doc"') < out.index("raise RuntimeError")
    assert mutate_source("def alpha(): pass\n", "missing", 1) is None


def test_pts_score_selection_uses_dynamic_truth():
    from eval.pts_dynamic import score_selection
    tests = [{"test_id": f"t{i}::test_{i}", "symbols": [], "duration": 1.0, "history": []}
             for i in range(9)]
    tests[4]["test_id"] = "t4::test_alpha"
    tests[4]["symbols"] = ["alpha"]
    probes = [{"symbol": "alpha", "path": "t4.py", "ground_truth_tests": ["t4::test_alpha"]}]
    out = score_selection(probes, tests, seed=0)
    assert out["selector"]["mrr"] == 1.0
    assert out["selector"]["recall@k"] >= out["random"]["recall@k"]


# -- cross-encoder non-fallback branch (F5) --------------------------------- #
def test_cross_encoder_rerank_uses_model_branch(tmp_path):
    from harness import retrieval_heavy
    idx = _repo(tmp_path)
    hits = HybridRetriever(idx).retrieve("beta_helper", limit=5)
    rr = retrieval_heavy.CrossEncoderReranker()
    rr.model = object()  # force the non-fallback path without a real runtime

    def fake_score(query, chunk):
        return 1.0 if (chunk and "beta" in chunk.path) else 0.0

    rr.score = fake_score
    out = rr.rerank("beta_helper", hits, idx.chunks)
    assert out and [h.score_type for h in out] == ["cross-encoder"] * len(out)
    assert out[0].path.endswith("b.py")


# -- optional real semantic encoder (F6) ------------------------------------ #
class _BagModel:
    """Stand-in sentence-transformers model: counts three marker words."""

    def encode(self, texts, **_kw):
        rows = []
        for t in texts:
            rows.append([float(t.count("alpha")), float(t.count("beta")),
                         float(t.count("gamma"))])
        return np.asarray(rows)


def test_semantic_encoder_ranks_by_embedding():
    from harness.retrieval_heavy import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(model=_BagModel())

    class C:
        def __init__(self, content):
            self.content = content

    docs = [C("beta beta helper"), C("alpha value"), C("gamma")]
    ranked = enc.rank("alpha", docs, limit=2)
    assert ranked and ranked[0][0] == 1
    assert np.isclose(np.linalg.norm(enc.encode("alpha beta")), 1.0)
    assert enc.document_vectors(docs).shape == (3, enc.dim)


def test_semantic_encoder_caches_unchanged_chunks():
    from harness.retrieval_heavy import SentenceTransformerEncoder
    calls = []

    class CountingModel(_BagModel):
        def encode(self, texts, **kw):
            calls.append(list(texts))
            return super().encode(texts, **kw)

    class C:
        def __init__(self, content):
            self.content = content
            self.content_hash = content

    enc = SentenceTransformerEncoder(model=CountingModel())
    chunks = [C("alpha"), C("beta")]
    first = enc.document_vectors(chunks)
    enc.document_vectors(chunks)
    assert len(calls) == 1  # second pass served from the hash cache
    enc.document_vectors([C("alpha"), C("gamma")])
    assert calls[-1] == ["gamma"]  # only the changed chunk is re-encoded
    assert np.allclose(enc.document_vectors(chunks), first)


def test_heavy_encoder_is_opt_in(monkeypatch):
    from harness import retrieval_heavy
    monkeypatch.delenv("RBG_EMBEDDING_MODEL", raising=False)
    assert retrieval_heavy.encoder() is None


def test_encoder_device_resolution(monkeypatch):
    from harness import retrieval_heavy
    monkeypatch.setenv("RBG_EMBEDDING_DEVICE", "cuda")
    assert retrieval_heavy._resolve_device() == "cuda"
    assert retrieval_heavy._resolve_device("cpu") == "cpu"  # explicit wins
    monkeypatch.delenv("RBG_EMBEDDING_DEVICE")
    assert retrieval_heavy._resolve_device() in {"cuda", "cpu"}


def test_encoder_from_state_dispatches_semantic_kind(monkeypatch):
    from harness import retrieval_heavy
    from harness.retrieval import encoder_from_state
    monkeypatch.setattr(retrieval_heavy, "available", lambda: True)
    monkeypatch.setattr(retrieval_heavy, "_load_model", lambda _p, _d: _BagModel())
    enc = encoder_from_state({"kind": "sentence-transformer", "model": "fake", "dim": 384},
                             np.zeros((0, 384)))
    assert isinstance(enc, retrieval_heavy.SentenceTransformerEncoder) and enc.dim == 384
    # Without the extra, a persisted semantic encoder must not be silently used.
    monkeypatch.setattr(retrieval_heavy, "available", lambda: False)
    with pytest.raises(RuntimeError):
        encoder_from_state({"kind": "sentence-transformer", "model": "fake", "dim": 384},
                           np.zeros((0, 384)))


# -- experience memory is causal (F3) --------------------------------------- #
def test_experience_memory_causes_resolution(tmp_path):
    import re
    from harness import db
    from harness.retrieval import EXPERIENCE_DIM, experience_encoder
    from harness.__main__ import FIXED_SHOP
    root = str(tmp_path / "exp-causal")
    _make_demo_repo(root)
    task = "fix the discount calculation so a 10 percent discount applies correctly"
    store = db.Store(root)
    store.save_experience(task_id="prior", repo_id="demo", revision="",
                          query=task, outcome="verified",
                          summary="plan: edit shop.py\nfiles: shop.py",
                          vector=experience_encoder(EXPERIENCE_DIM).encode(task))
    store.close()

    class MemoLLM(ScriptedLLM):
        """Edits a file only when the recalled experience names it — ignores the
        retrieval hits, so resolution is caused by the experience, not search."""
        def edit(self, task, plan, context, diagnostics=""):
            named = set()
            for h in context:
                if h.path == Orchestrator.EXPERIENCE_PATH:
                    named |= set(re.findall(r"[\w./-]+\.[a-z]+", h.content))
            return [EditOp("shop.py", FIXED_SHOP)] if "shop.py" in named else []

    on = Orchestrator(root, MemoLLM(), use_experience=True).run(task, task_id="b1")
    off = Orchestrator(root, MemoLLM(), use_experience=False).run(task, task_id="b2")
    assert on.success and not off.success


def test_patch_head_trains_from_regression_ledger(tmp_path):
    from harness import db as hdb
    from harness.verify import Ledger
    root = tmp_path / "patch-ledger"
    root.mkdir()
    (root / "m.py").write_text("def f():\n    return 1\n")
    (root / "test_m.py").write_text("def test_0():\n    assert f() == 1\n")
    idx = RepoIndex.build(str(root))
    store = hdb.Store(str(root))
    ledger = Ledger(store)
    for i in range(10):
        tid = f"t{i}"
        ledger.record(component="patch_features", task_id=tid, repo_id="r",
                      revision=idx.revision, model_name="patch-features-v1",
                      result={"feature_vector": [float(i) / 10.0] * 11})
        ledger.record(component="test_selector", task_id=tid, repo_id="r",
                      revision=idx.revision, model_name="test-ranker-heuristic-v1",
                      result={"changed_symbols": ["f"]})
        status = "failed" if i % 2 else "passed"
        ledger.record(component="verifier", task_id=tid, repo_id="r",
                      revision=idx.revision, model_name="pytest",
                      result={"outcomes": [{"test_id": f"test_m.py::test_{i}",
                                            "status": status}]})
    store.close()
    models = train_from_ledger(str(root), index=idx)
    assert models.patch_trained
    assert models.metrics["patch"]["n"] >= 8
    assert models.metrics["patch"]["positive_rate"] == 0.5
