from eval import core_bench, model_gate


def test_core_bench_scores_a_known_case(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target.py").write_text("def parse_config():\n    return {}\n")
    (repo / "other.py").write_text("def unrelated():\n    return 1\n")
    cases = [{"query": "parse config", "relevant_paths": ["target.py"]}]
    out = core_bench.evaluate(str(repo), cases, k=5)
    assert out["cases"] == 1 and out["mrr"] > 0


def test_model_gate_rejects_regression():
    base = {"recall_at_10": 0.8, "mrr": 0.7}
    cand = {"recall_at_10": 0.5, "mrr": 0.4}
    verdict = model_gate.gate(base, cand)
    assert verdict["ok"] is False
    assert "recall_regression" in verdict["failures"]
