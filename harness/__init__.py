"""Super Harness: one main generative LLM plus deterministic control-plane,
hybrid retrieval, conventional verification, and small learned models.

Only numpy is required.  The determinism baseline (index, retrieval, gates,
ledger) runs without any model service.
"""

from .index import Chunk, RepoIndex, Symbol
from .models import (
    Decision, FlakyModel, GBDT, PlattScaler, Policy, TaskFeatures, TestRank,
    TestSelector, extract_features,
)
from .orchestrator import EditOp, MainLLM, Orchestrator, Report
from .plan import Drift, Plan, PlanStep, audit_completion, parse_plan, scope_drift
from .retrieval import BM25, DenseEncoder, Hit, HybridRetriever, Reranker
from .verify import CommandResult, Ledger, Sandbox, TestOutcome, Verifier

__all__ = [
    "Chunk", "RepoIndex", "Symbol",
    "GBDT", "PlattScaler", "Policy", "TaskFeatures", "Decision",
    "TestSelector", "TestRank", "FlakyModel", "extract_features",
    "Orchestrator", "Report", "EditOp", "MainLLM",
    "Plan", "PlanStep", "Drift", "parse_plan", "scope_drift", "audit_completion",
    "BM25", "DenseEncoder", "HybridRetriever", "Reranker", "Hit",
    "Sandbox", "Verifier", "Ledger", "CommandResult", "TestOutcome",
]
