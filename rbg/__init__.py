"""RBG: one main generative LLM plus deterministic control-plane,
hybrid retrieval, conventional verification, and small learned models.

Only numpy is required.  The determinism baseline (index, retrieval, gates,
ledger) runs without any model service.
"""

from .index import Chunk, RepoIndex, Symbol
from .models import (
    Decision, DriftModel, FlakyModel, GBDT, PlattScaler, Policy, TaskFeatures,
    TestRank, TestSelector, drift_features, extract_features,
)
from .orchestrator import EditOp, MainLLM, Orchestrator, Report
from .plan import Drift, Plan, PlanStep, audit_completion, parse_plan, scope_drift
from .diffing import ChangedSet, git_changed
from .plan_drift import audit_document, section_validation
from .providers import OpenAICompatLLM, parse_edits
from .retrieval import BM25, DenseEncoder, Hit, HybridRetriever, Reranker
from .verify import CommandResult, Ledger, Sandbox, TestOutcome, Verifier

__all__ = [
    "Chunk", "RepoIndex", "Symbol",
    "GBDT", "PlattScaler", "Policy", "TaskFeatures", "Decision",
    "TestSelector", "TestRank", "FlakyModel", "extract_features",
    "DriftModel", "drift_features", "ChangedSet", "git_changed",
    "audit_document", "section_validation",
    "Orchestrator", "Report", "EditOp", "MainLLM",
    "Plan", "PlanStep", "Drift", "parse_plan", "scope_drift", "audit_completion",
    "OpenAICompatLLM", "parse_edits",
    "BM25", "DenseEncoder", "HybridRetriever", "Reranker", "Hit",
    "Sandbox", "Verifier", "Ledger", "CommandResult", "TestOutcome",
]
