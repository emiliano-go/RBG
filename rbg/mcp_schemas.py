"""Tool and resource registration for the MCP server."""

from __future__ import annotations

import json
import os

from .mcp_plan import (
    plan_append,
    plan_attach,
    plan_audit,
    plan_check,
    plan_complete,
    plan_create,
    plan_get,
    plan_modify,
    plan_note,
    plan_tree,
)
from .mcp_service import _index_summary
from .mcp_tools import (
    check_drift,
    db_stats,
    find_symbol,
    get_calibration,
    get_change_impact,
    get_diagnostics,
    get_task_status,
    propose_patch,
    rank_tests,
    read_chunk,
    retrieve_context,
    run_checks,
    search_code,
    train_models,
    vector_search,
)

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
MAX_LINE_BYTES = int(os.environ.get("RBG_MCP_MAX_LINE_BYTES", str(1024 * 1024)))

# MCP resource surface (spec 3.3C): read-only views over a repo's store.  The
# repo is passed as a query parameter so one server serves every snapshot.
RESOURCE_TEMPLATES = [
    {"uriTemplate": "rbg://repo/ledger{?root}", "name": "Execution ledger",
     "description": "Append-only task/trace ledger for a repository.",
     "mimeType": "application/json"},
    {"uriTemplate": "rbg://repo/models{?root}", "name": "Learned models",
     "description": "Training metrics and calibration state for the repo's models.",
     "mimeType": "application/json"},
    {"uriTemplate": "rbg://repo/db{?root}", "name": "Store statistics",
     "description": "libSQL store counts: chunks, vectors, symbols, ledger, models.",
     "mimeType": "application/json"},
    {"uriTemplate": "rbg://repo/index{?root}", "name": "Repository index",
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



