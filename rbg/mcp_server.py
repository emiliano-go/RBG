"""RBG MCP server: a thin JSON-RPC 2.0 adapter over stdio.

Transport, authorization gates, rate limiting, argument validation, and the
hard wall-clock deadline live here.  Indexing/retrieval live in
:mod:`rbg.mcp_service`, tools in :mod:`rbg.mcp_tools`, plan tools in
:mod:`rbg.mcp_plan`, and registration in :mod:`rbg.mcp_schemas`.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import threading
import time
import traceback

from .mcp_service import (  # noqa: F401 - re-exported service API
    IndexBuilding, _BUILD_LOCK, _BUILD_STATE, _FAST_RETRIEVER_CACHE, _INDEX_CACHE,
    _INDEX_SIGNATURE, _LEARNED_CACHE, _PLAN_DRIFT_LOGGED, _PLAN_DRIFT_LOGGED_MAX,
    _RESULT_CACHE, _RESULT_CACHE_MAX, _RETRIEVER_BUILD, _RETRIEVER_BUILD_LOCK,
    _RETRIEVER_CACHE, _STORE_CACHE, _TOKENS_CACHE, _authorize_root, _index,
    _learned, _retriever, _store,
)
from .mcp_tools import (  # noqa: F401 - re-exported tool API
    check_drift, db_stats, find_symbol, get_calibration, get_change_impact,
    get_diagnostics, get_task_status, propose_patch, rank_tests, read_chunk,
    retrieve_context, run_checks, search_code, train_models, vector_search,
)
from .mcp_plan import (  # noqa: F401 - re-exported plan API
    plan_append, plan_attach, plan_audit, plan_check, plan_complete, plan_create,
    plan_get, plan_modify, plan_note, plan_tree,
)
from .mcp_schemas import (MAX_LINE_BYTES, MAX_OUTPUT_BYTES, PROMPTS,
                          RESOURCE_TEMPLATES, TOOLS, _read_resource)

SERVER_NAME = "rbg-mcp"
SERVER_VERSION = "0.2.0"
DEFAULT_PROTOCOL = "2025-06-18"

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
        rate = float(os.environ.get("RBG_MCP_RATE_PER_MIN", "120")) / 60.0
        burst = float(os.environ.get("RBG_MCP_BURST", "30"))
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
    try:
        default_ms = int(os.environ.get("RBG_TOOL_TIMEOUT_MS", "60000"))
        max_ms = int(os.environ.get("RBG_TOOL_TIMEOUT_MAX_MS", "300000"))
        requested = int(args.get("timeout_ms") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid timeout configuration: {exc}") from exc
    if requested < 0:
        raise ValueError("timeout_ms must be >= 0")
    if default_ms <= 0 or max_ms <= 0 or default_ms > max_ms:
        raise RuntimeError(
            "invalid tool timeout configuration: require 0 < RBG_TOOL_TIMEOUT_MS "
            "<= RBG_TOOL_TIMEOUT_MAX_MS")
    # A caller may lower the budget but never raise it past the operator cap.
    timeout_ms = min(requested, max_ms) if requested else default_ms
    if (not hasattr(signal, "SIGALRM")
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
            tb = traceback.format_exc() if os.environ.get("RBG_MCP_DEBUG") else ""
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
            tb = traceback.format_exc() if os.environ.get("RBG_MCP_DEBUG") else ""
            response = _error(msg.get("id"), -32603,
                              f"internal error: {type(exc).__name__}: {exc}\n{tb}")
        _send(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
