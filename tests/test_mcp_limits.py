
from rbg import mcp_server


def test_invalid_arguments_are_rejected():
    spec = mcp_server.TOOLS["search_code"]
    assert mcp_server._validate_args(spec, {"root": "/r"}) != ""
    assert mcp_server._validate_args(spec, {"root": 1, "query": "q"}) != ""
    assert mcp_server._validate_args(spec, {"root": "/r", "query": "q"}) == ""


def test_rate_limiter_blocks_after_burst(monkeypatch):
    monkeypatch.setenv("RBG_MCP_RATE_PER_MIN", "1")
    monkeypatch.setenv("RBG_MCP_BURST", "1")
    mcp_server._RATE_STATE.clear()
    assert mcp_server._rate_ok("x") is True
    assert mcp_server._rate_ok("x") is False


def test_manifest_hash_is_stable_and_exposed():
    first = mcp_server._tool_manifest_hash()
    assert len(first) == 64 and first == mcp_server._tool_manifest_hash()
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["serverInfo"]["tool_manifest_sha256"] == first


def test_requested_timeout_is_capped_by_operator_max(monkeypatch):
    import time
    monkeypatch.setenv("RBG_TOOL_TIMEOUT_MS", "40")
    monkeypatch.setenv("RBG_TOOL_TIMEOUT_MAX_MS", "50")
    t0 = time.monotonic()
    try:
        mcp_server._call_with_deadline(lambda **_: time.sleep(2), {"timeout_ms": 10_000})
    except mcp_server.ToolTimeout:
        pass
    else:
        raise AssertionError("expected the operator cap to interrupt the call")
    assert time.monotonic() - t0 < 1.0


def test_invalid_timeout_configuration_is_rejected(monkeypatch):
    import pytest
    monkeypatch.setenv("RBG_TOOL_TIMEOUT_MS", "0")
    with pytest.raises(RuntimeError):
        mcp_server._call_with_deadline(lambda **_: None, {})
    monkeypatch.setenv("RBG_TOOL_TIMEOUT_MS", "100")
    with pytest.raises(ValueError):
        mcp_server._call_with_deadline(lambda **_: None, {"timeout_ms": -1})
