
from harness import mcp_server


def test_invalid_arguments_are_rejected():
    spec = mcp_server.TOOLS["search_code"]
    assert mcp_server._validate_args(spec, {"root": "/r"}) != ""
    assert mcp_server._validate_args(spec, {"root": 1, "query": "q"}) != ""
    assert mcp_server._validate_args(spec, {"root": "/r", "query": "q"}) == ""


def test_rate_limiter_blocks_after_burst(monkeypatch):
    monkeypatch.setenv("HARNESS_MCP_RATE_PER_MIN", "1")
    monkeypatch.setenv("HARNESS_MCP_BURST", "1")
    mcp_server._RATE_STATE.clear()
    assert mcp_server._rate_ok("x") is True
    assert mcp_server._rate_ok("x") is False


def test_manifest_hash_is_stable_and_exposed():
    first = mcp_server._tool_manifest_hash()
    assert len(first) == 64 and first == mcp_server._tool_manifest_hash()
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["serverInfo"]["tool_manifest_sha256"] == first
