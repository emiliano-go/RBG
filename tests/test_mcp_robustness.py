"""The stdio server must survive malformed input and bound large results."""

import json
import os
import subprocess
import sys

from harness import mcp_server

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(lines: bytes, **env):
    boost = {**os.environ, "PYTHONPATH": REPO, **env}
    proc = subprocess.run(
        [sys.executable, "-m", "harness.mcp_server"],
        input=lines, capture_output=True, cwd=REPO, env=boost, timeout=60,
        check=False)
    assert proc.returncode == 0, proc.stderr.decode()
    return [json.loads(line) for line in proc.stdout.decode().splitlines() if line]


def test_non_object_messages_get_an_error_and_the_server_survives():
    out = _run(b'[]\nnull\n42\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n')
    assert len(out) == 4
    assert out[0]["error"]["code"] == -32600
    assert out[1]["error"]["code"] == -32600
    assert out[2]["error"]["code"] == -32600
    assert out[3]["result"] == {}


def test_invalid_json_gets_a_parse_error():
    out = _run(b'{not json\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n')
    assert out[0]["error"]["code"] == -32700
    assert out[1]["result"] == {}


def test_oversized_line_is_rejected_without_losing_the_next_request():
    out = _run(b'{"x":"' + b"a" * 500 + b'"}\n'
               b'{"jsonrpc":"2.0","id":3,"method":"ping"}\n',
               HARNESS_MCP_MAX_LINE_BYTES="64")
    assert out[0]["error"]["code"] == -32600
    assert "exceeds" in out[0]["error"]["message"]
    assert out[1]["result"] == {}


def test_tool_result_bounds_structured_content():
    big = {"rows": ["x" * 1024] * 1024}
    response = mcp_server._tool_result(1, big)
    result = response["result"]
    assert result["structuredContent"]["truncated"] is True
    assert result["structuredContent"]["bytes"] > mcp_server.MAX_OUTPUT_BYTES
    assert len(result["content"][0]["text"].encode()) <= mcp_server.MAX_OUTPUT_BYTES


def test_small_tool_result_is_unchanged():
    out = {"ok": True}
    result = mcp_server._tool_result(1, out)["result"]
    assert result["structuredContent"] == out
    assert json.loads(result["content"][0]["text"]) == out
