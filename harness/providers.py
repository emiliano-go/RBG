"""An OpenAI-compatible :class:`MainLLM`.

Only a chat-completions endpoint is assumed, so this works against OpenAI,
OpenRouter, vLLM, llama.cpp's server, Ollama's OpenAI shim, or anything else
speaking the same shape.  Configuration comes from the environment
(``RBG_LLM_BASE_URL``, ``RBG_LLM_API_KEY``, ``RBG_LLM_MODEL``) or the
constructor; the transport is stdlib ``urllib`` so no client package is needed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from .orchestrator import EditOp
from .retrieval import Hit

PLAN_SYSTEM = (
    "You are the single coding model in a controlled harness. Given a task and "
    "retrieved source, reply with a short numbered plan. Name the exact files "
    "each step touches at the end of the line. Do not write code yet."
)

EDIT_SYSTEM = (
    "You are the single coding model in a controlled harness. Apply the plan to "
    "the retrieved source. Reply with ONLY a JSON object of the form "
    '{"edits": [{"path": "relative/path", "content": "the complete new file"}]}. '
    "Each content is the full file, not a patch. Paths are relative to the repo "
    "root. If nothing should change, reply {\"edits\": []}."
)


class OpenAICompatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout_s: float = 120.0) -> None:
        self.model = model or os.environ.get("RBG_LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get(
            "RBG_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("RBG_LLM_API_KEY", "")
        self.timeout_s = timeout_s

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def plan(self, task: str, context: list[Hit]) -> str:
        return self._chat(PLAN_SYSTEM, _render(task, context))

    def edit(self, task: str, plan: str, context: list[Hit],
             diagnostics: str = "") -> list[EditOp]:
        return parse_edits(self._chat(EDIT_SYSTEM, _render(task, context, plan, diagnostics)))


def _render(task: str, context: list[Hit], plan: str = "", diagnostics: str = "") -> str:
    parts = [f"TASK:\n{task}"]
    if plan:
        parts.append(f"PLAN:\n{plan}")
    if diagnostics:
        parts.append(f"DIAGNOSTICS (previous attempt failed):\n{diagnostics}")
    ctx = "\n\n".join(
        f"--- {h.path}:{h.start_line}-{h.end_line} ({h.symbol or 'block'})\n{h.content}"
        for h in context)
    parts.append(f"RETRIEVED SOURCE:\n{ctx}")
    return "\n\n".join(parts)


def parse_edits(text: str) -> list[EditOp]:
    """Extract the edits JSON from a model reply.

    Accepts a bare ``{"edits": [...]}`` object, a bare list, or either wrapped in
    a ``` fence, with prose around it.
    """
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    payload = match.group(1) if match else text
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        start, end = payload.find("{"), payload.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON edits found in model reply: {text[:200]!r}")
        data = json.loads(payload[start:end + 1])
    items = data.get("edits", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"edits payload is not a list: {data!r}")
    return [EditOp(str(e["path"]), str(e["content"])) for e in items]
