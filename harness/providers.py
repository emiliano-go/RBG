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
import time
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
                 api_key: str | None = None, timeout_s: float | None = None) -> None:
        self.model = model or os.environ.get("RBG_LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get(
            "RBG_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("RBG_LLM_API_KEY", "")
        # Reasoning models can take minutes on a full-file edit prompt; 120s
        # (the old default) timed a live run out.  RBG_LLM_TIMEOUT overrides.
        self.timeout_s = (timeout_s if timeout_s is not None
                          else float(os.environ.get("RBG_LLM_TIMEOUT", "120")))

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        max_tokens = int(os.environ.get("RBG_LLM_MAX_TOKENS", "0") or 0)
        if max_tokens:
            payload["max_tokens"] = max_tokens  # reasoning models spend tokens thinking
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        # Hosted endpoints drop or rate-limit the occasional request; a live
        # measurement must not lose a task to one transient failure.
        last: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
            time.sleep(2 * (attempt + 1))
        raise last  # type: ignore[misc]

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


class ReplayLLM:
    """A :class:`MainLLM` driven by a transcript authored by a session model.

    The transcript is ``{"plan": str, "edits": [{"path", "content"}]}``.  Edits
    go through :func:`parse_edits`, so the harness exercises exactly the same
    contract as a live provider; only the transport is replaced by a file.
    """

    def __init__(self, transcript: dict) -> None:
        self.transcript = transcript

    @classmethod
    def from_file(cls, path: str) -> "ReplayLLM":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def plan(self, task: str, context: list[Hit]) -> str:
        return str(self.transcript.get("plan") or "")

    def edit(self, task: str, plan: str, context: list[Hit],
             diagnostics: str = "") -> list[EditOp]:
        return parse_edits(json.dumps({"edits": self.transcript.get("edits", [])}))


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
        start = payload.find("{")
        if start == -1:
            raise ValueError(f"no JSON edits found in model reply: {text[:200]!r}")
        # Take the *first* JSON value from here, not first-brace to last-brace:
        # models sometimes emit the object more than once, and a truncated
        # second copy must not poison the parse (``Extra data``).
        try:
            data, _ = json.JSONDecoder().raw_decode(payload[start:])
        except json.JSONDecodeError:
            raise ValueError(f"no JSON edits found in model reply: {text[:200]!r}")
    items = data.get("edits", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"edits payload is not a list: {data!r}")
    return [EditOp(str(e["path"]), str(e["content"])) for e in items]
