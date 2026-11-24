"""LLM agent scaffold (provider-agnostic).

Pass any ``complete(system: str, user: str) -> str`` callable, or a backend from
``cpbench.backends`` (which also counts tokens). The agent sends the rendered
observation and the action request, asks for one JSON object, and falls back to
a random legal action when parsing fails (counted by the engine as an invalid
action, which is itself a benchmark metric, and by ``self.parse_failures``).

The model may add a ``"thought"`` field; it is stored in the transcript under
god view only and lets analysts compare private intent with public speech.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .base import RandomAgent
from ..observation import render_text, render_request

SYSTEM = ("You are a player in a multi-agent social deduction game. Stay in character, pursue your "
          "role's objective, and reply with exactly one JSON object. You may include a private field "
          "\"thought\" with your reasoning; other players never see it.")


def build_prompt(obs: dict, history_events: int = 80) -> str:
    return render_text(obs, history_events) + "\n\n" + render_request(obs["request"])


class LLMAgent(RandomAgent):
    def __init__(self, agent_id, board, complete: Callable[[str, str], str], seed=None,
                 history_events: int = 80, system: str = SYSTEM, keep_raw: bool = True):
        super().__init__(agent_id, board, seed)
        self.complete = complete
        self.history_events = history_events
        self.system = system
        self.keep_raw = keep_raw
        self.raw: list[dict] = []
        self.calls = 0
        self.parse_failures = 0
        self.errors = 0

    def _ask(self, prompt: str):
        if hasattr(self.complete, "complete"):          # a cpbench.backends.Backend
            c = self.complete.complete(self.system, prompt)
            return c.text, c.error, (c.prompt_tokens, c.completion_tokens)
        return self.complete(self.system, prompt), None, None

    def act(self, obs: dict) -> dict:
        prompt = build_prompt(obs, self.history_events)
        reply, error, toks = self._ask(prompt)
        self.calls += 1
        self.errors += int(error is not None)
        if self.keep_raw:
            self.raw.append({"round": obs["round"], "kind": obs["request"]["kind"], "prompt_chars": len(prompt),
                             "reply": reply, "error": error, "tokens": toks})
        action = parse_json(reply)
        if action is None:
            self.parse_failures += 1
            return super().act(obs)
        if "thought" in action:
            action["_thought"] = action.pop("thought")
        return action

    def stats(self) -> dict:
        return {"calls": self.calls, "parse_failures": self.parse_failures, "errors": self.errors}


_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_json(text: str) -> dict | None:
    """First JSON object in the reply; tolerates <think> blocks, code fences and trailing prose."""
    if not text:
        return None
    text = _THINK.sub("", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start:i + 1])
                        return v if isinstance(v, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
