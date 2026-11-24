"""LLM backends used by ``LLMAgent`` and by the LLM judge.

Every backend exposes ``complete(system, user) -> Completion`` and keeps
thread-safe token counters, so an experiment can report tokens and cost.

- ``openai``: any OpenAI-compatible ``/chat/completions`` endpoint. This covers a
  local or cluster **vLLM** server (open-weight Qwen, with or without LoRA
  adapters), OpenAI, Gemini (OpenAI-compatible endpoint), DeepSeek, Together.
  Uses only the standard library.
- ``anthropic``: Claude models through the official ``anthropic`` SDK.
- ``mock``: no model; answers with a random *legal* JSON action parsed from the
  request. Used by the tests and to dry-run the whole pipeline without a GPU.

Build one from a config dict with ``make_backend``.
"""
from __future__ import annotations

import ast
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# USD per 1M tokens (input, output), standard tier, checked 2026-11-15. Used only for cost estimates.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "deepseek-v4-pro": (1.32, 3.96),
    "deepseek-flash": (0.30, 1.20),
}


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None


@dataclass
class Usage:
    calls: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, c: Completion):
        with self._lock:
            self.calls += 1
            self.errors += int(c.error is not None)
            self.prompt_tokens += c.prompt_tokens
            self.completion_tokens += c.completion_tokens
            self.seconds += c.latency_s

    def to_dict(self, model: str | None = None) -> dict:
        d = {"calls": self.calls, "errors": self.errors, "prompt_tokens": self.prompt_tokens,
             "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 1)}
        if model in PRICES:
            pin, pout = PRICES[model]
            d["est_cost_usd"] = round((self.prompt_tokens * pin + self.completion_tokens * pout) / 1e6, 4)
        return d


class Backend:
    model: str = "?"

    def __init__(self):
        self.usage = Usage()

    def complete(self, system: str, user: str) -> Completion:
        c = self._complete(system, user)
        self.usage.add(c)
        return c

    def _complete(self, system: str, user: str) -> Completion:
        raise NotImplementedError

    # LLMAgent accepts any callable(system, user) -> str; backends are callable too
    def __call__(self, system: str, user: str) -> str:
        return self.complete(system, user).text


def _retry(fn, max_retries: int, what: str):
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (network errors, 429, 5xx)
            retriable = True
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404, 422):
                retriable = False
            if not retriable or attempt == max_retries:
                raise
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)


class OpenAICompatBackend(Backend):
    """POST {base_url}/chat/completions. ``extra_body`` is merged into the request (e.g. for vLLM +
    Qwen: ``{"chat_template_kwargs": {"enable_thinking": false}}``)."""

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", api_key: str | None = None,
                 api_key_env: str | None = "OPENAI_API_KEY", temperature: float | None = 0.7,
                 max_tokens: int = 512, top_p: float | None = None, extra_body: dict | None = None,
                 timeout: float = 300, max_retries: int = 6, max_tokens_field: str = "max_tokens"):
        super().__init__()
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None) or "EMPTY"
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.max_tokens_field = max_tokens_field   # OpenAI reasoning models want "max_completion_tokens"
        self.extra_body, self.timeout, self.max_retries = extra_body or {}, timeout, max_retries

    def _complete(self, system: str, user: str) -> Completion:
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                self.max_tokens_field: self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        body.update(self.extra_body)
        data = json.dumps(body).encode()

        def call():
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())

        t0 = time.time()
        try:
            out = _retry(call, self.max_retries, self.model)
        except Exception as e:  # noqa: BLE001
            return Completion("", latency_s=time.time() - t0, error=f"{type(e).__name__}: {e}")
        msg = (out.get("choices") or [{}])[0].get("message") or {}
        u = out.get("usage") or {}
        return Completion(msg.get("content") or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                          time.time() - t0)


class AnthropicBackend(Backend):
    """Claude through the official SDK. Thinking stays adaptive; depth is set with ``effort``.
    Server-side refusal fallbacks are on by default (beta ``server-side-fallback-2026-07-01``)."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 4096,
                 fallbacks: bool = True, max_retries: int = 6, timeout: float = 600):
        super().__init__()
        import anthropic  # optional dependency
        self.client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.model, self.effort, self.max_tokens, self.fallbacks = model, effort, max_tokens, fallbacks

    def _complete(self, system: str, user: str) -> Completion:
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                  messages=[{"role": "user", "content": user}], output_config={"effort": self.effort})
        t0 = time.time()
        try:
            if self.fallbacks:
                r = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                     fallbacks="default", **kw)
            else:
                r = self.client.messages.create(**kw)
        except Exception as e:  # noqa: BLE001
            return Completion("", latency_s=time.time() - t0, error=f"{type(e).__name__}: {e}")
        if r.stop_reason == "refusal":
            return Completion("", r.usage.input_tokens, r.usage.output_tokens, time.time() - t0, error="refusal")
        text = "".join(b.text for b in r.content if b.type == "text")
        return Completion(text, r.usage.input_tokens, r.usage.output_tokens, time.time() - t0)


class MockBackend(Backend):
    """Random legal action read from the 'Details:' line of the request. ``invalid_rate`` of the
    answers are garbage, to exercise the invalid-action path."""

    model = "mock"

    def __init__(self, seed: int = 0, invalid_rate: float = 0.05, **_):
        super().__init__()
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.invalid_rate = invalid_rate

    def _complete(self, system: str, user: str) -> Completion:
        with self.lock:
            rng = random.Random(self.rng.random())
        text = mock_answer(user, rng, self.invalid_rate)
        return Completion(text, prompt_tokens=len(system + user) // 4, completion_tokens=len(text) // 4)


def mock_answer(user: str, rng: random.Random, invalid_rate: float = 0.0) -> str:
    if rng.random() < invalid_rate:
        return "I am not sure what to do."
    kind = re.search(r"ACTION REQUIRED: (\w+)", user)
    det = re.search(r"Details: (\{.*\})", user)
    kind = kind.group(1) if kind else ""
    try:
        d = ast.literal_eval(det.group(1)) if det else {}
    except (ValueError, SyntaxError):
        d = {}
    a: dict = {}
    if kind == "place":
        a = {"station": rng.choice(d["legal"])}
    elif kind == "thief_move":
        a = {"path": rng.choice(d["legal"])}
    elif kind == "chat":
        if rng.random() < 0.5:
            a = {"channel": rng.choice(d["channels"]), "text": "I think the thieves are near the centre."}
    elif kind == "police_action":
        a = {"type": rng.choice(d["legal_types"]), "station": rng.choice(d["legal_stations"])}
    elif kind == "report":
        a = {"team_claim": rng.choice(d["legal_claims"]), "share_public": rng.random() < 0.3}
    elif kind == "vote":
        a = {"target": rng.choice(d["legal"])}
    a["thought"] = "mock reasoning"
    return json.dumps(a)


def make_backend(spec: dict) -> Backend:
    spec = dict(spec)
    typ = spec.pop("type")
    if typ == "openai":
        return OpenAICompatBackend(**spec)
    if typ == "anthropic":
        return AnthropicBackend(**spec)
    if typ == "mock":
        return MockBackend(**spec)
    raise ValueError(f"unknown backend type {typ}")
