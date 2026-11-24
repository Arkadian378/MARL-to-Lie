"""Text generation for rollouts. Three interchangeable backends:

- ``hf``:          transformers ``generate`` on the training model itself. Works everywhere
                   (CPU tests, small models); slow for 9B+ models.
- ``vllm``:        vLLM inside the training process, on the same GPU, LoRA adapters hot-loaded
                   from disk after every update. The recommended setting on one A100/H100 80 GB.
- ``vllm_server``: an external vLLM OpenAI server (another GPU or node) started with
                   ``--enable-lora`` and ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True``; adapters are
                   loaded through ``/v1/load_lora_adapter`` from a shared filesystem.

All return, for each prompt, ``(text, completion_token_ids)``. Requests name an adapter (or None
for the base model). ``register(name, path)`` tells the generator where the newest weights of that
adapter are on disk; calling it again with a new path (after an update) swaps the weights.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class HFGenerator:
    def __init__(self, policy, max_new_tokens=384, temperature=1.0, top_p=1.0, batch_size=8, **_):
        self.policy, self.batch_size = policy, batch_size
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p

    def register(self, name: str, path: str | None):
        pass        # the HF generator uses the adapters living in the training model


    def generate(self, prompts: list[str], adapters: list[str | None]) -> list[tuple[str, list[int]]]:
        out: list = [None] * len(prompts)
        groups: dict = {}
        for i, a in enumerate(adapters):
            groups.setdefault(a, []).append(i)
        for a, idx in groups.items():
            res = self.policy.generate(a, [prompts[i] for i in idx], self.max_new_tokens,
                                       self.temperature, self.top_p, self.batch_size)
            for i, r in zip(idx, res):
                out[i] = r
        return out


class VLLMGenerator:
    def __init__(self, model_name: str, max_new_tokens=384, temperature=1.0, top_p=1.0, max_lora_rank=32,
                 max_loras=8, gpu_memory_utilization=0.35, max_model_len=8192, dtype="bfloat16",
                 enable_prefix_caching=True, **kw):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        self._LoRARequest = LoRARequest
        self.llm = LLM(model=model_name, enable_lora=True, max_lora_rank=max_lora_rank, max_loras=max_loras,
                       gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, dtype=dtype,
                       enable_prefix_caching=enable_prefix_caching, **kw)
        self.sp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        self.reqs: dict[str, object] = {}
        self.paths: dict[str, str] = {}
        self.next_id = 1

    def register(self, name: str, path: str | None):
        if path and self.paths.get(name) != path:
            # a new integer id forces vLLM to load the new weights instead of its cached copy
            self.reqs[name] = self._LoRARequest(f"{name}-{self.next_id}", self.next_id, os.path.abspath(path))
            self.paths[name] = path
            self.next_id += 1

    def generate(self, prompts, adapters):
        lr = [self.reqs[a] if a else None for a in adapters]
        outs = self.llm.generate(prompts, self.sp, lora_request=lr, use_tqdm=False)
        return [(o.outputs[0].text, list(o.outputs[0].token_ids)) for o in outs]


class VLLMServerGenerator:
    def __init__(self, base_url: str, base_model_name: str, tokenizer, max_new_tokens=384, temperature=1.0,
                 top_p=1.0, threads=64, timeout=600, **_):
        self.base_url, self.base_model = base_url.rstrip("/"), base_model_name
        self.tok, self.threads, self.timeout = tokenizer, threads, timeout
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p
        self.loaded: dict[str, str] = {}      # adapter name -> served lora name
        self.paths: dict[str, str] = {}
        self.n = 0
        self.lock = threading.Lock()

    def _post(self, path, body):
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            txt = r.read()
            return json.loads(txt) if txt.strip().startswith(b"{") else {}

    def register(self, name: str, path: str | None):
        if not path:
            return
        with self.lock:
            if self.paths.get(name) == path:
                return
            self.n += 1
            lname = f"{name}-{self.n}"
            self._post("/load_lora_adapter", {"lora_name": lname, "lora_path": os.path.abspath(path)})
            old = self.loaded.get(name)
            self.loaded[name], self.paths[name] = lname, path
            if old:      # free the previous version on the server
                try:
                    self._post("/unload_lora_adapter", {"lora_name": old})
                except Exception:  # noqa: BLE001
                    pass

    def _one(self, prompt, adapter):
        model = self.loaded[adapter] if adapter else self.base_model
        r = self._post("/completions", {"model": model, "prompt": prompt, "max_tokens": self.max_new_tokens,
                                        "temperature": self.temperature, "top_p": self.top_p})
        text = r["choices"][0]["text"]
        return text, self.tok(text, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]

    def generate(self, prompts, adapters):
        with ThreadPoolExecutor(self.threads) as ex:
            return list(ex.map(self._one, prompts, adapters))


def make_generator(kind: str, policy, model_name: str, **kw):
    if kind == "hf":
        return HFGenerator(policy, **kw)
    if kind == "vllm":
        return VLLMGenerator(model_name, **kw)
    if kind == "vllm_server":
        return VLLMServerGenerator(tokenizer=policy.tok, base_model_name=kw.pop("base_model_name", model_name), **kw)
    raise ValueError(kind)
