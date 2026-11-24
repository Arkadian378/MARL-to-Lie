"""Trainable policy: one base model in memory with one LoRA adapter per trained role,
plus frozen adapter snapshots (the self-play population).

The reference policy for the KL term is the base model itself (adapters disabled).
"""
from __future__ import annotations

import os
import shutil

import torch
import torch.nn.functional as F


def _dtype(name: str):
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}[name]


class Policy:
    def __init__(self, model_name: str, adapters: dict[str, str | None], lora: dict | None = None,
                 dtype: str = "bf16", device: str | None = None, load_in_4bit: bool = False,
                 gradient_checkpointing: bool = True, chat_template_kwargs: dict | None = None,
                 attn_implementation: str | None = None, trust_remote_code: bool = False,
                 tokenizer=None, model=None):
        """``adapters``: {adapter_name: init_path or None (fresh LoRA)}."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, PeftModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu" and dtype in ("bf16", "bfloat16", "fp16"):
            dtype = "fp32"
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        if model is None:
            kw = dict(torch_dtype=_dtype(dtype), trust_remote_code=trust_remote_code)
            if attn_implementation:
                kw["attn_implementation"] = attn_implementation
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=_dtype(dtype),
                    bnb_4bit_use_double_quant=True)
                kw["device_map"] = {"": self.device}
            model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
            if load_in_4bit:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
            else:
                model.to(self.device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        lora = dict(lora or {})
        self.lora_cfg = LoraConfig(
            r=lora.get("r", 32), lora_alpha=lora.get("alpha", 64), lora_dropout=lora.get("dropout", 0.0),
            target_modules=lora.get("target_modules", "all-linear"),
            exclude_modules=lora.get("exclude_modules", r".*(visual|vision|mtp).*"),
            task_type="CAUSAL_LM")
        names = list(adapters)
        first = names[0]
        if adapters[first]:
            self.model = PeftModel.from_pretrained(model, adapters[first], adapter_name=first, is_trainable=True)
        else:
            self.model = get_peft_model(model, self.lora_cfg, adapter_name=first)
        for n in names[1:]:
            if adapters[n]:
                self.model.load_adapter(adapters[n], adapter_name=n, is_trainable=True)
            else:
                self.model.add_adapter(n, self.lora_cfg)
        self.trainable = names
        self.frozen: set[str] = set()

    # ------------------------------------------------------------ adapters
    def params_of(self, adapter: str):
        key = f".{adapter}."
        return [p for n, p in self.model.named_parameters() if "lora_" in n and key in n]

    def set_adapter(self, name: str):
        self.model.set_adapter(name)

    def load_frozen(self, name: str, path: str):
        if name in self.frozen or name in self.trainable:
            return
        self.model.load_adapter(path, adapter_name=name, is_trainable=False)
        self.frozen.add(name)

    def save_adapter(self, name: str, path: str) -> str:
        """Writes the adapter so that ``path`` itself holds adapter_config.json (vLLM/PEFT loadable)."""
        tmp = path + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        self.model.save_pretrained(tmp, selected_adapters=[name])
        src = tmp if os.path.exists(os.path.join(tmp, "adapter_config.json")) else os.path.join(tmp, name)
        shutil.rmtree(path, ignore_errors=True)
        shutil.move(src, path)
        shutil.rmtree(tmp, ignore_errors=True)
        return path

    # ------------------------------------------------------------ text
    def chat_prompt(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            **self.chat_template_kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------ generation (HF backend)
    @torch.no_grad()
    def generate(self, adapter: str | None, prompts: list[str], max_new_tokens: int, temperature: float,
                 top_p: float, batch_size: int = 8) -> list[tuple[str, list[int]]]:
        self.model.eval()
        self.model.config.use_cache = True
        out = []
        ctx = self.model.disable_adapter() if adapter is None else _null()
        if adapter is not None:
            self.set_adapter(adapter)
        with ctx:
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i:i + batch_size]
                enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                          temperature=max(temperature, 1e-5), top_p=top_p,
                                          pad_token_id=self.tok.pad_token_id)
                for row in gen[:, enc["input_ids"].shape[1]:].tolist():
                    ids = [t for t in row if t != self.tok.pad_token_id]
                    if self.tok.eos_token_id in ids:
                        ids = ids[:ids.index(self.tok.eos_token_id) + 1]
                    out.append((self.tok.decode(ids, skip_special_tokens=True), ids))
        return out

    # ------------------------------------------------------------ log-probabilities
    def token_logprobs(self, batch: list[tuple[list[int], list[int]]], adapter: str | None,
                       grad: bool) -> list[torch.Tensor]:
        """Log-probabilities of each completion token given its prompt. Left padding, so only
        the last ``max completion length`` logits are computed (saves memory on long prompts)."""
        seqs = [p + c for p, c in batch]
        L = max(len(s) for s in seqs)
        K = max(len(c) for _, c in batch)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s)
            mask[i, L - len(s):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        pos = (mask.cumsum(1) - 1).clamp(min=0)
        ctx = self.model.disable_adapter() if adapter is None else _null()
        if adapter is not None:
            self.set_adapter(adapter)
        with ctx, torch.set_grad_enabled(grad):
            logits = self.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                logits_to_keep=K + 1, use_cache=False).logits
            logits = logits[:, :-1].float()                    # predicts the last K tokens
            targets = ids[:, -K:]
            lp = torch.gather(F.log_softmax(logits, -1), 2, targets.unsqueeze(-1)).squeeze(-1)
        return [lp[i, K - len(c):] for i, (_, c) in enumerate(batch)]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
