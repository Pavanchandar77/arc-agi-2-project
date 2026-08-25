"""Frozen open-model client for the elevation experiment.

Preferred checkpoint: thinkingmachines/Inkling-Small (open weights, ~276B MoE).
That is not a local-runnable default. The project local alternative is
Qwen/Qwen2.5-1.5B-Instruct, the same Instruct checkpoint used by src/train.py.

The model is frozen: no training, no TTT, no per-task weight updates.
Temperature, max tokens, and call budget are matched across M0–M3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

PREFERRED_INKLING = "thinkingmachines/Inkling-Small"
LOCAL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"


def resolve_model_name(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("HRPS_MODEL") or os.environ.get("HRPS_OPEN_MODEL")
    if env:
        return env
    return LOCAL_DEFAULT


def inking_is_local_default() -> bool:
    return False


@dataclass
class ModelTurn:
    text: str
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    backend: str = "none"
    model_name: str = ""


class FrozenOpenModel(Protocol):
    name: str
    backend: str

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> ModelTurn: ...


@dataclass
class ScriptedModel:
    """Deterministic stand-in for tests. Not an elevation result."""

    responses: list[str] = field(default_factory=list)
    name: str = "scripted"
    backend: str = "fake"
    calls: int = 0

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> ModelTurn:
        self.calls += 1
        if not self.responses:
            text = ""
        else:
            text = self.responses.pop(0)
        return ModelTurn(
            text=text,
            n_prompt_tokens=len(prompt.split()),
            n_completion_tokens=len(text.split()),
            backend=self.backend,
            model_name=self.name,
        )


@dataclass
class CallbackModel:
    fn: Callable[[str], str]
    name: str = "callback"
    backend: str = "fake"
    calls: int = 0

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> ModelTurn:
        self.calls += 1
        text = self.fn(prompt)
        return ModelTurn(
            text=text,
            n_prompt_tokens=len(prompt.split()),
            n_completion_tokens=len(text.split()),
            backend=self.backend,
            model_name=self.name,
        )


class HuggingFaceModel:
    """Lazy transformers causal LM. Requires torch in the runtime.

    Optional adapter_path loads Bond LoRA/QLoRA on top of the frozen base.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_name = model_name
        self.name = model_name if not adapter_path else f"{model_name}+bond"
        self.backend = "hf" if not adapter_path else "hf_bond"
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cuda = torch.cuda.is_available()
        dtype = torch.float32
        if cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.device = device or ("cuda" if cuda else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if cuda else None,
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        if not cuda:
            self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, *, max_tokens: int, temperature: float) -> ModelTurn:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_prompt = int(inputs["input_ids"].shape[-1])
        gen_kwargs: dict = {
            "max_new_tokens": int(max_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][n_prompt:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return ModelTurn(
            text=completion,
            n_prompt_tokens=n_prompt,
            n_completion_tokens=int(new_tokens.shape[-1]),
            backend="hf",
            model_name=self.name,
        )


def try_load_open_model(
    model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> tuple[Optional[FrozenOpenModel], str]:
    """Load the frozen open model. Never downloads a 276B checkpoint by accident.

    Returns (model, status). status is 'ok', 'no_torch', 'no_transformers',
    'inkling_too_large', or an error string. adapter_path loads Bond LoRA.
    """
    name = resolve_model_name(model_name)
    if name.lower().startswith("thinkingmachines/inkling") or name.lower() == "inkling-small":
        allow = os.environ.get("HRPS_ALLOW_INKLING", "").strip() in {"1", "true", "yes"}
        if not allow:
            return (
                None,
                "inkling_too_large: Inkling-Small is ~276B MoE; set HRPS_ALLOW_INKLING=1 to override, "
                f"or use {LOCAL_DEFAULT}",
            )
    try:
        import torch  # noqa: F401
    except Exception:
        return None, "no_torch"
    try:
        import transformers  # noqa: F401
    except Exception:
        return None, "no_transformers"
    try:
        model = HuggingFaceModel(name, adapter_path=adapter_path)
        return model, "ok"
    except Exception as exc:
        return None, f"load_failed:{type(exc).__name__}:{exc}"
