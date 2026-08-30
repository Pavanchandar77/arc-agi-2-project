"""Transformers 5 / current TRL compatibility for Bond-4B native-precision LoRA.

Inspects installed signatures at runtime. Does not download models.
Does not implement QLoRA: this stack loads native fp16/bf16 + LoRA.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import logging
import sys
import types
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

TORCHAO_MIN = (0, 16, 0)
NATIVE_LORA = "native_precision_lora"


def param_names(obj: Any) -> set[str]:
    try:
        target = obj if inspect.isfunction(obj) or inspect.ismethod(obj) else getattr(obj, "__init__", obj)
        return set(inspect.signature(target).parameters)
    except (TypeError, ValueError):
        return set()


def filter_to_signature(obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    names = param_names(obj)
    banned = {"tokenizer", "dataset_text_field", "torch_dtype"}
    try:
        target = obj if inspect.isfunction(obj) else obj.__init__
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        has_var_kw = False
    if not names or has_var_kw:
        return {k: v for k, v in kwargs.items() if k not in banned}
    return {k: v for k, v in kwargs.items() if k in names and k != "self" and k not in banned}


def parse_version_tuple(version: str) -> tuple[int, int, int]:
    raw = (version or "0").split("+")[0].split(".")
    nums: list[int] = []
    for part in raw[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def from_pretrained_dtype_kwargs(dtype: Any, params: Optional[set[str]] = None) -> dict[str, Any]:
    """Transformers 5 removed torch_dtype in favor of dtype on some loaders."""
    if params is None:
        try:
            from transformers import AutoModelForCausalLM

            params = param_names(AutoModelForCausalLM.from_pretrained)
        except Exception:
            params = {"dtype", "torch_dtype"}
    if "dtype" in params:
        return {"dtype": dtype}
    if "torch_dtype" in params:
        return {"torch_dtype": dtype}
    return {}


def warmup_kwargs(params: set[str], warmup_ratio: float) -> dict[str, Any]:
    """Keep the numeric 0.05. Do not convert a ratio into an integer step count.

    Older TrainingArguments: warmup_ratio=0.05 means 5% of training steps.
    Transformers 5 Kaggle runtime: warmup_ratio was removed; pass the same
    0.05 as warmup_steps. A float in [0, 1) is a ratio, not 0.05 steps.
    """
    if "warmup_ratio" in params:
        return {"warmup_ratio": warmup_ratio}
    if "warmup_steps" in params:
        return {"warmup_steps": warmup_ratio}
    return {}


def eval_strategy_kwargs(params: set[str], strategy: str) -> dict[str, Any]:
    if "eval_strategy" in params:
        return {"eval_strategy": strategy}
    if "evaluation_strategy" in params:
        return {"evaluation_strategy": strategy}
    return {}


def max_length_kwargs(params: set[str], max_seq_length: int) -> dict[str, Any]:
    if "max_length" in params:
        return {"max_length": max_seq_length}
    if "max_seq_length" in params:
        return {"max_seq_length": max_seq_length}
    return {}


def build_sft_config_kwargs(
    params: set[str],
    *,
    output_dir: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    max_seq_length: int,
    num_train_epochs: int,
    logging_steps: int,
    save_steps: int,
    has_eval: bool,
    use_fp16: bool,
    use_bf16: bool,
    seed: int,
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": num_train_epochs,
        "logging_steps": logging_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 2,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": "adamw_torch",
        "gradient_checkpointing": gradient_checkpointing,
        "weight_decay": 0.01,
        "seed": seed,
        "report_to": "none",
    }
    # TRL >= 0.20 defaults completion_only_loss=True and refuses a
    # formatting_func alongside it, because formatting produces a plain
    # language-modelling dataset with no prompt/completion boundary to mask on.
    # We format, so the loss is over the whole sequence.
    if "completion_only_loss" in params:
        kw["completion_only_loss"] = False
    kw.update(warmup_kwargs(params, warmup_ratio))
    kw.update(max_length_kwargs(params, max_seq_length))
    strategy = "steps" if has_eval else "no"
    kw.update(eval_strategy_kwargs(params, strategy))
    if has_eval:
        kw["eval_steps"] = save_steps
    elif "eval_steps" in params:
        kw["eval_steps"] = None
    return {k: v for k, v in kw.items() if k in params or not params}


def build_sft_trainer_kwargs(
    params: set[str],
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    formatting_func: Optional[Callable],
) -> dict[str, Any]:
    """Current TRL: processing_class + SFTConfig.max_length. Never pass tokenizer= or max_seq_length=."""
    kw: dict[str, Any] = {"model": model, "args": args, "train_dataset": train_dataset}
    if "eval_dataset" in params:
        kw["eval_dataset"] = eval_dataset
    if "processing_class" in params:
        kw["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kw["tokenizer"] = tokenizer
    if formatting_func is not None and "formatting_func" in params:
        kw["formatting_func"] = formatting_func
    # Explicitly never attach obsolete keys even if a signature still lists them.
    kw.pop("dataset_text_field", None)
    kw.pop("max_seq_length", None)
    if "processing_class" in kw:
        kw.pop("tokenizer", None)
    if params:
        kw = {k: v for k, v in kw.items() if k in params}
    return kw


# A base checkpoint ships no chat template, and apply_chat_template raises
# rather than degrading. Patching our own formatter is not enough: TRL calls
# apply_chat_template independently inside _prepare_dataset, and so does any
# other consumer. The only fix that covers them all is to give the tokenizer a
# template when it has none.
FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def ensure_chat_template(tokenizer: Any) -> bool:
    """Install a plain role-prefixed template if the tokenizer lacks one.

    Returns True if a template was installed. Instruct checkpoints already have
    one and are left untouched.
    """
    if getattr(tokenizer, "chat_template", None):
        return False
    try:
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    except Exception:
        return False
    logger.info("tokenizer had no chat_template; installed the plain fallback")
    return True


def _flatten_messages(msgs: list[dict[str, Any]], eos: str) -> str:
    """Plain-text rendering for base checkpoints that ship no chat template."""
    parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs]
    return "\n".join(parts) + eos


def make_formatting_func(tokenizer: Any) -> Callable[[dict[str, Any]], Any]:
    """Chat-template formatter. Single conversational example returns a str.

    Base (non-instruct) checkpoints have no chat template, and
    apply_chat_template raises rather than degrading, so fall back to a plain
    role-prefixed rendering instead of failing the whole training run.
    """

    def render(msgs: list[dict[str, Any]]) -> str:
        eos = getattr(tokenizer, "eos_token", "") or ""
        if not getattr(tokenizer, "chat_template", None):
            return _flatten_messages(msgs, eos)
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    def formatting_prompts_func(example: dict[str, Any]):
        if "messages" in example:
            msgs = example["messages"]
            if msgs and isinstance(msgs[0], dict):
                return render(msgs)
            if msgs and isinstance(msgs[0], list):
                return [render(m) for m in msgs]
        if "prompt" in example and "completion" in example:
            prompt, completion = example["prompt"], example["completion"]
            eos = getattr(tokenizer, "eos_token", "") or ""
            if isinstance(prompt, list):
                return [f"{p}\n{c}{eos}" for p, c in zip(prompt, completion)]
            return f"{prompt}\n{completion}{eos}"
        return ""

    return formatting_prompts_func


def preview_formatted_example(example: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    fn = make_formatting_func(tokenizer)
    formatted = fn(example)
    text = formatted[0] if isinstance(formatted, list) else formatted
    n_tokens = None
    try:
        ids = tokenizer(text, add_special_tokens=False)
        n_tokens = len(ids["input_ids"]) if isinstance(ids, dict) else len(ids)
    except Exception:
        n_tokens = None
    return {
        "kind": type(formatted).__name__,
        "is_str": isinstance(formatted, str),
        "chars": len(text or ""),
        "n_tokens": n_tokens,
        "preview": (text or "")[:400],
        "looks_like_python_repr": (text or "").lstrip().startswith(("{", "[", "<"))
        and "role" not in (text or "")[:80]
        and "user" not in (text or "").lower()[:120],
    }


def neutralize_incompatible_torchao() -> dict[str, Any]:
    """Ordinary LoRA does not use torchao. PEFT may still import it.

    Policy: if torchao is missing, continue. If torchao >= 0.16.0, keep it.
    If torchao < 0.16.0, stub it before PEFT import so native LoRA can run.
    Preferred operator fix: `pip uninstall -y torchao`.
    """
    rec: dict[str, Any] = {
        "policy": NATIVE_LORA,
        "required_if_present": ">=0.16.0",
        "present": False,
        "version": None,
        "action": "none",
    }
    try:
        import importlib.metadata as md

        ver = md.version("torchao")
    except Exception:
        rec["action"] = "not_installed"
        return rec
    rec["present"] = True
    rec["version"] = ver
    if parse_version_tuple(ver) >= TORCHAO_MIN:
        rec["action"] = "kept_compatible"
        return rec
    for name in list(sys.modules):
        if name == "torchao" or name.startswith("torchao."):
            del sys.modules[name]
    stub = types.ModuleType("torchao")
    stub.__version__ = f"{ver}+disabled-for-native-lora"
    stub.__file__ = None
    # A module built by hand has __spec__ = None, and importlib.util.find_spec
    # raises ValueError on that rather than returning it. transformers calls
    # find_spec("torchao") while importing, so a stub without a spec turns this
    # guard into the crash it exists to prevent.
    stub.__spec__ = importlib.machinery.ModuleSpec("torchao", loader=None)

    def _missing(name: str):
        raise ImportError(
            f"torchao.{name} disabled: {ver} is incompatible with PEFT. "
            "This run is native-precision LoRA. pip uninstall -y torchao"
        )

    stub.__getattr__ = _missing  # type: ignore[attr-defined]
    dtypes = types.ModuleType("torchao.dtypes")
    dtypes.__spec__ = importlib.machinery.ModuleSpec("torchao.dtypes", loader=None)
    dtypes.AffineQuantizedTensor = type("AffineQuantizedTensor", (), {})  # type: ignore[attr-defined]
    sys.modules["torchao"] = stub
    sys.modules["torchao.dtypes"] = dtypes
    rec["action"] = "disabled_incompatible"
    rec["note"] = (
        f"torchao {ver} is < 0.16.0 and crashes PEFT LoRA injection. "
        "Stubbed because this run is native-precision LoRA, not torchao quantization. "
        "Preferred: pip uninstall -y torchao"
    )
    logger.warning(rec["note"])
    return rec
