"""Offline LLM predictor for ARC tasks, with optional per-task test-time training.

Built for the Kaggle competition environment, which means:

* **Weights come from disk.** `local_files_only=True` everywhere. On Kaggle the
  model is a Dataset mounted under `/kaggle/input/`; nothing is downloaded.
* **Every call is budgeted.** Loading, adaptation, and generation all check a
  deadline, because the notebook is killed at the wall clock regardless of what
  the model is doing.
* **Nothing raises.** A failure to load, adapt, generate, or parse returns
  `None` for that slot and the caller falls back.

Test-time training reuses `src.test_time_train`: adapt LoRA weights on the
task's own augmented demonstrations, predict, then restore the base weights
exactly, so no task can leak into the next.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.grid import Grid, as_grid, is_valid_grid

TaskDict = dict[str, Any]


@dataclass
class LlmConfig:
    """Everything the LLM layer needs. Paths are local; nothing is fetched."""

    model_path: str
    adapter_path: Optional[str] = None
    device: Optional[str] = None
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 0
    # Test-time training. steps=0 disables it and the model stays frozen.
    ttt_steps: int = 0
    ttt_lr: float = 5e-4
    ttt_batch_size: int = 2
    ttt_augmentations: int = 8
    grid_format: str = "compact"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "ttt_steps": self.ttt_steps,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass
class LlmStats:
    loaded: bool = False
    load_error: str = ""
    n_tasks: int = 0
    n_adapted: int = 0
    n_generated: int = 0
    n_parse_failures: int = 0
    n_deadline_skips: int = 0
    seconds_loading: float = 0.0
    seconds_adapting: float = 0.0
    seconds_generating: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "load_error": self.load_error,
            "n_tasks": self.n_tasks,
            "n_adapted": self.n_adapted,
            "n_generated": self.n_generated,
            "n_parse_failures": self.n_parse_failures,
            "n_deadline_skips": self.n_deadline_skips,
            "seconds_loading": round(self.seconds_loading, 2),
            "seconds_adapting": round(self.seconds_adapting, 2),
            "seconds_generating": round(self.seconds_generating, 2),
            "errors": self.errors[:10],
        }


class LlmSolver:
    """Wraps a causal LM as a two-attempt ARC predictor.

    `model` and `tokenizer` may be injected directly, which is how the tests
    drive this without any weights on disk.
    """

    def __init__(
        self,
        config: LlmConfig,
        *,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.stats = LlmStats(loaded=model is not None)
        self._base_state: Optional[dict[str, Any]] = None
        self._device = config.device or "cpu"

    # -- lifecycle ---------------------------------------------------------

    def load(self, *, deadline: Optional[float] = None) -> bool:
        """Load weights from disk. Returns False rather than raising."""
        if self.model is not None:
            self.stats.loaded = True
            return True
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cfg = self.config
            kw = {"local_files_only": True, "trust_remote_code": True}
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, **kw)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            cuda = torch.cuda.is_available()
            dtype = torch.float32
            if cuda:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self._device = cfg.device or ("cuda" if cuda else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                device_map="auto" if cuda else None,
                dtype=dtype,
                **kw,
            )
            if cfg.adapter_path:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(
                    self.model, cfg.adapter_path, local_files_only=True, is_trainable=cfg.ttt_steps > 0
                )
            if not cuda:
                self.model.to(self._device)
            self.model.eval()
            self.stats.loaded = True
        except Exception as exc:
            self.stats.load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.stats.loaded = False
        finally:
            self.stats.seconds_loading += time.perf_counter() - started
        if self.stats.loaded and self.config.ttt_steps > 0:
            self._capture_base_state()
        return self.stats.loaded

    def _capture_base_state(self) -> None:
        try:
            from src.test_time_train import capture_trainable_state

            self._base_state = capture_trainable_state(self.model)
            if not self._base_state:
                # Nothing trainable means TTT would silently do nothing.
                self.config = _with_ttt_disabled(self.config)
                self.stats.errors.append("ttt disabled: no trainable parameters")
        except Exception as exc:
            self.config = _with_ttt_disabled(self.config)
            self.stats.errors.append(f"ttt disabled: {type(exc).__name__}")

    # -- prediction --------------------------------------------------------

    def predict(
        self,
        task: TaskDict,
        test_idx: int = 0,
        *,
        n_attempts: int = 2,
        deadline: Optional[float] = None,
    ) -> list[Optional[Grid]]:
        """Up to `n_attempts` grids for one test input. Entries may be None."""
        if not self.stats.loaded or self.model is None:
            return [None] * n_attempts
        if deadline is not None and time.perf_counter() >= deadline:
            self.stats.n_deadline_skips += 1
            return [None] * n_attempts
        self.stats.n_tasks += 1

        adapted = False
        if self.config.ttt_steps > 0 and self._base_state is not None:
            adapted = self._adapt(task, deadline=deadline)

        out: list[Optional[Grid]] = []
        try:
            for attempt in range(n_attempts):
                if deadline is not None and time.perf_counter() >= deadline:
                    self.stats.n_deadline_skips += 1
                    break
                # Attempt 1 greedy, later attempts sampled, per the ARC pass@2 rules.
                temperature = 0.0 if attempt == 0 else self.config.temperature
                grid = self._generate_one(task, test_idx, temperature, attempt)
                out.append(grid)
        finally:
            if adapted:
                self._restore()
        while len(out) < n_attempts:
            out.append(None)
        return out

    def _adapt(self, task: TaskDict, *, deadline: Optional[float]) -> bool:
        started = time.perf_counter()
        try:
            from src.test_time_train import adapt_model_to_task

            cfg = self.config
            adapt_model_to_task(
                self.model,
                self.tokenizer,
                task,
                ttt_steps=cfg.ttt_steps,
                learning_rate=cfg.ttt_lr,
                batch_size=cfg.ttt_batch_size,
                device=self._device,
                num_augmentations=cfg.ttt_augmentations,
                seed=cfg.seed,
            )
            self.stats.n_adapted += 1
            return True
        except Exception as exc:
            self.stats.errors.append(f"adapt: {type(exc).__name__}: {str(exc)[:120]}")
            return False
        finally:
            self.stats.seconds_adapting += time.perf_counter() - started

    def _restore(self) -> None:
        """Strict reset. A leak here would silently contaminate every later task."""
        try:
            from src.test_time_train import restore_trainable_state

            restore_trainable_state(self.model, self._base_state, device=self._device)
            self.model.eval()
        except Exception as exc:
            self.stats.errors.append(f"restore: {type(exc).__name__}")

    def _generate_one(
        self, task: TaskDict, test_idx: int, temperature: float, attempt: int
    ) -> Optional[Grid]:
        started = time.perf_counter()
        try:
            import torch

            from src.data import DEFAULT_SYSTEM_PROMPT, task_to_prompt, text_to_grid

            user_prompt, _ = task_to_prompt(
                task, test_idx=test_idx, include_test_output=False,
                grid_format=self.config.grid_format,
            )
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if getattr(self.tokenizer, "chat_template", None):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            n_prompt = int(inputs["input_ids"].shape[-1])
            gen: dict[str, Any] = {
                "max_new_tokens": int(self.config.max_new_tokens),
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature and temperature > 0:
                gen.update(
                    do_sample=True, temperature=float(temperature), top_p=float(self.config.top_p)
                )
            else:
                gen["do_sample"] = False
            torch.manual_seed(self.config.seed + attempt)
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen)
            completion = self.tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True)
            self.stats.n_generated += 1
            parsed = text_to_grid(completion)
            if parsed is None or not is_valid_grid(parsed):
                self.stats.n_parse_failures += 1
                return None
            return as_grid(parsed)
        except Exception as exc:
            self.stats.errors.append(f"generate: {type(exc).__name__}: {str(exc)[:120]}")
            return None
        finally:
            self.stats.seconds_generating += time.perf_counter() - started


def _with_ttt_disabled(cfg: LlmConfig) -> LlmConfig:
    from dataclasses import replace

    return replace(cfg, ttt_steps=0)


def find_model_dir(*roots: str) -> Optional[str]:
    """Locate a model directory in a Kaggle Dataset mount.

    A model directory is one containing a config.json alongside weights.
    """
    from pathlib import Path

    search = [Path(r) for r in (roots or ("/kaggle/input", "models"))]
    for base in search:
        if not base.is_dir():
            continue
        if (base / "config.json").is_file():
            return str(base)
        for cfg in sorted(base.glob("**/config.json")):
            folder = cfg.parent
            if any(folder.glob("*.safetensors")) or any(folder.glob("*.bin")):
                return str(folder)
    return None
