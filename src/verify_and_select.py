"""Offline self-verification and pass@2 candidate selection for ARC-AGI-2.

Competition submissions run fully sandboxed (no internet, no external APIs).
This module uses only the task's own known training pairs as a consistency
signal for ranking generated test-output candidates.

Idea
----
After (optional) test-time training, generate *more than 2* candidate output
grids for the held-out test input, using different decoding temperatures /
seeds.  For each candidate, verify that the SAME adapted model can still
reproduce the task's known demonstration pairs when those pairs are presented
as if they were the test problem (leave-one-out).

Optionally, the candidate itself is inserted as an extra demonstration
("hypothesized rule example").  A candidate that contradicts the true rule
tends to poison reconstruction of the known pairs and is ranked down.

ARC scoring allows exactly 2 submitted attempts per test case (pass@2).
We rank candidates by the consistency check and submit the top 2 unique
valid grids, rather than blindly submitting greedy + one arbitrary sample.

This is local/offline only: no tools, no program interpreters, no APIs.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.data import (
    DEFAULT_SYSTEM_PROMPT,
    grids_equal,
    is_valid_grid,
    task_to_chat_messages,
    text_to_grid,
)

logger = logging.getLogger(__name__)

# Signature: generate_fn(prompt: str, gen_config: Dict[str, Any]) -> str
GenerateFn = Callable[[str, Dict[str, Any]], str]


# ============================================================================
# 1. Decoding schedule
# ============================================================================

@dataclass(frozen=True)
class GenerationConfig:
    """One decoding attempt (temperature / seed / greedy vs sample)."""

    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }


# Default: greedy plus a spread of temperatures and a repeated mid temperature
# with a second seed.  More than 2 so ranking has something to choose from.
DEFAULT_CANDIDATE_SCHEDULE: Tuple[GenerationConfig, ...] = (
    GenerationConfig(do_sample=False, temperature=0.0, seed=0),
    GenerationConfig(do_sample=True, temperature=0.2, seed=1),
    GenerationConfig(do_sample=True, temperature=0.4, seed=2),
    GenerationConfig(do_sample=True, temperature=0.7, seed=3),
    GenerationConfig(do_sample=True, temperature=0.7, seed=4),
    GenerationConfig(do_sample=True, temperature=1.0, seed=5),
)

_EXTRA_TEMPERATURES: Tuple[float, ...] = (0.3, 0.5, 0.8, 1.2)


def build_candidate_schedule(n_candidates: int) -> List[GenerationConfig]:
    """Return `n_candidates` decoding configs, extending the default if needed."""
    if n_candidates <= 0:
        return []
    schedule: List[GenerationConfig] = list(DEFAULT_CANDIDATE_SCHEDULE)
    extra_i = 0
    while len(schedule) < n_candidates:
        schedule.append(
            GenerationConfig(
                do_sample=True,
                temperature=_EXTRA_TEMPERATURES[extra_i % len(_EXTRA_TEMPERATURES)],
                seed=100 + extra_i,
            )
        )
        extra_i += 1
    return schedule[:n_candidates]


# ============================================================================
# 2. Candidate / probe records
# ============================================================================

@dataclass
class ProbeResult:
    """Result of presenting one known training pair as if it were the test."""

    pair_index: int
    exact_match: bool
    cell_accuracy: float
    valid_parse: bool
    predicted_grid: Optional[List[List[int]]]
    raw_text: str


@dataclass
class Candidate:
    """One test-output hypothesis plus its offline consistency score."""

    raw_text: str
    grid: Optional[List[List[int]]]
    gen_config: GenerationConfig
    consistency_score: float = 0.0
    mean_cell_accuracy: float = 0.0
    n_train_exact: int = 0
    n_train_total: int = 0
    n_valid_probes: int = 0
    is_valid_grid: bool = False
    probe_results: List[ProbeResult] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "consistency_score": self.consistency_score,
            "mean_cell_accuracy": self.mean_cell_accuracy,
            "n_train_exact": self.n_train_exact,
            "n_train_total": self.n_train_total,
            "n_valid_probes": self.n_valid_probes,
            "is_valid_grid": self.is_valid_grid,
            "gen_config": self.gen_config.to_dict(),
        }


# ============================================================================
# 3. Grid helpers (kept local to avoid an evaluate.py circular import)
# ============================================================================

def grid_cell_accuracy(
    pred: Optional[Sequence[Sequence[int]]],
    gt: Sequence[Sequence[int]],
) -> float:
    """Fraction of matching cells when shapes match; else 0."""
    if pred is None or not is_valid_grid(pred) or not is_valid_grid(gt):
        return 0.0
    if len(pred) != len(gt) or len(pred[0]) != len(gt[0]):
        return 0.0
    total = len(gt) * len(gt[0])
    if total == 0:
        return 1.0
    correct = sum(
        1
        for r in range(len(gt))
        for c in range(len(gt[0]))
        if pred[r][c] == gt[r][c]
    )
    return correct / total


def grid_fingerprint(grid: Optional[Sequence[Sequence[int]]]) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """Hashable identity for a parsed grid (used to drop duplicate submissions)."""
    if grid is None:
        return None
    try:
        return tuple(tuple(int(c) for c in row) for row in grid)
    except (TypeError, ValueError):
        return None


# ============================================================================
# 4. Leave-one-out probe tasks
# ============================================================================

def build_probe_task(
    task: Dict[str, Any],
    held_out_idx: int,
    candidate_pair: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a probe task that presents one known training pair as the test.

    Demonstrations are the remaining training pairs.  When `candidate_pair`
    is provided (the hypothesized test input/output), it is appended as an
    extra demonstration so a contradictory hypothesis can poison reconstruction.

    If that would leave zero demonstrations (single-pair task, no candidate),
    the held-out pair is reused as its own demonstration (self-reproduction).
    """
    train_pairs = list(task.get("train", []))
    if held_out_idx < 0 or held_out_idx >= len(train_pairs):
        raise IndexError(
            f"held_out_idx {held_out_idx} out of range for {len(train_pairs)} train pairs"
        )

    held = train_pairs[held_out_idx]
    remaining = [p for i, p in enumerate(train_pairs) if i != held_out_idx]
    if candidate_pair is not None:
        remaining = remaining + [candidate_pair]
    if not remaining:
        remaining = [held]

    return {
        "train": remaining,
        "test": [{"input": held["input"], "output": held["output"]}],
    }


def consistency_score_from_probe_results(
    probe_results: Sequence[ProbeResult],
) -> Tuple[float, float, int, int]:
    """Aggregate probe results into (exact_rate, mean_cell_acc, n_exact, n_valid)."""
    n = len(probe_results)
    if n == 0:
        return 0.0, 0.0, 0, 0
    n_exact = sum(1 for p in probe_results if p.exact_match)
    n_valid = sum(1 for p in probe_results if p.valid_parse)
    mean_cell = sum(p.cell_accuracy for p in probe_results) / n
    return n_exact / n, mean_cell, n_exact, n_valid


# ============================================================================
# 5. Prompting & generation (GPU path is lazy-imported; tests inject generate_fn)
# ============================================================================

def format_inference_prompt(
    task: Dict[str, Any],
    tokenizer: Any = None,
    test_idx: int = 0,
) -> str:
    """Format a task as an inference prompt (no test output revealed)."""
    messages = task_to_chat_messages(
        task,
        test_idx=test_idx,
        include_test_output=False,
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_msg}\nOutput:\n"


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    gen_config: GenerationConfig,
    device: str = "cpu",
    max_new_tokens: int = 512,
    generate_fn: Optional[GenerateFn] = None,
) -> str:
    """Generate a completion string.  `generate_fn` bypasses torch (unit tests)."""
    cfg = gen_config.to_dict() if isinstance(gen_config, GenerationConfig) else dict(gen_config)
    if generate_fn is not None:
        return generate_fn(prompt, cfg)

    import torch

    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        random.seed(int(seed))

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)

    do_sample = bool(cfg.get("do_sample", False))
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "do_sample": do_sample,
    }
    if do_sample:
        temperature = float(cfg.get("temperature", 0.7))
        gen_kwargs["temperature"] = max(temperature, 1e-6)
        gen_kwargs["top_p"] = float(cfg.get("top_p", 0.9))

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ============================================================================
# 6. Consistency scoring of one candidate
# ============================================================================

def score_candidate_consistency(
    task: Dict[str, Any],
    candidate_grid: Optional[List[List[int]]],
    model: Any,
    tokenizer: Any,
    gen_config: GenerationConfig,
    device: str = "cpu",
    max_new_tokens: int = 512,
    condition_on_candidate: bool = True,
    generate_fn: Optional[GenerateFn] = None,
    test_idx: int = 0,
) -> Tuple[List[ProbeResult], float, float, int, int]:
    """Probe every training pair as a fake test input; return aggregated scores.

    When `condition_on_candidate` is True and `candidate_grid` is a valid grid,
    the hypothesized (test_input, candidate) pair is appended as an extra
    demonstration.  That is what makes the score *candidate-specific*: a
    contradictory hypothesis can break reconstruction of known pairs.
    """
    train_pairs = list(task.get("train", []))
    test_pairs = list(task.get("test", []))

    candidate_pair: Optional[Dict[str, Any]] = None
    if (
        condition_on_candidate
        and candidate_grid is not None
        and is_valid_grid(candidate_grid)
        and test_pairs
        and 0 <= test_idx < len(test_pairs)
        and "input" in test_pairs[test_idx]
    ):
        candidate_pair = {
            "input": test_pairs[test_idx]["input"],
            "output": candidate_grid,
        }

    probe_results: List[ProbeResult] = []
    for held_out_idx, pair in enumerate(train_pairs):
        probe_task = build_probe_task(
            task,
            held_out_idx=held_out_idx,
            candidate_pair=candidate_pair,
        )
        prompt = format_inference_prompt(probe_task, tokenizer=tokenizer, test_idx=0)
        raw = generate_completion(
            model,
            tokenizer,
            prompt,
            gen_config,
            device=device,
            max_new_tokens=max_new_tokens,
            generate_fn=generate_fn,
        )
        pred = text_to_grid(raw)
        gt = pair["output"]
        valid = pred is not None and is_valid_grid(pred)
        exact = grids_equal(pred, gt)
        probe_results.append(
            ProbeResult(
                pair_index=held_out_idx,
                exact_match=bool(exact),
                cell_accuracy=grid_cell_accuracy(pred, gt),
                valid_parse=bool(valid),
                predicted_grid=pred,
                raw_text=raw,
            )
        )

    exact_rate, mean_cell, n_exact, n_valid = consistency_score_from_probe_results(probe_results)
    return probe_results, exact_rate, mean_cell, n_exact, n_valid


def generate_and_score_candidate(
    task: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    gen_config: GenerationConfig,
    test_idx: int = 0,
    device: str = "cpu",
    max_new_tokens: int = 512,
    condition_on_candidate: bool = True,
    generate_fn: Optional[GenerateFn] = None,
) -> Candidate:
    """Generate one test-output candidate and score it on the training pairs."""
    prompt = format_inference_prompt(task, tokenizer=tokenizer, test_idx=test_idx)
    raw = generate_completion(
        model,
        tokenizer,
        prompt,
        gen_config,
        device=device,
        max_new_tokens=max_new_tokens,
        generate_fn=generate_fn,
    )
    grid = text_to_grid(raw)
    valid = grid is not None and is_valid_grid(grid)

    probes, exact_rate, mean_cell, n_exact, n_valid = score_candidate_consistency(
        task=task,
        candidate_grid=grid if valid else None,
        model=model,
        tokenizer=tokenizer,
        gen_config=gen_config,
        device=device,
        max_new_tokens=max_new_tokens,
        condition_on_candidate=condition_on_candidate,
        generate_fn=generate_fn,
        test_idx=test_idx,
    )

    return Candidate(
        raw_text=raw,
        grid=grid if valid else None,
        gen_config=gen_config,
        consistency_score=exact_rate,
        mean_cell_accuracy=mean_cell,
        n_train_exact=n_exact,
        n_train_total=len(probes),
        n_valid_probes=n_valid,
        is_valid_grid=bool(valid),
        probe_results=probes,
    )


# ============================================================================
# 7. Ranking & selection (pure; this is what the unit tests pin down)
# ============================================================================

def candidate_rank_tuple(candidate: Candidate) -> Tuple[Any, ...]:
    """Higher is better.  Used with `sorted(..., reverse=True)`."""
    greedy_bonus = 0 if candidate.gen_config.do_sample else 1
    # Lower temperature is more deterministic; negate so reverse-sort prefers it.
    neg_temperature = -float(candidate.gen_config.temperature)
    return (
        1 if candidate.is_valid_grid else 0,
        float(candidate.consistency_score),
        float(candidate.mean_cell_accuracy),
        int(candidate.n_train_exact),
        int(candidate.n_valid_probes),
        greedy_bonus,
        neg_temperature,
    )


def rank_and_select(
    candidates: Sequence[Candidate],
    n_submit: int = 2,
) -> List[Candidate]:
    """Pick up to `n_submit` candidates for ARC pass@2.

    Ranking (high to low):
      1. Parseable valid grid
      2. Train-pair exact-match consistency
      3. Mean cell accuracy on train probes
      4. Number of exact / valid probes
      5. Greedy decoding, then lower temperature

    Duplicate parsed grids are skipped so the two attempts are distinct
    whenever a distinct runner-up exists.  If everything collapses to one
    unique grid (or there are fewer than `n_submit` candidates), the list
    is padded from the remaining ranked candidates, allowing duplicates,
    so the caller can always submit two strings.
    """
    if n_submit <= 0 or not candidates:
        return []

    ranked = sorted(candidates, key=candidate_rank_tuple, reverse=True)

    selected: List[Candidate] = []
    seen_keys = set()
    for cand in ranked:
        fp = grid_fingerprint(cand.grid)
        key: Any = fp if fp is not None else ("__raw__", cand.raw_text)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(cand)
        if len(selected) >= n_submit:
            break

    if len(selected) < n_submit:
        for cand in ranked:
            if cand in selected:
                continue
            selected.append(cand)
            if len(selected) >= n_submit:
                break

    return selected[:n_submit]


def _has_enough_perfect(candidates: Sequence[Candidate], n_submit: int) -> bool:
    """True once we already have `n_submit` unique valid grids at consistency 1.0."""
    seen = set()
    n_perfect = 0
    for cand in candidates:
        if not cand.is_valid_grid or cand.consistency_score < 1.0:
            continue
        fp = grid_fingerprint(cand.grid)
        if fp is None or fp in seen:
            continue
        seen.add(fp)
        n_perfect += 1
        if n_perfect >= n_submit:
            return True
    return False


# ============================================================================
# 8. End-to-end selection (and TTT-wrapped variant)
# ============================================================================

def select_verified_attempts(
    task: Dict[str, Any],
    model: Any = None,
    tokenizer: Any = None,
    test_idx: int = 0,
    n_candidates: int = 6,
    n_submit: int = 2,
    candidate_schedule: Optional[Sequence[GenerationConfig]] = None,
    device: str = "cpu",
    max_new_tokens: int = 512,
    condition_on_candidate: bool = True,
    early_stop_perfect: bool = True,
    generate_fn: Optional[GenerateFn] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Generate many candidates, rank by consistency, return the top 2 strings.

    Returns:
        (attempt_1_text, attempt_2_text, info_dict)
    """
    schedule = (
        list(candidate_schedule)
        if candidate_schedule is not None
        else build_candidate_schedule(n_candidates)
    )
    if candidate_schedule is None:
        schedule = schedule[:n_candidates]

    candidates: List[Candidate] = []
    for cfg in schedule:
        cand = generate_and_score_candidate(
            task=task,
            model=model,
            tokenizer=tokenizer,
            gen_config=cfg,
            test_idx=test_idx,
            device=device,
            max_new_tokens=max_new_tokens,
            condition_on_candidate=condition_on_candidate,
            generate_fn=generate_fn,
        )
        candidates.append(cand)
        logger.info(
            "Candidate seed=%s temp=%.2f sample=%s valid=%s consistency=%.2f "
            "(%d/%d train exact)",
            cfg.seed,
            cfg.temperature,
            cfg.do_sample,
            cand.is_valid_grid,
            cand.consistency_score,
            cand.n_train_exact,
            cand.n_train_total,
        )
        if early_stop_perfect and _has_enough_perfect(candidates, n_submit):
            logger.info(
                "Early-stop: %d unique perfect-consistency candidates.", n_submit
            )
            break

    selected = rank_and_select(candidates, n_submit=n_submit)
    att_1 = selected[0].raw_text if selected else ""
    if len(selected) >= 2:
        att_2 = selected[1].raw_text
    elif selected:
        att_2 = selected[0].raw_text
    else:
        att_2 = ""

    info: Dict[str, Any] = {
        "n_generated": len(candidates),
        "n_selected": len(selected),
        "n_submit": n_submit,
        "condition_on_candidate": condition_on_candidate,
        "early_stopped": early_stop_perfect and _has_enough_perfect(candidates, n_submit),
        "selected": [c.summary_dict() for c in selected],
        "all_scores": [c.consistency_score for c in candidates],
        "all_valid": [c.is_valid_grid for c in candidates],
        "num_valid_candidates": sum(1 for c in candidates if c.is_valid_grid),
        "training_consistency": {
            "consistency_score": selected[0].consistency_score if selected else 0.0,
            "attempt_scores": [c.consistency_score for c in selected],
            "n_train_exact": selected[0].n_train_exact if selected else 0,
            "n_train_total": selected[0].n_train_total if selected else 0,
        },
    }
    return att_1, att_2, info


def predict_task_with_verified_selection(
    model: Any,
    tokenizer: Any,
    task: Dict[str, Any],
    base_state: Optional[Dict[str, Any]] = None,
    test_idx: int = 0,
    use_ttt: bool = True,
    ttt_steps: int = 30,
    learning_rate: float = 5e-4,
    device: str = "cuda",
    max_new_tokens: int = 512,
    n_candidates: int = 6,
    n_submit: int = 2,
    candidate_schedule: Optional[Sequence[GenerationConfig]] = None,
    condition_on_candidate: bool = True,
    early_stop_perfect: bool = True,
    generate_fn: Optional[GenerateFn] = None,
    ttt_seed: int = 42,
    num_candidates: Optional[int] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Optionally TTT-adapt, run verified candidate selection, always restore weights.

    Weight restoration is in a `finally` block so a later task can never
    inherit this task's adapter drift (same contract as `predict_task_with_ttt`).
    """
    from src.test_time_train import adapt_model_to_task, restore_trainable_state

    if num_candidates is not None:
        n_candidates = num_candidates

    try:
        if use_ttt and ttt_steps > 0 and generate_fn is None:
            adapt_model_to_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                ttt_steps=ttt_steps,
                learning_rate=learning_rate,
                device=device,
                seed=ttt_seed,
            )
        if model is not None and hasattr(model, "eval"):
            model.eval()

        return select_verified_attempts(
            task=task,
            model=model,
            tokenizer=tokenizer,
            test_idx=test_idx,
            n_candidates=n_candidates,
            n_submit=n_submit,
            candidate_schedule=candidate_schedule,
            device=device,
            max_new_tokens=max_new_tokens,
            condition_on_candidate=condition_on_candidate,
            early_stop_perfect=early_stop_perfect,
            generate_fn=generate_fn,
        )
    finally:
        if model is not None and base_state is not None:
            restore_trainable_state(model, base_state, device=device)
            if hasattr(model, "eval"):
                model.eval()


# Alias used by evaluate.py
predict_task_with_verification = predict_task_with_verified_selection
