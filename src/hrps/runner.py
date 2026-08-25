"""Active Bond/HRPS episode runner.

The model decides; HRPS executes and returns exact consequences. This is not
a one-shot proposer. Termination is always one of: committed, max_calls,
max_tokens, timeout, parse_failure_budget, direct_done.

Kind: exact budgets and logging.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary, set_active_library
from src.hrps.agent import SYSTEM_M0, EpisodeResult, score_attempts
from src.hrps.env import Action, HrpsEnv
from src.hrps.identity import SYSTEM_BOND_DIRECT, is_bond_identity
from src.hrps.model import FrozenOpenModel
from src.hrps.schema import SYSTEM_BOND_JSON, compact_observation, parse_strict_action
from src.hrps.task import ArcTask


@dataclass(frozen=True)
class RunnerBudget:
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_tokens_per_call: int = 256
    max_model_calls: int = 8
    max_total_tokens: int = 4096
    max_seconds: float = 30.0
    max_program_depth: int = 3
    max_consecutive_invalid: int = 3


def _rss_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


@dataclass
class Interaction:
    step: int
    system: str
    prompt_chars: int
    model_output: str
    parsed_action: Optional[dict[str, Any]]
    observation: dict[str, Any]
    accepted: bool
    n_prompt_tokens: int
    n_completion_tokens: int
    elapsed: float
    error: str = ""
    hypothesis: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "system": self.system,
            "prompt_chars": self.prompt_chars,
            "model_output": self.model_output,
            "parsed_action": self.parsed_action,
            "hypothesis": self.hypothesis,
            "observation": self.observation,
            "accepted": self.accepted,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_completion_tokens": self.n_completion_tokens,
            "elapsed": round(self.elapsed, 6),
            "error": self.error,
        }


@dataclass
class RunnerResult:
    system: str
    task_id: str
    termination: str
    episode: EpisodeResult
    interactions: list[Interaction] = field(default_factory=list)
    n_invalid_actions: int = 0
    n_hypothesis_rejections: int = 0
    saw_underconstraint_flags: bool = False
    committed_underconstrained: bool = False
    hrps_symbolic_seconds: float = 0.0
    n_verified_candidates: int = 0
    identity: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        base = self.episode.as_dict()
        base.update(
            {
                "system": self.system,
                "termination": self.termination,
                "n_invalid_actions": self.n_invalid_actions,
                "n_hypothesis_rejections": self.n_hypothesis_rejections,
                "saw_underconstraint_flags": self.saw_underconstraint_flags,
                "committed_underconstrained": self.committed_underconstrained,
                "hrps_symbolic_seconds": round(self.hrps_symbolic_seconds, 6),
                "n_verified_candidates": self.n_verified_candidates,
                "n_interactions": len(self.interactions),
                "n_reasoning_cycles": max(0, len(self.interactions)),
                "identity": self.identity,
                "seed": self.seed,
                "interactions": [i.as_dict() for i in self.interactions],
            }
        )
        return base


def _model_identity(model: FrozenOpenModel, seed: int) -> dict[str, Any]:
    if hasattr(model, "provenance"):
        rec = model.provenance()  # type: ignore[attr-defined]
        rec["seed"] = seed
        return rec
    return {
        "public_name": getattr(model, "name", ""),
        "is_bond": is_bond_identity(model),
        "backend": getattr(model, "backend", ""),
        "seed": seed,
    }


def _hypothesis_text(parsed: Optional[dict[str, Any]]) -> str:
    if not parsed:
        return ""
    args = parsed.get("arguments") or {}
    act = parsed.get("action") or ""
    if act in {"revise_hypothesis", "propose_program", "reject_hypothesis"}:
        return str(args.get("text") or args.get("reason") or args.get("note") or args.get("program") or "")
    return ""


def run_direct(
    task: ArcTask,
    model: FrozenOpenModel,
    budget: RunnerBudget,
    *,
    system: str,
) -> RunnerResult:
    env = HrpsEnv(task)
    started = time.perf_counter()
    observe = env.observe()
    interactions: list[Interaction] = []
    n_prompt = n_comp = 0
    tokens_used = 0
    termination = "direct_done"
    t_sym = 0.0
    for step in range(2):
        if time.perf_counter() - started >= budget.max_seconds:
            termination = "timeout"
            break
        if tokens_used >= budget.max_total_tokens:
            termination = "max_tokens"
            break
        sys_prompt = SYSTEM_BOND_DIRECT if is_bond_identity(model) else SYSTEM_M0
        prompt = sys_prompt + "\n\n" + observe.text + "\n\nEmit the output grid for TEST 0."
        turn = model.generate(
            prompt,
            max_tokens=budget.max_tokens_per_call,
            temperature=budget.temperature,
            top_p=budget.top_p,
            seed=budget.seed + step,
        )
        n_prompt += turn.n_prompt_tokens
        n_comp += turn.n_completion_tokens
        tokens_used += turn.n_prompt_tokens + turn.n_completion_tokens
        t0 = time.perf_counter()
        fb = env.step(Action("answer", turn.text, turn.text))
        t_sym += time.perf_counter() - t0
        interactions.append(
            Interaction(
                step=step,
                system=system,
                prompt_chars=len(prompt),
                model_output=turn.text,
                parsed_action={"action": "answer"},
                observation=compact_observation(fb),
                accepted=fb.accepted,
                n_prompt_tokens=turn.n_prompt_tokens,
                n_completion_tokens=turn.n_completion_tokens,
                elapsed=time.perf_counter() - started,
                error="" if fb.accepted else str(fb.data.get("error")),
            )
        )
        if len(env.answer_attempts) >= 2:
            termination = "direct_done"
            break
    ep = _finalize(task, env, model, system, started, n_prompt, n_comp, len(interactions), "M0")
    return RunnerResult(
        system=system,
        task_id=task.task_id,
        termination=termination,
        episode=ep,
        interactions=interactions,
        hrps_symbolic_seconds=t_sym,
        n_verified_candidates=sum(1 for c in env.candidates if c.joint_exact),
        identity=_model_identity(model, budget.seed),
        seed=budget.seed,
    )


def run_hrps(
    task: ArcTask,
    model: FrozenOpenModel,
    budget: RunnerBudget,
    *,
    system: str,
    library: Optional[AbstractionLibrary] = None,
    enable_h: bool = False,
) -> RunnerResult:
    from src.hrps.bond_overseer import run_overseer

    return run_overseer(task, model, budget, system=system, library=library, enable_h=enable_h)


def run_system(
    task: ArcTask,
    model: FrozenOpenModel,
    system: str,
    budget: RunnerBudget,
    library: Optional[AbstractionLibrary] = None,
) -> RunnerResult:
    if system not in {"base_direct", "base_hrps", "bond_direct", "bond_hrps"}:
        raise ValueError(f"unknown system {system}")
    if system.endswith("_direct"):
        return run_direct(task, model, budget, system=system)
    return run_hrps(task, model, budget, system=system, library=library)


def _finalize(
    task: ArcTask,
    env: HrpsEnv,
    model: FrozenOpenModel,
    system: str,
    started: float,
    n_prompt: int,
    n_comp: int,
    n_calls: int,
    condition: str,
) -> EpisodeResult:
    attempts = env.finalize_attempts()
    programs = [p.serialize() for p in env.finalize_programs()] if not system.endswith("_direct") else []
    test_exact, pass2 = score_attempts(task, attempts)
    joint = any(c.joint_exact for c in env.candidates)
    if system.endswith("_direct"):
        solved = pass2
        failure = "solved" if solved else "direct_mismatch"
    else:
        solved = bool(pass2 and joint)
        if solved:
            failure = "solved"
        elif joint and not pass2:
            failure = "joint_demo_failed_to_transfer"
        elif not joint:
            failure = "no_joint_program"
        else:
            failure = "unsolved"
    t_joint = None
    if joint:
        t_joint = round(time.perf_counter() - started, 6)
    return EpisodeResult(
        condition=condition,
        task_id=task.task_id,
        split=task.split,
        solved=solved,
        joint_demo_exact=joint,
        test_exact=test_exact,
        pass2=pass2,
        programs=programs,
        attempts=attempts,
        valid_formal_action_rate=env.valid_action_rate(),
        n_distinct_hypotheses=env.distinct_hypotheses(),
        n_hypothesis_revisions=env.hypothesis_revisions(),
        n_representation_requests=env.n_inspect,
        n_verifier_calls=env.n_apply,
        n_contradiction_resolutions=env.n_contradiction_resolutions,
        residual_trace=list(env.residual_trace),
        accepted_actions=env.n_accepted,
        rejected_actions=env.n_rejected,
        time_to_first_joint_exact=t_joint,
        n_final_candidate_diversity=len(set(programs)),
        peak_memory=_rss_bytes(),
        n_model_calls=n_calls,
        n_prompt_tokens=n_prompt,
        n_completion_tokens=n_comp,
        wall_clock=time.perf_counter() - started,
        model_name=getattr(model, "name", ""),
        backend=getattr(model, "backend", ""),
        failure=failure,
        notes=[],
        telemetry={
            "system": system,
            "n_commit": env.n_commit,
            "n_hypothesize": env.n_hypothesize,
            "n_rejections": env.n_rejections,
        },
    )
