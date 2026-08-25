"""Matched M0–M3 loops. The model decides; HRPS executes and feeds back.

M0  Direct answer from raw serialized grids.
M1  One-shot formal DSL proposal; env executes and verifies.
M2  Active loop: observe, inspect, apply, residual, revise, commit.
M3  Active loop plus training-only exact abstractions (H).

Executor and verifier are identical in M1–M3. Test labels are hidden
until the elevation harness scores the two committed attempts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary, set_active_library
from src.hrps.env import Action, HrpsEnv, parse_answer_grids, parse_model_actions
from src.hrps.model import FrozenOpenModel, ModelTurn
from src.hrps.task import ArcTask


@dataclass(frozen=True)
class ElevationBudget:
    temperature: float = 0.0
    max_tokens: int = 512
    max_calls: int = 8
    max_seconds: float = 30.0
    max_program_depth: int = 3


SYSTEM_M0 = (
    "You solve ARC-AGI puzzles. Study the demonstrations and emit the output grid "
    "for each test input. Output ONLY the grid as space-separated integers, one row per line. "
    "No explanations."
)

SYSTEM_M1 = (
    "You solve ARC-AGI puzzles by writing an HRPS DSL program. "
    "The program is executed exactly on every demonstration and must match every demo output. "
    "Respond with a single program using operators composed with ' | '. Example: rot180 "
    "or crop_fg:0 | left_half. Depth at most 3. Do not emit grids."
)

SYSTEM_M2 = (
    "You are an active hypothesis-testing agent inside HRPS. "
    "You observe demonstrations, form a transformation hypothesis, inspect structure, "
    "apply exact DSL actions, read residuals and underconstraint flags, and revise. "
    "The verifier is exact feedback, not the solution. Joint demo exactness does not "
    "guarantee test transfer — if flags mention disjoint palettes, unseen test colors, "
    "or marker/blob role collision, the hypothesis is underconstrained; revise it. "
    "Emit one action per turn:\n"
    "  HYPOTHESIZE <text>\n"
    "  INSPECT colors|shapes|objects|catalog|underconstraint\n"
    "  APPLY <program>\n"
    "  COMMIT <program>\n"
    "When you have a coherent rule that is jointly exact and not flagged as underconstrained, COMMIT it."
)


def _rss_bytes() -> Optional[int]:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


@dataclass
class EpisodeResult:
    condition: str
    task_id: str
    split: str
    solved: bool
    joint_demo_exact: bool
    test_exact: list[bool]
    pass2: bool
    programs: list[str]
    attempts: list[list[list[list[int]]]]
    valid_formal_action_rate: float
    n_distinct_hypotheses: int
    n_hypothesis_revisions: int
    n_representation_requests: int
    n_verifier_calls: int
    n_contradiction_resolutions: int
    residual_trace: list[int]
    accepted_actions: int
    rejected_actions: int
    time_to_first_joint_exact: Optional[float]
    n_final_candidate_diversity: int
    peak_memory: Optional[int]
    n_model_calls: int
    n_prompt_tokens: int
    n_completion_tokens: int
    wall_clock: float
    model_name: str
    backend: str
    failure: str
    notes: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "task_id": self.task_id,
            "split": self.split,
            "solved": self.solved,
            "joint_demo_exact": self.joint_demo_exact,
            "test_exact": self.test_exact,
            "pass2": self.pass2,
            "programs": self.programs,
            "valid_formal_action_rate": round(self.valid_formal_action_rate, 6),
            "n_distinct_hypotheses": self.n_distinct_hypotheses,
            "n_hypothesis_revisions": self.n_hypothesis_revisions,
            "n_representation_requests": self.n_representation_requests,
            "n_verifier_calls": self.n_verifier_calls,
            "n_contradiction_resolutions": self.n_contradiction_resolutions,
            "residual_trace": self.residual_trace,
            "accepted_actions": self.accepted_actions,
            "rejected_actions": self.rejected_actions,
            "time_to_first_joint_exact": self.time_to_first_joint_exact,
            "n_final_candidate_diversity": self.n_final_candidate_diversity,
            "peak_memory": self.peak_memory,
            "n_model_calls": self.n_model_calls,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_completion_tokens": self.n_completion_tokens,
            "wall_clock": round(self.wall_clock, 6),
            "model_name": self.model_name,
            "backend": self.backend,
            "failure": self.failure,
            "notes": self.notes,
            "telemetry": self.telemetry,
        }


def score_attempts(task: ArcTask, attempts: list[list[list[list[int]]]]) -> tuple[list[bool], bool]:
    """Exact test-transfer, pass@2. Uses gold only at scoring time."""
    test_exact: list[bool] = []
    gts = task.test_outputs()
    n_test = task.n_test
    for i in range(n_test):
        gt = gts[i]
        ok = False
        if gt is not None:
            for att in attempts:
                if i >= len(att):
                    continue
                pred = att[i]
                if pred is not None and tuple(tuple(r) for r in pred) == gt:
                    ok = True
                    break
        test_exact.append(ok)
    return test_exact, all(test_exact) if test_exact else False


def _t0_joint(env: HrpsEnv, started: float) -> Optional[float]:
    if any(c.joint_exact for c in env.candidates):
        return round(time.perf_counter() - started, 6)
    return None


def run_episode(
    task: ArcTask,
    model: FrozenOpenModel,
    condition: str,
    budget: ElevationBudget,
    library: Optional[AbstractionLibrary] = None,
) -> EpisodeResult:
    condition = condition.upper()
    if condition not in {"M0", "M1", "M2", "M3"}:
        raise ValueError(f"unknown condition {condition}")
    enable_h = condition == "M3"
    lib = library or AbstractionLibrary()
    if enable_h:
        set_active_library(lib)
    else:
        set_active_library(AbstractionLibrary())
    env = HrpsEnv(task, library=lib, enable_h=enable_h, max_depth=budget.max_program_depth)
    started = time.perf_counter()
    deadline = started + budget.max_seconds
    n_calls = 0
    n_prompt = 0
    n_comp = 0
    peak = _rss_bytes()
    t_joint: Optional[float] = None
    notes: list[str] = []

    observe = env.observe()
    catalog = env.catalog_text() if condition != "M0" else ""

    def _call(prompt: str) -> ModelTurn:
        nonlocal n_calls, n_prompt, n_comp, peak
        n_calls += 1
        turn = model.generate(
            prompt, max_tokens=budget.max_tokens, temperature=budget.temperature
        )
        n_prompt += turn.n_prompt_tokens
        n_comp += turn.n_completion_tokens
        rss = _rss_bytes()
        if rss is not None:
            peak = max(peak or 0, rss)
        return turn

    def remaining() -> bool:
        return n_calls < budget.max_calls and time.perf_counter() < deadline

    if condition == "M0":
        prompt = _prompt_m0(observe.text)
        while remaining() and len(env.answer_attempts) < 2:
            turn = _call(prompt)
            actions = parse_model_actions(turn.text)
            answer_acts = [a for a in actions if a.kind == "answer"]
            if answer_acts:
                for a in answer_acts:
                    env.step(a)
            else:
                env.step(Action(kind="answer", payload=turn.text, raw=turn.text))
            if not parse_answer_grids(turn.text, n_test=task.n_test):
                notes.append("m0_parse_failed")
        if not env.answer_attempts:
            notes.append("m0_no_grid")
    elif condition == "M1":
        prompt = _prompt_m1(observe.text, catalog)
        # Two proposals max, matched call cap.
        while remaining() and len(env.candidates) < 2:
            turn = _call(prompt)
            actions = parse_model_actions(turn.text)
            if not actions:
                env.step(Action(kind="apply", payload=turn.text.strip().splitlines()[0] if turn.text.strip() else "", raw=turn.text))
            else:
                for a in actions:
                    if a.kind in {"apply", "commit"}:
                        env.step(Action("apply", a.payload or a.raw, a.raw))
                    elif a.kind == "hypothesize":
                        env.step(a)
            if t_joint is None:
                t_joint = _t0_joint(env, started)
        # Auto-commit best jointly exact (or best residual) — M1 is one-shot, not a loop.
        for prog in env.finalize_programs():
            if all(c.program.serialize() != prog.serialize() or c.source != "commit" for c in env.candidates):
                env.step(Action("commit", prog.serialize(), prog.serialize()))
    else:
        # M2 / M3 active loop
        transcript: list[str] = [observe.text, "CATALOG:\n" + catalog]
        sys_prefix = SYSTEM_M2
        if enable_h:
            sys_prefix += "\nYou may use abs:<name> macros from the catalog. They are exact compositions, not new primitives."
        while remaining():
            prompt = sys_prefix + "\n\n" + "\n\n".join(transcript[-8:]) + "\n\nYour action:"
            turn = _call(prompt)
            actions = parse_model_actions(turn.text)
            if not actions:
                # Try as apply, else record reject via unknown.
                line = turn.text.strip().splitlines()[0] if turn.text.strip() else ""
                actions = [Action("apply", line, turn.text)]
            fb_texts = []
            committed_now = False
            for a in actions:
                fb = env.step(a)
                fb_texts.append(fb.text)
                if a.kind == "commit" and fb.accepted:
                    committed_now = True
            transcript.append("MODEL:\n" + turn.text)
            transcript.append("HRPS:\n" + "\n".join(fb_texts))
            if t_joint is None:
                t_joint = _t0_joint(env, started)
            if committed_now and len(env.committed) >= 2:
                break
            if committed_now and len(env.committed) >= 1 and any(c.joint_exact for c in env.candidates):
                # One jointly exact commit is enough to stop; second slot is duplicated.
                break

    attempts = env.finalize_attempts()
    programs = [p.serialize() for p in env.finalize_programs()] if condition != "M0" else []
    test_exact, pass2 = score_attempts(task, attempts)
    joint = any(c.joint_exact for c in env.candidates)
    # Competition-aligned solved: exact test transfer. For M1–M3 also require
    # a jointly verified program so lucky unverified grids are not counted as HRPS solves.
    if condition == "M0":
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

    wall = time.perf_counter() - started
    diversity = len({p for p in programs})
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
        n_final_candidate_diversity=diversity,
        peak_memory=peak,
        n_model_calls=n_calls,
        n_prompt_tokens=n_prompt,
        n_completion_tokens=n_comp,
        wall_clock=wall,
        model_name=getattr(model, "name", ""),
        backend=getattr(model, "backend", ""),
        failure=failure,
        notes=notes,
        telemetry={
            "n_commit": env.n_commit,
            "n_hypothesize": env.n_hypothesize,
            "n_candidates": len(env.candidates),
            "enable_h": enable_h,
            "budget": {
                "temperature": budget.temperature,
                "max_tokens": budget.max_tokens,
                "max_calls": budget.max_calls,
                "max_seconds": budget.max_seconds,
            },
        },
    )


def _prompt_m0(observe_text: str) -> str:
    return (
        SYSTEM_M0
        + "\n\n"
        + observe_text
        + "\n\nEmit the output grid for TEST 0 (and TEST 1 if present)."
    )


def _prompt_m1(observe_text: str, catalog: str) -> str:
    return (
        SYSTEM_M1
        + "\n\n"
        + catalog
        + "\n\n"
        + observe_text
        + "\n\nPropose one DSL program."
    )
