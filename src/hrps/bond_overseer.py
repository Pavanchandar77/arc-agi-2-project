"""Bond overseer: the model-controlled HRPS reasoning cycle.

  Bond sees task + memory
  Bond writes a hypothesis and selects a typed tool
  HRPS executes exactly
  Bond interprets residuals (the model, not HRPS, supplies meaning)
  Bond revises, rejects, or commits

This is not one-shot program proposal.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from src.hrps.abstractions import AbstractionLibrary, set_active_library
from src.hrps.arc_adapter import ArcHRPSEnvironment
from src.hrps.bond_memory import BondMemory
from src.hrps.bond_tools import dispatch_tool
from src.hrps.core import HRPSEnvironment, HRPSState
from src.hrps.model import FrozenOpenModel
from src.hrps.runner import (
    Interaction,
    RunnerBudget,
    RunnerResult,
    _finalize,
    _hypothesis_text,
    _model_identity,
)
from src.hrps.schema import SYSTEM_BOND_JSON, parse_strict_action
from src.hrps.task import ArcTask


def run_overseer(
    task: ArcTask,
    model: FrozenOpenModel,
    budget: RunnerBudget,
    *,
    system: str,
    library: Optional[AbstractionLibrary] = None,
    enable_h: bool = False,
) -> RunnerResult:
    lib = library or AbstractionLibrary()
    if enable_h:
        set_active_library(lib)
    else:
        set_active_library(AbstractionLibrary())
    arc_env = ArcHRPSEnvironment(
        task, library=lib, enable_h=enable_h, max_depth=budget.max_program_depth
    )
    environment: HRPSEnvironment = arc_env
    env = arc_env.inner
    memory = BondMemory(task_id=task.task_id)
    started = time.perf_counter()
    deadline = started + budget.max_seconds
    raw = environment.reset()
    catalog = environment.catalog()
    interactions: list[Interaction] = []
    n_prompt = n_comp = tokens_used = 0
    n_invalid = consecutive_invalid = 0
    termination = "max_calls"
    t_sym = 0.0
    saw_flags = False
    committed_under = False

    for step in range(budget.max_model_calls):
        if time.perf_counter() >= deadline:
            termination = "timeout"
            break
        if tokens_used >= budget.max_total_tokens:
            termination = "max_tokens"
            break
        state = HRPSState(
            task_id=task.task_id,
            domain=environment.domain,
            observation_text=raw.text,
            catalog=catalog,
            memory_snapshot=memory.snapshot(),
            step=step,
        )
        prompt = (
            SYSTEM_BOND_JSON
            + "\n\n"
            + state.prompt_block()
            + "\n\n"
            + memory.prompt_block()
            + "\n\nJSON action:"
        )
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
        parsed = parse_strict_action(turn.text)
        if not parsed.ok or parsed.action is None:
            n_invalid += 1
            consecutive_invalid += 1
            obs = {"status": "rejected", "error": parsed.error, "uses_test_labels": False, "earned": False}
            env.total_actions += 1
            env.n_rejected += 1
            memory.record_action(step, None, obs)
            t_sym += time.perf_counter() - t0
            interactions.append(
                Interaction(
                    step=step,
                    system=system,
                    prompt_chars=len(prompt),
                    model_output=turn.text,
                    parsed_action=None,
                    observation=obs,
                    accepted=False,
                    n_prompt_tokens=turn.n_prompt_tokens,
                    n_completion_tokens=turn.n_completion_tokens,
                    elapsed=time.perf_counter() - started,
                    error=parsed.error,
                )
            )
            if consecutive_invalid >= budget.max_consecutive_invalid:
                termination = "parse_failure_budget"
                break
            continue
        consecutive_invalid = 0
        act = {"action": parsed.action.action, "arguments": parsed.action.arguments}
        obs = dispatch_tool(environment, act)
        t_sym += time.perf_counter() - t0
        memory.record_action(step, act, obs)
        flags = obs.get("underconstraint_flags") or []
        if flags:
            saw_flags = True
            memory.interpret("underconstraint flags present; jointly exact may not transfer")
        if act["action"] == "commit_candidates" and obs.get("status") == "ok" and flags:
            committed_under = True
        interactions.append(
            Interaction(
                step=step,
                system=system,
                prompt_chars=len(prompt),
                model_output=turn.text,
                parsed_action=act,
                hypothesis=_hypothesis_text(act),
                observation=obs,
                accepted=obs.get("status") == "ok",
                n_prompt_tokens=turn.n_prompt_tokens,
                n_completion_tokens=turn.n_completion_tokens,
                elapsed=time.perf_counter() - started,
                error="" if obs.get("status") == "ok" else str(obs.get("error") or ""),
            )
        )
        if act["action"] == "commit_candidates" and obs.get("status") == "ok" and len(env.committed) >= 1:
            termination = "committed"
            break
    else:
        termination = "max_calls"

    ep = _finalize(task, env, model, system, started, n_prompt, n_comp, len(interactions), "M2")
    result = RunnerResult(
        system=system,
        task_id=task.task_id,
        termination=termination,
        episode=ep,
        interactions=interactions,
        n_invalid_actions=n_invalid,
        n_hypothesis_rejections=max(env.n_rejections, memory.n_rejections()),
        saw_underconstraint_flags=saw_flags,
        committed_underconstrained=committed_under,
        hrps_symbolic_seconds=t_sym,
        n_verified_candidates=sum(1 for c in env.candidates if c.joint_exact),
        identity=_model_identity(model, budget.seed),
        seed=budget.seed,
    )
    result.episode.telemetry["bond_memory"] = memory.as_dict()
    result.episode.telemetry["overseer"] = True
    return result
