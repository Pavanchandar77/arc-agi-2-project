"""Bond-4B runtime: exact search + verifier + optional model loop.

Public identity is Bond. The only foundation for this system is
Qwen/Qwen3.5-4B. Direct answering never sees HRPS. HRPS arms run
Bond-L1 guided search, then optionally the active overseer, then
commit the best jointly exact unconstrained programs (at most two).

Kind: exact commit policy. Search order is heuristic.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Optional

from src.hrps.agent import score_attempts
from src.hrps.bond_overseer import run_overseer
from src.hrps.bond_reward import verifier_reward
from src.hrps.dsl import Program, replay
from src.hrps.env import Action, HrpsEnv, parse_program
from src.hrps.guided_search import guided_search
from src.hrps.identity import PUBLIC_NAME
from src.hrps.language import FOUNDATION_HF_ID, LANGUAGE_DEPTH, LANGUAGE_ID, bond_l1_budget
from src.hrps.model import FrozenOpenModel, ModelTurn
from src.hrps.runner import RunnerBudget, RunnerResult, _finalize, _model_identity
from src.hrps.search import SearchBudget
from src.hrps.task import ArcTask


class _SearchModel:
    name = "bond_l1_search"
    backend = "search"

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float = 1.0, seed: int = 0) -> ModelTurn:
        return ModelTurn(text="", backend="search", model_name=self.name)


def _program_attempts(task: ArcTask, programs: list[Program]) -> list[list[list[list[int]]]]:
    attempts: list[list[list[list[int]]]] = []
    for prog in programs[:2]:
        grids: list[list[list[int]]] = []
        ok = True
        for p in task.test:
            pred = replay(prog, p.input)
            if pred is None:
                ok = False
                break
            grids.append([list(row) for row in pred])
        if ok and grids:
            attempts.append(grids)
    return attempts


def _accept_program(task: ArcTask, ser: str) -> tuple[Optional[Program], dict[str, Any]]:
    prog, err = parse_program(ser, max_depth=LANGUAGE_DEPTH)
    if prog is None or err:
        return None, {}
    return prog, verifier_reward(task, prog)


def run_bond_engine(
    task: ArcTask,
    model: Optional[FrozenOpenModel] = None,
    *,
    system: str = "bond_hrps",
    search_budget: Optional[SearchBudget] = None,
    loop_budget: Optional[RunnerBudget] = None,
    run_overseer_loop: bool = True,
) -> RunnerResult:
    """Solve one task. Search always runs. Overseer runs if a model is given."""
    started = time.perf_counter()
    search_budget = search_budget or bond_l1_budget()
    loop_budget = loop_budget or RunnerBudget(
        max_model_calls=8,
        max_seconds=max(5.0, float(search_budget.max_seconds)),
        max_program_depth=LANGUAGE_DEPTH,
        seed=0,
    )
    guided = guided_search(task, model=model, budget=search_budget)
    env = HrpsEnv(task, max_depth=LANGUAGE_DEPTH)
    env.observe()
    chosen: list[Program] = []
    unconstrained: list[Program] = []
    fallback: list[Program] = []
    for ser in guided.programs:
        prog, rec = _accept_program(task, ser)
        if prog is None:
            continue
        env.step(Action("apply", ser))
        if rec.get("joint_demo_exact") and not rec.get("underconstraint_flags"):
            env.step(Action("commit", ser))
            unconstrained.append(prog)
        elif rec.get("joint_demo_exact"):
            fallback.append(prog)
    chosen = (unconstrained + fallback)[:2]
    overseer_res: Optional[RunnerResult] = None
    if run_overseer_loop and model is not None and len(unconstrained) < 2:
        overseer_res = run_overseer(task, model, loop_budget, system=system, enable_h=False)
        have = {p.serialize() for p in chosen}
        for ser in overseer_res.episode.programs:
            if ser in have:
                continue
            prog, rec = _accept_program(task, ser)
            if prog is None:
                continue
            if rec.get("joint_demo_exact") and not rec.get("underconstraint_flags"):
                chosen.append(prog)
                have.add(ser)
            elif rec.get("joint_demo_exact") and len(chosen) < 2:
                chosen.append(prog)
                have.add(ser)
            if len(chosen) >= 2:
                break
        chosen = chosen[:2]
    attempts = _program_attempts(task, chosen)
    env.answer_attempts = attempts[:2]
    dummy = model or _SearchModel()
    n_calls = guided.n_model_proposals
    n_prompt = n_comp = 0
    if overseer_res is not None:
        n_calls += overseer_res.episode.n_model_calls
        n_prompt = overseer_res.episode.n_prompt_tokens
        n_comp = overseer_res.episode.n_completion_tokens
    ep = _finalize(task, env, dummy, system, started, n_prompt, n_comp, n_calls, "L")
    test_exact, pass2 = score_attempts(task, attempts[:2])
    joint = bool(chosen)
    solved = bool(pass2 and joint)
    ep.solved = solved
    ep.pass2 = pass2
    ep.test_exact = test_exact
    ep.joint_demo_exact = joint
    ep.programs = [p.serialize() for p in chosen]
    ep.attempts = attempts[:2]
    if solved:
        ep.failure = "solved"
    elif joint:
        ep.failure = "joint_demo_failed_to_transfer"
    else:
        ep.failure = "no_joint_program"
    ep.notes = [
        f"language={LANGUAGE_ID}",
        f"foundation={FOUNDATION_HF_ID}",
        f"public_name={PUBLIC_NAME}",
        f"search_solved={guided.search_solved}",
    ]
    ep.telemetry = {
        **(ep.telemetry or {}),
        "guided": guided.as_dict(),
        "n_unconstrained": len(unconstrained),
    }
    return RunnerResult(
        system=system,
        task_id=task.task_id,
        termination="committed" if chosen else "no_joint_program",
        episode=ep,
        interactions=overseer_res.interactions if overseer_res else [],
        n_verified_candidates=len(chosen),
        identity=_model_identity(dummy, loop_budget.seed),
        seed=loop_budget.seed,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Bond-4B engine (Bond-L1 search + verifier)")
    p.add_argument("--task-id", type=str, default=None)
    p.add_argument("--n", type=int, default=0, help="synthesize n tasks and solve (CPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=2.0)
    args = p.parse_args(argv)
    if args.n:
        from src.hrps.synthesize import synthesize_batch

        batch = synthesize_batch(args.n, seed=args.seed)
        n_ok = 0
        for task, program in batch:
            res = run_bond_engine(
                task,
                None,
                search_budget=bond_l1_budget(nodes=400, seconds=args.seconds, frontier=4000),
                run_overseer_loop=False,
            )
            n_ok += int(res.episode.solved)
            print(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "teacher": program.serialize(),
                        "solved": res.episode.solved,
                        "programs": res.episode.programs,
                    }
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "n": len(batch),
                    "solved": n_ok,
                    "language": LANGUAGE_ID,
                    "foundation": FOUNDATION_HF_ID,
                    "public_name": PUBLIC_NAME,
                }
            )
        )
        return 0
    if args.task_id:
        from src.hrps.task import DEFAULT_DATA_ROOT, load_task_file

        path = DEFAULT_DATA_ROOT / "training" / f"{args.task_id}.json"
        task = load_task_file(path, "training")
        res = run_bond_engine(task, None, run_overseer_loop=False)
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "solved": res.episode.solved,
                    "programs": res.episode.programs,
                    "failure": res.episode.failure,
                    "language": LANGUAGE_ID,
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps({"public_name": PUBLIC_NAME, "language": LANGUAGE_ID, "foundation": FOUNDATION_HF_ID}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
