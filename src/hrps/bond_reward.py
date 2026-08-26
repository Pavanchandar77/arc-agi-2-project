"""Verifier-closed reward for Bond-4B.

Reward uses joint demonstration exactness and gold-free underconstraint
flags. Test labels are never read. Transfer on synthetic tasks is known
because the generating program is retained separately from the reward.

Kind: exact.
"""

from __future__ import annotations

from typing import Any

from src.hrps.dsl import Program, replay
from src.hrps.env import gold_free_constraint_feedback
from src.hrps.residual import joint_residual
from src.hrps.task import ArcTask


def verifier_reward(task: ArcTask, program: Program) -> dict[str, Any]:
    preds = tuple(replay(program, p.input) for p in task.train)
    gts = task.train_outputs()
    residual = joint_residual(preds, gts, spec=None)
    constraint = gold_free_constraint_feedback(task, program)
    flags = list(constraint.get("underconstraint_flags") or [])
    joint = bool(residual.all_exact)
    if not joint:
        reward = 0.0
    elif flags:
        reward = 0.15
    else:
        reward = 1.0
        # Shorter jointly exact programs are preferred, never using test gold.
        reward += max(0.0, 0.1 - 0.02 * program.depth())
    return {
        "reward": round(reward, 6),
        "joint_demo_exact": joint,
        "underconstraint_flags": flags,
        "pixel_residual": residual.pixel_total,
        "program": program.serialize(),
        "uses_test_labels": False,
        "criterion": (
            "Reward is 1 for jointly exact unconstrained programs, 0.15 if jointly "
            "exact but underconstrained, else 0. Test gold is not an input."
        ),
    }


def rank_candidates(task: ArcTask, programs: list[Program]) -> list[dict[str, Any]]:
    scored = [verifier_reward(task, p) for p in programs]
    scored.sort(key=lambda r: (-r["reward"], r["pixel_residual"], len(r["program"])))
    return scored
