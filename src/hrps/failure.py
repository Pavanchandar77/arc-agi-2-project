"""Automatic failure taxonomy. Kind: heuristic classification of traces."""

from __future__ import annotations

from typing import Optional

FAILURE_CATEGORIES = (
    "solved",
    "representation",
    "DSL_expressiveness",
    "candidate_generation",
    "executor",
    "consistency",
    "signature",
    "quotient",
    "bounds",
    "search_explosion",
    "timeout",
)


def classify_failure(
    *,
    solved: bool,
    timed_out: bool,
    hit_node_limit: bool,
    hit_frontier_limit: bool,
    enumerated_exhausted: bool,
    n_ops_generated: int,
    n_executor_rejects: int,
    n_partial_consistency: int,
    n_exact_demos_best: int,
    n_demos: int,
    representation_instability: float,
    used_object_ops: bool,
) -> str:
    if solved:
        return "solved"
    if timed_out:
        return "timeout"
    if hit_node_limit or hit_frontier_limit:
        return "search_explosion"
    if n_ops_generated == 0:
        return "candidate_generation"
    if n_executor_rejects > 0 and enumerated_exhausted and n_exact_demos_best == 0:
        return "executor"
    if 0 < n_exact_demos_best < n_demos:
        return "consistency"
    if enumerated_exhausted:
        if used_object_ops and representation_instability >= 4.0:
            return "representation"
        return "DSL_expressiveness"
    if used_object_ops and representation_instability >= 8.0:
        return "representation"
    if timed_out:
        return "timeout"
    return "search_explosion"
