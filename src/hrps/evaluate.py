"""Run the HRPS search microscope on an ARC-AGI-2 split.

Default split is training. Public evaluation is never used as a tuning loop;
pass --split evaluation only for a frozen hold-out measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.hrps.dsl import operator_catalog, stage_config
from src.hrps.kinds import COMPONENT_KIND
from src.hrps.search import SearchBudget, TaskResult, search_task
from src.hrps.task import ArcTask, iter_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def result_to_row(res: TaskResult) -> dict[str, Any]:
    return {
        "task_id": res.task_id,
        "split": res.split,
        "stage": res.stage,
        "solved": res.solved,
        "runtime": res.runtime,
        "failure_category": res.failure_category,
        "programs": res.programs,
        "test_exact": res.test_exact,
        "telemetry": res.telemetry,
        "task_stats": res.task_stats,
        "candidates": res.candidate_summaries,
    }


def attempts_to_submission_entry(res: TaskResult) -> list[dict[str, Any]]:
    """Kaggle pass@2 schema: one object per test input, each with attempt_1/2."""
    n_test = len(res.attempts[0]) if res.attempts else 0
    out = []
    for i in range(n_test):
        a1 = res.attempts[0][i] if len(res.attempts) > 0 else [[0]]
        a2 = res.attempts[1][i] if len(res.attempts) > 1 else a1
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_solved = sum(1 for r in rows if r["solved"])
    runtimes = [r["runtime"] for r in rows]
    expansions = [r["telemetry"]["nodes_expanded"] for r in rows]
    generated = [r["telemetry"]["nodes_generated"] for r in rows]
    dups = [r["telemetry"]["duplicate_states"] for r in rows]
    unique = [r["telemetry"]["unique_states"] for r in rows]
    inst = [r["task_stats"]["representation_instability"] for r in rows]
    fails = Counter(r["failure_category"] for r in rows)
    ttf = [
        r["telemetry"]["time_to_first_exact_demonstration_solution"]
        for r in rows
        if r["telemetry"]["time_to_first_exact_demonstration_solution"] is not None
    ]
    expensive = sorted(rows, key=lambda r: r["telemetry"]["nodes_expanded"], reverse=True)[:15]
    slow = sorted(rows, key=lambda r: r["runtime"], reverse=True)[:10]

    def _mean(xs: list) -> Optional[float]:
        return round(sum(xs) / len(xs), 6) if xs else None

    return {
        "tasks_evaluated": n,
        "solved": n_solved,
        "solve_rate": round(n_solved / n, 6) if n else 0.0,
        "runtime_sec_total": round(sum(runtimes), 4),
        "runtime_sec_mean": _mean(runtimes),
        "runtime_sec_median": round(statistics.median(runtimes), 6) if runtimes else None,
        "nodes_expanded_mean": _mean(expansions),
        "nodes_expanded_median": round(statistics.median(expansions), 6) if expansions else None,
        "nodes_expanded_max": max(expansions) if expansions else 0,
        "nodes_generated_mean": _mean(generated),
        "duplicate_states_mean": _mean(dups),
        "unique_states_mean": _mean(unique),
        "time_to_first_exact_mean": _mean(ttf),
        "n_with_exact_demo_solution": len(ttf),
        "failure_taxonomy": dict(fails),
        "representation_instability_mean": _mean(inst),
        "representation_instability_median": round(statistics.median(inst), 6) if inst else None,
        "most_expensive_by_expansions": [
            {
                "task_id": r["task_id"],
                "expanded": r["telemetry"]["nodes_expanded"],
                "generated": r["telemetry"]["nodes_generated"],
                "runtime": r["runtime"],
                "failure": r["failure_category"],
                "area": r["task_stats"]["max_grid_area"],
                "instability": r["task_stats"]["representation_instability"],
            }
            for r in expensive
        ],
        "slowest_tasks": [
            {"task_id": r["task_id"], "runtime": r["runtime"], "failure": r["failure_category"]}
            for r in slow
        ],
    }


def _next_bottleneck(summary: dict[str, Any]) -> str:
    fails = summary.get("failure_taxonomy", {})
    ranked = sorted(((k, v) for k, v in fails.items() if k != "solved"), key=lambda kv: -kv[1])
    if not ranked:
        return "all_evaluated_tasks_solved"
    top, n = ranked[0]
    mapping = {
        "search_explosion": "search growth under the finite DSL; measure F vs A node counts before adding operators",
        "timeout": "per-task wall clock; raise budget only after confirming residual factorization reduces expansions",
        "DSL_expressiveness": "closed operator set cannot represent the joint map; do not add tricks until this dominates explosion",
        "consistency": "programs fit some demonstrations or fail held-out test inputs; joint residual / multi-test is the bottleneck",
        "representation": "segmentation hypotheses disagree; expand the hypothesis bank before new ops",
        "candidate_generation": "operator generator produced nothing legal",
        "executor": "operators rejected on preconditions or invalid grids",
        "signature": "continuation signatures",
        "quotient": "state merging",
        "bounds": "admissible bound pruning",
    }
    return f"{top} (n={n}): {mapping.get(top, top)}"


def run_split(
    split: str = "training",
    stage: str = "G",
    max_tasks: Optional[int] = None,
    budget: Optional[SearchBudget] = None,
    data_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    budget = budget or SearchBudget()
    out_dir = Path(out_dir) if out_dir is not None else REPO_ROOT / "artifacts" / f"hrps_phase1_{stage}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    submission: dict[str, Any] = {}
    t0 = time.perf_counter()
    tasks = list(iter_split(split, data_root=data_root, max_tasks=max_tasks))
    jsonl_path = out_dir / "tasks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for i, task in enumerate(tasks, 1):
            res = search_task(task, stage=stage, budget=budget)
            row = result_to_row(res)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            submission[res.task_id] = attempts_to_submission_entry(res)
            if i % 20 == 0 or i == len(tasks):
                n_ok = sum(1 for r in rows if r["solved"])
                print(
                    f"[{stage} {split}] {i}/{len(tasks)} solved={n_ok} "
                    f"last={task.task_id}:{res.failure_category} "
                    f"exp={res.telemetry['nodes_expanded']} t={res.runtime:.3f}s",
                    flush=True,
                )
    summary = summarize(rows)
    report = {
        "architecture": {
            "pipeline": [
                "ARC task",
                "parser",
                "typed symbolic representation",
                "joint residual",
                "candidate operations",
                "exact executor",
                "joint consistency",
                "continuation-safe signature",
                "admissible remaining-cost bound",
                "budgeted best-first / branch-and-bound",
                "exact replay + pass@2",
            ],
            "stage": stage,
            "stage_config": {
                **{k: v for k, v in stage_config(stage).__dict__.items() if k != "object_specs"},
                "object_specs": [s.spec_id for s in stage_config(stage).object_specs],
            },
            "component_kinds": {k: v.value for k, v in COMPONENT_KIND.items()},
            "implemented_operators": operator_catalog(),
            "not_in_phase_1": ["training-only abstraction library (H)", "Qwen proposal model (I)"],
        },
        "budget": {
            "max_depth": budget.max_depth,
            "max_nodes": budget.max_nodes,
            "max_seconds": budget.max_seconds,
            "max_frontier": budget.max_frontier,
            "max_ops_per_node": budget.max_ops_per_node,
        },
        "split": split,
        "wall_clock_sec": round(time.perf_counter() - t0, 4),
        "summary": summary,
        "highest_leverage_next_bottleneck": _next_bottleneck(summary),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
    return report


def run_ablation(
    stages: list[str],
    split: str,
    max_tasks: int,
    budget: SearchBudget,
    out_dir: Path,
) -> dict[str, Any]:
    """Matched-budget A–G comparison on the same task prefix."""
    tasks = list(iter_split(split, max_tasks=max_tasks))
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage in stages:
        rows = []
        print(f"=== ablation stage {stage} n={len(tasks)} ===", flush=True)
        for task in tasks:
            res = search_task(task, stage=stage, budget=budget)
            rows.append(result_to_row(res))
        by_stage[stage] = rows
    compact = {
        stage: {
            "solve_rate": summarize(rows)["solve_rate"],
            "solved": summarize(rows)["solved"],
            "nodes_expanded_mean": summarize(rows)["nodes_expanded_mean"],
            "nodes_generated_mean": summarize(rows)["nodes_generated_mean"],
            "duplicate_states_mean": summarize(rows)["duplicate_states_mean"],
            "runtime_sec_mean": summarize(rows)["runtime_sec_mean"],
            "failure_taxonomy": summarize(rows)["failure_taxonomy"],
            "n_with_exact_demo_solution": summarize(rows)["n_with_exact_demo_solution"],
        }
        for stage, rows in by_stage.items()
    }
    payload = {
        "tasks": [t.task_id for t in tasks],
        "budget": budget.__dict__,
        "stages": compact,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HRPS Phase-1 search microscope")
    p.add_argument("--split", default="training", choices=["training", "evaluation"])
    p.add_argument("--stage", default="G", help="A–G")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--max-nodes", type=int, default=400)
    p.add_argument("--max-seconds", type=float, default=1.25)
    p.add_argument("--max-frontier", type=int, default=4000)
    p.add_argument("--max-ops-per-node", type=int, default=40)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--ablate", type=str, default="", help="comma stages, e.g. A,B,C,D,E,F,G")
    p.add_argument("--ablate-tasks", type=int, default=40)
    args = p.parse_args(argv)

    budget = SearchBudget(
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        max_seconds=args.max_seconds,
        max_frontier=args.max_frontier,
        max_ops_per_node=args.max_ops_per_node,
    )
    out = Path(args.out_dir) if args.out_dir else REPO_ROOT / "artifacts" / f"hrps_phase1_{args.stage}_{args.split}"
    report = run_split(
        split=args.split,
        stage=args.stage,
        max_tasks=args.max_tasks,
        budget=budget,
        out_dir=out,
    )
    print(json.dumps({k: report[k] for k in ("summary", "highest_leverage_next_bottleneck", "budget")}, indent=2))
    if args.ablate:
        stages = [s.strip().upper() for s in args.ablate.split(",") if s.strip()]
        abl = run_ablation(stages, args.split, args.ablate_tasks, budget, out)
        print(json.dumps(abl["stages"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
