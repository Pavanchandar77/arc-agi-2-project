"""M0–M3 open-model capability-elevation experiment.

Thesis: the same frozen open model solves more ARC tasks when it can
actively reason through HRPS than when it reasons from raw grids.

Conditions (matched checkpoint, temperature, max tokens, call/time caps):

  M0  Direct answer from raw serialized grids.
  M1  One-shot formal DSL proposal (diagnostic: can the model speak HRPS).
  M2  Active HRPS loop with execution feedback. Main thesis test.
  M3  Active loop plus verified training-only abstractions.

Also reports a symbolic HRPS-without-model baseline on the same tasks.

No public-evaluation tuning. No task-specific patches. aabf363d is not
special-cased. Test labels are hidden from the loop.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary, load_library_json, set_active_library
from src.hrps.agent import ElevationBudget, EpisodeResult, run_episode
from src.hrps.evaluate import REPO_ROOT
from src.hrps.model import (
    LOCAL_DEFAULT,
    PREFERRED_INKLING,
    FrozenOpenModel,
    ScriptedModel,
    resolve_model_name,
    try_load_open_model,
)
from src.hrps.search import SearchBudget, search_task
from src.hrps.separability import DEFAULT_N, DEFAULT_OFFSET, held_out_training_ids
from src.hrps.task import DEFAULT_DATA_ROOT, ArcTask, load_task_file

CONDITIONS = ("M0", "M1", "M2", "M3")
DEFAULT_H_PATH = REPO_ROOT / "artifacts" / "hrps_separability_h" / "abstractions.json"


def load_held_out_tasks(
    offset: int = DEFAULT_OFFSET,
    n: int = DEFAULT_N,
    data_root: Optional[Path] = None,
) -> list[ArcTask]:
    ids = held_out_training_ids(offset=offset, n=n, data_root=data_root)
    folder = (data_root or DEFAULT_DATA_ROOT) / "training"
    return [load_task_file(folder / f"{tid}.json", "training") for tid in ids]


def symbolic_baseline(task: ArcTask, budget: Optional[SearchBudget] = None) -> dict[str, Any]:
    """HRPS without a model: finite-DSL search (stage G, D3 language)."""
    budget = budget or SearchBudget(max_depth=3, max_nodes=200, max_seconds=0.45, max_frontier=4000, max_ops_per_node=36)
    set_active_library(AbstractionLibrary())
    res = search_task(task, stage="G", budget=budget)
    return {
        "condition": "M_sym",
        "task_id": task.task_id,
        "solved": res.solved,
        "joint_demo_exact": int(res.telemetry.get("n_verified_programs") or 0) > 0,
        "test_exact": res.test_exact,
        "programs": res.programs,
        "failure": res.failure_category,
        "runtime": res.runtime,
        "nodes_expanded": res.telemetry.get("nodes_expanded"),
    }


def _summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_solved = sum(1 for r in rows if r.get("solved"))
    n_joint = sum(1 for r in rows if r.get("joint_demo_exact"))
    n_pass2 = sum(1 for r in rows if r.get("pass2"))
    fails = Counter(r.get("failure") for r in rows)

    def _mean(key: str) -> Optional[float]:
        xs = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(xs) / len(xs), 6) if xs else None

    return {
        "n": n,
        "solved": n_solved,
        "solve_rate": round(n_solved / n, 6) if n else 0.0,
        "joint_demo_exact": n_joint,
        "pass2": n_pass2,
        "pass2_rate": round(n_pass2 / n, 6) if n else 0.0,
        "failure_taxonomy": dict(fails),
        "valid_formal_action_rate_mean": _mean("valid_formal_action_rate"),
        "n_distinct_hypotheses_mean": _mean("n_distinct_hypotheses"),
        "n_hypothesis_revisions_mean": _mean("n_hypothesis_revisions"),
        "n_representation_requests_mean": _mean("n_representation_requests"),
        "n_verifier_calls_mean": _mean("n_verifier_calls"),
        "n_contradiction_resolutions_mean": _mean("n_contradiction_resolutions"),
        "n_model_calls_mean": _mean("n_model_calls"),
        "wall_clock_mean": _mean("wall_clock"),
        "n_prompt_tokens_sum": sum(r.get("n_prompt_tokens") or 0 for r in rows),
        "n_completion_tokens_sum": sum(r.get("n_completion_tokens") or 0 for r in rows),
        "solved_ids": [r["task_id"] for r in rows if r.get("solved")],
        "joint_not_transferred": [
            r["task_id"] for r in rows if r.get("joint_demo_exact") and not r.get("pass2")
        ],
    }


def elevation_verdict(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    m0 = summaries.get("M0", {})
    m1 = summaries.get("M1", {})
    m2 = summaries.get("M2", {})
    m3 = summaries.get("M3", {})
    m0s = m0.get("solved", 0)
    m2s = m2.get("solved", 0)
    m3s = m3.get("solved", 0)
    delta = m2s - m0s
    loop_over_oneshot = m2s - m1.get("solved", 0)
    if m2.get("n", 0) == 0:
        claim = "no_m2_measurement"
    elif m0.get("n") != m2.get("n"):
        claim = "unmatched_task_counts"
    elif delta > 0 and loop_over_oneshot > 0:
        claim = "active_hrps_elevated_beyond_direct_and_oneshot"
    elif delta > 0:
        claim = "active_hrps_elevated_vs_direct_baseline"
    elif delta == 0 and m2s == 0:
        claim = "no_elevation_on_this_slice"
    elif delta == 0:
        claim = "active_loop_matched_direct_baseline"
    else:
        claim = "active_loop_regressed_versus_direct_baseline"
    return {
        "claim": claim,
        "primary_delta_m2_minus_m0": delta,
        "secondary_delta_m2_minus_m1": loop_over_oneshot,
        "m0_solved": m0s,
        "m1_solved": m1.get("solved", 0),
        "m2_solved": m2s,
        "m3_solved": m3s,
        "m3_minus_m2": m3s - m2s,
        "elevated_vs_direct": delta > 0,
        "loop_beat_oneshot": loop_over_oneshot > 0,
        "criterion": (
            "Primary elevation is M2 exact test-transfer versus M0 on the same "
            "frozen model, tasks, and matched decoding budget. M1 is a diagnostic "
            "for whether the model can speak the DSL. M3 tests whether training-only "
            "abstractions further elevate the active loop. Valid syntax alone is not elevation."
        ),
    }


def run_elevation(
    tasks: list[ArcTask],
    model: FrozenOpenModel,
    *,
    conditions: tuple[str, ...] = CONDITIONS,
    budget: Optional[ElevationBudget] = None,
    library: Optional[AbstractionLibrary] = None,
    include_symbolic: bool = True,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    budget = budget or ElevationBudget()
    library = library or AbstractionLibrary()
    out_dir = Path(out_dir) if out_dir is not None else REPO_ROOT / "artifacts" / "hrps_elevation"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    rows_by: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
    if include_symbolic:
        rows_by["M_sym"] = []
    t0 = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            if include_symbolic:
                sym = symbolic_baseline(task)
                rows_by["M_sym"].append(sym)
                fh.write(json.dumps(sym) + "\n")
                fh.flush()
            for cond in conditions:
                ep = run_episode(task, model, cond, budget, library=library if cond == "M3" else AbstractionLibrary())
                row = ep.as_dict()
                rows_by[cond].append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"[{cond}] {task.task_id} solved={ep.solved} joint={ep.joint_demo_exact} "
                    f"fail={ep.failure} calls={ep.n_model_calls} t={ep.wall_clock:.3f}s",
                    flush=True,
                )
    summaries = {k: _summarize_condition(v) for k, v in rows_by.items()}
    report = {
        "thesis": (
            "Hierarchical Residual Program Synthesis elevates open-model reasoning "
            "by turning the model into an active hypothesis-testing agent operating "
            "inside a structured, executable, feedback-rich environment."
        ),
        "model": {
            "preferred": PREFERRED_INKLING,
            "local_default": LOCAL_DEFAULT,
            "used": getattr(model, "name", ""),
            "backend": getattr(model, "backend", ""),
            "frozen": True,
        },
        "budget": {
            "temperature": budget.temperature,
            "max_tokens": budget.max_tokens,
            "max_calls": budget.max_calls,
            "max_seconds": budget.max_seconds,
            "max_program_depth": budget.max_program_depth,
        },
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "conditions": list(conditions),
        "n_abstractions": len(library),
        "wall_clock_sec": round(time.perf_counter() - t0, 4),
        "summaries": summaries,
        "verdict": elevation_verdict(summaries),
        "competition_boundaries": [
            "no internet during evaluation",
            "no hidden labels in the loop",
            "no public-evaluation tuning",
            "no arbitrary unrestricted code execution",
            "exactly two final outputs per test input",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HRPS open-model elevation M0–M3")
    p.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--conditions", type=str, default="M0,M1,M2,M3")
    p.add_argument("--model", type=str, default=None, help="HF id; default Qwen2.5-1.5B-Instruct")
    p.add_argument("--backend", type=str, default="auto", choices=("auto", "hf", "fake"))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--abstractions", type=str, default=str(DEFAULT_H_PATH))
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "artifacts" / "hrps_elevation"))
    p.add_argument("--no-symbolic", action="store_true")
    p.add_argument("--task-id", type=str, default=None, help="run a single training task id")
    args = p.parse_args(argv)

    conditions = tuple(c.strip().upper() for c in args.conditions.split(",") if c.strip())
    budget = ElevationBudget(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_calls=args.max_calls,
        max_seconds=args.max_seconds,
    )
    library = AbstractionLibrary()
    abs_path = Path(args.abstractions)
    if abs_path.is_file():
        library = load_library_json(abs_path)

    if args.task_id:
        folder = DEFAULT_DATA_ROOT / "training"
        tasks = [load_task_file(folder / f"{args.task_id}.json", "training")]
    else:
        tasks = load_held_out_tasks(offset=args.offset, n=args.n)

    model: Optional[FrozenOpenModel] = None
    status = ""
    if args.backend == "fake":
        model = ScriptedModel(responses=[])
        status = "fake"
    elif args.backend in {"auto", "hf"}:
        model, status = try_load_open_model(args.model)
        if model is None and args.backend == "hf":
            print(f"failed to load open model: {status}", flush=True)
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.out_dir) / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "reason": status,
                        "preferred_model": PREFERRED_INKLING,
                        "local_default": LOCAL_DEFAULT,
                        "resolved": resolve_model_name(args.model),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 2

    if model is None:
        print(
            f"open model unavailable ({status}); writing blocked report. "
            "Tests cover M0–M3 with ScriptedModel. Install torch+transformers "
            f"to run {LOCAL_DEFAULT}. Inkling-Small is not a local default.",
            flush=True,
        )
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "blocked",
            "reason": status,
            "preferred_model": PREFERRED_INKLING,
            "local_default": LOCAL_DEFAULT,
            "resolved": resolve_model_name(args.model),
            "n_tasks": len(tasks),
            "task_ids": [t.task_id for t in tasks],
            "conditions": list(conditions),
            "budget": {
                "temperature": budget.temperature,
                "max_tokens": budget.max_tokens,
                "max_calls": budget.max_calls,
                "max_seconds": budget.max_seconds,
            },
            "note": (
                "The elevation harness is implemented. A live M0-vs-M2 number "
                "requires a frozen open-model runtime. This environment has no torch."
            ),
        }
        (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 2

    report = run_elevation(
        tasks,
        model,
        conditions=conditions,
        budget=budget,
        library=library,
        include_symbolic=not args.no_symbolic,
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(report["verdict"], indent=2), flush=True)
    print(json.dumps(report["summaries"], indent=2)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
