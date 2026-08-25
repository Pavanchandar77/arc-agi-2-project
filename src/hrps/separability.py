"""Language-vs-search separability experiment.

A–G mechanisms, DSL, depth, and per-node operator cap stay frozen.
Phase-1 conclusions live in src.hrps.conclusions and must not be patched
around with Qwen, extra heuristics, or unmeasured operators.

This module only varies SearchBudget and records whether a task is:

  solved_at_low_budget
  solved_only_at_higher_budget
  not_expressible          — frontier exhausted with no joint solution
  unsolved_search_still_truncated — highest budget still hit a cap

The held-out slice is a later block of official training IDs so it is not the
prefix used while debugging operator order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary, set_active_library
from src.hrps.dsl import Op, Program, infer_colormap, replay, stage_config
from src.hrps.evaluate import REPO_ROOT, result_to_row
from src.hrps.grid import colors_present, shape
from src.hrps.search import SearchBudget, search_task
from src.hrps.task import ArcTask, iter_split, load_task_file, DEFAULT_DATA_ROOT

# Frozen production budget from the 1000-task G measurement (2.5%).
LOW_BUDGET_ID = "B0"

# D3 baseline language. Experiment D4 raises only max_depth; ops/node stay 36.
FROZEN_DEPTH = 3
FROZEN_OPS_PER_NODE = 36

_BUDGET_RUNGS: tuple[tuple[str, int, float, int], ...] = (
    ("B0", 200, 0.45, 4000),
    ("B1", 1000, 3.0, 12000),
    ("B2", 5000, 15.0, 40000),
    ("B3", 25000, 40.0, 100000),
)


def make_budget_ladder(max_depth: int = FROZEN_DEPTH) -> tuple[tuple[str, SearchBudget], ...]:
    """Same node/time/frontier rungs; only program depth changes."""
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    return tuple(
        (
            bid,
            SearchBudget(
                max_depth=max_depth,
                max_nodes=nodes,
                max_seconds=seconds,
                max_frontier=frontier,
                max_ops_per_node=FROZEN_OPS_PER_NODE,
            ),
        )
        for bid, nodes, seconds, frontier in _BUDGET_RUNGS
    )


BUDGET_LADDER: tuple[tuple[str, SearchBudget], ...] = make_budget_ladder(FROZEN_DEPTH)

SEPARABILITY_CLASSES = (
    "solved_at_low_budget",
    "solved_only_at_higher_budget",
    "not_expressible",
    "unsolved_search_still_truncated",
)

# First 60 sorted IDs were the Phase-1 A/F/G prefix. Skip well past that.
DEFAULT_OFFSET = 400
DEFAULT_N = 40
DEFAULT_STAGES = ("A", "F", "G")


def held_out_training_ids(
    offset: int = DEFAULT_OFFSET,
    n: int = DEFAULT_N,
    data_root: Optional[Path] = None,
) -> list[str]:
    ids = [t.task_id for t in iter_split("training", data_root=data_root)]
    if offset < 0 or offset >= len(ids):
        raise ValueError(f"offset {offset} out of range for {len(ids)} training tasks")
    return ids[offset : offset + n]


def classify_budget_trace(trace: list[dict[str, Any]]) -> str:
    """Classify one task from its low→high budget records for a single stage."""
    if not trace:
        raise ValueError("empty budget trace")
    first_solved: Optional[str] = None
    exhausted_unsolved = False
    for rec in trace:
        tel = rec["telemetry"]
        if rec["solved"] and first_solved is None:
            first_solved = rec["budget_id"]
            break
        if tel.get("enumerated_exhausted") and not rec["solved"]:
            exhausted_unsolved = True
            break
    if first_solved == LOW_BUDGET_ID:
        return "solved_at_low_budget"
    if first_solved is not None:
        return "solved_only_at_higher_budget"
    if exhausted_unsolved:
        return "not_expressible"
    return "unsolved_search_still_truncated"


def _singleton_cells(grid) -> list[tuple[int, int, int]]:
    """Non-zero cells whose 4-neighborhood (in-grid) has no other non-zero."""
    h, w = shape(grid)
    out = []
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == 0:
                continue
            nbr = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                    nbr += 1
            if nbr == 0:
                out.append((r, c, v))
    return out


def _corner_nonzero(grid) -> list[tuple[str, int]]:
    h, w = shape(grid)
    corners = {
        "tl": grid[0][0],
        "tr": grid[0][w - 1],
        "bl": grid[h - 1][0],
        "br": grid[h - 1][w - 1],
    }
    return [(name, val) for name, val in corners.items() if val != 0]


def colormap_from_program(text: str) -> Optional[dict[int, int]]:
    if not text.startswith("apply_colormap:"):
        return None
    payload = text.split(":", 1)[1]
    # First op only (depth-1 colormap).
    payload = payload.split(" | ")[0]
    pairs = {}
    for tok in payload.split(";"):
        if not tok or "-" not in tok:
            continue
        a, b = tok.split("-", 1)
        pairs[int(a)] = int(b)
    return pairs


def characterize_colormap_constraint(task: ArcTask, program_text: str) -> dict[str, Any]:
    """What a jointly consistent colormap failed to pin down (if anything)."""
    mapping = colormap_from_program(program_text)
    demo_in_palettes = [set(colors_present(p.input)) for p in task.train]
    demo_out_palettes = [set(colors_present(p.output)) for p in task.train]  # type: ignore[arg-type]
    test_in_palettes = [set(colors_present(p.input)) for p in task.test]
    train_in_union: set[int] = set().union(*demo_in_palettes) if demo_in_palettes else set()
    test_unseen = sorted(set().union(*test_in_palettes) - train_in_union) if test_in_palettes else []

    per_demo_maps = []
    disjoint_support = True
    used: set[int] = set()
    for p in task.train:
        local = infer_colormap((p.input,), (p.output,))  # type: ignore[arg-type]
        srcs = set()
        if local:
            srcs = {a for a, b in local if a != b}
            overlap = srcs & used
            if overlap:
                disjoint_support = False
            used |= srcs
        per_demo_maps.append(
            {
                "map": [[a, b] for a, b in (local or ())],
                "input_colors": sorted(colors_present(p.input)),
                "output_colors": sorted(colors_present(p.output)),  # type: ignore[arg-type]
                "singletons": _singleton_cells(p.input),
                "corner_nonzero": _corner_nonzero(p.input),
            }
        )

    test_replay = []
    for p in task.test:
        prog = Program(tuple(Op.deserialize(tok) for tok in program_text.split(" | ") if tok != "identity"))
        pred = replay(prog, p.input)
        gt = p.output
        test_replay.append(
            {
                "pred_equals_gt": pred is not None and gt is not None and pred == gt,
                "input_colors": sorted(colors_present(p.input)),
                "gt_colors": sorted(colors_present(gt)) if gt is not None else None,
                "pred_colors": sorted(colors_present(pred)) if pred is not None else None,
                "singletons": _singleton_cells(p.input),
                "corner_nonzero": _corner_nonzero(p.input),
                "unseen_input_colors": sorted(set(colors_present(p.input)) - train_in_union),
            }
        )

    role_collision = []
    # A color that is a singleton/key in one pair and a blob color in another.
    singleton_colors = set()
    blob_colors = set()
    for p in list(task.train) + list(task.test):
        sc = {v for _, _, v in _singleton_cells(p.input)}
        allc = set(colors_present(p.input)) - {0}
        singleton_colors |= sc
        blob_colors |= allc - sc
    role_collision = sorted(singleton_colors & blob_colors)

    return {
        "program": program_text,
        "inferred_joint_map": [[k, mapping[k]] for k in sorted(mapping)] if mapping else None,
        "disjoint_per_demo_color_support": disjoint_support and mapping is not None,
        "per_demo_maps": per_demo_maps,
        "test_unseen_input_colors": test_unseen,
        "role_collision_colors": role_collision,
        "test_replay": test_replay,
        "constraint_failure": _constraint_failure_label(
            mapping is not None,
            disjoint_support,
            test_unseen,
            role_collision,
            test_replay,
        ),
    }


def _constraint_failure_label(
    is_colormap: bool,
    disjoint_support: bool,
    test_unseen: list[int],
    role_collision: list[int],
    test_replay: list[dict[str, Any]],
) -> str:
    if not test_replay:
        return "no_test"
    if all(r.get("pred_equals_gt") for r in test_replay):
        return "constrained_and_transferred"
    if not is_colormap:
        return "joint_program_failed_to_transfer"
    parts = []
    if disjoint_support:
        parts.append("joint_map_is_union_of_disjoint_per_demo_palettes")
    if test_unseen:
        parts.append("test_introduces_unseen_input_colors")
    if role_collision:
        parts.append("color_plays_marker_role_in_one_pair_and_blob_role_in_another")
    return "+".join(parts) if parts else "colormap_underconstrained_for_other_reasons"


def analyze_joint_demo_row(row: dict[str, Any], task: ArcTask) -> dict[str, Any]:
    prog = row["programs"][0] if row.get("programs") else "identity"
    family = prog.split(":")[0].split(" | ")[0]
    rec: dict[str, Any] = {
        "task_id": row["task_id"],
        "solved": row["solved"],
        "failure_category": row["failure_category"],
        "program": prog,
        "family": family,
        "n_train": task.n_train,
        "n_test": task.n_test,
        "ttf": row["telemetry"].get("time_to_first_exact_demonstration_solution"),
        "nodes_expanded": row["telemetry"].get("nodes_expanded"),
        "description_length": row["telemetry"].get("description_length"),
    }
    if prog.startswith("apply_colormap") or "apply_colormap" in prog:
        rec["colormap"] = characterize_colormap_constraint(task, prog)
    else:
        rec["colormap"] = None
        rec["constraint_failure"] = (
            "constrained_and_transferred" if row["solved"] else "joint_program_failed_to_transfer"
        )
    return rec


def language_ceiling(classes: Counter) -> dict[str, Any]:
    n = sum(classes.values())
    solved = classes["solved_at_low_budget"] + classes["solved_only_at_higher_budget"]
    inexpr = classes["not_expressible"]
    trunc = classes["unsolved_search_still_truncated"]
    decided = solved + inexpr
    return {
        "n": n,
        "solved_any_budget": solved,
        "not_expressible": inexpr,
        "still_truncated": trunc,
        "decided": decided,
        "accuracy_lower_bound": round(solved / n, 6) if n else 0.0,
        "accuracy_upper_bound": round((solved + trunc) / n, 6) if n else 0.0,
        "ceiling_among_exhausted": round(solved / decided, 6) if decided else None,
        "search_truncation_share_of_unsolved": (
            round(trunc / (inexpr + trunc), 6) if (inexpr + trunc) else None
        ),
        "verdict": _ceiling_verdict(solved, inexpr, trunc, n),
    }


def _ceiling_verdict(solved: int, inexpr: int, trunc: int, n: int) -> str:
    if n == 0:
        return "no_tasks"
    if trunc == 0 and inexpr > solved:
        return "primarily_dsl_expressiveness_ceiling"
    if trunc == 0 and solved >= inexpr:
        return "language_ceiling_reached_mostly_expressible"
    if trunc > 0 and inexpr == 0 and solved == 0:
        return "primarily_search_truncation_ceiling"
    if trunc > inexpr and solved == 0:
        return "primarily_search_truncation_unresolved"
    if inexpr > 0 and trunc > 0:
        return "both_truncation_and_expressiveness"
    if trunc > 0 and solved > 0 and inexpr == 0:
        return "solved_or_still_truncated_no_exhausted_negatives"
    return "mixed"


def run_separability(
    stages: tuple[str, ...] = DEFAULT_STAGES,
    offset: int = DEFAULT_OFFSET,
    n: int = DEFAULT_N,
    out_dir: Optional[Path] = None,
    data_root: Optional[Path] = None,
    max_depth: int = FROZEN_DEPTH,
    abstractions: AbstractionLibrary | None = None,
) -> dict[str, Any]:
    ids = held_out_training_ids(offset=offset, n=n, data_root=data_root)
    folder = (data_root or DEFAULT_DATA_ROOT) / "training"
    tasks = [load_task_file(folder / f"{tid}.json", "training") for tid in ids]
    out_dir = Path(out_dir) if out_dir is not None else REPO_ROOT / "artifacts" / "hrps_separability"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    ladder = make_budget_ladder(max_depth)
    library = abstractions or AbstractionLibrary()
    set_active_library(library)
    abs_tuple = library.items

    traces: dict[str, dict[str, list[dict[str, Any]]]] = {s: {t.task_id: [] for t in tasks} for s in stages}
    t0 = time.perf_counter()
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for stage in stages:
            print(f"=== separability D{max_depth} stage {stage} n={len(tasks)} ===", flush=True)
            for task in tasks:
                for budget_id, budget in ladder:
                    res = search_task(
                        task,
                        stage=stage,
                        budget=budget,
                        cfg=stage_config(stage, abstractions=abs_tuple),
                    )
                    row = result_to_row(res)
                    row["budget_id"] = budget_id
                    row["budget"] = {
                        "max_depth": budget.max_depth,
                        "max_nodes": budget.max_nodes,
                        "max_seconds": budget.max_seconds,
                        "max_frontier": budget.max_frontier,
                        "max_ops_per_node": budget.max_ops_per_node,
                    }
                    traces[stage][task.task_id].append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    tel = row["telemetry"]
                    print(
                        f"  [{stage} {budget_id}] {task.task_id} solved={row['solved']} "
                        f"exh={tel.get('enumerated_exhausted')} exp={tel.get('nodes_expanded')} "
                        f"t={row['runtime']:.3f}s {row['failure_category']}",
                        flush=True,
                    )
                    if row["solved"] or tel.get("enumerated_exhausted"):
                        break

    per_stage: dict[str, Any] = {}
    for stage in stages:
        class_by_task = {
            tid: classify_budget_trace(trace) for tid, trace in traces[stage].items()
        }
        classes = Counter(class_by_task.values())
        order = {bid: i for i, (bid, _) in enumerate(ladder)}
        n_tasks = len(traces[stage])
        by_budget = []
        for budget_id, _ in ladder:
            idx = order[budget_id]
            executed = []
            cumulative_solved = 0
            cumulative_exhausted = 0
            first_solved_here = 0
            for tid, trace in traces[stage].items():
                rec_here = next((r for r in trace if r["budget_id"] == budget_id), None)
                if rec_here is not None:
                    executed.append(rec_here)
                    if rec_here["solved"] and all(
                        not r["solved"] for r in trace if order[r["budget_id"]] < idx
                    ):
                        first_solved_here += 1
                if any(r["solved"] and order[r["budget_id"]] <= idx for r in trace):
                    cumulative_solved += 1
                if any(
                    r["telemetry"].get("enumerated_exhausted") and order[r["budget_id"]] <= idx
                    for r in trace
                ):
                    cumulative_exhausted += 1
            quotients = [
                r["telemetry"].get("quotient_ratio")
                for r in executed
                if r["telemetry"].get("quotient_ratio") is not None
            ]
            ttfs = [
                r["telemetry"].get("time_to_first_exact_demonstration_solution")
                for r in executed
                if r["telemetry"].get("time_to_first_exact_demonstration_solution") is not None
            ]
            expansions = [r["telemetry"]["nodes_expanded"] for r in executed]
            generated = [r["telemetry"]["nodes_generated"] for r in executed]
            unique = [r["telemetry"]["unique_states"] for r in executed]
            dups = [r["telemetry"]["duplicate_states"] for r in executed]
            runtimes = [r["runtime"] for r in executed]

            def _mean(xs: list) -> Optional[float]:
                return round(sum(xs) / len(xs), 6) if xs else None

            by_budget.append(
                {
                    "budget_id": budget_id,
                    "n_executed_this_rung": len(executed),
                    "cumulative_solved": cumulative_solved,
                    "cumulative_solve_rate": round(cumulative_solved / n_tasks, 6) if n_tasks else 0.0,
                    "first_solved_at_this_rung": first_solved_here,
                    "cumulative_exhausted": cumulative_exhausted,
                    "nodes_expanded_mean": _mean(expansions),
                    "nodes_generated_mean": _mean(generated),
                    "unique_states_mean": _mean(unique),
                    "duplicate_states_mean": _mean(dups),
                    "quotient_ratio_mean": _mean(quotients),
                    "ttf_mean": _mean(ttfs),
                    "n_with_first_solution": len(ttfs),
                    "runtime_sec_mean": _mean(runtimes),
                }
            )
        per_stage[stage] = {
            "class_counts": dict(classes),
            "class_by_task": class_by_task,
            "language_ceiling": language_ceiling(classes),
            "by_budget": by_budget,
        }

    report = {
        "protocol": {
            "held_out_offset": offset,
            "held_out_n": n,
            "task_ids": ids,
            "stages": list(stages),
            "frozen": {
                "depth": max_depth,
                "baseline_depth": FROZEN_DEPTH,
                "ops_per_node": FROZEN_OPS_PER_NODE,
                "dsl_operators": "unchanged",
                "heuristics": "unchanged",
                "language_change": (
                    "H_training_only_abstractions"
                    if abs_tuple
                    else (
                        "none"
                        if max_depth == FROZEN_DEPTH
                        else f"max_depth {FROZEN_DEPTH} -> {max_depth} only"
                    )
                ),
                "n_abstractions": len(abs_tuple),
            },
            "budgets": [
                {"id": bid, **b.__dict__} for bid, b in ladder
            ],
            "note": (
                "Held-out slice is training IDs[offset:offset+n] in sorted order. "
                "The first 60 sorted IDs were the Phase-1 A/F/G debugging prefix and are excluded."
            ),
        },
        "wall_clock_sec": round(time.perf_counter() - t0, 4),
        "stages": per_stage,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_depth_reports(baseline: dict[str, Any], raised: dict[str, Any]) -> dict[str, Any]:
    """Compare two A/F/G ladder reports that differ only in max_depth."""
    stages = sorted(set(baseline["stages"]) & set(raised["stages"]))
    out: dict[str, Any] = {
        "baseline_depth": baseline.get("protocol", {}).get("frozen", {}).get("depth"),
        "raised_depth": raised.get("protocol", {}).get("frozen", {}).get("depth"),
        "same_task_ids": baseline.get("protocol", {}).get("task_ids")
        == raised.get("protocol", {}).get("task_ids"),
        "stages": {},
    }
    for stage in stages:
        b = baseline["stages"][stage]
        r = raised["stages"][stage]
        b_map = b["class_by_task"]
        r_map = r["class_by_task"]
        b_solved = {t for t, c in b_map.items() if c.startswith("solved")}
        r_solved = {t for t, c in r_map.items() if c.startswith("solved")}
        newly = sorted(r_solved - b_solved)
        lost = sorted(b_solved - r_solved)
        out["stages"][stage] = {
            "baseline_class_counts": b["class_counts"],
            "raised_class_counts": r["class_counts"],
            "baseline_ceiling": b["language_ceiling"],
            "raised_ceiling": r["language_ceiling"],
            "newly_solved": newly,
            "lost_solved": lost,
            "n_newly_solved": len(newly),
            "n_lost_solved": len(lost),
            "higher_budget_solves_raised": r["class_counts"].get("solved_only_at_higher_budget", 0),
        }
    return out


def analyze_joint_demo_artifact(
    jsonl_path: Path,
    data_root: Optional[Path] = None,
) -> dict[str, Any]:
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    joint = [r for r in rows if r.get("telemetry", {}).get("joint_verified")]
    folder = (data_root or DEFAULT_DATA_ROOT) / "training"
    analyses = []
    for row in joint:
        task = load_task_file(folder / f"{row['task_id']}.json", "training")
        analyses.append(analyze_joint_demo_row(row, task))
    families = Counter(a["family"] for a in analyses)
    cmap_fail = [
        a for a in analyses if a.get("colormap") and not a["solved"]
    ]
    return {
        "n_joint_demo_solutions": len(analyses),
        "n_test_transfer_success": sum(1 for a in analyses if a["solved"]),
        "n_test_transfer_fail": sum(1 for a in analyses if not a["solved"]),
        "families": dict(families),
        "transfer_failures": [
            {
                "task_id": a["task_id"],
                "program": a["program"],
                "constraint_failure": a["colormap"]["constraint_failure"] if a.get("colormap") else a.get("constraint_failure"),
                "colormap": a.get("colormap"),
            }
            for a in analyses
            if not a["solved"]
        ],
        "all": analyses,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HRPS language-vs-search separability")
    p.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--stages", default="A,F,G")
    p.add_argument("--depth", type=int, default=FROZEN_DEPTH, help="max program depth; D4 uses 4")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Path to a prior ladder summary.json (usually D3) to diff against",
    )
    p.add_argument(
        "--joint-demo-jsonl",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "hrps_phase1_G_training" / "tasks.jsonl"),
    )
    p.add_argument("--skip-ladder", action="store_true")
    p.add_argument("--skip-joint-demo", action="store_true")
    p.add_argument(
        "--enable-h",
        action="store_true",
        help="Mine training-only abstractions from traces, excluding the held-out slice",
    )
    p.add_argument(
        "--traces",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "hrps_phase1_G_training" / "tasks.jsonl"),
        help="G training traces used to mine H (solved=True only; held-out IDs excluded)",
    )
    args = p.parse_args(argv)
    if args.enable_h:
        default_out = REPO_ROOT / "artifacts" / "hrps_separability_h"
    elif args.depth == FROZEN_DEPTH:
        default_out = REPO_ROOT / "artifacts" / "hrps_separability"
    else:
        default_out = REPO_ROOT / "artifacts" / f"hrps_separability_d{args.depth}"
    out = Path(args.out_dir) if args.out_dir else default_out
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_joint_demo:
        joint_path = Path(args.joint_demo_jsonl)
        if joint_path.exists():
            joint = analyze_joint_demo_artifact(joint_path)
            (out / "joint_demo_analysis.json").write_text(json.dumps(joint, indent=2), encoding="utf-8")
            print(json.dumps({k: joint[k] for k in joint if k != "all"}, indent=2))
        else:
            print(f"joint-demo jsonl not found: {joint_path}", flush=True)

    library = AbstractionLibrary()
    if args.enable_h:
        from src.hrps.abstractions import mine_from_jsonl, transfer_report

        exclude = held_out_training_ids(offset=args.offset, n=args.n)
        library = mine_from_jsonl(Path(args.traces), exclude)
        (out / "abstractions.json").write_text(json.dumps(library.as_dict(), indent=2), encoding="utf-8")
        folder = DEFAULT_DATA_ROOT / "training"
        held_tasks = [load_task_file(folder / f"{tid}.json", "training") for tid in exclude]
        xfer = transfer_report(library, held_tasks)
        (out / "h_transfer.json").write_text(json.dumps(xfer, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "n_abstractions": len(library),
                    "exclude_n": len(exclude),
                    "transfer": {k: xfer[k] for k in xfer if k != "hits"},
                    "abstraction_names": [a.name for a in library.items],
                },
                indent=2,
            ),
            flush=True,
        )

    if not args.skip_ladder:
        stages = tuple(s.strip().upper() for s in args.stages.split(",") if s.strip())
        report = run_separability(
            stages=stages,
            offset=args.offset,
            n=args.n,
            out_dir=out,
            max_depth=args.depth,
            abstractions=library if args.enable_h else None,
        )
        compact: dict[str, Any] = {
            "protocol": report["protocol"],
            "wall_clock_sec": report["wall_clock_sec"],
            "stages": {
                s: {
                    "class_counts": report["stages"][s]["class_counts"],
                    "language_ceiling": report["stages"][s]["language_ceiling"],
                    "by_budget": report["stages"][s]["by_budget"],
                }
                for s in report["stages"]
            },
        }
        if args.compare_with:
            baseline = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
            diff = compare_depth_reports(baseline, report)
            compact["vs_baseline"] = diff
            (out / "vs_baseline.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")
        print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
