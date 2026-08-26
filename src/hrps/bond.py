"""Bond: an actively trained open reasoning model-system.

  Qwen foundation
    + HRPS-generated reasoning episodes
    + learned Bond adapter
    + active HRPS reasoning loop
    + exact executor and verifier
  = Bond-Qwen

The 1.5B local run is Bond-Qwen1.5B-smoke, not final Bond.
Qwen3.5-4B is the first serious foundation (remote GPU).
Inkling-Small is a configurable future backend and is not required locally.

Never label a wrapper as a trained model. Never silently swap checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary, load_library_json
from src.hrps.backend import FOUNDATIONS, hardware_gate, load_backend, resolve_foundation
from src.hrps.identity import adapter_is_complete
from src.hrps.elevation import DEFAULT_H_PATH, REPO_ROOT, load_held_out_tasks
from src.hrps.episodes import (
    ACTION_SCHEMA,
    BondEpisode,
    assert_training_safe,
    file_sha256,
    generate_bond_episodes,
    write_episodes,
)
from src.hrps.model import FrozenOpenModel, ScriptedModel
from src.hrps.runner import RunnerBudget, run_system
from src.hrps.schema import BOND_ACTIONS
from src.hrps.separability import DEFAULT_N, DEFAULT_OFFSET, held_out_training_ids
from src.hrps.task import DEFAULT_DATA_ROOT, ArcTask, load_task_file

BOND_DIR = REPO_ROOT / "artifacts" / "bond"
BOND_ADAPTER_DIR = REPO_ROOT / "models" / "bond_adapter"
BOND_VERSION = "bond-v0"
SYSTEMS = ("base_direct", "base_hrps", "bond_direct", "bond_hrps")

CPU_TRAIN_CONFIG: dict[str, Any] = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "seed": 42,
    "quantization": "none",
    "device": "cpu",
}

DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "optim": "adamw_torch",
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.05,
    "seed": 42,
    "quantization": "none",
}


def code_revision() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def bond_manifest(
    *,
    foundation: dict[str, Any],
    adapter_dir: Optional[Path],
    train_config: dict[str, Any],
    episode_summary: dict[str, Any],
    held_out_ids: list[str],
    status: str,
    notes: list[str],
    losses: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    adapter_present = bool(
        adapter_dir and Path(adapter_dir).exists() and any(Path(adapter_dir).iterdir())
    )
    smoke = foundation.get("id") == "qwen1.5b_smoke"
    return {
        "public_name": "Bond",
        "name": "Bond",
        "version": BOND_VERSION,
        "status": status,
        "artifact_class": "smoke_not_final_bond" if smoke else "experimental",
        "is_final_bond": False,
        "retain_foundation_weights": True,
        "license": {
            "note": (
                "Bond is a fine-tune of an open Qwen foundation. Keep the foundation "
                "checkpoint. Record the foundation identifier here; do not use that "
                "name as the runtime identity."
            ),
            "foundation_hf_id": foundation.get("hf_id"),
        },
        "thesis": (
            "Bond is an actively trained open reasoning model-system whose parameters "
            "are improved through verified HRPS reasoning experience, and whose inference "
            "is amplified by the same executable cognitive substrate."
        ),
        "foundation": {
            **foundation,
            "weights_hash": None,
            "hash_note": "filled after the base checkpoint is materialized on disk",
        },
        "adapter": {
            "path": str(adapter_dir) if adapter_dir else None,
            "present": adapter_present,
            "kind": "lora",
            "trained": status == "adapter_trained",
        },
        "training": train_config,
        "losses": losses or {},
        "code_revision": code_revision(),
        "data_provenance": {
            "split": "training",
            "held_out_excluded": held_out_ids,
            "public_evaluation_used": False,
            "episodes": episode_summary,
        },
        "schemas": {
            "legacy_actions": ACTION_SCHEMA,
            "json_actions": list(BOND_ACTIONS),
            "inference_controller": "src.hrps.runner.run_system",
            "executor": "src.hrps.dsl.replay",
            "verifier": "src.hrps.residual.joint_residual + gold_free_constraint_feedback",
        },
        "notes": notes,
    }


def generate_and_write(
    out_dir: Path = BOND_DIR,
    library: Optional[AbstractionLibrary] = None,
    max_strategy_episodes: int = 15,
) -> tuple[list[BondEpisode], dict[str, Any]]:
    held = held_out_training_ids()
    episodes = generate_bond_episodes(library=library, max_strategy_episodes=max_strategy_episodes)
    assert_training_safe(episodes, held)
    summary = write_episodes(episodes, out_dir)
    summary["held_out_excluded"] = held
    return episodes, summary


def train_bond_adapter(
    sft_path: Path,
    *,
    model_name: str,
    output_dir: Path,
    config: Optional[dict[str, Any]] = None,
    foundation_id: str = "qwen1.5b_smoke",
) -> dict[str, Any]:
    """Train Bond LoRA. Fails clearly without torch. Never swaps the foundation."""
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if config:
        cfg.update(config)
    spec = resolve_foundation(foundation_id if foundation_id != "custom" else model_name)
    if spec["hf_id"] != model_name and foundation_id != "custom":
        # Keep the declared HF id authoritative.
        model_name = spec["hf_id"]
    blocked = hardware_gate(spec)
    if blocked:
        return {"status": "blocked", "reason": blocked, "output_dir": str(output_dir), "is_final_bond": False}
    from src.hrps.backend import free_ram_gate

    ram_block = free_ram_gate(spec)
    if ram_block:
        return {"status": "blocked", "reason": ram_block, "output_dir": str(output_dir), "is_final_bond": False}
    try:
        import torch  # noqa: F401
    except Exception:
        return {
            "status": "blocked",
            "reason": "no_torch",
            "output_dir": str(output_dir),
            "is_final_bond": False,
            "message": "PyTorch is required to train a Bond adapter. This is not a trained Bond checkpoint.",
        }
    if not sft_path.is_file():
        return {"status": "blocked", "reason": "missing_sft", "output_dir": str(output_dir), "is_final_bond": False}
    from src.train import train as sft_train

    sft_train(
        train_file=str(sft_path),
        val_file=None,
        model_name=model_name,
        output_dir=str(output_dir),
        max_seq_length=int(cfg["max_seq_length"]),
        lora_r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        seed=int(cfg["seed"]),
    )
    cfg["adapter_kind"] = "native_precision_lora"
    cfg["quantization"] = "none"
    weights_ok = adapter_is_complete(Path(output_dir))
    (Path(output_dir) / "BOND_TRAIN.json").write_text(
        json.dumps(
            {
                "foundation": spec,
                "hf_id": model_name,
                "is_final_bond": False,
                "adapter_kind": "native_precision_lora",
                "adapter_weights_present": weights_ok,
                "artifact_class": "smoke_not_final_bond" if spec["id"] == "qwen1.5b_smoke" else "experimental",
                "config": cfg,
                "sft_sha256": file_sha256(sft_path),
                "code_revision": code_revision(),
                "note": (
                    "Not a learned Bond checkpoint unless adapter_weights_present is true "
                    "and the adapter reloads on Qwen/Qwen3.5-4B."
                    if spec["id"] == "qwen3.5_4b"
                    else "Not a learned Bond checkpoint unless adapter weights were saved and reloaded."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "config": cfg,
        "foundation": spec,
        "is_final_bond": False,
        "adapter_kind": "native_precision_lora",
        "adapter_weights_present": weights_ok,
        "artifact_class": "smoke_not_final_bond" if spec["id"] == "qwen1.5b_smoke" else "experimental",
    }


def bond_deltas(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def _solved(key: str) -> int:
        return int(summaries.get(key, {}).get("solved") or 0)

    base_d = _solved("base_direct")
    base_h = _solved("base_hrps")
    bond_d = _solved("bond_direct")
    bond_h = _solved("bond_hrps")
    d_train = bond_d - base_d
    d_sub = bond_h - bond_d
    d_full = bond_h - base_d
    harness_only = base_h - base_d
    if d_train > 0 and d_sub > 0:
        claim = "learned_and_substrate_gains"
    elif d_train > 0 and d_sub == 0:
        claim = "learned_gain_without_extra_substrate_gain"
    elif d_train <= 0 and d_sub > 0:
        claim = "substrate_gain_without_learned_direct_gain"
    elif d_full > 0:
        claim = "system_gain_unallocated"
    else:
        claim = "no_bond_gain"
    return {
        "claim": claim,
        "learned_model_gain": d_train > 0,
        "substrate_gain": d_sub > 0,
        "base_direct": base_d,
        "base_hrps": base_h,
        "bond_direct": bond_d,
        "bond_hrps": bond_h,
        "delta_train_bond_direct_minus_base_direct": d_train,
        "delta_substrate_bond_hrps_minus_bond_direct": d_sub,
        "delta_harness_base_hrps_minus_base_direct": harness_only,
        "delta_full_bond_hrps_minus_base_direct": d_full,
        "criterion": (
            "A learned-model gain requires bond_direct > base_direct after adapter "
            "weights changed. A substrate gain requires bond_hrps > bond_direct. "
            "A wrapper-only score is not a Bond result."
        ),
    }


def _summarize_runner_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_solved = sum(1 for r in rows if r.get("solved"))
    n_joint = sum(1 for r in rows if r.get("joint_demo_exact"))
    n_pass2 = sum(1 for r in rows if r.get("pass2"))

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
        "valid_formal_action_rate_mean": _mean("valid_formal_action_rate"),
        "n_distinct_hypotheses_mean": _mean("n_distinct_hypotheses"),
        "n_hypothesis_revisions_mean": _mean("n_hypothesis_revisions"),
        "n_hypothesis_rejections_mean": _mean("n_hypothesis_rejections"),
        "n_representation_requests_mean": _mean("n_representation_requests"),
        "n_verifier_calls_mean": _mean("n_verifier_calls"),
        "n_contradiction_resolutions_mean": _mean("n_contradiction_resolutions"),
        "n_model_calls_mean": _mean("n_model_calls"),
        "n_prompt_tokens_sum": sum(r.get("n_prompt_tokens") or 0 for r in rows),
        "n_completion_tokens_sum": sum(r.get("n_completion_tokens") or 0 for r in rows),
        "hrps_symbolic_seconds_sum": round(sum(r.get("hrps_symbolic_seconds") or 0 for r in rows), 6),
        "wall_clock_mean": _mean("wall_clock"),
        "peak_memory_max": max((r.get("peak_memory") or 0) for r in rows) if rows else None,
        "solved_ids": [r["task_id"] for r in rows if r.get("solved")],
        "terminations": {r.get("termination"): None for r in rows},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "system",
        "task_id",
        "solved",
        "joint_demo_exact",
        "pass2",
        "failure",
        "termination",
        "n_model_calls",
        "n_prompt_tokens",
        "n_completion_tokens",
        "n_distinct_hypotheses",
        "n_hypothesis_revisions",
        "n_hypothesis_rejections",
        "n_representation_requests",
        "n_verifier_calls",
        "n_contradiction_resolutions",
        "n_invalid_actions",
        "valid_formal_action_rate",
        "time_to_first_joint_exact",
        "hrps_symbolic_seconds",
        "wall_clock",
        "peak_memory",
        "saw_underconstraint_flags",
        "committed_underconstrained",
        "n_verified_candidates",
        "n_reasoning_cycles",
        "seed",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def probe_aabf363d(
    base_model: FrozenOpenModel,
    bond_model: Optional[FrozenOpenModel],
    budget: RunnerBudget,
) -> dict[str, Any]:
    path = DEFAULT_DATA_ROOT / "training" / "aabf363d.json"
    if not path.is_file():
        return {"status": "missing_task"}
    task = load_task_file(path, "training")
    models = [("base_direct", base_model), ("base_hrps", base_model)]
    if bond_model is not None:
        models.extend([("bond_direct", bond_model), ("bond_hrps", bond_model)])
    out = {}
    for label, model in models:
        res = run_system(task, model, label, budget)
        out[label] = {
            "solved": res.episode.solved,
            "joint_demo_exact": res.episode.joint_demo_exact,
            "pass2": res.episode.pass2,
            "programs": res.episode.programs,
            "saw_underconstraint_flags": res.saw_underconstraint_flags,
            "committed_underconstrained": res.committed_underconstrained,
            "n_hypothesis_rejections": res.n_hypothesis_rejections,
            "recognized_underconstraint": bool(
                res.saw_underconstraint_flags or res.n_hypothesis_rejections
            )
            and not res.committed_underconstrained,
        }
    return {"task_id": "aabf363d", "note": "training-split diagnostic, not holdout accuracy", "systems": out}


def _as_runner_budget(budget: object) -> RunnerBudget:
    if isinstance(budget, RunnerBudget):
        return budget
    if budget is None:
        return RunnerBudget()
    return RunnerBudget(
        temperature=float(getattr(budget, "temperature", 0.0) or 0.0),
        max_tokens_per_call=int(getattr(budget, "max_tokens", None) or getattr(budget, "max_tokens_per_call", 256)),
        max_model_calls=int(getattr(budget, "max_calls", None) or getattr(budget, "max_model_calls", 8)),
        max_seconds=float(getattr(budget, "max_seconds", 30.0) or 30.0),
        max_program_depth=int(getattr(budget, "max_program_depth", 3) or 3),
        seed=int(getattr(budget, "seed", 0) or 0),
    )


def run_bond_eval(
    tasks: list[ArcTask],
    *,
    base_model: FrozenOpenModel,
    bond_model: Optional[FrozenOpenModel],
    budget: Optional[RunnerBudget] = None,
    out_dir: Optional[Path] = None,
    include_aabf_probe: bool = True,
) -> dict[str, Any]:
    budget = _as_runner_budget(budget)
    out_dir = Path(out_dir) if out_dir is not None else BOND_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by: dict[str, list[dict[str, Any]]] = {s: [] for s in SYSTEMS}
    flat: list[dict[str, Any]] = []
    jsonl = out_dir / "runs.jsonl"
    traj = out_dir / "trajectories.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh, traj.open("w", encoding="utf-8") as th:
        for task in tasks:
            pairs = [
                ("base_direct", base_model),
                ("base_hrps", base_model),
            ]
            if bond_model is not None:
                pairs.extend([("bond_direct", bond_model), ("bond_hrps", bond_model)])
            for label, model in pairs:
                res = run_system(task, model, label, budget)
                row = res.as_dict()
                rows_by[label].append(row)
                flat.append(row)
                fh.write(json.dumps({k: v for k, v in row.items() if k != "interactions"}) + "\n")
                th.write(json.dumps({"system": label, "task_id": task.task_id, "interactions": row.get("interactions")}) + "\n")
                fh.flush()
                print(
                    f"[{label}] {task.task_id} solved={res.episode.solved} "
                    f"term={res.termination} calls={res.episode.n_model_calls} "
                    f"t={res.episode.wall_clock:.3f}s",
                    flush=True,
                )
    _write_csv(out_dir / "runs.csv", flat)
    summaries = {k: _summarize_runner_rows(v) for k, v in rows_by.items() if v}
    paired = []
    by_task: dict[str, dict[str, Any]] = {}
    for r in flat:
        by_task.setdefault(r["task_id"], {})[r["system"]] = {
            "solved": r.get("solved"),
            "joint_demo_exact": r.get("joint_demo_exact"),
            "termination": r.get("termination"),
        }
    for tid, rec in sorted(by_task.items()):
        paired.append({"task_id": tid, **rec})
    aabf = probe_aabf363d(base_model, bond_model, budget) if include_aabf_probe else {}
    report = {
        "systems": list(SYSTEMS),
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "budget": {
            "temperature": budget.temperature,
            "top_p": budget.top_p,
            "seed": budget.seed,
            "max_tokens_per_call": budget.max_tokens_per_call,
            "max_model_calls": budget.max_model_calls,
            "max_total_tokens": budget.max_total_tokens,
            "max_seconds": budget.max_seconds,
        },
        "base_model": getattr(base_model, "name", ""),
        "bond_model": getattr(bond_model, "name", None) if bond_model else None,
        "bond_present": bond_model is not None,
        "adapter_trained": bool(bond_model is not None and getattr(bond_model, "backend", "") in {"hf_bond", "fake_adapter"}),
        "summaries": summaries,
        "deltas": bond_deltas(summaries),
        "paired_outcomes": paired,
        "aabf363d_probe": aabf,
        "is_final_bond": False,
        "honesty": (
            "Scores here are not a learned Bond result unless adapter weights were "
            "trained and bond_direct is compared to base_direct."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Bond: generate episodes, train adapter, evaluate")
    p.add_argument(
        "command",
        choices=("generate", "train", "eval", "smoke", "all", "package", "merge", "bundle"),
    )
    p.add_argument("--model", type=str, default=None, help="HF id override")
    p.add_argument("--foundation", type=str, default="qwen1.5b_smoke", help="qwen1.5b_smoke | qwen3.5_4b | inkling_small")
    p.add_argument("--adapter", type=str, default=str(BOND_ADAPTER_DIR))
    p.add_argument("--episodes", type=str, default=str(BOND_DIR / "episodes.jsonl"))
    p.add_argument("--output-dir", dest="out_dir", type=str, default=str(BOND_DIR))
    p.add_argument("--holdout-spec", type=str, default="training[400:440]")
    p.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--abstractions", type=str, default=str(DEFAULT_H_PATH))
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--scale", choices=("smoke", "train"), default="smoke")
    p.add_argument("--foundation-dir", type=str, default="")
    p.add_argument("--merge-out", type=str, default=str(REPO_ROOT / "models" / "bond_merged"))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = resolve_foundation(args.foundation)
    model_name = args.model or spec["hf_id"]
    if args.model and args.model != spec["hf_id"] and args.foundation != "custom":
        print(
            f"refusing silent foundation swap: --foundation {args.foundation} is {spec['hf_id']}, "
            f"--model is {args.model}. Pass --foundation custom to use an unregistered id.",
            flush=True,
        )
        return 2
    library = AbstractionLibrary()
    abs_path = Path(args.abstractions)
    if abs_path.is_file():
        library = load_library_json(abs_path)
    held = held_out_training_ids(offset=args.offset, n=args.n)
    notes: list[str] = []
    episode_summary: dict[str, Any] = {}

    if args.command in {"package", "merge", "bundle"}:
        from src.hrps.package import merge_bond_checkpoint, write_bond_package, write_remote_train_bundle

        if args.command == "package":
            write_bond_package(
                out_dir / "release",
                adapter_dir=Path(args.adapter) if Path(args.adapter).exists() else None,
                manifest={"public_name": "Bond", "foundation": spec, "holdout": held},
            )
            print(f"wrote Bond package under {out_dir / 'release'}", flush=True)
            return 0
        if args.command == "merge":
            rec = merge_bond_checkpoint(
                Path(args.foundation_dir or "models/foundation"),
                Path(args.adapter),
                Path(args.merge_out),
            )
            print(json.dumps(rec, indent=2), flush=True)
            return 0 if rec.get("status") == "ok" else 2
        write_remote_train_bundle(
            out_dir / "remote_train_bundle",
            episodes_path=Path(args.episodes),
            holdout_ids=held,
            command={
                "foundation": "qwen3.5_4b",
                "adapter": "models/bond_qwen35_4b",
                "seed": args.seed,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "max_seq_length": args.max_seq_length,
                "holdout_spec": args.holdout_spec,
            },
        )
        print(f"wrote remote training bundle under {out_dir / 'remote_train_bundle'}", flush=True)
        return 0

    if args.command in {"generate"} or (args.command == "train" and args.regenerate):
        if args.scale == "train":
            dest = out_dir / "train_scale"
            episodes, episode_summary = generate_and_write(
                dest, library=library, max_strategy_episodes=200
            )
            notes.append(
                f"train-scale generated {len(episodes)} episodes; smoke 62-set was not overwritten"
            )
        else:
            pin = out_dir / "DATASET_HASH.txt"
            if pin.is_file() and not args.regenerate:
                notes.append("refusing to overwrite pinned 62-episode smoke dataset; pass --regenerate")
                episode_summary = {
                    "n_episodes": 62,
                    "episodes_sha256": pin.read_text(encoding="utf-8").strip(),
                    "pinned": True,
                }
            else:
                episodes, episode_summary = generate_and_write(out_dir, library=library)
                notes.append(f"generated {len(episodes)} smoke episodes excluding {len(held)} held-out ids")
        print(json.dumps(episode_summary.get("kinds"), indent=2), flush=True)
    elif (out_dir / "episodes.jsonl").is_file():
        episode_summary = {
            "n_episodes": sum(1 for line in (out_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()),
            "episodes_sha256": file_sha256(out_dir / "episodes.jsonl"),
            "sft_sha256": file_sha256(out_dir / "sft.jsonl") if (out_dir / "sft.jsonl").is_file() else None,
            "sft_actions_sha256": file_sha256(out_dir / "sft_actions.jsonl") if (out_dir / "sft_actions.jsonl").is_file() else None,
        }
        notes.append("using existing episodes; not regenerated")

    cfg = dict(DEFAULT_TRAIN_CONFIG)
    cfg.update(
        {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.epochs,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "holdout_spec": args.holdout_spec,
            "foundation_id": spec["id"],
            "hf_id": model_name,
        }
    )

    adapter_status: dict[str, Any] = {"status": "skipped"}
    if args.command in {"train", "all"}:
        sft = out_dir / "sft_actions.jsonl"
        if not sft.is_file():
            sft = out_dir / "sft.jsonl"
        adapter_status = train_bond_adapter(
            sft,
            model_name=model_name,
            output_dir=Path(args.adapter),
            config=cfg,
            foundation_id=spec["id"],
        )
        notes.append(f"adapter_train={adapter_status.get('status')}")
        if adapter_status.get("status") != "ok":
            notes.append(str(adapter_status.get("reason")))
            print(json.dumps(adapter_status, indent=2), flush=True)

    if args.command in {"eval", "smoke", "all"}:
        budget = RunnerBudget(seed=args.seed)
        if args.command == "smoke":
            # Scripted four-way path: no torch required.
            from src.hrps.backend import load_scripted_adapter, save_scripted_adapter
            from src.hrps.task import parse_task

            task = parse_task(
                "synth_rot180",
                {
                    "train": [
                        {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                        {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
                    ],
                    "test": [{"input": [[1, 0], [0, 2]], "output": [[2, 0], [0, 1]]}],
                },
                "training",
            )
            base = ScriptedModel(
                responses=["1 0\n0 2", "1 0\n0 2"]
                + [
                    json.dumps({"action": "execute_program", "arguments": {"program": "rot90"}}),
                    json.dumps({"action": "commit_candidates", "arguments": {"program": "rot90"}}),
                ]
            )
            adapter_dir = out_dir / "scripted_adapter"
            save_scripted_adapter(
                [
                    "2 0\n0 1",
                    "2 0\n0 1",
                    json.dumps({"action": "revise_hypothesis", "arguments": {"text": "rotate 180"}}),
                    json.dumps({"action": "inspect_objects", "arguments": {}}),
                    json.dumps({"action": "execute_program", "arguments": {"program": "rot180"}}),
                    json.dumps({"action": "commit_candidates", "arguments": {"program": "rot180"}}),
                ],
                adapter_dir,
            )
            # Reload to prove save/load.
            bond = load_scripted_adapter(adapter_dir, name="bond_scripted")
            report = run_bond_eval(
                [task],
                base_model=base,
                bond_model=bond,
                budget=budget,
                out_dir=out_dir / "eval_smoke",
                include_aabf_probe=False,
            )
            print(json.dumps(report["deltas"], indent=2), flush=True)
            notes.append("smoke four-way eval used scripted adapter; not a learned Bond result")
        else:
            base, status, _ = load_backend(spec["id"], seed=args.seed, adapter_path=None, as_bond=False)
            bond = None
            if adapter_status.get("status") == "ok" or Path(args.adapter).exists():
                bond, bstatus, _ = load_backend(
                    spec["id"], seed=args.seed, adapter_path=args.adapter, as_bond=True
                )
                if bond is None:
                    notes.append(bstatus)
                    print(bstatus, flush=True)
            if base is None:
                notes.append(f"eval_blocked={status}")
                print(f"eval blocked: {status}", flush=True)
            else:
                tasks = load_held_out_tasks(offset=args.offset, n=args.n)
                eval_dir = out_dir if out_dir.resolve() != BOND_DIR.resolve() else out_dir / "eval"
                report = run_bond_eval(
                    tasks,
                    base_model=base,
                    bond_model=bond,
                    budget=budget,
                    out_dir=eval_dir,
                )
                print(json.dumps(report["deltas"], indent=2), flush=True)

    status = "episodes_ready"
    if adapter_status.get("status") == "ok":
        status = "adapter_trained"
    elif adapter_status.get("status") == "blocked":
        status = "episodes_ready_adapter_blocked"
    manifest = bond_manifest(
        foundation=spec,
        adapter_dir=Path(args.adapter),
        train_config=cfg,
        episode_summary=episode_summary,
        held_out_ids=held,
        status=status,
        notes=notes,
        losses=adapter_status.get("losses") if isinstance(adapter_status, dict) else None,
    )
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Bond manifest status={status} artifact_class={manifest['artifact_class']}", flush=True)
    if args.command == "generate":
        return 0
    if args.command == "smoke":
        return 0
    if status == "episodes_ready_adapter_blocked" and args.command in {"train", "all"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
