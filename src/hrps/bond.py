"""Bond: an actively trained open reasoning model-system.

Bond is not a passive wrapper around Inkling-Small. It is:

  Inkling-Small (or the local Qwen alternative)
    + HRPS-generated reasoning episodes
    + learned LoRA/QLoRA adapter
    + active HRPS inference loop

HRPS remains the executable substrate. Bond's adapter improves decisions
inside that substrate. The verifier is exact feedback, not the answer.

Training uses only official training tasks, excluding the held-out
diagnostic slice. Public evaluation is never used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.hrps.abstractions import AbstractionLibrary, load_library_json
from src.hrps.agent import ElevationBudget, run_episode
from src.hrps.elevation import (
    DEFAULT_H_PATH,
    REPO_ROOT,
    _summarize_condition,
    load_held_out_tasks,
)
from src.hrps.episodes import (
    ACTION_SCHEMA,
    BondEpisode,
    assert_training_safe,
    generate_bond_episodes,
    write_episodes,
)
from src.hrps.model import (
    LOCAL_DEFAULT,
    PREFERRED_INKLING,
    FrozenOpenModel,
    resolve_model_name,
    try_load_open_model,
)
from src.hrps.separability import DEFAULT_N, DEFAULT_OFFSET, held_out_training_ids
from src.hrps.task import ArcTask

BOND_DIR = REPO_ROOT / "artifacts" / "bond"
BOND_ADAPTER_DIR = REPO_ROOT / "models" / "bond_adapter"
BOND_VERSION = "bond-v0"

SYSTEMS = ("base_direct", "base_hrps", "bond_direct", "bond_hrps")

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


def base_checkpoint_record(model_name: str) -> dict[str, Any]:
    return {
        "preferred_foundation": PREFERRED_INKLING,
        "local_foundation": LOCAL_DEFAULT,
        "resolved_foundation": model_name,
        "weights_hash": None,
        "hash_note": "hash is filled after the base checkpoint is materialized on disk",
        "inkling_local": False,
    }


def bond_manifest(
    *,
    model_name: str,
    adapter_dir: Optional[Path],
    train_config: dict[str, Any],
    episode_summary: dict[str, Any],
    held_out_ids: list[str],
    status: str,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "name": "Bond",
        "version": BOND_VERSION,
        "status": status,
        "thesis": (
            "Bond is an actively trained open reasoning model-system whose parameters "
            "are improved through verified HRPS reasoning experience, and whose inference "
            "is amplified by the same executable cognitive substrate."
        ),
        "foundation": base_checkpoint_record(model_name),
        "adapter": {
            "path": str(adapter_dir) if adapter_dir else None,
            "present": bool(adapter_dir and Path(adapter_dir).exists() and any(Path(adapter_dir).iterdir()) if adapter_dir and Path(adapter_dir).exists() else False),
            "kind": "lora",
        },
        "training": train_config,
        "data_provenance": {
            "split": "training",
            "held_out_excluded": held_out_ids,
            "public_evaluation_used": False,
            "episodes": episode_summary,
        },
        "schemas": {
            "actions": ACTION_SCHEMA,
            "inference_controller": "src.hrps.agent.run_episode M0/M2",
            "executor": "src.hrps.dsl.replay",
            "verifier": "src.hrps.residual.joint_residual + gold_free_constraint_feedback",
        },
        "notes": notes,
    }


def generate_and_write(
    out_dir: Path = BOND_DIR,
    library: Optional[AbstractionLibrary] = None,
) -> tuple[list[BondEpisode], dict[str, Any]]:
    held = held_out_training_ids()
    episodes = generate_bond_episodes(library=library)
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
) -> dict[str, Any]:
    """Train Bond LoRA on HRPS episode conversations. Requires torch + GPU."""
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if config:
        cfg.update(config)
    try:
        import torch  # noqa: F401
    except Exception:
        return {"status": "blocked", "reason": "no_torch", "output_dir": str(output_dir)}
    if not sft_path.is_file():
        return {"status": "blocked", "reason": "missing_sft", "output_dir": str(output_dir)}
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
    return {"status": "ok", "output_dir": str(output_dir), "config": cfg}


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
        "base_direct": base_d,
        "base_hrps": base_h,
        "bond_direct": bond_d,
        "bond_hrps": bond_h,
        "delta_train_bond_direct_minus_base_direct": d_train,
        "delta_substrate_bond_hrps_minus_bond_direct": d_sub,
        "delta_harness_base_hrps_minus_base_direct": harness_only,
        "delta_full_bond_hrps_minus_base_direct": d_full,
        "criterion": (
            "Bond succeeds when Bond+HRPS solves more held-out tasks than the "
            "unchanged foundation under matched information and compute. Strong "
            "evidence also requires Bond direct > foundation direct (learned) and "
            "Bond+HRPS > Bond direct (substrate)."
        ),
    }


def run_bond_eval(
    tasks: list[ArcTask],
    *,
    base_model: FrozenOpenModel,
    bond_model: Optional[FrozenOpenModel],
    budget: Optional[ElevationBudget] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Any]:
    budget = budget or ElevationBudget()
    out_dir = Path(out_dir) if out_dir is not None else BOND_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by: dict[str, list[dict[str, Any]]] = {s: [] for s in SYSTEMS}
    jsonl = out_dir / "runs.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for task in tasks:
            pairs = [
                ("base_direct", base_model, "M0"),
                ("base_hrps", base_model, "M2"),
            ]
            if bond_model is not None:
                pairs.extend(
                    [
                        ("bond_direct", bond_model, "M0"),
                        ("bond_hrps", bond_model, "M2"),
                    ]
                )
            for label, model, cond in pairs:
                ep = run_episode(task, model, cond, budget)
                row = ep.as_dict()
                row["system"] = label
                rows_by[label].append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"[{label}] {task.task_id} solved={ep.solved} fail={ep.failure} "
                    f"calls={ep.n_model_calls} t={ep.wall_clock:.3f}s",
                    flush=True,
                )
    summaries = {k: _summarize_condition(v) for k, v in rows_by.items() if v}
    report = {
        "systems": list(SYSTEMS),
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "budget": {
            "temperature": budget.temperature,
            "max_tokens": budget.max_tokens,
            "max_calls": budget.max_calls,
            "max_seconds": budget.max_seconds,
        },
        "base_model": getattr(base_model, "name", ""),
        "bond_model": getattr(bond_model, "name", None) if bond_model else None,
        "summaries": summaries,
        "deltas": bond_deltas(summaries),
        "bond_present": bond_model is not None,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Bond: generate episodes, train adapter, evaluate")
    p.add_argument("command", choices=("generate", "train", "eval", "all"))
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--adapter", type=str, default=str(BOND_ADAPTER_DIR))
    p.add_argument("--out-dir", type=str, default=str(BOND_DIR))
    p.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--abstractions", type=str, default=str(DEFAULT_H_PATH))
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model_name(args.model)
    library = AbstractionLibrary()
    abs_path = Path(args.abstractions)
    if abs_path.is_file():
        library = load_library_json(abs_path)
    held = held_out_training_ids(offset=args.offset, n=args.n)
    notes: list[str] = []

    episode_summary: dict[str, Any] = {}
    if args.command in {"generate", "all", "train"}:
        episodes, episode_summary = generate_and_write(out_dir, library=library)
        notes.append(f"generated {len(episodes)} episodes excluding {len(held)} held-out ids")
        print(json.dumps(episode_summary["kinds"], indent=2), flush=True)

    adapter_status = {"status": "skipped"}
    if args.command in {"train", "all"}:
        adapter_status = train_bond_adapter(
            out_dir / "sft.jsonl",
            model_name=model_name,
            output_dir=Path(args.adapter),
        )
        notes.append(f"adapter_train={adapter_status.get('status')}")
        if adapter_status.get("status") != "ok":
            notes.append(str(adapter_status.get("reason")))

    if args.command in {"eval", "all"}:
        base, status = try_load_open_model(model_name)
        bond = None
        if adapter_status.get("status") == "ok" or Path(args.adapter).exists():
            bond, bstatus = try_load_open_model(model_name, adapter_path=args.adapter)
            if bond is None:
                notes.append(f"bond_load={bstatus}")
        if base is None:
            notes.append(f"eval_blocked={status}")
            print(f"eval blocked: {status}", flush=True)
        else:
            tasks = load_held_out_tasks(offset=args.offset, n=args.n)
            report = run_bond_eval(tasks, base_model=base, bond_model=bond, out_dir=out_dir / "eval")
            print(json.dumps(report["deltas"], indent=2), flush=True)

    status = "episodes_ready"
    if adapter_status.get("status") == "ok":
        status = "adapter_trained"
    elif adapter_status.get("status") == "blocked":
        status = "episodes_ready_adapter_blocked"
    manifest = bond_manifest(
        model_name=model_name,
        adapter_dir=Path(args.adapter),
        train_config=DEFAULT_TRAIN_CONFIG,
        episode_summary=episode_summary,
        held_out_ids=held,
        status=status,
        notes=notes,
    )
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Bond manifest status={status}", flush=True)
    return 0 if status != "episodes_ready_adapter_blocked" or args.command == "generate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
