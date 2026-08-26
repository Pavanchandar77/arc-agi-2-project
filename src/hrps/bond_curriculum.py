"""Build a holdout-clean Bond-4B curriculum from Bond-L1 synthesis + search.

Official evaluation is never read. Held-out training[400:440] is never
included. Synthetic task ids are synth_* and cannot collide with official
hashes used as holdout.

Kind: learned inventory of traces; env replay is exact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.hrps.elevation import REPO_ROOT
from src.hrps.episodes import BondEpisode, teacher_direct_episode, teacher_hrps_episode, write_episodes
from src.hrps.guided_search import guided_search
from src.hrps.language import FOUNDATION_HF_ID, LANGUAGE_ID, bond_l1_budget
from src.hrps.separability import held_out_training_ids
from src.hrps.synthesize import synthesize_batch

CURRICULUM_DIR = REPO_ROOT / "artifacts" / "bond" / "curriculum"
TRAIN_SCALE_SFT = REPO_ROOT / "artifacts" / "bond" / "train_scale" / "sft_actions.jsonl"
MERGED_SFT = CURRICULUM_DIR / "sft_merged.jsonl"


def _task_ids_from_jsonl(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.is_file():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = rec.get("task_id")
        if tid:
            ids.add(str(tid))
    return ids


def verify_holdout_clean(paths: list[Path]) -> dict[str, Any]:
    held = set(held_out_training_ids())
    ids: set[str] = set()
    for path in paths:
        ids |= _task_ids_from_jsonl(path)
    leak = sorted(ids & held)
    return {
        "ok": not leak,
        "n_ids": len(ids),
        "held_out_leaks": leak,
        "public_evaluation_used": False,
        "holdout_spec": "training[400:440]",
    }


def curriculum_artifacts_ready(
    *,
    curriculum_dir: Optional[Path] = None,
    merged_path: Optional[Path] = None,
    train_scale: Optional[Path] = None,
) -> dict[str, Any]:
    """Cheap check: files exist, holdout-clean, no eval split. Does not resynthesize."""
    curriculum_dir = Path(curriculum_dir) if curriculum_dir is not None else CURRICULUM_DIR
    merged_path = Path(merged_path) if merged_path is not None else MERGED_SFT
    train_scale = Path(train_scale) if train_scale is not None else TRAIN_SCALE_SFT
    sft = curriculum_dir / "sft_actions.jsonl"
    meta = curriculum_dir / "CURRICULUM.json"
    missing = [str(p) for p in (train_scale, sft, meta, merged_path) if not p.is_file()]
    rec: dict[str, Any] = {
        "ready": False,
        "missing": missing,
        "curriculum_dir": str(curriculum_dir),
        "merged": str(merged_path),
    }
    if missing:
        rec["reason"] = "missing_artifacts"
        return rec
    hold = verify_holdout_clean([sft, train_scale, merged_path])
    rec["holdout"] = hold
    if not hold["ok"]:
        rec["reason"] = "holdout_leak"
        return rec
    rec["ready"] = True
    rec["reason"] = "reused"
    rec["n_sft_merged"] = sum(1 for line in merged_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rec


def _episodes_from_batch(
    batch,
    *,
    search_verify: bool = True,
    seconds: float = 1.5,
) -> list[BondEpisode]:
    held = set(held_out_training_ids())
    out: list[BondEpisode] = []
    budget = bond_l1_budget(nodes=400, seconds=seconds, frontier=4000)
    for task, program in batch:
        if task.task_id in held or task.split == "evaluation":
            continue
        ser = program.serialize()
        if search_verify:
            g = guided_search(task, budget=budget)
            if not g.joint_demo_exact:
                continue
        ep = teacher_hrps_episode(
            task,
            ser,
            test_transfer=True,
            kind_hint="success_trajectory",
            include_competing=False,
        )
        if ep is None:
            continue
        ep.held_out = False
        out.append(ep)
        direct = teacher_direct_episode(task, ser, True)
        if direct is not None:
            direct.held_out = False
            out.append(direct)
    return out


def build_curriculum(
    n: int = 128,
    *,
    seed: int = 0,
    out_dir: Optional[Path] = None,
    search_verify: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir) if out_dir is not None else CURRICULUM_DIR
    batch = synthesize_batch(n, seed=seed)
    episodes = _episodes_from_batch(batch, search_verify=search_verify)
    summary = write_episodes(episodes, out_dir)
    summary.update(
        {
            "language": LANGUAGE_ID,
            "foundation_hf_id": FOUNDATION_HF_ID,
            "n_synthetic_tasks": len(batch),
            "held_out_excluded": True,
            "public_evaluation_used": False,
            "search_verify": search_verify,
        }
    )
    (out_dir / "CURRICULUM.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def merge_sft(paths: list[Path], dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as fh:
        for path in paths:
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    fh.write(line.strip() + "\n")
                    n += 1
    rec = {"n_sft": n, "sources": [str(p) for p in paths], "dest": str(dest), "foundation_hf_id": FOUNDATION_HF_ID}
    (dest.parent / "MERGED.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec
