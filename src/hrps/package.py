"""Bond release package, merge (never overwrite foundation), remote train bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from src.hrps.episodes import file_sha256
from src.hrps.identity import PUBLIC_NAME, QWEN_LICENSE_NOTE, adapter_dir_hash
from src.hrps.schema import ACTION_SCHEMA as JSON_ACTION_SCHEMA
from src.hrps.schema import BOND_ACTIONS

LAYOUT = {
    "public_name": PUBLIC_NAME,
    "bond_model": "learned adapter or merged weights; foundation checkpoint stays separate and must not be deleted",
    "bond_controller": "src.hrps.runner.run_system",
    "bond_interface": "src.hrps.schema",
    "bond_hrps": "src.hrps.env + src.hrps.representation",
    "bond_executor": "src.hrps.dsl.replay",
    "bond_verifier": "src.hrps.residual.joint_residual",
    "bond_manifest": "bond_manifest.json",
}


def merge_bond_checkpoint(
    foundation_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    *,
    internal_name: str = "Bond-Qwen35-4B-merged",
) -> dict[str, Any]:
    """Merge adapter into a *copy* of the foundation. Never overwrite the foundation."""
    foundation_dir = Path(foundation_dir)
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    if output_dir.resolve() == foundation_dir.resolve():
        return {
            "status": "blocked",
            "reason": "refusing to overwrite the foundation checkpoint",
            "foundation_dir": str(foundation_dir),
        }
    if not adapter_dir.exists():
        return {"status": "blocked", "reason": "Bond adapter not found", "adapter_dir": str(adapter_dir)}
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # noqa: F401
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        # Preserve a non-destructive copy of the adapter for the package.
        dest = output_dir / "bond_model"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(adapter_dir, dest)
        rec = {
            "status": "blocked",
            "reason": "no_torch",
            "public_name": PUBLIC_NAME,
            "internal_name": internal_name,
            "overwrote_foundation": False,
            "copied_adapter_to": str(dest),
            "note": "Merge on a GPU machine with peft. Foundation directory was not modified.",
        }
        (output_dir / "MERGE.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec

    tokenizer = AutoTokenizer.from_pretrained(str(foundation_dir), trust_remote_code=True, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(foundation_dir), trust_remote_code=True, local_files_only=True)
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()
    dest = output_dir / "bond_model"
    dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(dest))
    tokenizer.save_pretrained(str(dest))
    rec = {
        "status": "ok",
        "public_name": PUBLIC_NAME,
        "internal_name": internal_name,
        "overwrote_foundation": False,
        "merged_dir": str(dest),
        "foundation_dir": str(foundation_dir),
        "adapter_hash": adapter_dir_hash(adapter_dir),
    }
    (output_dir / "MERGE.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def write_bond_package(
    out_dir: Path,
    *,
    adapter_dir: Optional[Path] = None,
    manifest: Optional[dict[str, Any]] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bond_controller").mkdir(exist_ok=True)
    (out_dir / "bond_interface").mkdir(exist_ok=True)
    (out_dir / "bond_hrps").mkdir(exist_ok=True)
    (out_dir / "bond_executor").mkdir(exist_ok=True)
    (out_dir / "bond_verifier").mkdir(exist_ok=True)
    (out_dir / "bond_model").mkdir(exist_ok=True)
    (out_dir / "LAYOUT.json").write_text(json.dumps(LAYOUT, indent=2), encoding="utf-8")
    (out_dir / "bond_interface" / "schema.json").write_text(
        json.dumps({"actions": list(BOND_ACTIONS), "schema": JSON_ACTION_SCHEMA}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "bond_controller" / "README.txt").write_text(
        "Active Bond reasoning loop: src.hrps.runner.run_system\n"
        "Bond observes, hypothesizes, acts through HRPS, interprets residuals, revises, commits.\n",
        encoding="utf-8",
    )
    (out_dir / "bond_executor" / "README.txt").write_text(
        "Exact DSL executor: src.hrps.dsl.replay\n", encoding="utf-8"
    )
    (out_dir / "bond_verifier" / "README.txt").write_text(
        "Exact verifier: src.hrps.residual.joint_residual + gold_free_constraint_feedback\n",
        encoding="utf-8",
    )
    (out_dir / "bond_hrps" / "README.txt").write_text(
        "HRPS substrate: src.hrps.env representations, state, residuals.\n"
        "Not the semantic answer source.\n",
        encoding="utf-8",
    )
    (out_dir / "LICENSE_ATTRIBUTION.txt").write_text(QWEN_LICENSE_NOTE + "\n", encoding="utf-8")
    if adapter_dir and Path(adapter_dir).exists():
        dest = out_dir / "bond_model" / "adapter"
        if dest.exists():
            shutil.rmtree(dest)
        if Path(adapter_dir).is_dir():
            shutil.copytree(adapter_dir, dest)
        else:
            shutil.copy2(adapter_dir, dest)
    payload = dict(manifest or {})
    payload.setdefault("public_name", PUBLIC_NAME)
    payload.setdefault("layout", LAYOUT)
    payload.setdefault("license", {"note": QWEN_LICENSE_NOTE})
    payload["adapter_hash"] = adapter_dir_hash(Path(adapter_dir)) if adapter_dir else None
    (out_dir / "bond_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_dir


def write_remote_train_bundle(
    out_dir: Path,
    *,
    episodes_path: Path,
    holdout_ids: list[str],
    command: dict[str, Any],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = {
        "public_name": PUBLIC_NAME,
        "internal_artifact": "Bond-Qwen35-4B-adapter",
        "foundation_hf_id": "Qwen/Qwen3.5-4B",
        "command": command,
        "holdout_spec": "training[400:440]",
        "holdout_ids": holdout_ids,
        "episodes_path": str(episodes_path),
        "episodes_sha256": file_sha256(episodes_path) if Path(episodes_path).is_file() else None,
        "do_not_train_locally_on_8gb_cpu": True,
        "do_not_overwrite_foundation": True,
        "note": "Recorded configuration. Do not silently change it at train time.",
    }
    (out_dir / "COMMAND.json").write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    (out_dir / "holdout.json").write_text(json.dumps({"ids": holdout_ids}, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "Stage 1 remote Bond adapter training (GPU).\n"
        "Public identity: Bond\n"
        "Internal artifact: Bond-Qwen35-4B-adapter\n"
        "Foundation stays on disk; merge writes a new directory.\n\n"
        "python -m src.hrps.bond train --foundation qwen3.5_4b "
        "--adapter models/bond_qwen35_4b --seed 42 --lora-r 16 --lora-alpha 32 "
        "--lora-dropout 0.05 --epochs 3 --learning-rate 2e-4 --max-seq-length 2048 "
        "--holdout-spec training[400:440]\n",
        encoding="utf-8",
    )
    return out_dir
