"""Lightning / Kaggle Bond-4B trainer. Native-precision LoRA on Qwen/Qwen3-4B.

Does not regenerate Bond-L1 curriculum when a holdout-clean artifact already
exists. Pass --regenerate to rebuild. Does not swap the foundation.

  python scripts/train_bond_lightning.py
  python scripts/train_bond_lightning.py --regenerate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.hrps.backend import probe_hardware, resolve_foundation
from src.hrps.bond import train_bond_adapter
from src.hrps.bond_curriculum import (
    CURRICULUM_DIR,
    MERGED_SFT,
    TRAIN_SCALE_SFT,
    build_curriculum,
    curriculum_artifacts_ready,
    merge_sft,
    verify_holdout_clean,
)
from src.hrps.identity import PUBLIC_NAME, adapter_is_complete
from src.hrps.language import FOUNDATION_HF_ID, LANGUAGE_ID

PINNED = {
    "public_name": PUBLIC_NAME,
    "language": LANGUAGE_ID,
    "foundation": "qwen3.5_4b",
    "hf_id": FOUNDATION_HF_ID,
    "adapter": str(REPO / "models" / "bond_qwen3_4b"),
    "train_scale": str(TRAIN_SCALE_SFT),
    "merged_sft": str(MERGED_SFT),
    "seed": 42,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "epochs": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "holdout_spec": "training[400:440]",
    "curriculum_n": 256,
}


def prepare_sft(*, regenerate: bool = False) -> dict:
    train_scale = Path(PINNED["train_scale"])
    if not train_scale.is_file():
        return {"status": "blocked", "reason": "missing_train_scale", "path": str(train_scale)}
    ready = curriculum_artifacts_ready(
        curriculum_dir=CURRICULUM_DIR,
        merged_path=Path(PINNED["merged_sft"]),
        train_scale=train_scale,
    )
    if ready.get("ready") and not regenerate:
        hold = ready["holdout"]
        print(json.dumps({"phase": "curriculum", "action": "reused", **ready}, indent=2))
        return {"status": "ok", "action": "reused", "holdout": hold, "merged": PINNED["merged_sft"]}
    print(
        json.dumps(
            {
                "phase": "curriculum",
                "action": "build" if regenerate or not ready.get("ready") else "reused",
                "regenerate": regenerate,
                "prior": ready,
                "n": PINNED["curriculum_n"],
                "language": LANGUAGE_ID,
            },
            indent=2,
        )
    )
    cur = build_curriculum(PINNED["curriculum_n"], seed=PINNED["seed"], search_verify=True)
    print(json.dumps({"curriculum": {k: cur.get(k) for k in ("n_episodes", "n_sft", "n_synthetic_tasks", "language")}}, indent=2))
    merged = merge_sft(
        [train_scale, CURRICULUM_DIR / "sft_actions.jsonl"],
        Path(PINNED["merged_sft"]),
    )
    hold = verify_holdout_clean(
        [train_scale, CURRICULUM_DIR / "sft_actions.jsonl", Path(PINNED["merged_sft"])]
    )
    if not hold["ok"]:
        return {"status": "blocked", "reason": "holdout_leak", "holdout": hold}
    print(json.dumps({"merged": merged, "holdout": hold}, indent=2))
    return {"status": "ok", "action": "built", "holdout": hold, "merged": PINNED["merged_sft"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bond-4B Lightning/Kaggle native-precision LoRA")
    p.add_argument(
        "--regenerate",
        action="store_true",
        help="Rebuild Bond-L1 curriculum even if a holdout-clean artifact exists.",
    )
    args = p.parse_args(argv)

    spec = resolve_foundation(PINNED["foundation"])
    if spec["hf_id"] != PINNED["hf_id"] or spec["hf_id"] != "Qwen/Qwen3-4B":
        print(json.dumps({"status": "blocked", "reason": "foundation_id_mismatch", "pinned": PINNED["hf_id"], "resolved": spec["hf_id"]}))
        return 2
    prep = prepare_sft(regenerate=args.regenerate)
    if prep.get("status") != "ok":
        print(json.dumps(prep, indent=2))
        return 2
    hw = probe_hardware()
    if not hw["cuda"]:
        rec = {
            "status": "blocked",
            "reason": (
                f"hardware_blocked: {PINNED['hf_id']} needs CUDA. "
                "Curriculum/SFT verified. Train on Kaggle/Lightning with this same script."
            ),
            "hardware": hw,
            "pinned": PINNED,
            "sft": prep,
            "is_final_bond": False,
            "public_name": PUBLIC_NAME,
            "experimental_note": (
                "Experimental Bond-Qwen35-4B native-precision LoRA adapter. Not AGI. Not a frontier result."
            ),
        }
        print(json.dumps(rec, indent=2))
        (REPO / "artifacts" / "bond").mkdir(parents=True, exist_ok=True)
        (REPO / "artifacts" / "bond" / "LAST_BLOCK.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return 2
    result = train_bond_adapter(
        Path(PINNED["merged_sft"]),
        model_name=PINNED["hf_id"],
        output_dir=Path(PINNED["adapter"]),
        foundation_id=PINNED["foundation"],
        config={
            "lora_r": PINNED["lora_r"],
            "lora_alpha": PINNED["lora_alpha"],
            "lora_dropout": PINNED["lora_dropout"],
            "num_train_epochs": PINNED["epochs"],
            "learning_rate": PINNED["learning_rate"],
            "max_seq_length": PINNED["max_seq_length"],
            "seed": PINNED["seed"],
        },
    )
    result["language"] = LANGUAGE_ID
    result["public_name"] = PUBLIC_NAME
    result["is_final_bond"] = False
    result["adapter_is_complete"] = adapter_is_complete(Path(PINNED["adapter"]))
    result["experimental_note"] = (
        "Experimental Bond-Qwen35-4B native-precision LoRA adapter. Not AGI. Not a frontier result. "
        "A learned Bond checkpoint requires adapter_config.json plus adapter weights, then four-way eval."
    )
    print(json.dumps(result, indent=2))
    if result.get("status") == "ok" and not result["adapter_is_complete"]:
        return 2
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
