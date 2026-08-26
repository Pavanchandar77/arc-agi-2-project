"""Lightning / remote GPU Bond-4B trainer.

Foundation is Qwen/Qwen3.5-4B only. Builds a Bond-L1 curriculum, merges
it with the holdout-clean train_scale SFT, then trains the Bond adapter.

Do not run this on the Iris Xe laptop. Do not swap the foundation.
Do not write these weights to a 0.5B/1.5B directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.hrps.backend import probe_hardware, resolve_foundation
from src.hrps.bond import train_bond_adapter
from src.hrps.bond_curriculum import CURRICULUM_DIR, build_curriculum, merge_sft
from src.hrps.identity import PUBLIC_NAME
from src.hrps.language import FOUNDATION_HF_ID, LANGUAGE_ID

PINNED = {
    "public_name": PUBLIC_NAME,
    "language": LANGUAGE_ID,
    "foundation": "qwen3.5_4b",
    "hf_id": FOUNDATION_HF_ID,
    "adapter": str(REPO / "models" / "bond_qwen35_4b"),
    "train_scale": str(REPO / "artifacts" / "bond" / "train_scale" / "sft_actions.jsonl"),
    "merged_sft": str(REPO / "artifacts" / "bond" / "curriculum" / "sft_merged.jsonl"),
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


def main() -> int:
    spec = resolve_foundation(PINNED["foundation"])
    if spec["hf_id"] != PINNED["hf_id"]:
        print(json.dumps({"status": "blocked", "reason": "foundation_id_mismatch", "pinned": PINNED["hf_id"]}))
        return 2
    print(json.dumps({"phase": "curriculum", "n": PINNED["curriculum_n"], "language": LANGUAGE_ID}, indent=2))
    cur = build_curriculum(PINNED["curriculum_n"], seed=PINNED["seed"], search_verify=True)
    print(json.dumps({"curriculum": {k: cur.get(k) for k in ("n_episodes", "n_sft", "n_synthetic_tasks", "language")}}, indent=2))
    merged = merge_sft(
        [
            Path(PINNED["train_scale"]),
            CURRICULUM_DIR / "sft_actions.jsonl",
        ],
        Path(PINNED["merged_sft"]),
    )
    print(json.dumps({"merged": merged}, indent=2))
    hw = probe_hardware()
    if not hw["cuda"]:
        rec = {
            "status": "blocked",
            "reason": (
                f"hardware_blocked: {PINNED['hf_id']} needs CUDA. "
                "Curriculum is written. Train on Lightning with this same script."
            ),
            "hardware": hw,
            "pinned": PINNED,
            "is_final_bond": False,
            "public_name": PUBLIC_NAME,
        }
        print(json.dumps(rec, indent=2))
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
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
