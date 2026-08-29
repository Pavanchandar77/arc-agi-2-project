"""CUDA-only Stage 1 launcher. Pinned config. Never swaps the foundation.

Run this on an NVIDIA GPU machine (24–48 GB VRAM practical).
This file must not be used to train Qwen2.5-1.5B and still write
models/bond_qwen3_4b.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.hrps.backend import probe_hardware, resolve_foundation
from src.hrps.bond import train_bond_adapter

PINNED = {
    "foundation": "qwen3.5_4b",
    "hf_id": "Qwen/Qwen3-4B",
    "adapter": str(REPO / "models" / "bond_qwen3_4b"),
    "episodes": str(REPO / "artifacts" / "bond" / "train_scale" / "sft_actions.jsonl"),
    "seed": 42,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "epochs": 3,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "holdout_spec": "training[400:440]",
}


def main() -> int:
    spec = resolve_foundation(PINNED["foundation"])
    if spec["hf_id"] != PINNED["hf_id"]:
        print(json.dumps({"status": "blocked", "reason": "foundation_id_mismatch", "pinned": PINNED["hf_id"], "resolved": spec["hf_id"]}))
        return 2
    hw = probe_hardware()
    if not hw["cuda"]:
        rec = {
            "status": "blocked",
            "reason": (
                f"hardware_blocked: {PINNED['hf_id']} needs CUDA. "
                f"This host: {hw}. Do not silently switch the foundation."
            ),
            "output_dir": PINNED["adapter"],
            "is_final_bond": False,
            "pinned": PINNED,
            "hardware": hw,
        }
        print(json.dumps(rec, indent=2))
        (REPO / "artifacts" / "bond" / "LAST_BLOCK.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return 2
    sft = Path(PINNED["episodes"])
    if not sft.is_file():
        print(json.dumps({"status": "blocked", "reason": "missing_sft", "path": str(sft)}))
        return 2
    result = train_bond_adapter(
        sft,
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
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
