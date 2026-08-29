"""Laptop Bond LoRA: CPU only, small Qwen, not Qwen3-4B.

Close Discord (~4GB) and Chrome (~3GB) first. Needs ~3GB+ free RAM
and a few GB disk for torch + the 0.5B weights.

  pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu
  pip install transformers peft trl datasets accelerate
  python scripts/train_bond_cpu.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.hrps.backend import free_ram_gate, probe_hardware, resolve_foundation
from src.hrps.bond import CPU_TRAIN_CONFIG, train_bond_adapter

PINNED = {
    "foundation": "qwen05b_cpu",
    "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "adapter": str(REPO / "models" / "bond_qwen05b_cpu"),
    "episodes": str(REPO / "artifacts" / "bond" / "sft_actions.jsonl"),
    **CPU_TRAIN_CONFIG,
}


def main() -> int:
    spec = resolve_foundation(PINNED["foundation"])
    hw = probe_hardware()
    print(json.dumps({"hardware": hw, "pinned": {"hf_id": PINNED["hf_id"], "device": "cpu"}}, indent=2))
    ram_block = free_ram_gate(spec)
    if ram_block:
        print(json.dumps({"status": "blocked", "reason": ram_block, "is_final_bond": False}, indent=2))
        return 2
    sft = Path(PINNED["episodes"])
    if not sft.is_file():
        sft = REPO / "artifacts" / "bond" / "sft.jsonl"
    result = train_bond_adapter(
        sft,
        model_name=PINNED["hf_id"],
        output_dir=Path(PINNED["adapter"]),
        foundation_id=PINNED["foundation"],
        config=dict(CPU_TRAIN_CONFIG),
    )
    result["is_final_bond"] = False
    result["artifact_class"] = "cpu_laptop_not_4b"
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
