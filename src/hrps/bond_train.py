"""Bond SFT/LoRA/QLoRA entry point.

  python -m src.hrps.bond_train --foundation Qwen/Qwen3.8-27B ...

SFT first. Do not mix GRPO/DPO into this baseline. Never overwrite the
foundation. Never download 27B onto the laptop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.hrps.backend import probe_hardware, resolve_foundation
from src.hrps.bond import BOND_DIR, train_bond_adapter
from src.hrps.elevation import REPO_ROOT
from src.hrps.identity import PUBLIC_NAME


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Bond SFT adapter training")
    p.add_argument("--foundation", type=str, default="Qwen/Qwen3.8-27B")
    p.add_argument("--episodes", type=str, default=str(BOND_DIR / "train_scale" / "sft_actions.jsonl"))
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "models" / "bond_qwen38_27b"))
    p.add_argument("--method", choices=("sft", "lora", "qlora"), default="qlora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--holdout-spec", type=str, default="training[400:440]")
    args = p.parse_args(argv)

    spec = resolve_foundation(args.foundation)
    hw = probe_hardware()
    print(
        json.dumps(
            {
                "public_name": PUBLIC_NAME,
                "foundation_hf_id": spec["hf_id"],
                "method": args.method,
                "hardware": hw,
                "is_final_bond": False,
            },
            indent=2,
        )
    )
    if spec.get("refuse_local_download") and not hw["cuda"]:
        rec = {
            "status": "blocked",
            "reason": (
                f"refuse_local_download: {spec['hf_id']} must be trained on a remote GPU. "
                "This laptop must not download the 27B checkpoint."
            ),
            "output_dir": args.output_dir,
            "is_final_bond": False,
            "pinned": {
                "foundation": spec["hf_id"],
                "method": args.method,
                "seed": args.seed,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "max_seq_length": args.max_seq_length,
                "holdout_spec": args.holdout_spec,
            },
        }
        print(json.dumps(rec, indent=2))
        Path(BOND_DIR).mkdir(parents=True, exist_ok=True)
        (Path(BOND_DIR) / "LAST_BLOCK.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return 2

    sft = Path(args.episodes)
    if sft.suffix == ".jsonl" and "episodes.jsonl" in sft.name:
        alt = sft.parent / "sft_actions.jsonl"
        if alt.is_file():
            sft = alt
    result = train_bond_adapter(
        sft,
        model_name=spec["hf_id"],
        output_dir=Path(args.output_dir),
        foundation_id=spec["id"],
        config={
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "num_train_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "quantization": "4bit" if args.method == "qlora" else "none",
            "method": args.method,
            "holdout_spec": args.holdout_spec,
        },
    )
    result["is_final_bond"] = False
    result["public_name"] = PUBLIC_NAME
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
