"""Bond SFT native-precision LoRA entry point.

First real experiment: Qwen/Qwen3.5-4B on a remote NVIDIA GPU.
0.5B/1.5B = laptop smoke only. 27B = later ceiling, not this run.

SFT first. Never overwrite the foundation. Never label 0.5B/1.5B as 4B Bond.
This path is native fp16/bf16 + LoRA, not QLoRA / BitsAndBytes.
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
    p.add_argument("--foundation", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--episodes", type=str, default=str(BOND_DIR / "train_scale" / "sft_actions.jsonl"))
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "models" / "bond_qwen35_4b"))
    p.add_argument("--method", choices=("sft", "lora", "qlora"), default="lora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--holdout-spec", type=str, default="training[400:440]")
    args = p.parse_args(argv)

    spec = resolve_foundation(args.foundation)
    hw = probe_hardware()
    method = args.method
    if method == "qlora":
        method = "lora"
        print(
            json.dumps(
                {
                    "warning": (
                        "qlora was requested but this trainer is native-precision LoRA "
                        "(fp16/bf16 + PEFT). BitsAndBytes QLoRA is not implemented. "
                        "Continuing as lora."
                    )
                }
            )
        )
    print(
        json.dumps(
            {
                "public_name": PUBLIC_NAME,
                "foundation_hf_id": spec["hf_id"],
                "method": method,
                "adapter_kind": "native_precision_lora",
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
            "quantization": "none",
            "method": method,
            "adapter_kind": "native_precision_lora",
            "qlora_requested": args.method == "qlora",
            "holdout_spec": args.holdout_spec,
        },
    )
    result["is_final_bond"] = False
    result["public_name"] = PUBLIC_NAME
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
