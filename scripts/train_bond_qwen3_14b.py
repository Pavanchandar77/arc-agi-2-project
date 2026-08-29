"""Remote Stage-1 Bond SFT on Qwen3-14B. Do not run on the laptop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.hrps.bond_train import main as train_main

if __name__ == "__main__":
    raise SystemExit(
        train_main(
            [
                "--foundation",
                "Qwen/Qwen3-14B",
                "--episodes",
                str(REPO / "artifacts" / "bond" / "train_scale" / "sft_actions.jsonl"),
                "--output-dir",
                str(REPO / "models" / "bond_qwen3_14b"),
                "--method",
                "qlora",
                "--seed",
                "42",
                "--epochs",
                "1",
                "--learning-rate",
                "2e-4",
                "--lora-r",
                "16",
                "--lora-alpha",
                "32",
                "--lora-dropout",
                "0.05",
                "--max-seq-length",
                "4096",
                "--holdout-spec",
                "training[400:440]",
            ]
        )
    )
