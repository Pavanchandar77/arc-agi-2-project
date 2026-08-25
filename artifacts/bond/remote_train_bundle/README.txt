Stage 1 remote Bond adapter training (NVIDIA CUDA GPU).
Public identity: Bond
Internal artifact: Bond-Qwen35-4B-adapter
Foundation: Qwen/Qwen3.5-4B (do not swap)
Do not train this on the Iris Xe laptop. Copy the repo to a 24–48 GB GPU box.

pip install -r requirements-gpu.txt
python scripts/train_bond_qwen35_4b.py

Or:

python -m src.hrps.bond train --foundation qwen3.5_4b --adapter models/bond_qwen35_4b --episodes artifacts/bond/train_scale/sft_actions.jsonl --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --epochs 3 --learning-rate 2e-4 --max-seq-length 2048 --holdout-spec training[400:440]
