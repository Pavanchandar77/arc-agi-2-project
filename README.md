# ARC-AGI-2 Fine-Tuning & Evaluation Suite

A modular, end-to-end framework for fine-tuning Large Language Models (LLMs) on ARC-AGI & ARC-AGI-2 (Abstraction and Reasoning Corpus) tasks using LoRA/QLoRA and Hugging Face TRL.

---

## Project Structure

```
arc-agi-2-project/
├── src/
│   ├── __init__.py
│   ├── data.py          # Grid serialization, D8 symmetries, color bijections, task augmentations
│   ├── build_dataset.py # Builds arc_train.jsonl / arc_val.jsonl with augmentations (CPU, local)
│   ├── train.py         # 4-bit QLoRA SFTTrainer training loop (GPU, Colab & local)
│   └── evaluate.py      # ARC benchmark exact-match scoring (2-attempt Pass@2 metric)
├── requirements.txt     # Dependencies (torch, transformers, peft, trl, bitsandbytes, datasets)
└── README.md            # Quickstart guide and documentation
```

---

## Quickstart

### 1. Installation

```bash
# Clone or navigate to the repository
cd arc-agi-2-project

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Build Dataset (Local / CPU)

The dataset builder automatically downloads the official ARC benchmark (or accepts custom task folders) and applies task augmentations:
- **Dihedral Group (D8)**: 8 spatial symmetries (rotations, reflections, transpositions).
- **Color Permutations**: Bijective color mappings preserving spatial structure.
- **Demonstration Shuffling**: Randomizes in-context example order to prevent positional bias.

```bash
# Build dataset from official ARC repository with 8x augmentation factor
python src/build_dataset.py --output-dir data/processed --aug-factor 8

# Or build from a local directory of ARC JSON tasks
python src/build_dataset.py --data-dir path/to/arc_tasks --output-dir data/processed --aug-factor 8

# Quick synthetic dataset generation (for testing/debugging without downloading)
python src/build_dataset.py --use-synthetic --output-dir data/processed
```

Outputs generated:
- `data/processed/arc_train.jsonl`
- `data/processed/arc_val.jsonl`

---

### 3. Fine-Tuning on GPU (Google Colab / Local Workstation)

Train a base model (e.g., `Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`) using 4-bit QLoRA:

```bash
python src/train.py \
    --train-file data/processed/arc_train.jsonl \
    --val-file data/processed/arc_val.jsonl \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --output-dir models/arc_lora_adapter \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 4 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32
```

#### Running on Google Colab:
```python
# Cell 1: Clone repo and install requirements
!git clone https://github.com/your-username/arc-agi-2-project.git
%cd arc-agi-2-project
!pip install -r requirements.txt

# Cell 2: Build dataset with augmentations
!python src/build_dataset.py --output-dir data/processed --aug-factor 8

# Cell 3: Run 4-bit QLoRA fine-tuning (fits on free T4 / A100 GPU)
!python src/train.py \
    --train-file data/processed/arc_train.jsonl \
    --val-file data/processed/arc_val.jsonl \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --output-dir models/arc_lora_adapter \
    --epochs 3
```

---

### 4. Benchmark Evaluation & Scoring

Evaluates according to official ARC benchmark rules:
- **2 Attempts per Test Problem**:
  - **Attempt 1**: Greedy decoding (`temperature=0.0`)
  - **Attempt 2**: Temperature sampling (`temperature=0.7`, `top_p=0.9`)
- **Scoring**: A problem is solved if **either** attempt achieves 100% exact match across all grid dimensions and cells.

```bash
# Evaluate fine-tuned model with LoRA adapter
python src/evaluate.py \
    --val-file data/processed/arc_val.jsonl \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --adapter-path models/arc_lora_adapter \
    --output-report eval_results.json

# Zero-shot baseline evaluation (without adapter)
python src/evaluate.py \
    --val-file data/processed/arc_val.jsonl \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --zero-shot
```

#### Metrics Reported:
- **ARC Score (Pass@2 Exact Match %)**: Official competition benchmark metric.
- **Pass@1 Accuracy %**: Single-shot exact match rate.
- **Grid Dimension Match Rate %**: Accuracy of predicting correct output height and width.
- **Average Cell-Level Accuracy %**: Percentage of correct cell color predictions.
- **Parse Failure Rate %**: Frequency of unparseable model responses.

---

## Core Module API Reference

### `src.data`

```python
from src.data import grid_to_text, text_to_grid, augment_task, random_color_map, apply_d8_transform

# Grid serialization & robust parsing
grid = [[0, 1], [2, 3]]
text = grid_to_text(grid, format_style="compact")
# "0 1\n2 3"

parsed_grid = text_to_grid("Output:\n```\n0 1\n2 3\n```")
# [[0, 1], [2, 3]]

# D8 Dihedral Symmetries (0 to 7)
rot90 = apply_d8_transform(grid, op=1)

# Color Permutations
color_map = random_color_map(preserve_background=True)
```

---

## License

MIT License
