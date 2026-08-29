# ARC-AGI-2 Solver & Kaggle Submission Suite

An offline symbolic solver for ARC-AGI, a Kaggle submission runner built to
survive the competition environment, and an LLM fine-tuning stack for the work
that the symbolic layer cannot reach.

## Kaggle in one command

```bash
python -m src.kaggle_run          # finds the data, writes submission.json
```

Full notebook instructions, including the no-upload bundle route, are in
[`kaggle/README.md`](kaggle/README.md). The runner is CPU-only, needs no
network, and holds its wall-clock deadline.

## Where this actually stands

Measured on the real benchmarks, `--per-task-seconds 30`, 4 CPU workers, pass@2:

| Benchmark | DSL search alone | Solver bank + search |
|---|---|---|
| ARC-AGI-1 training (400) | — | ~17% |
| ARC-AGI-1 evaluation (400) | — | ~10% |
| ARC-AGI-2 training (1000) | 2.5% | ~10% |
| **ARC-AGI-2 evaluation (120)** | **0%** | **~0%** |

Read the last row first: **ARC Prize scores against the ARC-AGI-2 private
evaluation set**, which resembles that public eval split, not the training
split. The symbolic layer contributes essentially nothing there, and that is
not a budget problem — ARC-AGI-2 was explicitly designed so that
single-transformation program search fails on it.

For calibration: the strongest purely symbolic ARC result ever recorded is
roughly 40% on ARC-AGI-**1** (icecuber, 2020), from a large hand-tuned C++ DSL.
Every approach that has scored meaningfully on ARC-AGI-2 pairs a neural model
with test-time training. So the symbolic bank here is a cheap, exact,
always-correct-when-it-fires floor — not the thing that produces a score. The
LLM path (`src/train.py`, `src/test_time_train.py`, `scripts/train_bond_*.py`)
is where the score comes from.

## Project structure

```
arc-agi-2-project/
├── src/
│   ├── kaggle_run.py      # Kaggle runner: discovery, deadline, workers, schema validation
│   ├── arc_solve.py       # One task in, two attempts out. Never raises.
│   ├── hrps/
│   │   ├── solvers.py     # Exact train-verified solver bank (the symbolic floor)
│   │   ├── search.py      # Instrumented finite-DSL best-first search
│   │   └── ...            # Grids, objects, residuals, Bond/HRPS reasoning stack
│   ├── train.py           # QLoRA SFT training loop (GPU)
│   ├── test_time_train.py # Per-task test-time training
│   └── evaluate.py        # Pass@2 exact-match scoring
├── kaggle/                # Notebook, bundle, and Kaggle-specific docs
└── scripts/               # Bundle builder, GPU preflight, Bond training runners
```

## The solver bank

`src/hrps/solvers.py` enumerates hypotheses from a fixed set of families and
keeps only those that reproduce **every** demonstration pair exactly. A rule
that cannot replay the training pairs never predicts, so the bank's answers are
verified rather than guessed; when nothing verifies, it abstains and the
fallback layer supplies a well-formed grid.

Families: D8 transforms with learned colour maps, mosaic tiling, fractal
self-tiling, uniform and content-derived scaling, cellwise functions over
separator-split panels, panel selection, symmetry repair and occlusion fill,
object selection/recolouring/filtering, colour remapping by frequency rank,
row-column deduplication, borders, denoising, constant outputs, and a
last-resort local neighbourhood lookup.

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
