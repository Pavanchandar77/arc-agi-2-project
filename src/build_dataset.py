"""Dataset Builder for ARC-AGI-2 Fine-Tuning.

Builds `arc_train.jsonl` and `arc_val.jsonl` from ARC-AGI-2 task JSON files.
Key Requirements:
- Splits strictly by TASK (not by individual example) so augmented versions
  of a task never appear in both train and validation splits.
- Applies D8 dihedral symmetries, color permutations, and demonstration shuffling
  to training tasks only.
- Formats examples in ChatML format ready for SFTTrainer fine-tuning.

Usage:
    python src/build_dataset.py
    python src/build_dataset.py --data-dir ARC-AGI-2/data/training --output-dir data/processed --aug-factor 8
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path when running directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    DEFAULT_SYSTEM_PROMPT,
    generate_task_augmentations,
    is_valid_grid,
    task_to_chat_messages,
    task_to_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Synthetic Generator (Fallback / Debugging)
# ============================================================================

def create_synthetic_arc_task(task_type: str = "color_invert", rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Generate a valid synthetic ARC task for quick testing."""
    r = rng if rng is not None else random.Random()
    
    def make_random_grid(h: int, w: int, num_colors: int = 3) -> List[List[int]]:
        colors = [0] + r.sample(range(1, 10), min(num_colors, 9))
        return [[r.choice(colors) for _ in range(w)] for _ in range(h)]

    train_pairs = []
    for _ in range(3):
        h, w = r.randint(3, 6), r.randint(3, 6)
        inp = make_random_grid(h, w)
        if task_type == "flip_v":
            out = inp[::-1]
        elif task_type == "flip_h":
            out = [row[::-1] for row in inp]
        elif task_type == "replace_color":
            out = [[1 if val == 0 else 0 for val in row] for row in inp]
        else:  # rotate 90
            out = [[inp[h - 1 - ri][ci] for ri in range(h)] for ci in range(w)]
        train_pairs.append({"input": inp, "output": out})

    test_h, test_w = r.randint(3, 6), r.randint(3, 6)
    test_inp = make_random_grid(test_h, test_w)
    if task_type == "flip_v":
        test_out = test_inp[::-1]
    elif task_type == "flip_h":
        test_out = [row[::-1] for row in test_inp]
    elif task_type == "replace_color":
        test_out = [[1 if val == 0 else 0 for val in row] for row in test_inp]
    else:
        test_out = [[test_inp[test_h - 1 - ri][ci] for ri in range(test_h)] for ci in range(test_w)]

    return {
        "train": train_pairs,
        "test": [{"input": test_inp, "output": test_out}]
    }


def generate_synthetic_tasks(num_tasks: int = 40) -> Dict[str, Dict[str, Any]]:
    """Create a dictionary of synthetic ARC tasks."""
    task_types = ["flip_v", "flip_h", "replace_color", "rot90"]
    tasks = {}
    for i in range(num_tasks):
        ttype = task_types[i % len(task_types)]
        tasks[f"synthetic_{i:04d}"] = create_synthetic_arc_task(ttype)
    return tasks


# ============================================================================
# Loading & Task-Level Splitting
# ============================================================================

def load_tasks_from_directory(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all valid JSON ARC tasks from a directory recursively."""
    tasks: Dict[str, Dict[str, Any]] = {}
    for json_file in sorted(dir_path.rglob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "train" in data and "test" in data:
                tasks[json_file.stem] = data
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    return tasks


def split_tasks_by_id(
    tasks: Dict[str, Dict[str, Any]],
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Split tasks strictly by TASK ID (ensuring zero data leakage across train and val)."""
    task_ids = sorted(tasks.keys())
    rng = random.Random(seed)
    rng.shuffle(task_ids)

    split_idx = int(len(task_ids) * (1.0 - val_ratio))
    # Ensure at least 1 val task if there are tasks
    if split_idx == len(task_ids) and len(task_ids) > 1:
        split_idx = len(task_ids) - 1

    train_ids = set(task_ids[:split_idx])
    val_ids = set(task_ids[split_idx:])

    train_tasks = {tid: tasks[tid] for tid in train_ids}
    val_tasks = {tid: tasks[tid] for tid in val_ids}

    # Integrity verification
    assert train_ids.isdisjoint(val_ids), "Train and Validation task ID sets must be strictly disjoint!"

    return train_tasks, val_tasks


# ============================================================================
# Dataset Building & JSONL Serialization
# ============================================================================

def build_jsonl_dataset(
    train_tasks: Dict[str, Dict[str, Any]],
    val_tasks: Dict[str, Dict[str, Any]],
    output_dir: Path,
    aug_factor: int = 8,
    permute_colors: bool = True,
    apply_symmetries: bool = True,
    grid_format: str = "compact",
    seed: int = 42
) -> Tuple[Path, Path, int, int]:
    """Generate and write formatted train and validation JSONL files.
    
    Returns:
        (train_path, val_path, train_example_count, val_example_count)
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out_path = output_dir / "arc_train.jsonl"
    val_out_path = output_dir / "arc_val.jsonl"

    # 1. Process Training Tasks (with Augmentations)
    train_records = []
    logger.info(f"Processing {len(train_tasks)} train tasks with x{aug_factor} augmentations...")
    for task_id, task in train_tasks.items():
        augmented_variants = generate_task_augmentations(
            task,
            num_augmentations=aug_factor,
            permute_colors=permute_colors,
            apply_symmetries=apply_symmetries,
            shuffle_demonstrations=True,
            rng=rng
        )
        for aug_idx, aug_task in enumerate(augmented_variants):
            for test_idx in range(len(aug_task.get("test", []))):
                messages = task_to_chat_messages(
                    aug_task,
                    test_idx=test_idx,
                    include_test_output=True,
                    grid_format=grid_format
                )
                user_p, target_out = task_to_prompt(
                    aug_task,
                    test_idx=test_idx,
                    include_test_output=True,
                    grid_format=grid_format
                )
                train_records.append({
                    "task_id": f"{task_id}_aug{aug_idx}_test{test_idx}",
                    "parent_task_id": task_id,
                    "messages": messages,
                    "prompt": user_p,
                    "completion": target_out
                })

    # 2. Process Validation Tasks (Clean / Zero-leakage, unaugmented)
    val_records = []
    logger.info(f"Processing {len(val_tasks)} validation tasks (no augmentation)...")
    for task_id, task in val_tasks.items():
        for test_idx in range(len(task.get("test", []))):
            messages = task_to_chat_messages(
                task,
                test_idx=test_idx,
                include_test_output=True,
                grid_format=grid_format
            )
            user_p, target_out = task_to_prompt(
                task,
                test_idx=test_idx,
                include_test_output=True,
                grid_format=grid_format
            )
            val_records.append({
                "task_id": f"{task_id}_test{test_idx}",
                "parent_task_id": task_id,
                "messages": messages,
                "prompt": user_p,
                "completion": target_out
            })

    # Shuffle training records for i.i.d. SGD
    rng.shuffle(train_records)

    # Write files
    with open(train_out_path, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    with open(val_out_path, "w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Saved {train_out_path} ({len(train_records)} examples)")
    logger.info(f"Saved {val_out_path} ({len(val_records)} examples)")

    return train_out_path, val_out_path, len(train_records), len(val_records)


def main():
    parser = argparse.ArgumentParser(description="Build ARC-AGI-2 fine-tuning datasets by task-level split.")
    parser.add_argument("--data-dir", type=str, default="ARC-AGI-2/data/training", help="Path to ARC-AGI-2 training folder.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for JSONL files.")
    parser.add_argument("--aug-factor", type=int, default=8, help="Number of augmented variants per training task.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio for task-level split.")
    parser.add_argument("--no-color-permute", action="store_true", help="Disable color permutation augmentation.")
    parser.add_argument("--no-symmetries", action="store_true", help="Disable D8 dihedral symmetry augmentations.")
    parser.add_argument("--grid-format", type=str, choices=["compact", "brackets", "delimited"], default="compact", help="Grid serialization format.")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic tasks for dry run/testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    args = parser.parse_args()

    data_dir_path = Path(args.data_dir)
    output_dir_path = Path(args.output_dir)

    if args.use_synthetic:
        logger.info("Using synthetic ARC dataset...")
        all_tasks = generate_synthetic_tasks(num_tasks=40)
    elif data_dir_path.exists():
        logger.info(f"Loading tasks from {data_dir_path}...")
        all_tasks = load_tasks_from_directory(data_dir_path)
    else:
        logger.warning(f"Data directory {data_dir_path} not found. Checking fallback paths...")
        # Check standard fallback
        alt_path = Path("ARC-AGI-2/data/training")
        if alt_path.exists():
            logger.info(f"Loading tasks from {alt_path}...")
            all_tasks = load_tasks_from_directory(alt_path)
        else:
            logger.warning("No ARC dataset found on disk. Generating synthetic dataset for demo.")
            all_tasks = generate_synthetic_tasks(num_tasks=40)

    total_tasks = len(all_tasks)
    if total_tasks == 0:
        logger.error(f"No tasks found in {data_dir_path}! Exiting.")
        sys.exit(1)

    logger.info(f"Total base ARC tasks loaded: {total_tasks}")

    # Split strictly by TASK
    train_tasks, val_tasks = split_tasks_by_id(all_tasks, val_ratio=args.val_ratio, seed=args.seed)
    logger.info(f"Split by task: {len(train_tasks)} train tasks ({100*(1-args.val_ratio):.1f}%), {len(val_tasks)} validation tasks ({100*args.val_ratio:.1f}%)")

    # Build dataset
    train_path, val_path, train_count, val_count = build_jsonl_dataset(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        output_dir=output_dir_path,
        aug_factor=args.aug_factor,
        permute_colors=not args.no_color_permute,
        apply_symmetries=not args.no_symmetries,
        grid_format=args.grid_format,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("           ARC-AGI-2 DATASET BUILD SUMMARY")
    print("=" * 60)
    print(f"Total Base Tasks Loaded     : {total_tasks}")
    print(f"Training Base Tasks         : {len(train_tasks)}")
    print(f"Validation Base Tasks       : {len(val_tasks)}")
    print(f"Augmentation Multiplier     : x{args.aug_factor}")
    print(f"Final Train Examples (JSONL): {train_count} -> {train_path}")
    print(f"Final Val Examples (JSONL)  : {val_count} -> {val_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
