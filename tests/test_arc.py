"""Unit and integration tests for ARC-AGI-2 Suite."""

import json
import pytest
from pathlib import Path

from src.data import (
    apply_color_map,
    apply_d8_transform,
    augment_task,
    generate_task_augmentations,
    grid_to_text,
    grids_equal,
    is_valid_grid,
    random_color_map,
    task_to_chat_messages,
    task_to_prompt,
    text_to_grid,
)
from src.build_dataset import (
    build_jsonl_dataset,
    create_synthetic_arc_task,
    generate_synthetic_tasks,
    split_tasks_by_id,
)
from src.evaluate import calculate_cell_accuracy, evaluate_task_predictions


# ============================================================================
# 1. Grid Validation & Roundtrip Tests
# ============================================================================

def test_is_valid_grid():
    assert is_valid_grid([[0, 1], [2, 3]])
    assert not is_valid_grid([])
    assert not is_valid_grid([[]])
    assert not is_valid_grid([[0, 1], [2]])  # Non-rectangular
    assert not is_valid_grid([[0, 10]])  # Invalid color (>9)
    assert not is_valid_grid([["a", "b"]])  # Non-integer


def test_grid_serialization_and_deserialization():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]
    ]

    # Compact format
    compact_str = grid_to_text(grid, format_style="compact")
    assert compact_str == "0 1 2\n3 4 5\n6 7 8"
    assert text_to_grid(compact_str) == grid

    # Bracket format
    bracket_str = grid_to_text(grid, format_style="brackets")
    assert text_to_grid(bracket_str) == grid

    # Delimited format
    delim_str = grid_to_text(grid, format_style="delimited")
    assert text_to_grid(delim_str) == grid

    # Model markdown code block wrapping
    code_block_text = f"Here is the predicted output grid:\n```\n{compact_str}\n```\nExplanation: all colors shifted."
    assert text_to_grid(code_block_text) == grid

    # JSON code block wrapping
    json_block = f"```json\n{bracket_str}\n```"
    assert text_to_grid(json_block) == grid


def test_grids_equal():
    g1 = [[1, 2], [3, 4]]
    g2 = [[1, 2], [3, 4]]
    g3 = [[1, 2], [3, 5]]
    g4 = [[1, 2, 0], [3, 4, 0]]
    assert grids_equal(g1, g2)
    assert not grids_equal(g1, g3)
    assert not grids_equal(g1, g4)
    assert not grids_equal(g1, None)


# ============================================================================
# 2. Spatial Symmetries (D8) Tests
# ============================================================================

def test_d8_transformations():
    grid = [
        [1, 2],
        [3, 4]
    ]

    # 0: Identity
    assert apply_d8_transform(grid, 0) == [[1, 2], [3, 4]]
    
    # 1: Rot 90 CW
    assert apply_d8_transform(grid, 1) == [[3, 1], [4, 2]]

    # 2: Rot 180
    assert apply_d8_transform(grid, 2) == [[4, 3], [2, 1]]

    # 3: Rot 270 CW
    assert apply_d8_transform(grid, 3) == [[2, 4], [1, 3]]

    # 4: Flip Horizontal (left-right)
    assert apply_d8_transform(grid, 4) == [[2, 1], [4, 3]]

    # 5: Flip Vertical (up-down)
    assert apply_d8_transform(grid, 5) == [[3, 4], [1, 2]]

    # 6: Transpose (main diagonal)
    assert apply_d8_transform(grid, 6) == [[1, 3], [2, 4]]

    # 7: Anti-Transpose
    assert apply_d8_transform(grid, 7) == [[4, 2], [3, 1]]


# ============================================================================
# 3. Color Permutations Tests
# ============================================================================

def test_color_permutations():
    cmap = random_color_map(preserve_background=True)
    assert cmap[0] == 0  # Background preserved
    assert set(cmap.keys()) == set(range(10))
    assert set(cmap.values()) == set(range(10))  # Bijection

    grid = [[0, 1], [2, 3]]
    transformed = apply_color_map(grid, cmap)
    assert transformed[0][0] == 0
    assert transformed[0][1] == cmap[1]
    assert transformed[1][0] == cmap[2]
    assert transformed[1][1] == cmap[3]


# ============================================================================
# 4. Task Augmentations & Prompt Formatting Tests
# ============================================================================

def test_task_augmentations():
    task = create_synthetic_arc_task(task_type="flip_h")
    aug_variants = generate_task_augmentations(task, num_augmentations=4)
    assert len(aug_variants) == 4
    
    for aug in aug_variants:
        assert "train" in aug and "test" in aug
        assert len(aug["train"]) == len(task["train"])
        assert len(aug["test"]) == len(task["test"])


def test_task_to_prompt_and_chat_messages():
    task = create_synthetic_arc_task(task_type="flip_v")
    prompt_str, target_str = task_to_prompt(task, include_test_output=True)
    assert "Demonstration 1:" in prompt_str
    assert "Test Problem:" in prompt_str
    assert target_str is not None

    messages = task_to_chat_messages(task, include_test_output=True)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == target_str


# ============================================================================
# 5. Dataset Builder Integration Test
# ============================================================================

def test_build_jsonl_dataset(tmp_path: Path):
    all_tasks = generate_synthetic_tasks(num_tasks=6)
    train_tasks, val_tasks = split_tasks_by_id(all_tasks, val_ratio=0.33, seed=42)
    output_dir = tmp_path / "processed"
    
    train_p, val_p, train_count, val_count = build_jsonl_dataset(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        output_dir=output_dir,
        aug_factor=2
    )

    assert train_p.exists()
    assert val_p.exists()

    with open(train_p, "r", encoding="utf-8") as f:
        train_lines = [json.loads(line) for line in f]
    
    assert len(train_lines) == len(train_tasks) * 2
    assert "messages" in train_lines[0]
    assert "task_id" in train_lines[0]


# ============================================================================
# 6. Evaluation Metrics Tests
# ============================================================================

def test_evaluation_metrics():
    gt = [
        [1, 2],
        [3, 4]
    ]
    exact_match_str = "1 2\n3 4"
    partial_match_str = "1 2\n3 0"
    wrong_shape_str = "1 2 3\n4 5 6"
    invalid_str = "I cannot solve this."

    # Test exact match on Attempt 1
    eval_res_1 = evaluate_task_predictions(gt, exact_match_str, invalid_str)
    assert eval_res_1["exact_match"] is True
    assert eval_res_1["em_attempt_1"] is True
    assert eval_res_1["best_cell_accuracy"] == 1.0

    # Test exact match on Attempt 2 (Pass@2)
    eval_res_2 = evaluate_task_predictions(gt, partial_match_str, exact_match_str)
    assert eval_res_2["exact_match"] is True
    assert eval_res_2["em_attempt_1"] is False
    assert eval_res_2["em_attempt_2"] is True

    # Test partial match
    eval_res_3 = evaluate_task_predictions(gt, partial_match_str, wrong_shape_str)
    assert eval_res_3["exact_match"] is False
    assert eval_res_3["cell_accuracy_1"] == 0.75  # 3 out of 4 cells
    assert eval_res_3["shape_match"] is True

    # Cell accuracy helper
    assert calculate_cell_accuracy([[1, 2], [3, 4]], gt) == 1.0
    assert calculate_cell_accuracy([[1, 2], [3, 9]], gt) == 0.75
    assert calculate_cell_accuracy([[1, 2, 3]], gt) == 0.0
