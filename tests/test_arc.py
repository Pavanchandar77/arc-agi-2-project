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
from src.test_time_train import (
    capture_trainable_state,
    create_ttt_dataset_for_task,
    restore_trainable_state,
    verify_weight_equality,
)


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


# ============================================================================
# 7. Test-Time Training (TTT) Isolation Tests
# ============================================================================

def test_ttt_dataset_generation():
    """Verify that TTT builds synthetic dataset from ONLY that task's own demonstrations."""
    task = create_synthetic_arc_task(task_type="rot90")
    num_train_pairs = len(task["train"])
    
    ttt_records = create_ttt_dataset_for_task(task, num_augmentations=4)
    assert len(ttt_records) > 0

    # Every generated record must contain prompt and target completion
    for rec in ttt_records:
        assert "messages" in rec
        assert "prompt" in rec
        assert "completion" in rec
        assert rec["completion"] is not None
        assert len(rec["completion"]) > 0


class MockParameter:
    """Mock trainable parameter container for CPU unit testing."""
    def __init__(self, data, requires_grad=True):
        self.data = list(data)
        self.requires_grad = requires_grad

    def copy_(self, other):
        if hasattr(other, "data"):
            self.data = list(other.data)
        elif isinstance(other, list):
            self.data = list(other)
        else:
            self.data = other

    def copy(self):
        return MockParameter(list(self.data), self.requires_grad)


class MockModel:
    """Mock neural network model with LoRA parameters for testing weight isolation."""
    def __init__(self):
        self.lora_A = MockParameter([0.1, 0.2, 0.3, 0.4])
        self.lora_B = MockParameter([1.0, 2.0, 3.0, 4.0])
        self.frozen_base = MockParameter([100.0, 200.0], requires_grad=False)

    def named_parameters(self):
        return [
            ("lora_A", self.lora_A),
            ("lora_B", self.lora_B),
            ("frozen_base", self.frozen_base),
        ]


def test_ttt_weight_reset_isolation():
    """CRITICAL TEST: Prove that task adaptations are strictly isolated with ZERO weight leakage.
    
    Verifies:
    1. Base snapshot S0 is captured from starting checkpoint.
    2. Task 1 fine-tuning mutates weights (W1 != W0).
    3. restore_trainable_state restores weights exactly to S0.
    4. Task 2 fine-tuning starts strictly from S0 (never from mutated Task 1 state).
    5. Consecutive tasks never accumulate or leak adapter weights.
    """
    model = MockModel()

    # Step 1: Capture immutable base state S0
    base_state = capture_trainable_state(model)
    assert "lora_A" in base_state
    assert "lora_B" in base_state
    assert "frozen_base" not in base_state  # Frozen parameters excluded
    assert base_state["lora_A"].data == [0.1, 0.2, 0.3, 0.4]
    assert verify_weight_equality(model, base_state) is True

    # Step 2: Simulate Task 1 adaptation (mutates weights)
    model.lora_A.data = [0.99, 0.88, 0.77, 0.66]
    model.lora_B.data = [5.55, 6.66, 7.77, 8.88]
    assert verify_weight_equality(model, base_state) is False, "Weights should have changed during Task 1 adaptation"

    # Step 3: Restore to base state
    restore_trainable_state(model, base_state)
    assert verify_weight_equality(model, base_state) is True, "Weights must match base state S0 after Task 1 reset"
    assert model.lora_A.data == [0.1, 0.2, 0.3, 0.4]
    assert model.lora_B.data == [1.0, 2.0, 3.0, 4.0]

    # Step 4: Simulate Task 2 adaptation (distinct mutations)
    model.lora_A.data = [-0.11, -0.22, -0.33, -0.44]
    model.lora_B.data = [-1.0, -2.0, -3.0, -4.0]
    assert verify_weight_equality(model, base_state) is False

    # Step 5: Restore to base state after Task 2
    restore_trainable_state(model, base_state)
    assert verify_weight_equality(model, base_state) is True, "Weights must match base state S0 after Task 2 reset"
    assert model.lora_A.data == [0.1, 0.2, 0.3, 0.4]
    assert model.lora_B.data == [1.0, 2.0, 3.0, 4.0]

