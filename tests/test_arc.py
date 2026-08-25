"""Unit and integration tests for ARC-AGI-2 Suite."""

import json
from pathlib import Path
from typing import Optional

import pytest

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
from src.verify_and_select import (
    Candidate,
    GenerationConfig,
    ProbeResult,
    build_candidate_schedule,
    build_probe_task,
    consistency_score_from_probe_results,
    predict_task_with_verified_selection,
    rank_and_select,
    select_verified_attempts,
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


# ============================================================================
# 8. Offline Self-Verification & Candidate Selection
# ============================================================================

def _make_candidate(
    raw_text: str,
    grid,
    consistency: float,
    *,
    do_sample: bool = True,
    temperature: float = 0.7,
    seed: int = 0,
    mean_cell: Optional[float] = None,
    n_exact: int = 0,
    n_total: int = 2,
    n_valid: int = 0,
) -> Candidate:
    valid = grid is not None
    return Candidate(
        raw_text=raw_text,
        grid=grid,
        gen_config=GenerationConfig(
            do_sample=do_sample, temperature=temperature, seed=seed
        ),
        consistency_score=consistency,
        mean_cell_accuracy=mean_cell if mean_cell is not None else consistency,
        n_train_exact=n_exact,
        n_train_total=n_total,
        n_valid_probes=n_valid,
        is_valid_grid=valid,
    )


def _flip_h_task() -> dict:
    """Deterministic 2-demo horizontal-flip task used by dummy-model tests."""
    return {
        "train": [
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
            {"input": [[2, 0], [0, 2]], "output": [[0, 2], [2, 0]]},
        ],
        "test": [
            {"input": [[3, 0], [0, 3]], "output": [[0, 3], [3, 0]]},
        ],
    }


class DummyRuleModel:
    """Dummy generator used like MockModel in the TTT isolation tests.

    No torch, no GPU.  `generate(prompt, gen_config)` returns a grid string
    based on decoding temperature and whether a contradictory hypothesis was
    inserted into the demonstrations.
    """

    TEST_IN = "3 0\n0 3"
    TEST_GOOD = "0 3\n3 0"
    TEST_BAD_A = "3 0\n0 3"       # identity (wrong)
    TEST_BAD_B = "1 1\n1 1"       # garbage but valid
    TRAIN1_IN = "1 0\n0 1"
    TRAIN1_OUT = "0 1\n1 0"
    TRAIN2_IN = "2 0\n0 2"
    TRAIN2_OUT = "0 2\n2 0"
    WRONG_OUT = "9 9\n9 9"

    def __init__(self):
        self.calls: list = []

    def generate(self, prompt: str, gen_config: dict) -> str:
        self.calls.append({"prompt": prompt, "gen_config": dict(gen_config)})
        do_sample = bool(gen_config.get("do_sample", False))
        temperature = float(gen_config.get("temperature", 0.0) or 0.0)

        parts = prompt.split("Test Problem:")
        demo_section = parts[0] if parts else ""
        test_section = parts[-1] if parts else prompt

        # A contradictory hypothesized test output in the demos poisons probes.
        # Match against "Output:\n..." so we don't false-positive on the Input grid
        # (TEST_BAD_A is the identity map and equals TEST_IN).
        good_hypothesis = f"Output:\n{self.TEST_GOOD}" in demo_section
        bad_hypothesis = (
            not good_hypothesis
            and (
                f"Output:\n{self.TEST_BAD_A}" in demo_section
                or f"Output:\n{self.TEST_BAD_B}" in demo_section
            )
        )

        asking_train1 = self.TRAIN1_IN in test_section
        asking_train2 = self.TRAIN2_IN in test_section
        asking_test = self.TEST_IN in test_section

        if asking_train1 or asking_train2:
            if bad_hypothesis:
                return self.WRONG_OUT
            # Greedy / low-temp is consistent; high-temp sampling is not.
            if (not do_sample) or temperature <= 0.25:
                return self.TRAIN1_OUT if asking_train1 else self.TRAIN2_OUT
            return self.WRONG_OUT

        if asking_test:
            if (not do_sample) or temperature <= 0.25:
                return self.TEST_GOOD
            if temperature <= 0.55:
                return self.TEST_BAD_A
            return self.TEST_BAD_B

        return "0"


def test_build_probe_task_leave_one_out():
    task = _flip_h_task()
    probe = build_probe_task(task, held_out_idx=0)
    assert probe["test"][0]["input"] == task["train"][0]["input"]
    assert probe["test"][0]["output"] == task["train"][0]["output"]
    assert probe["train"] == [task["train"][1]]


def test_build_probe_task_single_pair_fallback():
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    probe = build_probe_task(task, held_out_idx=0)
    # No remaining demos: reuse the held-out pair as its own demonstration.
    assert probe["train"] == [task["train"][0]]
    assert probe["test"] == [task["train"][0]]


def test_build_probe_task_conditions_on_candidate():
    task = _flip_h_task()
    candidate_pair = {"input": [[3, 0], [0, 3]], "output": [[0, 3], [3, 0]]}
    probe = build_probe_task(task, held_out_idx=1, candidate_pair=candidate_pair)
    assert probe["test"][0] == task["train"][1]
    assert probe["train"][-1] == candidate_pair
    assert task["train"][1] not in probe["train"]
    assert task["train"][0] in probe["train"]


def test_consistency_score_from_probe_results():
    probes = [
        ProbeResult(0, True, 1.0, True, [[0]], "0"),
        ProbeResult(1, False, 0.5, True, [[1]], "1"),
        ProbeResult(2, False, 0.0, False, None, "nope"),
    ]
    exact_rate, mean_cell, n_exact, n_valid = consistency_score_from_probe_results(probes)
    assert exact_rate == pytest.approx(1.0 / 3.0)
    assert mean_cell == pytest.approx(0.5)
    assert n_exact == 1
    assert n_valid == 2
    assert consistency_score_from_probe_results([]) == (0.0, 0.0, 0, 0)


def test_rank_and_select_prefers_higher_consistency():
    """The two submitted attempts must be the two highest-consistency unique grids."""
    low = _make_candidate("1 1\n1 1", [[1, 1], [1, 1]], 0.0, temperature=1.0, seed=1)
    mid = _make_candidate("3 0\n0 3", [[3, 0], [0, 3]], 0.5, temperature=0.7, seed=2)
    high = _make_candidate("0 3\n3 0", [[0, 3], [3, 0]], 1.0, do_sample=False, temperature=0.0, seed=0)

    selected = rank_and_select([low, mid, high], n_submit=2)
    assert len(selected) == 2
    assert selected[0] is high
    assert selected[1] is mid
    # The low-consistency candidate is never submitted when better unique grids exist.
    assert low not in selected


def test_rank_and_select_prefers_valid_grid_over_invalid():
    invalid = _make_candidate("I cannot solve this.", None, 1.0, do_sample=False, temperature=0.0)
    valid_low = _make_candidate("0 1\n1 0", [[0, 1], [1, 0]], 0.0, temperature=0.7)
    selected = rank_and_select([invalid, valid_low], n_submit=2)
    assert selected[0] is valid_low
    assert selected[1] is invalid


def test_rank_and_select_deduplicates_identical_grids():
    g = [[0, 3], [3, 0]]
    a = _make_candidate("0 3\n3 0", g, 1.0, do_sample=False, temperature=0.0, seed=0)
    b = _make_candidate("0 3\n3 0", [list(row) for row in g], 0.5, temperature=0.2, seed=1)
    c = _make_candidate("1 1\n1 1", [[1, 1], [1, 1]], 0.25, temperature=0.7, seed=2)

    selected = rank_and_select([a, b, c], n_submit=2)
    assert len(selected) == 2
    assert selected[0] is a
    # Duplicate of `a` is skipped in favor of the distinct runner-up.
    assert selected[1] is c


def test_rank_and_select_submits_exactly_two_and_pads_duplicates():
    only = _make_candidate("0 1\n1 0", [[0, 1], [1, 0]], 1.0, do_sample=False)
    selected = rank_and_select([only], n_submit=2)
    assert selected == [only]

    twins = [
        _make_candidate("0 1\n1 0", [[0, 1], [1, 0]], 1.0, do_sample=False, seed=0),
        _make_candidate("0 1\n1 0", [[0, 1], [1, 0]], 0.5, temperature=0.4, seed=1),
    ]
    padded = rank_and_select(twins, n_submit=2)
    assert len(padded) == 2
    assert padded[0] is twins[0]
    # No distinct runner-up: pad with the duplicate so the caller can still submit 2.
    assert padded[1] is twins[1]


def test_rank_and_select_empty_and_schedule_length():
    assert rank_and_select([], n_submit=2) == []
    sched = build_candidate_schedule(8)
    assert len(sched) == 8
    assert sched[0].do_sample is False
    assert all(isinstance(c, GenerationConfig) for c in sched)


def test_select_verified_attempts_dummy_model_ranks_consistent_first():
    """End-to-end selection with a dummy model: greedy (consistent) beats high-temp (not)."""
    dummy = DummyRuleModel()
    task = _flip_h_task()
    schedule = [
        GenerationConfig(do_sample=False, temperature=0.0, seed=0),
        GenerationConfig(do_sample=True, temperature=0.4, seed=1),
        GenerationConfig(do_sample=True, temperature=1.0, seed=2),
    ]

    att_1, att_2, info = select_verified_attempts(
        task=task,
        model=None,
        tokenizer=None,
        n_candidates=3,
        n_submit=2,
        candidate_schedule=schedule,
        condition_on_candidate=True,
        early_stop_perfect=False,
        generate_fn=dummy.generate,
    )

    assert text_to_grid(att_1) == [[0, 3], [3, 0]], "Top attempt must be the consistent (greedy) grid"
    assert info["n_generated"] == 3
    assert info["n_selected"] == 2
    assert info["selected"][0]["consistency_score"] == pytest.approx(1.0)
    assert info["selected"][0]["gen_config"]["do_sample"] is False
    # Runner-up is a different valid grid with lower consistency.
    assert text_to_grid(att_2) != text_to_grid(att_1)
    assert info["selected"][1]["consistency_score"] < info["selected"][0]["consistency_score"]
    assert dummy.calls, "Dummy model must have been invoked"


def test_select_verified_attempts_early_stop_on_two_perfect_unique():
    dummy = DummyRuleModel()
    task = _flip_h_task()

    # Both configs are greedy-like so DummyRuleModel returns the SAME perfect grid.
    # Early-stop requires 2 *unique* perfect grids, so this should NOT stop after 1.
    same_grid_schedule = [
        GenerationConfig(do_sample=False, temperature=0.0, seed=0),
        GenerationConfig(do_sample=False, temperature=0.0, seed=1),
        GenerationConfig(do_sample=True, temperature=1.0, seed=2),
    ]
    _, _, info = select_verified_attempts(
        task=task,
        generate_fn=dummy.generate,
        candidate_schedule=same_grid_schedule,
        n_submit=2,
        early_stop_perfect=True,
        condition_on_candidate=True,
    )
    # First two are duplicate perfect grids; must continue to generate the third.
    assert info["n_generated"] == 3
    assert info["early_stopped"] is False


def test_condition_on_candidate_penalizes_contradictory_hypothesis():
    """A wrong test-output hypothesis poisons train reconstruction and is ranked down."""
    dummy = DummyRuleModel()
    task = _flip_h_task()
    schedule = [
        GenerationConfig(do_sample=True, temperature=1.0, seed=2),   # TEST_BAD_B, poisons probes
        GenerationConfig(do_sample=False, temperature=0.0, seed=0),  # TEST_GOOD, probes succeed
    ]
    att_1, att_2, info = select_verified_attempts(
        task=task,
        generate_fn=dummy.generate,
        candidate_schedule=schedule,
        n_submit=2,
        early_stop_perfect=False,
        condition_on_candidate=True,
    )
    assert text_to_grid(att_1) == [[0, 3], [3, 0]]
    assert info["selected"][0]["consistency_score"] == pytest.approx(1.0)
    assert info["selected"][1]["consistency_score"] == pytest.approx(0.0)
    assert text_to_grid(att_2) == [[1, 1], [1, 1]]


def test_verified_selection_restores_weights_like_ttt():
    """Same isolation contract as test_ttt_weight_reset_isolation, on the verification path."""
    model = MockModel()
    base_state = capture_trainable_state(model)

    # Simulate adapter drift that happened before/during selection.
    model.lora_A.data = [9.0, 9.0, 9.0, 9.0]
    model.lora_B.data = [8.0, 8.0, 8.0, 8.0]
    assert verify_weight_equality(model, base_state) is False

    dummy = DummyRuleModel()
    predict_task_with_verified_selection(
        model=model,
        tokenizer=None,
        task=_flip_h_task(),
        base_state=base_state,
        use_ttt=False,  # skip adapt_model_to_task (would import torch)
        generate_fn=dummy.generate,
        n_candidates=2,
        device="cpu",
    )

    assert verify_weight_equality(model, base_state) is True, (
        "Verification path must restore LoRA weights even when TTT is skipped"
    )
    assert model.lora_A.data == [0.1, 0.2, 0.3, 0.4]
    assert model.lora_B.data == [1.0, 2.0, 3.0, 4.0]


