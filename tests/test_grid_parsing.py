"""Parsing grids out of real model output.

A parse failure here is invisible: it returns a plausible wrong grid rather than
an error, and the task is scored wrong with nothing pointing at the parser. The
cases below are the shapes a reasoning model actually emits.
"""

from __future__ import annotations

import pytest

from src.data import text_to_grid

FENCE = "```"


# --------------------------------------------------------------------------
# Prose must never contribute cells
# --------------------------------------------------------------------------


def test_a_counted_quantity_in_prose_is_not_a_grid():
    # Regression: "I count 4 objects" used to parse as [[4]] and return before
    # the real answer below it was ever reached.
    text = "I count 4 objects in the input.\n\n0 1\n2 3"
    assert text_to_grid(text) == [[0, 1], [2, 3]]


@pytest.mark.parametrize(
    "preamble",
    [
        "There are 3 distinct colours.",
        "Rotate by 90 degrees:",
        "Step 1: find the 2 panels.",
        "The grid is 5 by 5.",
        "Looking at example 1 and example 2:",
    ],
)
def test_digits_inside_sentences_are_ignored(preamble):
    assert text_to_grid(f"{preamble}\n\n7 8\n9 0") == [[7, 8], [9, 0]]


def test_prose_with_no_grid_returns_none():
    assert text_to_grid("I am not sure what the rule is here.") is None
    assert text_to_grid("The answer has 4 rows and 3 columns.") is None


# --------------------------------------------------------------------------
# The answer is the last grid, not the first
# --------------------------------------------------------------------------


def test_an_echoed_input_does_not_win_over_the_output():
    text = "Input:\n1 1\n1 1\nOutput:\n2 2\n2 2"
    assert text_to_grid(text) == [[2, 2], [2, 2]]


def test_the_last_fenced_block_wins():
    text = f"{FENCE}\n1 1\n1 1\n{FENCE}\nand the answer is\n{FENCE}\n9 9\n9 9\n{FENCE}"
    assert text_to_grid(text) == [[9, 9], [9, 9]]


def test_reasoning_before_the_answer_is_skipped():
    text = "Step 1: there are 3 colors.\nStep 2: tile it.\nAnswer:\n5 6\n7 8"
    assert text_to_grid(text) == [[5, 6], [7, 8]]


def test_the_last_json_array_wins():
    text = "First I thought [[1, 1], [1, 1]] but actually [[2, 2], [2, 2]]."
    assert text_to_grid(text) == [[2, 2], [2, 2]]


# --------------------------------------------------------------------------
# Formatting variants
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "0 1\n2 3",
        "0,1\n2,3",
        "| 0 1 |\n| 2 3 |",
        "[0 1]\n[2 3]",
        f"{FENCE}\n0 1\n2 3\n{FENCE}",
        f"{FENCE}json\n[[0, 1], [2, 3]]\n{FENCE}",
        "[[0, 1], [2, 3]]",
        "  0   1  \n  2   3  ",
    ],
)
def test_common_renderings_all_parse(text):
    assert text_to_grid(text) == [[0, 1], [2, 3]]


def test_row_labels_are_stripped_not_counted_as_cells():
    # Regression: "Row 0: 1 2" used to yield [[0, 1, 2], ...].
    assert text_to_grid("Row 0: 1 2\nRow 1: 3 4") == [[1, 2], [3, 4]]
    assert text_to_grid("0: 1 2\n1: 3 4") == [[1, 2], [3, 4]]


def test_trailing_commentary_is_ignored():
    assert text_to_grid("0 1\n2 3\n\nThis matches the pattern.") == [[0, 1], [2, 3]]


# --------------------------------------------------------------------------
# Rejection
# --------------------------------------------------------------------------


def test_ragged_rows_are_rejected():
    assert text_to_grid("0 1 2\n3 4") is None


def test_empty_and_non_string_inputs_are_rejected():
    assert text_to_grid("") is None
    assert text_to_grid(None) is None  # type: ignore[arg-type]
    assert text_to_grid("   \n\n  ") is None


def test_a_grid_larger_than_thirty_is_rejected():
    assert text_to_grid("\n".join(" ".join("1" for _ in range(31)) for _ in range(3))) is None


def test_a_single_cell_answer_still_parses():
    assert text_to_grid("The answer is:\n5") == [[5]]


def test_a_full_size_grid_parses():
    rows = ["1 " * 30, "2 " * 30]
    grid = text_to_grid("\n".join(r.strip() for r in rows))
    assert grid is not None and len(grid) == 2 and len(grid[0]) == 30
