"""The solver bank must be exact: verified rules only, and no malformed output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.arc_solve import blank_entry, solve_task, submission_entry
from src.hrps.grid import as_grid, is_valid_grid
from src.hrps.solvers import Rule, fit_rules, apply_split, split_methods, verify
from src.hrps.task import parse_task


def mk(task_id: str, train: list[tuple[list, list]], test: list[tuple[list, list | None]]):
    payload = {
        "train": [{"input": i, "output": o} for i, o in train],
        "test": [({"input": i, "output": o} if o is not None else {"input": i}) for i, o in test],
    }
    return parse_task(task_id, payload, "test")


def rule_names(task) -> set[str]:
    return {r.name for r in fit_rules(task)}


def predicted(task) -> list[list[list[int]]]:
    return solve_task(task, seconds=5.0).attempts[0]


# --------------------------------------------------------------------------
# Individual families
# --------------------------------------------------------------------------


def test_d8_rotation_is_found_and_applied():
    task = mk(
        "rot180",
        [
            ([[1, 2, 3], [4, 5, 6]], [[6, 5, 4], [3, 2, 1]]),
            ([[7, 0, 8], [0, 9, 0]], [[0, 9, 0], [8, 0, 7]]),
        ],
        [([[7, 8], [9, 1]], [[1, 9], [8, 7]])],
    )
    assert any(n.startswith("d8_colormap:rot180") for n in rule_names(task))
    assert predicted(task)[0] == [[1, 9], [8, 7]]


def test_colormap_composed_with_identity():
    task = mk(
        "cmap",
        [([[1, 1], [2, 2]], [[3, 3], [4, 4]]), ([[2, 1], [1, 2]], [[4, 3], [3, 4]])],
        [([[1, 2], [2, 1]], [[3, 4], [4, 3]])],
    )
    assert predicted(task)[0] == [[3, 4], [4, 3]]


def test_mirror_tiling_two_by_two():
    def build(g):
        top = [row + list(reversed(row)) for row in g]
        return top + list(reversed(top))

    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    c = [[9, 1], [2, 3]]
    task = mk("tile", [(a, build(a)), (b, build(b))], [(c, build(c))])
    assert any(n.startswith("tile_d8") for n in rule_names(task))
    assert predicted(task)[0] == build(c)


def test_fractal_self_tiling():
    def build(g, bg=0):
        h, w = len(g), len(g[0])
        rows = []
        for i in range(h):
            for r in range(h):
                acc = []
                for j in range(w):
                    acc.extend(g[r] if g[i][j] != bg else [bg] * w)
                rows.append(acc)
        return rows

    a = [[1, 0], [0, 1]]
    b = [[0, 2], [2, 2]]
    c = [[3, 3], [0, 3]]
    task = mk("fractal", [(a, build(a)), (b, build(b))], [(c, build(c))])
    assert predicted(task)[0] == build(c)


def test_panel_cellwise_learns_xor():
    def pair(left, right):
        grid = [lr + [5] + rr for lr, rr in zip(left, right)]
        out = [[2 if (x != 0) != (y != 0) else 0 for x, y in zip(lr, rr)] for lr, rr in zip(left, right)]
        return grid, out

    t1 = pair([[1, 0], [0, 1]], [[1, 1], [0, 0]])
    t2 = pair([[0, 0], [1, 1]], [[1, 0], [1, 0]])
    t3 = pair([[1, 1], [0, 0]], [[0, 1], [1, 0]])
    task = mk("xor", [t1, t2], [t3])
    assert any(n.startswith("panel_cellwise") for n in rule_names(task))
    assert predicted(task)[0] == t3[1]


def test_symmetry_repair_fills_occlusion():
    base = [
        [1, 2, 3, 2, 1],
        [2, 4, 5, 4, 2],
        [3, 5, 6, 5, 3],
        [2, 4, 5, 4, 2],
        [1, 2, 3, 2, 1],
    ]

    def occlude(cells):
        g = [row[:] for row in base]
        for r, c in cells:
            g[r][c] = 9
        return g

    task = mk(
        "sym",
        [(occlude([(0, 0)]), base), (occlude([(1, 1), (0, 2)]), base)],
        [(occlude([(3, 4), (4, 4)]), base)],
    )
    assert any(n.startswith("sym_repair_full") for n in rule_names(task))
    assert predicted(task)[0] == base


def test_symmetry_repair_abstains_on_an_unrecoverable_centre():
    # The centre of a fully symmetric odd grid is a fixed point of every
    # symmetry, so no orbit can supply its colour. Abstain rather than guess.
    base = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    holed = [[1, 2, 1], [2, 9, 2], [1, 2, 1]]
    task = mk("centre", [(holed, base)], [(holed, base)])
    assert not any(n.startswith("sym_repair") for n in rule_names(task))


def test_object_recolor_by_area_rank():
    def build(sizes):
        g = [[0] * 9 for _ in range(3)]
        col = 0
        for n in sizes:
            for k in range(n):
                g[0][col + k] = 3
            col += n + 1
        return g

    def target(g):
        # Largest run becomes 2, the other becomes 1.
        out = [row[:] for row in g]
        runs = []
        c = 0
        while c < 9:
            if out[0][c] == 3:
                s = c
                while c < 9 and out[0][c] == 3:
                    c += 1
                runs.append((s, c))
            else:
                c += 1
        biggest = max(runs, key=lambda r: r[1] - r[0])
        for s, e in runs:
            v = 2 if (s, e) == biggest else 1
            for k in range(s, e):
                out[0][k] = v
        return out

    a, b, c = build([3, 1]), build([1, 4]), build([2, 5])
    task = mk("recolor", [(a, target(a)), (b, target(b))], [(c, target(c))])
    assert any(n.startswith("obj_recolor") for n in rule_names(task))
    assert predicted(task)[0] == target(c)


def test_constant_output_requires_two_agreeing_demos():
    const = [[7, 7], [7, 7]]
    task = mk("const", [([[1]], const), ([[2, 3]], const)], [([[9, 9, 9]], const)])
    assert predicted(task)[0] == const


# --------------------------------------------------------------------------
# The exactness gate
# --------------------------------------------------------------------------


def test_verify_rejects_a_rule_that_misses_one_demo():
    task = mk("x", [([[1]], [[1]]), ([[2]], [[3]])], [([[4]], [[4]])])
    always_identity = Rule("id", 0, lambda g: g)
    assert not verify(always_identity, task)


def test_no_rule_fires_on_an_inconsistent_task():
    # Same input, two different outputs: nothing can verify.
    task = mk("bad", [([[1]], [[2]]), ([[1]], [[3]])], [([[1]], None)])
    assert fit_rules(task) == []


def test_rule_that_raises_is_treated_as_failure():
    task = mk("x", [([[1]], [[1]])], [([[2]], None)])

    def boom(_grid):
        raise RuntimeError("nope")

    assert not verify(Rule("boom", 0, boom), task)


def test_fit_rules_respects_deadline():
    import time

    task = mk("x", [([[1, 2], [3, 4]], [[4, 3], [2, 1]])], [([[1, 1], [1, 1]], None)])
    assert fit_rules(task, deadline=time.perf_counter() - 1.0) == []


# --------------------------------------------------------------------------
# Panel splitting
# --------------------------------------------------------------------------


def test_separator_split_finds_three_panels():
    grid = as_grid([[1, 5, 2, 5, 3], [1, 5, 2, 5, 3]])
    split = apply_split(grid, "sep:5")
    assert split is not None
    assert (split.nrows, split.ncols) == (1, 3)
    assert split.panels[0] == ((1,), (1,))
    assert split.panels[2] == ((3,), (3,))


def test_equal_split_rejects_indivisible_grid():
    assert apply_split(as_grid([[1, 2, 3]]), "eq:1x2") is None


def test_every_split_method_parses():
    grid = as_grid([[1, 2], [3, 4]])
    for method in split_methods():
        apply_split(grid, method)  # must not raise


# --------------------------------------------------------------------------
# Submission schema totality
# --------------------------------------------------------------------------


def test_solve_task_always_returns_two_valid_attempts():
    task = mk(
        "hard",
        [([[1, 2, 3], [4, 5, 6]], [[9]]), ([[6, 5, 4], [3, 2, 1]], [[8]])],
        [([[1, 1, 1], [2, 2, 2]], None), ([[3, 3, 3], [4, 4, 4]], None)],
    )
    outcome = solve_task(task, seconds=2.0)
    a1, a2 = outcome.attempts
    assert len(a1) == len(a2) == 2
    for grid in a1 + a2:
        assert is_valid_grid(grid)


def test_submission_entry_shape_matches_test_count():
    task = mk("t", [([[1]], [[1]])], [([[2]], None), ([[3]], None)])
    entry = submission_entry(solve_task(task, seconds=1.0))
    assert len(entry) == 2
    assert set(entry[0]) == {"attempt_1", "attempt_2"}


def test_blank_entry_is_schema_valid():
    entry = blank_entry(3)
    assert len(entry) == 3
    assert all(e["attempt_1"] == [[0]] and e["attempt_2"] == [[0]] for e in entry)


def test_solver_never_emits_an_oversized_grid():
    big = [[1] * 30 for _ in range(30)]
    task = mk("big", [(big, big)], [(big, None)])
    for grid in sum(solve_task(task, seconds=2.0).attempts, []):
        assert 1 <= len(grid) <= 30 and 1 <= len(grid[0]) <= 30
