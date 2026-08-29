"""The one-command trainer's decisions, without running a training job."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_bond import (
    MODEL_LADDER,
    _has_module,
    _slug,
    _truncate,
    choose_model,
    detect_environment,
    missing_packages,
)


def test_has_module_is_false_for_a_missing_parent_package():
    # find_spec raises rather than returning None here; the helper must not.
    assert _has_module("google.colab") in {True, False}
    assert _has_module("definitely_not_installed_xyz.sub") is False
    assert _has_module("json") is True


def test_detect_environment_reports_without_raising():
    info = detect_environment()
    assert info["environment"] in {"local", "colab", "kaggle"}
    assert "cuda" in info


def test_missing_packages_returns_a_list():
    assert isinstance(missing_packages(), list)


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------


def test_ladder_is_ordered_by_descending_vram_and_ends_at_zero():
    needs = [need for need, *_ in MODEL_LADDER]
    assert needs == sorted(needs, reverse=True)
    assert needs[-1] == 0, "the ladder must have a floor entry that always matches"


@pytest.mark.parametrize(
    "free_gb,expected",
    [
        (80.0, "Qwen/Qwen3-14B"),
        (40.0, "Qwen/Qwen3-8B"),
        (22.0, "Qwen/Qwen3-4B"),
        (15.0, "Qwen/Qwen2.5-3B-Instruct"),
        (8.0, "Qwen/Qwen2.5-1.5B-Instruct"),
        (4.0, "Qwen/Qwen2.5-0.5B-Instruct"),
    ],
)
def test_model_is_chosen_by_available_vram(free_gb, expected):
    model, batch, accum = choose_model({"cuda": True, "vram_free_gb": free_gb}, None)
    assert model == expected
    assert batch >= 1 and accum >= 1


def test_no_gpu_falls_back_to_the_smallest_model():
    model, _, _ = choose_model({"cuda": False}, None)
    assert model == MODEL_LADDER[-1][1]


def test_explicit_model_on_the_ladder_keeps_its_tuned_batch_size():
    model, batch, accum = choose_model({"cuda": True, "vram_free_gb": 4.0}, "Qwen/Qwen3-4B")
    assert model == "Qwen/Qwen3-4B"
    assert (batch, accum) == (2, 8)


def test_explicit_model_off_the_ladder_gets_conservative_settings():
    model, batch, accum = choose_model({"cuda": True, "vram_free_gb": 80.0}, "/local/path")
    assert model == "/local/path"
    assert batch == 1


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_slug_is_filesystem_safe():
    assert _slug("Qwen/Qwen2.5-1.5B-Instruct") == "bond_qwen2_5_1_5b_instruct"
    assert "/" not in _slug("org/Model.Name-v2")


def test_truncate_keeps_the_first_n_rows(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(f'{{"i":{i}}}' for i in range(50)) + "\n")
    _truncate(path, 10)
    assert len(path.read_text().strip().splitlines()) == 10


def test_truncate_is_a_noop_when_the_file_is_already_short(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"i":0}\n{"i":1}\n')
    _truncate(path, 10)
    assert len(path.read_text().strip().splitlines()) == 2
