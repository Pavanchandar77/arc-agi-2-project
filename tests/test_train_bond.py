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
    # Its own tuned pair, not the one the (much smaller) VRAM would have picked.
    tuned = {model: (bs, ga) for _, model, bs, ga in MODEL_LADDER}["Qwen/Qwen3-4B"]
    model, batch, accum = choose_model({"cuda": True, "vram_free_gb": 4.0}, "Qwen/Qwen3-4B")
    assert model == "Qwen/Qwen3-4B"
    assert (batch, accum) == tuned


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


# --------------------------------------------------------------------------
# Truncation preflight
# --------------------------------------------------------------------------


def _write_rows(path: Path, char_lengths: list[int]) -> None:
    import json as _json

    with path.open("w", encoding="utf-8") as fh:
        for n in char_lengths:
            fh.write(_json.dumps({"messages": [{"role": "user", "content": "x" * n}]}) + "\n")


def test_truncation_rate_counts_examples_over_the_budget(tmp_path: Path):
    from scripts.train_bond import CHARS_PER_TOKEN, truncation_rate

    budget = 100
    over, under = int(budget * CHARS_PER_TOKEN) + 50, int(budget * CHARS_PER_TOKEN) - 50
    path = tmp_path / "d.jsonl"
    _write_rows(path, [over, over, under, under])
    rate, longest = truncation_rate(path, budget)
    assert rate == 0.5
    assert longest == over


def test_truncation_rate_falls_to_zero_as_the_budget_grows(tmp_path: Path):
    from scripts.train_bond import truncation_rate

    path = tmp_path / "d.jsonl"
    _write_rows(path, [2000, 4000, 8000])
    assert truncation_rate(path, 512)[0] > truncation_rate(path, 8192)[0]
    assert truncation_rate(path, 8192)[0] == 0.0


def test_truncation_rate_survives_a_malformed_line(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"messages":[{"role":"user","content":"xx"}]}\nnot json\n', encoding="utf-8")
    from scripts.train_bond import truncation_rate

    rate, longest = truncation_rate(path, 1024)
    assert rate == 0.0 and longest == 2


def test_truncation_rate_on_an_empty_file_is_zero(tmp_path: Path):
    from scripts.train_bond import truncation_rate

    path = tmp_path / "d.jsonl"
    path.write_text("", encoding="utf-8")
    assert truncation_rate(path, 1024) == (0.0, 0)


def test_default_sequence_length_fits_the_bulk_of_arc_examples():
    # 2048 was measured to cut 16-22% of examples mid-grid; the default moved up.
    from scripts.train_bond import DEFAULT_MAX_SEQ_LENGTH

    assert DEFAULT_MAX_SEQ_LENGTH >= 4096


def test_ladder_keeps_a_constant_effective_batch():
    from scripts.train_bond import MODEL_LADDER

    effective = {bs * ga for _, _, bs, ga in MODEL_LADDER}
    assert len(effective) == 1, f"effective batch varies across the ladder: {effective}"


# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------


def test_colab_is_not_reported_as_kaggle(monkeypatch, tmp_path):
    # A bare /kaggle directory exists on some Colab images. The google.colab
    # module is the definitive signal and must win.
    import scripts.train_bond as tb

    monkeypatch.setattr(tb, "_has_module", lambda name: name == "google.colab")
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising=False)
    assert tb.detect_environment()["environment"] == "colab"


def test_kaggle_is_detected_by_its_own_variables(monkeypatch):
    import scripts.train_bond as tb

    monkeypatch.setattr(tb, "_has_module", lambda name: False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert tb.detect_environment()["environment"] == "kaggle"


def test_a_plain_machine_is_local(monkeypatch):
    import scripts.train_bond as tb

    monkeypatch.setattr(tb, "_has_module", lambda name: False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising=False)
    monkeypatch.setattr(tb.Path, "is_dir", lambda self: False)
    assert tb.detect_environment()["environment"] == "local"


# --------------------------------------------------------------------------
# torchao: PEFT reads distribution metadata, so the package must actually go
# --------------------------------------------------------------------------


def test_an_absent_torchao_needs_no_action(monkeypatch):
    import scripts.train_bond as tb

    monkeypatch.setattr(tb, "_dist_version", lambda name: None)
    assert tb.remove_incompatible_torchao() == "absent"


def test_a_new_enough_torchao_is_kept(monkeypatch):
    import scripts.train_bond as tb

    monkeypatch.setattr(tb, "_dist_version", lambda name: (0, 16, 0))
    assert tb.remove_incompatible_torchao().startswith("kept")


def test_an_old_torchao_is_uninstalled(monkeypatch):
    import scripts.train_bond as tb

    calls = []
    versions = iter([(0, 10, 0), None])  # before, then after the uninstall
    monkeypatch.setattr(tb, "_dist_version", lambda name: next(versions))
    monkeypatch.setattr(tb, "run", lambda cmd, check=True: calls.append(cmd) or 0)
    assert tb.remove_incompatible_torchao().startswith("removed")
    assert any("uninstall" in part for cmd in calls for part in cmd)


def test_an_old_torchao_can_be_kept_with_a_warning(monkeypatch):
    import scripts.train_bond as tb

    monkeypatch.setattr(tb, "_dist_version", lambda name: (0, 10, 0))
    monkeypatch.setattr(tb, "run", lambda cmd, check=True: pytest.fail("must not uninstall"))
    assert "left alone" in tb.remove_incompatible_torchao(allowed=False)


def test_a_failed_uninstall_stops_the_run_rather_than_training_into_a_crash(monkeypatch):
    import scripts.train_bond as tb

    monkeypatch.setattr(tb, "_dist_version", lambda name: (0, 10, 0))
    monkeypatch.setattr(tb, "run", lambda cmd, check=True: 0)
    with pytest.raises(SystemExit) as excinfo:
        tb.remove_incompatible_torchao()
    assert "pip uninstall" in str(excinfo.value)


def test_version_parsing_handles_local_and_short_versions(monkeypatch):
    import importlib.metadata as md

    import scripts.train_bond as tb

    monkeypatch.setattr(md, "version", lambda name: "0.10.0+cu128")
    assert tb._dist_version("x") == (0, 10, 0)
    monkeypatch.setattr(md, "version", lambda name: "1.2")
    assert tb._dist_version("x") == (1, 2)
