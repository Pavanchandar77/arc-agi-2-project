"""Bond identity: public Bond name, adapter required, foundation not overwritten."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hrps.backend import load_backend, load_scripted_adapter, save_scripted_adapter
from src.hrps.identity import (
    ADAPTER_MISSING,
    PUBLIC_NAME,
    load_bond,
    wrap_foundation,
)
from src.hrps.kinds import Kind, kind_of
from src.hrps.model import ScriptedModel
from src.hrps.package import merge_bond_checkpoint, write_bond_package, write_remote_train_bundle


def test_identity_component_labeled():
    assert kind_of("bond_identity") is Kind.EXACT


def test_cpu_foundation_does_not_require_cuda():
    from src.hrps.backend import hardware_gate, resolve_foundation

    spec = resolve_foundation("laptop")
    assert spec["id"] == "qwen05b_cpu"
    assert spec["hf_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert spec["requires_gpu"] is False
    assert hardware_gate(spec) is None


def test_qwen35_gate_requires_cuda_and_reports_measured_ram():
    from src.hrps.backend import hardware_gate, probe_hardware, resolve_foundation

    hw = probe_hardware()
    spec = resolve_foundation("qwen3.5_4b")
    blocked = hardware_gate(spec)
    if hw["cuda"]:
        assert blocked is None
    else:
        assert blocked is not None
        assert "CUDA" in blocked
        assert "8GB-RAM job" not in blocked
        if hw["ram_gb"] is not None:
            assert f"{hw['ram_gb']:.1f}GB" in blocked


def test_bond_refuses_missing_adapter(tmp_path: Path):
    inner = ScriptedModel(responses=["1 0\n0 2"])
    model, status = load_bond(
        inner,
        adapter_path=tmp_path / "missing",
        foundation_id="qwen3.5_4b",
        foundation_hf_id="Qwen/Qwen3-4B",
    )
    assert model is None
    assert status == ADAPTER_MISSING
    bare, st, _ = load_backend("qwen1.5b_smoke", as_bond=True, adapter_path=str(tmp_path / "nope"))
    assert bare is None
    assert ADAPTER_MISSING in st


def test_foundation_handle_is_not_named_bond():
    inner = ScriptedModel(responses=["ok"])
    f = wrap_foundation(inner, foundation_id="qwen3.5_4b", foundation_hf_id="Qwen/Qwen3-4B")
    assert f.is_bond is False
    assert f.name != PUBLIC_NAME
    assert "Qwen" not in f.name


def test_bond_handle_hides_qwen_in_runtime_name(tmp_path: Path):
    save_scripted_adapter(["2 0\n0 1"], tmp_path)
    bond = load_scripted_adapter(tmp_path)
    assert bond.is_bond is True
    assert bond.name in {PUBLIC_NAME, "Bond-smoke"}
    assert "Qwen" not in bond.name
    prov = bond.provenance()
    assert "foundation_hf_id" in prov  # private provenance only


def test_merge_refuses_to_overwrite_foundation(tmp_path: Path):
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    (foundation / "weights.bin").write_text("base", encoding="utf-8")
    rec = merge_bond_checkpoint(foundation, tmp_path / "no_adapter", foundation)
    assert rec["status"] == "blocked"
    assert "overwrite" in rec["reason"]
    assert (foundation / "weights.bin").read_text(encoding="utf-8") == "base"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("transformers") is not None,
    reason="covers the fallback taken when transformers is absent; with it "
    "installed the real merge path runs and rejects these placeholder files",
)
def test_merge_without_torch_copies_adapter_to_new_dir(tmp_path: Path):
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    (foundation / "weights.bin").write_text("base", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.bin").write_text("lora", encoding="utf-8")
    out = tmp_path / "merged"
    rec = merge_bond_checkpoint(foundation, adapter, out)
    assert rec["overwrote_foundation"] is False
    assert rec["public_name"] == PUBLIC_NAME
    assert (foundation / "weights.bin").read_text(encoding="utf-8") == "base"
    assert (out / "bond_model" / "adapter.bin").is_file()


def test_package_layout(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "a.bin").write_text("x", encoding="utf-8")
    root = write_bond_package(tmp_path / "release", adapter_dir=adapter, manifest={"public_name": "Bond"})
    for name in (
        "bond_model",
        "bond_controller",
        "bond_interface",
        "bond_hrps",
        "bond_executor",
        "bond_verifier",
        "bond_manifest.json",
        "LICENSE_ATTRIBUTION.txt",
        "LAYOUT.json",
    ):
        assert (root / name).exists()
    man = (root / "bond_manifest.json").read_text(encoding="utf-8")
    assert '"public_name": "Bond"' in man


def test_remote_bundle_records_command(tmp_path: Path):
    eps = tmp_path / "episodes.jsonl"
    eps.write_text("{}\n", encoding="utf-8")
    dest = write_remote_train_bundle(
        tmp_path / "bundle",
        episodes_path=eps,
        holdout_ids=["73182012"],
        command={"foundation": "qwen3.5_4b", "seed": 42},
    )
    cmd = (dest / "COMMAND.json").read_text(encoding="utf-8")
    assert "qwen3.5_4b" in cmd
    assert "do_not_overwrite_foundation" in cmd
    assert "Bond" in cmd
