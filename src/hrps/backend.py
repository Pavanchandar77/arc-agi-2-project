"""Bond model-backend registry (private foundation ids).

Public runtime identity is Bond, and only when a Bond adapter is loaded.
Foundation Hugging Face ids stay here for weight loading and the private
manifest. Never silently substitute a different checkpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.hrps.identity import (
    ADAPTER_MISSING,
    PUBLIC_NAME,
    wrap_foundation,
    wrap_scripted_bond,
)
from src.hrps.model import (
    CPU_FOUNDATION,
    CPU_LABEL,
    DEEPSEEK_FOUNDATION,
    DEEPSEEK_LABEL,
    PREFERRED_INKLING,
    PRIMARY_FOUNDATION,
    PRIMARY_LABEL,
    SMALL_EXPERIMENT_FOUNDATION,
    SMALL_LABEL,
    SMOKE_FOUNDATION,
    SMOKE_LABEL,
    FrozenOpenModel,
    ScriptedModel,
    try_load_open_model,
)

FOUNDATIONS: dict[str, dict[str, Any]] = {
    "qwen05b_cpu": {
        "id": "qwen05b_cpu",
        "hf_id": CPU_FOUNDATION,
        "label": CPU_LABEL,
        "public_name": PUBLIC_NAME,
        "is_final_bond": False,
        "local_ok": True,
        "requires_gpu": False,
        "min_ram_gb": 6,
        "min_avail_gb": 3.0,
        "notes": "Laptop CPU LoRA path (Iris Xe / no CUDA). Not a 4B Bond result.",
    },
    "qwen1.5b_smoke": {
        "id": "qwen1.5b_smoke",
        "hf_id": SMOKE_FOUNDATION,
        "label": SMOKE_LABEL,
        "public_name": PUBLIC_NAME,
        "is_final_bond": False,
        "local_ok": True,
        "requires_gpu": False,
        "min_ram_gb": 10,
        "min_avail_gb": 6.0,
        "notes": "Local 1.5B CPU/GPU smoke. Close Discord/Chrome first. Not a 4B Bond result.",
    },
    "qwen3.5_4b": {
        "id": "qwen3.5_4b",
        "hf_id": SMALL_EXPERIMENT_FOUNDATION,
        "label": SMALL_LABEL,
        "public_name": PUBLIC_NAME,
        "is_final_bond": False,
        "local_ok": False,
        "requires_gpu": True,
        "min_ram_gb": 16,
        "first_real_bond_experiment": True,
        "notes": "First real Bond experiment: remote CUDA LoRA/QLoRA. Not laptop smoke. 27B is later.",
    },
    "qwen38_27b": {
        "id": "qwen38_27b",
        "hf_id": PRIMARY_FOUNDATION,
        "label": PRIMARY_LABEL,
        "public_name": PUBLIC_NAME,
        "is_final_bond": False,
        "local_ok": False,
        "requires_gpu": True,
        "refuse_local_download": True,
        "min_ram_gb": 64,
        "min_disk_gb": 80,
        "notes": "Later high-capability Bond foundation. Not the first experiment. Never download to the laptop.",
    },
    "deepseek_v4_flash": {
        "id": "deepseek_v4_flash",
        "hf_id": DEEPSEEK_FOUNDATION,
        "label": DEEPSEEK_LABEL,
        "public_name": PUBLIC_NAME,
        "is_final_bond": False,
        "local_ok": False,
        "requires_gpu": True,
        "refuse_local_download": True,
        "min_ram_gb": 256,
        "notes": "Maximum-capability cluster experiment. Not required for Bond-27B SFT.",
    },
    "inkling_small": {
        "id": "inkling_small",
        "hf_id": os.environ.get("HRPS_INKLING_ID") or PREFERRED_INKLING,
        "label": "Inkling-Small",
        "public_name": PUBLIC_NAME,
        "is_final_bond": False,
        "local_ok": False,
        "requires_gpu": True,
        "min_ram_gb": 64,
        "requires_env": "HRPS_ALLOW_INKLING=1",
        "notes": "Configurable future backend. Not required for unit tests or 1.5B smoke.",
    },
}

ALIASES = {
    "cpu": "qwen05b_cpu",
    "laptop": "qwen05b_cpu",
    "qwen0.5b": "qwen05b_cpu",
    "qwen05b": "qwen05b_cpu",
    CPU_FOUNDATION.lower(): "qwen05b_cpu",
    "smoke": "qwen1.5b_smoke",
    "qwen1.5b": "qwen1.5b_smoke",
    SMOKE_FOUNDATION.lower(): "qwen1.5b_smoke",
    "qwen3.5-4b": "qwen3.5_4b",
    "qwen3.5_4b": "qwen3.5_4b",
    SMALL_EXPERIMENT_FOUNDATION.lower(): "qwen3.5_4b",
    "qwen3.8-27b": "qwen38_27b",
    "qwen38_27b": "qwen38_27b",
    "qwen3.8_27b": "qwen38_27b",
    PRIMARY_FOUNDATION.lower(): "qwen38_27b",
    "deepseek": "deepseek_v4_flash",
    "deepseek_v4_flash": "deepseek_v4_flash",
    DEEPSEEK_FOUNDATION.lower(): "deepseek_v4_flash",
    "inkling": "inkling_small",
    "inkling-small": "inkling_small",
    PREFERRED_INKLING.lower(): "inkling_small",
}


@dataclass(frozen=True)
class BackendSpec:
    foundation_id: str
    hf_id: str
    label: str
    adapter_path: Optional[str]
    local_files_only: bool
    seed: int
    temperature: float
    top_p: float
    is_final_bond: bool
    is_smoke: bool


def resolve_foundation(name: Optional[str]) -> dict[str, Any]:
    if not name:
        return FOUNDATIONS["qwen1.5b_smoke"]
    key = name.strip()
    aliased = ALIASES.get(key.lower(), key)
    if aliased in FOUNDATIONS:
        return FOUNDATIONS[aliased]
    # Bare HF id: do not relabel as Bond.
    return {
        "id": "custom",
        "hf_id": key,
        "label": key,
        "public_name": PUBLIC_NAME,
        "is_final_bond": False,
        "local_ok": True,
        "requires_gpu": False,
        "min_ram_gb": 0,
        "notes": "Unregistered HF id. Will not be labeled final Bond.",
    }


def probe_hardware() -> dict[str, Any]:
    """Measured host facts. Does not assume 8GB when RAM cannot be read."""
    ram_gb = None
    avail_gb = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_gb = round(vm.total / 1e9, 2)
        avail_gb = round(vm.available / 1e9, 2)
    except Exception:
        ram_gb, avail_gb = _windows_ram_gb()
    cuda = False
    torch_ok = False
    try:
        import torch

        torch_ok = True
        cuda = bool(torch.cuda.is_available())
    except Exception:
        torch_ok = False
        cuda = False
    gpu_name = _gpu_name()
    return {
        "ram_gb": ram_gb,
        "avail_gb": avail_gb,
        "torch": torch_ok,
        "cuda": cuda,
        "gpu_name": gpu_name,
    }


def _windows_ram_gb() -> tuple[Optional[float], Optional[float]]:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("s1", ctypes.c_ulonglong),
                ("s2", ctypes.c_ulonglong),
                ("s3", ctypes.c_ulonglong),
                ("s4", ctypes.c_ulonglong),
                ("s5", ctypes.c_ulonglong),
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None, None
        return round(m.ullTotalPhys / 1e9, 2), round(m.ullAvailPhys / 1e9, 2)
    except Exception:
        return None, None


def _gpu_name() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        name = out.decode("utf-8", errors="replace").strip().splitlines()[0].strip()
        return name or None
    except Exception:
        return None


def hardware_gate(spec: dict[str, Any]) -> Optional[str]:
    """Refuse silently-doomed loads. CUDA is required when requires_gpu is set."""
    if spec["id"] == "inkling_small":
        if os.environ.get("HRPS_ALLOW_INKLING", "").strip() not in {"1", "true", "yes"}:
            return (
                "hardware_blocked: Inkling-Small is not a local runtime. "
                "Set HRPS_ALLOW_INKLING=1 only on a machine that actually has the checkpoint. "
                f"For local work use {SMOKE_LABEL} ({SMOKE_FOUNDATION})."
            )
    hw = probe_hardware()
    if spec.get("requires_gpu") and not hw["cuda"]:
        ram = f"{hw['ram_gb']:.1f}GB RAM" if hw["ram_gb"] is not None else "RAM unknown"
        avail = f"{hw['avail_gb']:.1f}GB free" if hw["avail_gb"] is not None else "free RAM unknown"
        gpu = hw["gpu_name"] or "no NVIDIA GPU"
        torch_s = "torch present" if hw["torch"] else "no torch"
        return (
            f"hardware_blocked: {spec['hf_id']} needs CUDA. "
            f"This host: {ram}, {avail}, {gpu}, {torch_s}, cuda={hw['cuda']}. "
            f"Do not silently switch the foundation. On this laptop use "
            f"--foundation qwen05b_cpu or qwen1.5b_smoke after closing Discord/Chrome. "
            f"Qwen3.8-27B adapter training stays a remote NVIDIA job; do not download it here."
        )
    if spec.get("refuse_local_download") and not hw["cuda"]:
        return (
            f"refuse_local_download: {spec['hf_id']} must not be downloaded onto this laptop. "
            f"Prepare the training bundle and run adapter SFT on a remote GPU."
        )
    return None


def free_ram_gate(spec: dict[str, Any]) -> Optional[str]:
    """Train-time check. Tests do not call this, so a loaded desktop won't fail CI."""
    min_avail = spec.get("min_avail_gb")
    if not min_avail:
        return None
    hw = probe_hardware()
    if hw["avail_gb"] is not None and hw["avail_gb"] < float(min_avail):
        return (
            f"hardware_blocked: {spec['hf_id']} wants >={min_avail}GB free RAM "
            f"(this host has {hw['avail_gb']:.1f}GB free of {hw['ram_gb']}GB). "
            f"Quit Discord (~4GB) and Chrome (~3GB), then retry. "
            f"This is still a CPU Bond-smoke run, not Qwen3.5-4B."
        )
    return None


def load_backend(
    foundation: Optional[str] = None,
    *,
    adapter_path: Optional[str] = None,
    seed: int = 0,
    local_files_only: bool = True,
    allow_download: bool = False,
    as_bond: bool = False,
) -> tuple[Optional[FrozenOpenModel], str, dict[str, Any]]:
    """Load foundation weights, optionally the Bond adapter.

    as_bond=True requires a present adapter and returns a Bond identity.
    as_bond=False returns a foundation handle that is not named Bond.
    """
    spec = resolve_foundation(foundation)
    blocked = hardware_gate(spec)
    if blocked:
        return None, blocked, spec
    hf_id = spec["hf_id"]
    if as_bond:
        if not adapter_path or not Path(adapter_path).exists():
            return None, ADAPTER_MISSING, spec
        if Path(adapter_path).is_dir() and not any(Path(adapter_path).iterdir()):
            return None, ADAPTER_MISSING, spec
    elif adapter_path and not Path(adapter_path).exists():
        return None, ADAPTER_MISSING, spec
    offline = local_files_only and not allow_download
    model, status = try_load_open_model(
        hf_id,
        adapter_path=adapter_path if as_bond else None,
        local_files_only=offline,
        seed=seed,
    )
    if model is None:
        return None, status, spec
    if as_bond:
        from src.hrps.identity import load_bond

        bond, bstatus = load_bond(
            model,
            adapter_path=adapter_path,  # type: ignore[arg-type]
            foundation_id=spec["id"],
            foundation_hf_id=hf_id,
            seed=seed,
            is_smoke=spec["id"] == "qwen1.5b_smoke",
        )
        if bond is None:
            return None, bstatus, spec
        return bond, "ok", spec
    wrapped = wrap_foundation(
        model, foundation_id=spec["id"], foundation_hf_id=hf_id, seed=seed
    )
    return wrapped, "ok", spec


def save_scripted_adapter(responses: list[str], path: Path) -> None:
    """Deterministic stand-in adapter for tests. Not a trained Bond checkpoint."""
    import json

    path.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "scripted_adapter", "is_final_bond": False, "responses": list(responses)}
    (path / "scripted_adapter.json").write_text(json.dumps(payload), encoding="utf-8")


def load_scripted_adapter(path: Path, name: str = "scripted_bond"):
    import json

    payload = json.loads((Path(path) / "scripted_adapter.json").read_text(encoding="utf-8"))
    inner = ScriptedModel(responses=list(payload.get("responses") or []), name=name, backend="fake_adapter")
    return wrap_scripted_bond(inner, path)
