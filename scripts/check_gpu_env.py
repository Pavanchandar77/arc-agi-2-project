"""Preflight for Bond-4B native-precision LoRA on a GPU host (Kaggle / Lightning).

Fails clearly. Never swaps Qwen/Qwen3-4B. Never claims a learned Bond
checkpoint. device_map='auto' is model-parallel shard placement, not DDP.

Usage:
  python scripts/check_gpu_env.py
  python scripts/check_gpu_env.py --no-download
  HRPS_ALLOW_DOWNLOAD=1 python scripts/check_gpu_env.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FOUNDATION = "Qwen/Qwen3-4B"
EXPECTED_MODEL_TYPE = "qwen3_5"
SFT_PATH = REPO / "artifacts" / "bond" / "train_scale" / "sft_actions.jsonl"
ARC_TRAIN = REPO / "ARC-AGI-2" / "data" / "training"


def _pkg_version(name: str):
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bond-4B GPU preflight")
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Load Qwen/Qwen3-4B config from local cache only.",
    )
    args = p.parse_args(argv)

    errors: list[str] = []
    rec: dict = {
        "public_name": "Bond",
        "foundation_hf_id": FOUNDATION,
        "adapter_kind": "native_precision_lora",
        "device_map": "auto (model-parallel shard placement, not torchrun DDP)",
        "python": sys.version,
    }

    rec["torch"] = _pkg_version("torch")
    rec["transformers"] = _pkg_version("transformers")
    rec["trl"] = _pkg_version("trl")
    rec["peft"] = _pkg_version("peft")
    rec["torchao"] = _pkg_version("torchao")
    rec["datasets"] = _pkg_version("datasets")
    rec["accelerate"] = _pkg_version("accelerate")

    try:
        import torch

        rec["cuda_available"] = bool(torch.cuda.is_available())
        rec["cuda_version"] = getattr(torch.version, "cuda", None)
        rec["n_gpu"] = int(torch.cuda.device_count()) if rec["cuda_available"] else 0
        rec["gpus"] = []
        if rec["cuda_available"]:
            for i in range(rec["n_gpu"]):
                props = torch.cuda.get_device_properties(i)
                rec["gpus"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    }
                )
        else:
            errors.append("CUDA is not available. Bond-4B training will not run on this host.")
    except Exception as exc:
        rec["cuda_available"] = False
        errors.append(f"torch import failed: {exc}")

    if rec.get("torchao"):
        from src.hrps.hf_compat import TORCHAO_MIN, parse_version_tuple

        if parse_version_tuple(rec["torchao"]) < TORCHAO_MIN:
            errors.append(
                f"incompatible torchao {rec['torchao']} (< 0.16.0). "
                "This run is native-precision LoRA and does not need torchao. "
                "Run: pip uninstall -y torchao"
            )

    if not ARC_TRAIN.is_dir():
        errors.append(f"missing ARC training path: {ARC_TRAIN}")
    else:
        rec["arc_training"] = str(ARC_TRAIN)
        rec["arc_n_json"] = len(list(ARC_TRAIN.glob("*.json")))

    if not SFT_PATH.is_file():
        errors.append(f"missing SFT jsonl: {SFT_PATH}")
    else:
        rec["sft_path"] = str(SFT_PATH)
        rec["sft_n"] = sum(1 for line in SFT_PATH.read_text(encoding="utf-8").splitlines() if line.strip())

    deny = os.environ.get("HRPS_ALLOW_DOWNLOAD", "").strip().lower() in {"0", "false", "no"}
    local_only = bool(args.no_download) or deny
    rec["config_local_files_only"] = local_only
    rec["hrps_allow_download"] = not local_only
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            FOUNDATION,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        rec["model_type"] = getattr(cfg, "model_type", None)
        if rec["model_type"] != EXPECTED_MODEL_TYPE:
            errors.append(
                f"{FOUNDATION} config model_type={rec['model_type']!r}, expected {EXPECTED_MODEL_TYPE!r}. "
                "Install Transformers from current main so qwen3_5 is recognized. "
                "Do not switch the foundation."
            )
    except Exception as exc:
        hint = "Do not substitute another model. Keep Qwen/Qwen3-4B."
        if args.no_download:
            hint = "Config not in local cache. Re-run without --no-download (Kaggle default allows download)."
        errors.append(f"failed to load {FOUNDATION} config: {exc}. {hint}")

    rec["errors"] = errors
    rec["ok"] = not errors
    rec["is_final_bond"] = False
    rec["note"] = (
        "Preflight only. Adapter weights are not a learned Bond checkpoint until "
        "they are saved under models/bond_qwen3_4b and reloaded."
    )
    print(json.dumps(rec, indent=2))
    return 0 if rec["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
