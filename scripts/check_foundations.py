"""Verify every pinned foundation id resolves on the Hub, before booking GPU time.

An earlier revision of this repository pinned model ids that are not release
names, so `train_bond_*.py` could never have downloaded anything. Run this from
a machine with network access (Colab, a workstation, a Kaggle notebook with
internet on) and fix anything it reports before starting a training run.

    python scripts/check_foundations.py
    python scripts/check_foundations.py --id Qwen/Qwen3-4B --id my-org/my-model

Exit code is non-zero if any id fails to resolve, so it works as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def pinned_ids() -> dict[str, str]:
    from src.hrps.backend import FOUNDATIONS

    return {key: spec["hf_id"] for key, spec in FOUNDATIONS.items()}


def check(hf_id: str) -> dict[str, object]:
    """Resolve one id. Reports the reason rather than just a boolean."""
    try:
        from huggingface_hub import model_info
    except ImportError:
        return {"id": hf_id, "ok": False, "reason": "huggingface_hub not installed"}
    try:
        info = model_info(hf_id)
    except Exception as exc:
        name = type(exc).__name__
        reason = f"{name}: {str(exc)[:160]}"
        if "401" in str(exc) or "gated" in str(exc).lower():
            reason = f"gated or private; accept the licence and log in ({name})"
        elif "404" in str(exc) or "RepositoryNotFound" in name:
            reason = f"does not exist on the Hub ({name})"
        return {"id": hf_id, "ok": False, "reason": reason}
    files = [s.rfilename for s in (info.siblings or [])]
    has_weights = any(f.endswith((".safetensors", ".bin", ".gguf")) for f in files)
    return {
        "id": hf_id,
        "ok": has_weights,
        "gated": bool(getattr(info, "gated", False)),
        "reason": "" if has_weights else "resolves but carries no weight files",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="check that pinned foundation ids resolve")
    p.add_argument("--id", action="append", default=[], help="extra id to check")
    args = p.parse_args(argv)

    targets = pinned_ids()
    for extra in args.id:
        targets[f"cli:{extra}"] = extra

    rows = []
    for key, hf_id in targets.items():
        result = check(hf_id)
        result["foundation"] = key
        rows.append(result)
        mark = "ok  " if result["ok"] else "FAIL"
        detail = f"  {result['reason']}" if result.get("reason") else ""
        print(f"[{mark}] {key:<20} {hf_id}{detail}", flush=True)

    bad = [r for r in rows if not r["ok"]]
    print(json.dumps({"checked": len(rows), "failed": len(bad)}, indent=2))
    if bad:
        print(
            "\nFix each failing id in src/hrps/model.py (or set HRPS_MODEL) before training.",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
