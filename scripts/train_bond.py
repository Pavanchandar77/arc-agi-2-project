"""One command. Clone the repo, run this, get a trained adapter.

    python scripts/train_bond.py

Works on Colab, a Kaggle GPU notebook, or any CUDA box. It installs what is
missing, fetches the ARC data, picks a model that fits the GPU it actually
finds, builds the augmented dataset, runs LoRA SFT, and saves the adapter.

Every stage is skipped if its output already exists, so re-running after a
disconnect resumes rather than starting over. Nothing is silently guessed:
each stage prints what it decided and why.

Useful overrides, none required:

    --model Qwen/Qwen3-4B     pin the base model instead of auto-selecting
    --epochs 3                training epochs
    --aug-factor 8            augmented variants per task
    --max-tasks 200           smaller dataset for a quick smoke run
    --no-install              skip dependency installation
    --dry-run                 print the plan and exit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARC1_REPO = "https://github.com/fchollet/ARC-AGI.git"
ARC2_REPO = "https://github.com/arcprize/ARC-AGI-2.git"

# (min free VRAM in GB, model id, per-device batch size, grad accumulation).
# Ordered largest first; the first entry that fits wins. Batch sizes assume
# DEFAULT_MAX_SEQ_LENGTH and keep the effective batch at 16 throughout, so the
# learning rate stays comparable across the ladder. Activations dominate at
# these sequence lengths, which is why even 60 GB runs a per-device batch of 1.
MODEL_LADDER = (
    (60, "Qwen/Qwen3-14B", 1, 16),
    (36, "Qwen/Qwen3-8B", 1, 16),
    (20, "Qwen/Qwen3-4B", 1, 16),
    (12, "Qwen/Qwen2.5-3B-Instruct", 1, 16),
    (7, "Qwen/Qwen2.5-1.5B-Instruct", 2, 8),
    (0, "Qwen/Qwen2.5-0.5B-Instruct", 2, 8),
)

# ARC-AGI-2 grids are large. Measured over the built dataset, a 2048-token
# budget truncates 16-22% of examples mid-grid, which teaches the model to emit
# cut-off answers; 4096 truncates under 4%, and 8192 truncates none. 4096 is the
# default because 8192 doubles activation memory for the last few percent.
DEFAULT_MAX_SEQ_LENGTH = 4096

# Rough bytes-per-character of the serialized prompts under a BPE tokenizer on
# digit grids. Only used to warn about truncation before a run, never to size
# anything, so an approximation is fine.
CHARS_PER_TOKEN = 2.5

REQUIRED = ("torch", "transformers", "peft", "trl", "datasets", "accelerate")


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def run(cmd: list[str], *, check: bool = True) -> int:
    log("run", " ".join(cmd))
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _has_module(name: str) -> bool:
    """find_spec raises, not returns None, when a parent package is absent."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def detect_environment() -> dict:
    # Order matters. A bare /kaggle directory is not proof of the Kaggle
    # runtime - Colab images can carry one for dataset integration - so the
    # definitive google.colab module is checked first, and Kaggle is then
    # identified by the variables its kernels actually set.
    if _has_module("google.colab"):
        env = "colab"
    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        env = "kaggle"
    elif Path("/kaggle/working").is_dir() and Path("/kaggle/input").is_dir():
        env = "kaggle"
    else:
        env = "local"
    info = {"environment": env, "python": sys.version.split()[0], "cuda": False}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["n_gpu"] = torch.cuda.device_count()
            info["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info["vram_free_gb"] = round(free / 1e9, 1)
            info["vram_total_gb"] = round(total / 1e9, 1)
    except Exception:
        info["torch"] = None
    return info


def missing_packages() -> list[str]:
    return [p for p in REQUIRED if not _has_module(p)]


# PEFT refuses to inject LoRA when torchao is installed below this version. It
# reads the distribution metadata, so stubbing sys.modules cannot satisfy it -
# the package has to actually go. Nothing in this pipeline uses torchao: this is
# native-precision LoRA, not quantization.
TORCHAO_MIN_FOR_PEFT = (0, 16, 0)


def _dist_version(name: str) -> Optional[tuple[int, ...]]:
    try:
        import importlib.metadata as md

        raw = md.version(name).split("+")[0].split(".")
        return tuple(int("".join(c for c in part if c.isdigit()) or 0) for part in raw[:3])
    except Exception:
        return None


def remove_incompatible_torchao(*, allowed: bool = True) -> str:
    """Uninstall a torchao too old for PEFT. Returns what was decided."""
    version = _dist_version("torchao")
    if version is None:
        return "absent"
    if version >= TORCHAO_MIN_FOR_PEFT:
        return f"kept {'.'.join(map(str, version))}"
    pretty = ".".join(map(str, version))
    if not allowed:
        log("deps", f"WARNING: torchao {pretty} will make PEFT refuse to inject LoRA. "
                    f"Run: pip uninstall -y torchao")
        return f"incompatible {pretty}, left alone"
    log("deps", f"uninstalling torchao {pretty}: PEFT requires >= 0.16 and this "
                f"pipeline never uses it")
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    still = _dist_version("torchao")
    if still is not None and still < TORCHAO_MIN_FOR_PEFT:
        raise SystemExit(
            f"torchao {pretty} is still installed and PEFT will refuse to inject "
            f"LoRA. Remove it manually: pip uninstall -y torchao"
        )
    return f"removed {pretty}"


def install_dependencies(env: str) -> None:
    missing = missing_packages()
    if not missing:
        log("deps", "all present, nothing to install")
        return
    log("deps", f"installing {', '.join(missing)}")
    pip = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"]
    # torch is preinstalled on Colab and Kaggle and pinned to their CUDA build;
    # reinstalling it from PyPI is how those runtimes get broken.
    if "torch" in missing and env in {"colab", "kaggle"}:
        raise SystemExit(
            "torch is missing on a hosted runtime, which should not happen. "
            "Check that the notebook has an accelerator enabled, and do not "
            "pip install torch here - it would replace the runtime's CUDA build."
        )
    run(pip + missing)
    still = missing_packages()
    if still:
        raise SystemExit(f"still missing after install: {', '.join(still)}")
    log("deps", "install complete")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def ensure_data() -> Path:
    """Return a directory holding ARC task json, cloning it if needed."""
    for candidate in (REPO / "ARC-AGI-2" / "data" / "training", REPO / "ARC-AGI" / "data" / "training"):
        if candidate.is_dir() and any(candidate.glob("*.json")):
            log("data", f"using existing {candidate}")
            return candidate
    # A Kaggle notebook may already have the competition data mounted.
    for mount in sorted(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").is_dir() else []:
        hit = mount / "arc-agi_training_challenges.json"
        if hit.is_file():
            log("data", f"found Kaggle mount {hit}")
            return mount
    if shutil.which("git") is None:
        raise SystemExit("git not found and no ARC data present; clone ARC-AGI-2 manually")
    target = REPO / "ARC-AGI-2"
    log("data", f"cloning {ARC2_REPO}")
    run(["git", "clone", "--depth", "1", ARC2_REPO, str(target)])
    folder = target / "data" / "training"
    if not folder.is_dir():
        raise SystemExit(f"clone succeeded but {folder} is missing")
    return folder


def build_dataset(data_dir: Path, out_dir: Path, aug_factor: int, max_tasks: Optional[int]) -> tuple[Path, Path]:
    train_file = out_dir / "arc_train.jsonl"
    val_file = out_dir / "arc_val.jsonl"
    if train_file.is_file() and val_file.is_file():
        log("dataset", f"reusing {train_file} ({train_file.stat().st_size // 1024} KB)")
        return train_file, val_file
    cmd = [
        sys.executable, str(REPO / "src" / "build_dataset.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--aug-factor", str(aug_factor),
    ]
    run(cmd)
    if not train_file.is_file():
        raise SystemExit(f"dataset build produced no {train_file}")
    if max_tasks:
        _truncate(train_file, max_tasks)
    return train_file, val_file


def program_corpus(path: Path, *, val_fraction: float = 0.1) -> tuple[Path, Path]:
    """Split a harvested program corpus into train and validation files.

    The split is by task, not by row: the same task can yield several programs,
    and letting one land in train while another lands in validation would leak
    the answer across the split and make the eval loss a lie.
    """
    if not path.is_file():
        raise SystemExit(
            f"no program corpus at {path}. Build one first:\n"
            f"  python scripts/harvest_programs.py --splits training --out data/programs"
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty; harvest found no verified programs")
    task_ids = sorted({r.get("task_id", "") for r in rows})
    n_val = max(1, int(len(task_ids) * val_fraction)) if len(task_ids) > 1 else 0
    val_ids = set(task_ids[:n_val])
    train_rows = [r for r in rows if r.get("task_id") not in val_ids]
    val_rows = [r for r in rows if r.get("task_id") in val_ids]
    if not train_rows:  # tiny corpus: keep everything trainable
        train_rows, val_rows = rows, []
    out_dir = path.parent
    train_file = out_dir / "programs_train.jsonl"
    val_file = out_dir / "programs_val.jsonl"
    train_file.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n", encoding="utf-8")
    if val_rows:
        val_file.write_text("\n".join(json.dumps(r) for r in val_rows) + "\n", encoding="utf-8")
    log(
        "dataset",
        f"{len(rows)} verified programs over {len(task_ids)} tasks -> "
        f"{len(train_rows)} train / {len(val_rows)} val (split by task)",
    )
    return train_file, val_file


def truncation_rate(path: Path, max_seq_length: int) -> tuple[float, int]:
    """Estimated share of examples that will not fit, and the longest seen.

    A truncated ARC example loses the tail of the answer grid, so the model is
    trained to stop mid-output. That is worse than dropping the example, and it
    is silent, so it is worth an explicit check before a GPU run.
    """
    budget_chars = max_seq_length * CHARS_PER_TOKEN
    lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "messages" in row:
                lengths.append(sum(len(m.get("content", "")) for m in row["messages"]))
            elif "prompt" in row:
                lengths.append(len(row.get("prompt", "")) + len(row.get("completion", "")))
    if not lengths:
        return 0.0, 0
    over = sum(1 for v in lengths if v > budget_chars)
    return over / len(lengths), max(lengths)


def _truncate(path: Path, n: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()[:n]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("dataset", f"truncated {path.name} to {len(lines)} rows")


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------


def choose_model(info: dict, explicit: Optional[str]) -> tuple[str, int, int]:
    if explicit:
        for _, model, bs, ga in MODEL_LADDER:
            if model == explicit:
                return explicit, bs, ga
        log("model", f"{explicit} is not on the ladder; using conservative batch settings")
        return explicit, 1, 16
    if not info.get("cuda"):
        model, bs, ga = MODEL_LADDER[-1][1:]
        log("model", f"no CUDA: falling back to {model} (this will be slow and is only a smoke test)")
        return model, bs, ga
    free = float(info.get("vram_free_gb") or 0.0)
    for need, model, bs, ga in MODEL_LADDER:
        if free >= need:
            log("model", f"{free} GB free -> {model} (batch {bs} x accum {ga})")
            return model, bs, ga
    model, bs, ga = MODEL_LADDER[-1][1:]
    return model, bs, ga


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="one-command Bond training")
    p.add_argument("--model", default=None, help="base model id; auto-selected by VRAM if omitted")
    p.add_argument("--output-dir", default=None, help="where the adapter is written")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--aug-factor", type=int, default=8)
    p.add_argument("--max-tasks", type=int, default=None, help="truncate the dataset for a smoke run")
    p.add_argument(
        "--programs",
        default=None,
        help="train on a harvested program corpus (data/programs/programs.jsonl) "
             "instead of grid answers, so every label is one the verifier certified",
    )
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-install", action="store_true")
    p.add_argument("--keep-torchao", action="store_true",
                   help="warn about an incompatible torchao instead of removing it")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    started = time.perf_counter()
    info = detect_environment()
    log("env", json.dumps(info))

    if not args.no_install:
        install_dependencies(info["environment"])
        log("deps", f"torchao: {remove_incompatible_torchao(allowed=not args.keep_torchao)}")
        info = detect_environment()  # torch may have appeared
        log("env", json.dumps(info))

    if not info.get("cuda"):
        log("env", "WARNING: no GPU detected. Training will be extremely slow.")
        log("env", "On Colab: Runtime > Change runtime type > GPU. On Kaggle: Settings > Accelerator.")

    model_name, batch_size, grad_accum = choose_model(info, args.model)
    out_dir = Path(args.output_dir) if args.output_dir else REPO / "models" / _slug(model_name)

    if args.dry_run:
        print(json.dumps({
            "environment": info, "model": model_name, "batch_size": batch_size,
            "grad_accum": grad_accum, "output_dir": str(out_dir), "epochs": args.epochs,
        }, indent=2))
        return 0

    if (out_dir / "adapter_model.safetensors").is_file():
        log("train", f"adapter already exists at {out_dir}; delete it to retrain")
        return 0

    if args.programs:
        train_file, val_file = program_corpus(Path(args.programs))
    else:
        data_dir = ensure_data()
        train_file, val_file = build_dataset(
            data_dir, REPO / "data" / "processed", args.aug_factor, args.max_tasks
        )

    rate, longest = truncation_rate(train_file, args.max_seq_length)
    log(
        "dataset",
        f"longest example ~{int(longest / CHARS_PER_TOKEN)} tokens; "
        f"{rate:.1%} exceed max_seq_length={args.max_seq_length}",
    )
    if rate > 0.05:
        log(
            "dataset",
            f"WARNING: {rate:.1%} of examples will be cut mid-grid, training the "
            f"model to stop early. Raise --max-seq-length (8192 fits everything) "
            f"or accept the loss deliberately.",
        )

    # Long sequences are activation-bound; trading compute for memory is what
    # makes these lengths fit on a single consumer GPU at all.
    use_checkpointing = args.max_seq_length >= 4096
    log("train", f"{model_name} -> {out_dir} (gradient_checkpointing={use_checkpointing})")
    from src.train import train

    train(
        train_file=str(train_file),
        val_file=str(val_file) if val_file.is_file() else None,
        model_name=model_name,
        output_dir=str(out_dir),
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_train_epochs=args.epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        seed=args.seed,
        gradient_checkpointing=use_checkpointing,
    )

    mins = (time.perf_counter() - started) / 60
    log("done", f"adapter written to {out_dir} in {mins:.1f} min")
    print(
        "\nNext:\n"
        f"  1. Upload {out_dir} (and the base model) as a Kaggle Dataset.\n"
        "  2. Use kaggle/arc_prize_llm_notebook.py, which loads them offline.\n"
        "  3. Score it locally first:\n"
        "     python -m src.kaggle_llm_run --challenges <eval_challenges.json> \\\n"
        "       --solutions <eval_solutions.json> --model-path <base> \\\n"
        f"       --adapter-path {out_dir} --ttt-steps 20\n",
        flush=True,
    )
    return 0


def _slug(model_name: str) -> str:
    return "bond_" + model_name.split("/")[-1].replace(".", "_").replace("-", "_").lower()


if __name__ == "__main__":
    sys.exit(main())
