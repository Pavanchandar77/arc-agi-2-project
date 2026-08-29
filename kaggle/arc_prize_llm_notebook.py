"""ARC Prize submission notebook, symbolic + LLM. Paste these cells into Kaggle.

This is the GPU variant of `arc_prize_notebook.py`. It runs the symbolic pass
first, then spends the remaining time running a model over only the tasks the
symbolic pass could not verify.

Notebook settings: Internet **off**, Accelerator **GPU**, competition dataset
attached, and your model attached as a second Dataset.

Getting the weights in, since internet is off and nothing can be downloaded:

  1. On a machine with network access, download the model and save it locally
     (`AutoModelForCausalLM.from_pretrained(id).save_pretrained(dir)` plus the
     tokenizer), or fine-tune it and save the adapter.
  2. Upload that directory as a Kaggle Dataset.
  3. Attach it here. Cell 2 finds any directory holding a config.json beside
     safetensors, so the dataset slug does not matter.

Before you spend GPU hours, run `python scripts/check_foundations.py` somewhere
with a network to confirm your model id actually resolves on the Hub.
"""

# %% [cell] 1. Load the solver
import json
import os
import sys
import time
from pathlib import Path

t_start = time.time()


def _load_solver():
    for name in ("arc_bundle",):
        try:
            __import__(name)
            return f"bundle:{name}"
        except ImportError:
            pass
    for base in (Path("/kaggle/input"), Path("/kaggle/working"), Path(".")):
        if not base.is_dir():
            continue
        for hit in sorted(base.glob("*/src/kaggle_run.py")) + sorted(base.glob("src/kaggle_run.py")):
            root = str(hit.parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return f"path:{root}"
    raise RuntimeError("solver not found; see kaggle/README.md")


print("solver source:", _load_solver())
from src.hrps.llm_solver import find_model_dir  # noqa: E402
from src.kaggle_run import find_challenges, validate_submission  # noqa: E402


# %% [cell] 2. Check the environment before spending the budget
import torch  # noqa: E402

CHALLENGES = find_challenges()
MODEL_DIR = find_model_dir("/kaggle/input")

print("challenges:", CHALLENGES)
print("model dir :", MODEL_DIR)
print("cuda      :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# The run still produces a valid submission without a model, but it will score
# whatever the symbolic layer scores, which on ARC-AGI-2 is near zero.
if MODEL_DIR is None:
    print("\nWARNING: no model found. Attach your weights as a Dataset, or this "
          "is just the symbolic run.")


# %% [cell] 3. Run both phases
# The symbolic phase is cheap and its answers are verified against every
# demonstration, so it goes first and the LLM never overwrites what it solved.
# TTT_STEPS=0 leaves the model frozen; 20-40 is the usual range when adapting.
TOTAL_SECONDS = float(os.environ.get("ARC_TOTAL_SECONDS", 11 * 3600))
TTT_STEPS = 0

# Kaggle grades whatever lands in /kaggle/working; off Kaggle it does not exist,
# and the write would fail only at the final flush, after the whole run.
WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
SUBMISSION = WORKING / "submission.json"

argv = [
    "--challenges", str(CHALLENGES),
    "--output", str(SUBMISSION),
    "--total-seconds", str(TOTAL_SECONDS),
    "--symbolic-fraction", "0.15",
    "--symbolic-per-task", "20",
    "--llm-per-task", "90",
    "--ttt-steps", str(TTT_STEPS),
    "--max-new-tokens", "1024",
    "--workers", str(os.cpu_count() or 4),
]
if MODEL_DIR:
    argv += ["--model-path", MODEL_DIR]
else:
    argv += ["--no-llm"]

from src.kaggle_llm_run import main as run_both  # noqa: E402

exit_code = run_both(argv)
print("exit code:", exit_code)


# %% [cell] 4. Verify before submitting
submission = json.loads(SUBMISSION.read_text())
expected = sorted(json.loads(Path(CHALLENGES).read_text()))

problems = validate_submission(submission, expected)
print(f"tasks in submission: {len(submission)} / {len(expected)} expected")
print(f"schema problems: {problems if problems else 'none'}")
print(f"total notebook time: {(time.time() - t_start) / 60:.1f} min")
assert not problems, problems
assert len(submission) == len(expected), "submission is missing task ids"
print("submission.json is ready")
