"""Evaluation Pipeline for ARC-AGI-2 with Optional TTT and Offline Self-Verification.

Supports three evaluation modes (compare these three numbers on a GPU):
1. Direct Inference: Fast generation from the base LoRA model (greedy + one sample).
2. Test-Time Training (TTT) (--use-ttt): Per-puzzle adaptation, then greedy + one sample,
   with a strict weight reset between tasks.
3. TTT + Verification (--use-ttt --use-verification): After TTT, generate more than 2
   candidates, rank them by how well the same adapted model reconstructs the task's own
   known training pairs, and submit the top 2 unique grids (ARC pass@2).

Exact-Match Rule: Every cell and dimension must match exactly (100% match, no partial credit).
Verification is fully offline (no internet, no external APIs) and is legal in the
sandboxed competition environment.

Usage on GPU environment:
    # Direct evaluation (no TTT)
    python src/evaluate.py --val-file data/processed/arc_val.jsonl --adapter-path models/arc_qwen_1.5b_lora

    # Test-Time Training evaluation (TTT)
    python src/evaluate.py --val-file data/processed/arc_val.jsonl --adapter-path models/arc_qwen_1.5b_lora --use-ttt --ttt-steps 30

    # TTT + verification-based pass@2 candidate selection
    python src/evaluate.py --val-file data/processed/arc_val.jsonl --adapter-path models/arc_qwen_1.5b_lora --use-ttt --use-verification --n-candidates 6
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    DEFAULT_SYSTEM_PROMPT,
    grids_equal,
    is_valid_grid,
    text_to_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_cell_accuracy(pred: Optional[List[List[int]]], gt: List[List[int]]) -> float:
    """Calculate percentage of correctly predicted cells if dimensions match."""
    if pred is None or not is_valid_grid(pred) or not is_valid_grid(gt):
        return 0.0
    if len(pred) != len(gt) or len(pred[0]) != len(gt[0]):
        return 0.0
    
    total_cells = len(gt) * len(gt[0])
    if total_cells == 0:
        return 1.0
    
    correct = sum(
        1 for r in range(len(gt)) for c in range(len(gt[0]))
        if pred[r][c] == gt[r][c]
    )
    return correct / total_cells


def evaluate_task_predictions(
    ground_truth_grid: List[List[int]],
    attempt_1_raw: str,
    attempt_2_raw: Optional[str] = None,
) -> Dict[str, Any]:
    """Score predictions for a single ARC test problem. Strict Exact Match (no partial credit)."""
    grid_1 = text_to_grid(attempt_1_raw)
    grid_2 = text_to_grid(attempt_2_raw) if attempt_2_raw is not None else None

    em_1 = grids_equal(grid_1, ground_truth_grid)
    em_2 = grids_equal(grid_2, ground_truth_grid) if grid_2 is not None else False
    
    exact_match = em_1 or em_2

    gt_h, gt_w = len(ground_truth_grid), len(ground_truth_grid[0])
    shape_match_1 = (grid_1 is not None and len(grid_1) == gt_h and len(grid_1[0]) == gt_w)
    shape_match_2 = (grid_2 is not None and len(grid_2) == gt_h and len(grid_2[0]) == gt_w)

    cell_acc_1 = calculate_cell_accuracy(grid_1, ground_truth_grid)
    cell_acc_2 = calculate_cell_accuracy(grid_2, ground_truth_grid) if grid_2 is not None else 0.0

    return {
        "exact_match": exact_match,
        "em_attempt_1": em_1,
        "em_attempt_2": em_2,
        "shape_match": shape_match_1 or shape_match_2,
        "best_cell_accuracy": max(cell_acc_1, cell_acc_2),
        "cell_accuracy_1": cell_acc_1,
        "cell_accuracy_2": cell_acc_2,
        "attempt_1_grid": grid_1,
        "attempt_2_grid": grid_2,
        "ground_truth_grid": ground_truth_grid,
    }


def _eval_mode_label(use_ttt: bool, use_verification: bool) -> str:
    if use_ttt and use_verification:
        return "TTT+Verification"
    if use_verification:
        return "Direct+Verification"
    if use_ttt:
        return "TTT"
    return "Direct"


def run_evaluation(
    val_file: str = "data/processed/arc_val.jsonl",
    raw_tasks_dir: Optional[str] = "ARC-AGI-2/data/training",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: Optional[str] = "models/arc_qwen_1.5b_lora",
    output_report: str = "eval_results.json",
    use_ttt: bool = False,
    ttt_steps: int = 30,
    ttt_lr: float = 5e-4,
    use_verification: bool = False,
    n_candidates: int = 6,
    condition_on_candidate: bool = True,
    max_samples: Optional[int] = None,
):
    """Run ARC evaluation with optional TTT and/or verification-based selection."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    from src.test_time_train import (
        capture_trainable_state,
        predict_task_with_ttt,
        verify_weight_equality,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_label = _eval_mode_label(use_ttt, use_verification)
    if use_ttt:
        mode_str = f"{mode_label} (TTT {ttt_steps} steps"
        if use_verification:
            mode_str += f", {n_candidates} candidates, verification on"
        mode_str += ")"
    elif use_verification:
        mode_str = f"{mode_label} ({n_candidates} candidates, no TTT)"
    else:
        mode_str = "DIRECT Inference (No TTT)"
    logger.info(f"Starting ARC-AGI-2 evaluation in mode: {mode_str}")

    # 1. Load Tokenizer & Base Model
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model {model_name} on {device}...")
    torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # 2. Load Fine-Tuned LoRA Adapter
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Loading PEFT LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=use_ttt)
    else:
        logger.info("No adapter provided; evaluating base model.")

    # 3. Capture Base State Snapshot for TTT Weight Reset
    base_state = capture_trainable_state(model)
    logger.info(f"Captured base trainable state ({len(base_state)} LoRA tensors).")

    # 4. Load raw tasks dictionary if available for complete task structure in TTT
    raw_tasks = {}
    if raw_tasks_dir and os.path.exists(raw_tasks_dir):
        for jf in Path(raw_tasks_dir).rglob("*.json"):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    raw_tasks[jf.stem] = json.load(f)
            except Exception:
                pass

    # 5. Load Validation Dataset
    logger.info(f"Loading evaluation dataset from {val_file}...")
    dataset = load_dataset("json", data_files={"val": val_file})["val"]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_tasks = len(dataset)
    logger.info(f"Evaluating {total_tasks} held-out validation tasks ({mode_str})...")

    results = []
    total_exact_match = 0
    total_shape_matches = 0
    verification_metadata_list: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset, 1):
        task_id = sample.get("task_id", f"task_{idx}")
        parent_id = sample.get("parent_task_id", task_id.split("_")[0])
        messages = sample.get("messages", [])
        completion_str = sample.get("completion", "")

        gt_grid = text_to_grid(completion_str)
        if gt_grid is None:
            for m in messages:
                if m.get("role") == "assistant":
                    gt_grid = text_to_grid(m.get("content", ""))
                    break

        if gt_grid is None:
            logger.warning(f"Could not parse ground truth for {task_id}, skipping.")
            continue

        ver_info: Optional[Dict[str, Any]] = None
        consistency_str = ""

        if (use_ttt or use_verification) and parent_id in raw_tasks:
            task_dict = raw_tasks[parent_id]
            if use_verification:
                from src.verify_and_select import predict_task_with_verified_selection

                att_1_str, att_2_str, ver_info = predict_task_with_verified_selection(
                    model=model,
                    tokenizer=tokenizer,
                    task=task_dict,
                    base_state=base_state,
                    test_idx=0,
                    use_ttt=use_ttt,
                    ttt_steps=ttt_steps,
                    learning_rate=ttt_lr,
                    device=device,
                    n_candidates=n_candidates,
                    condition_on_candidate=condition_on_candidate,
                )
                ver_info = dict(ver_info)
                ver_info["task_id"] = task_id
                verification_metadata_list.append(ver_info)
                cscore = ver_info.get("training_consistency", {}).get("consistency_score", 0.0)
                consistency_str = f", consistency={cscore:.2f}"
            else:
                att_1_str, att_2_str = predict_task_with_ttt(
                    model=model,
                    tokenizer=tokenizer,
                    task=task_dict,
                    base_state=base_state,
                    test_idx=0,
                    ttt_steps=ttt_steps,
                    learning_rate=ttt_lr,
                    device=device,
                )
            # Verify weight isolation
            assert verify_weight_equality(model, base_state), f"Weight reset verification failed after task {task_id}!"
        else:
            if (use_ttt or use_verification) and parent_id not in raw_tasks:
                logger.warning(
                    f"Raw task '{parent_id}' not found in {raw_tasks_dir}; "
                    "falling back to direct inference for this sample."
                )
            # Direct Inference (No TTT)
            model.eval()
            if hasattr(tokenizer, "apply_chat_template"):
                input_prompt = tokenizer.apply_chat_template(
                    [m for m in messages if m.get("role") != "assistant"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                user_text = next((m["content"] for m in messages if m.get("role") == "user"), "")
                input_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_text}\nOutput:\n"

            inputs = tokenizer(input_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                out_1 = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_1 = out_1[0][inputs["input_ids"].shape[1]:]
            att_1_str = tokenizer.decode(gen_1, skip_special_tokens=True)

            with torch.no_grad():
                out_2 = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_2 = out_2[0][inputs["input_ids"].shape[1]:]
            att_2_str = tokenizer.decode(gen_2, skip_special_tokens=True)

        metrics = evaluate_task_predictions(gt_grid, att_1_str, att_2_str)
        metrics["task_id"] = task_id
        metrics["raw_attempt_1"] = att_1_str
        metrics["raw_attempt_2"] = att_2_str
        if ver_info is not None:
            metrics["verification"] = ver_info
        results.append(metrics)

        if metrics["exact_match"]:
            total_exact_match += 1
        if metrics["shape_match"]:
            total_shape_matches += 1

        status = "PASSED (Exact Match)" if metrics["exact_match"] else "FAILED"
        logger.info(
            f"[{idx}/{total_tasks}] {task_id}: {status} "
            f"(Best Cell Acc: {metrics['best_cell_accuracy']*100:.1f}%{consistency_str})"
        )

    num_eval = len(results)
    if num_eval == 0:
        logger.error("No valid examples evaluated.")
        return

    accuracy = (total_exact_match / num_eval) * 100.0
    shape_rate = (total_shape_matches / num_eval) * 100.0

    print("\n" + "=" * 60)
    print(f"       ARC-AGI-2 EVALUATION ({mode_label})")
    print("=" * 60)
    print(f"Total Validation Cases Evaluated : {num_eval}")
    print(f"Exact-Match Tasks Solved         : {total_exact_match}/{num_eval}")
    print(f"Exact-Match Accuracy             : {accuracy:.2f}% (No partial credit)")
    print(f"Grid Dimension Match Rate        : {shape_rate:.2f}%")

    if use_verification and verification_metadata_list:
        avg_consistency = sum(
            vm.get("training_consistency", {}).get("consistency_score", 0.0)
            for vm in verification_metadata_list
        ) / len(verification_metadata_list)
        avg_valid = sum(
            vm.get("num_valid_candidates", 0) for vm in verification_metadata_list
        ) / len(verification_metadata_list)
        print(f"Avg Training Consistency Score   : {avg_consistency:.2f}")
        print(f"Avg Valid Candidates per Task    : {avg_valid:.1f}")

    print("=" * 60 + "\n")

    report_payload = {
        "summary": {
            "mode": mode_label,
            "ttt_steps": ttt_steps if use_ttt else 0,
            "use_ttt": use_ttt,
            "use_verification": use_verification,
            "n_candidates": n_candidates if use_verification else 2,
            "condition_on_candidate": condition_on_candidate if use_verification else False,
            "total_evaluated": num_eval,
            "exact_match_count": total_exact_match,
            "exact_match_accuracy_percent": round(accuracy, 2),
            "dimension_match_rate_percent": round(shape_rate, 2),
        },
        "detailed_results": results,
    }

    if use_verification and verification_metadata_list:
        report_payload["verification_details"] = verification_metadata_list

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    logger.info(f"Evaluation report written to {output_report}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Qwen2.5 on ARC-AGI-2 (Direct, TTT, or TTT+Verification)."
    )
    parser.add_argument("--val-file", type=str, default="data/processed/arc_val.jsonl", help="Path to arc_val.jsonl")
    parser.add_argument("--raw-tasks-dir", type=str, default="ARC-AGI-2/data/training", help="Path to raw tasks directory.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier.")
    parser.add_argument("--adapter-path", type=str, default="models/arc_qwen_1.5b_lora", help="Path to LoRA adapter.")
    parser.add_argument("--use-ttt", action="store_true", help="Enable Test-Time Training (TTT) adaptation per task.")
    parser.add_argument("--ttt-steps", type=int, default=30, help="Number of gradient steps per task in TTT mode.")
    parser.add_argument("--ttt-lr", type=float, default=5e-4, help="Learning rate for per-task TTT adaptation.")
    parser.add_argument(
        "--use-verification",
        action="store_true",
        help="Rank >2 candidates by offline train-pair consistency and submit the top 2 (pass@2).",
    )
    parser.add_argument(
        "--n-candidates",
        "--num-candidates",
        dest="n_candidates",
        type=int,
        default=6,
        help="Number of test-output candidates to generate before verification ranking.",
    )
    parser.add_argument(
        "--condition-on-candidate",
        dest="condition_on_candidate",
        action="store_true",
        default=True,
        help="Insert the hypothesized test output as an extra demo while probing train pairs (default on).",
    )
    parser.add_argument(
        "--no-condition-on-candidate",
        dest="condition_on_candidate",
        action="store_false",
        help="Probe train pairs without inserting the candidate as an extra demonstration.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Max test samples to evaluate.")
    parser.add_argument("--output-report", type=str, default="eval_results.json", help="Path for JSON output report.")
    args = parser.parse_args()

    run_evaluation(
        val_file=args.val_file,
        raw_tasks_dir=args.raw_tasks_dir,
        model_name=args.model_name,
        adapter_path=args.adapter_path if os.path.exists(args.adapter_path) else None,
        output_report=args.output_report,
        use_ttt=args.use_ttt,
        ttt_steps=args.ttt_steps,
        ttt_lr=args.ttt_lr,
        use_verification=args.use_verification,
        n_candidates=args.n_candidates,
        condition_on_candidate=args.condition_on_candidate,
        max_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
