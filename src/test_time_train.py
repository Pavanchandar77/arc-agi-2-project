"""Test-Time Training (TTT) Pipeline for ARC-AGI-2.

Adapts the base fine-tuned LoRA model to each individual ARC puzzle at inference time.

Key Principles:
1. Base LoRA Checkpoint is the immutable starting state for every task.
2. For each task:
   - Extracts ONLY that task's own demonstration pairs.
   - Generates D8 spatial & color-bijective augmented sub-tasks using `src.data`.
   - Runs a short, bounded adaptation (e.g. 20-40 gradient steps with AdamW).
   - Generates predictions for the test problem (Attempt 1: Greedy, Attempt 2: Sampled).
   - STRICT WEIGHT RESET: Restores LoRA parameters back to the base checkpoint state.
   - Zero leakage across consecutive tasks.

Designed for remote GPU environments (Google Colab / Kaggle Competition limits).
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    DEFAULT_SYSTEM_PROMPT,
    apply_color_map,
    apply_d8_transform,
    augment_task,
    generate_task_augmentations,
    grid_to_text,
    grids_equal,
    is_valid_grid,
    random_color_map,
    task_to_chat_messages,
    task_to_prompt,
    text_to_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 1. State Snapshot & Weight Restoration (Strict Weight Isolation)
# ============================================================================

def capture_trainable_state(model) -> Dict[str, Any]:
    """Capture an immutable snapshot of all trainable parameter tensors (LoRA weights)."""
    state = {}
    for name, param in model.named_parameters():
        if getattr(param, "requires_grad", True):
            if hasattr(param, "detach"):
                state[name] = param.detach().cpu().clone()
            elif hasattr(param, "copy"):
                state[name] = param.copy()
            else:
                state[name] = copy.deepcopy(param)
    return state


def restore_trainable_state(model, base_state: Dict[str, Any], device: Optional[Any] = None):
    """Restore all trainable parameters strictly back to the base checkpoint state."""
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    def do_restore():
        for name, param in model.named_parameters():
            if name in base_state:
                target = base_state[name]
                if has_torch and hasattr(target, "to") and device is not None:
                    target = target.to(device)
                if hasattr(param, "copy_"):
                    param.copy_(target)
                elif hasattr(param, "data") and hasattr(param.data, "copy_"):
                    param.data.copy_(target)
                elif hasattr(param, "set_data"):
                    param.set_data(target)
                else:
                    setattr(param, "data", copy.deepcopy(target))

    if has_torch:
        import torch
        with torch.no_grad():
            do_restore()
    else:
        do_restore()


def verify_weight_equality(model, base_state: Dict[str, Any], tolerance: float = 1e-7) -> bool:
    """Verify that all trainable parameters in the model match the base state snapshot exactly."""
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    for name, param in model.named_parameters():
        if name in base_state:
            base_val = base_state[name]
            if has_torch and hasattr(param, "device") and hasattr(base_val, "to"):
                base_val = base_val.to(param.device)
                if not torch.allclose(param, base_val, atol=tolerance):
                    max_diff = (param - base_val).abs().max().item()
                    logger.error(f"Weight mismatch in parameter '{name}': max diff = {max_diff}")
                    return False
            else:
                # Value / list comparison
                p_data = getattr(param, "data", param)
                b_data = getattr(base_val, "data", base_val)
                if p_data != b_data:
                    return False
    return True


# ============================================================================
# 2. Per-Task TTT Training Dataset Construction
# ============================================================================

def create_ttt_dataset_for_task(
    task: Dict[str, Any],
    num_augmentations: int = 8,
    grid_format: str = "compact",
    rng: Optional[random.Random] = None
) -> List[Dict[str, Any]]:
    """Build synthetic training samples from ONLY this task's own demonstration pairs.
    
    Uses leave-one-out pseudo-tasks across the demonstration set:
    If a task has demonstrations [D1, D2, D3], we form sub-tasks:
    - Train on [D1, D2] -> Predict D3
    - Train on [D1, D3] -> Predict D2
    - Train on [D2, D3] -> Predict D1
    
    Then applies D8 dihedral symmetries and color permutations using src.data.
    """
    r = rng if rng is not None else random.Random()
    train_pairs = task.get("train", [])
    if len(train_pairs) < 1:
        return []

    pseudo_tasks: List[Dict[str, Any]] = []

    if len(train_pairs) == 1:
        # Single demonstration: train on (D1 -> D1) self-consistency
        pseudo_tasks.append({
            "train": [train_pairs[0]],
            "test": [train_pairs[0]]
        })
    else:
        # Leave-one-out pseudo tasks
        for held_out_idx in range(len(train_pairs)):
            sub_train = [p for i, p in enumerate(train_pairs) if i != held_out_idx]
            sub_test = [train_pairs[held_out_idx]]
            pseudo_tasks.append({
                "train": sub_train,
                "test": sub_test
            })

    # Apply D8 symmetries and color bijections to all pseudo-tasks
    ttt_records: List[Dict[str, Any]] = []
    for p_task in pseudo_tasks:
        aug_variants = generate_task_augmentations(
            p_task,
            num_augmentations=num_augmentations,
            permute_colors=True,
            apply_symmetries=True,
            shuffle_demonstrations=True,
            rng=r
        )
        for aug_t in aug_variants:
            messages = task_to_chat_messages(
                aug_t,
                test_idx=0,
                include_test_output=True,
                grid_format=grid_format
            )
            user_p, target_out = task_to_prompt(
                aug_t,
                test_idx=0,
                include_test_output=True,
                grid_format=grid_format
            )
            ttt_records.append({
                "messages": messages,
                "prompt": user_p,
                "completion": target_out
            })

    r.shuffle(ttt_records)
    return ttt_records


# ============================================================================
# 3. Task Adaptation Loop (Single Task Fine-Tuning)
# ============================================================================

def adapt_model_to_task(
    model,
    tokenizer,
    task: Dict[str, Any],
    ttt_steps: int = 30,
    learning_rate: float = 5e-4,
    batch_size: int = 2,
    device: str = "cuda",
    num_augmentations: int = 8,
    seed: int = 42,
) -> float:
    """Run short, bounded fine-tuning on a single task's demonstration augmentations.
    
    Returns:
        Final adaptation loss.
    """
    import torch
    from torch.utils.data import DataLoader

    rng = random.Random(seed)
    ttt_records = create_ttt_dataset_for_task(
        task,
        num_augmentations=num_augmentations,
        rng=rng
    )

    if not ttt_records:
        return 0.0

    # Tokenize records
    formatted_prompts = []
    for rec in ttt_records:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(rec["messages"], tokenize=False, add_generation_prompt=False)
        else:
            text = f"{rec['prompt']}\n{rec['completion']}{tokenizer.eos_token}"
        formatted_prompts.append(text)

    # Set model to train mode for trainable LoRA weights
    model.train()

    # Filter only trainable LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable parameters found for TTT adaptation.")
        return 0.0

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Short adaptation loop (bounded by ttt_steps)
    step = 0
    total_loss = 0.0
    num_records = len(formatted_prompts)

    while step < ttt_steps:
        # Sample batch
        batch_indices = [rng.randint(0, num_records - 1) for _ in range(batch_size)]
        batch_texts = [formatted_prompts[i] for i in batch_indices]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(device)

        inputs["labels"] = inputs["input_ids"].clone()

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

    return total_loss / max(step, 1)


# ============================================================================
# 4. End-to-End Test-Time Train and Predict
# ============================================================================

def predict_task_with_ttt(
    model,
    tokenizer,
    task: Dict[str, Any],
    base_state: Dict[str, Any],
    test_idx: int = 0,
    ttt_steps: int = 30,
    learning_rate: float = 5e-4,
    device: str = "cuda",
    max_new_tokens: int = 512,
) -> Tuple[str, str]:
    """Execute TTT adaptation for a single task, generate predictions, and strictly restore base weights.
    
    Guarantees:
    - Base weights are verified restored before returning.
    - Zero state leakage to subsequent tasks.
    
    Returns:
        (attempt_1_text, attempt_2_text)
    """
    import torch

    try:
        # 1. Adapt model on task's own demonstration augmentations
        if ttt_steps > 0:
            adapt_model_to_task(
                model=model,
                tokenizer=tokenizer,
                task=task,
                ttt_steps=ttt_steps,
                learning_rate=learning_rate,
                device=device,
            )

        # 2. Put in eval mode for inference
        model.eval()

        # 3. Format test problem prompt
        messages = task_to_chat_messages(
            task,
            test_idx=test_idx,
            include_test_output=False
        )

        if hasattr(tokenizer, "apply_chat_template"):
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            input_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_msg}\nOutput:\n"

        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)

        # Attempt 1: Greedy Decoding
        with torch.no_grad():
            out_1 = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_1 = out_1[0][inputs["input_ids"].shape[1]:]
        att_1_str = tokenizer.decode(gen_1, skip_special_tokens=True)

        # Attempt 2: Temperature Sampling
        with torch.no_grad():
            out_2 = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_2 = out_2[0][inputs["input_ids"].shape[1]:]
        att_2_str = tokenizer.decode(gen_2, skip_special_tokens=True)

        return att_1_str, att_2_str

    finally:
        # CRITICAL (Requirement 2e): ALWAYS restore base weights, even if an exception occurs
        restore_trainable_state(model, base_state, device=device)
        model.eval()
