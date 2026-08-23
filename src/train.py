"""Fine-Tuning Pipeline for ARC-AGI-2 with LoRA & TRL SFTTrainer.

Model: Qwen/Qwen2.5-1.5B-Instruct (Instruct checkpoint with built-in ChatML template).
Architecture & Optimization:
- Full precision / native mixed precision (no 4-bit/8-bit quantization or bitsandbytes)
- LoRA adapter (r=16, lora_alpha=32, targeting q/k/v/o/gate/up/down proj)
- Optimizer: optim="adamw_torch"
- use_gradient_checkpointing=False
- TRL SFTTrainer for ChatML instruction tuning

Note: Designed to run on remote GPU environment (Google Colab / Kaggle / Cloud GPU).
Do not execute locally on CPU.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train(
    train_file: str = "data/processed/arc_train.jsonl",
    val_file: Optional[str] = "data/processed/arc_val.jsonl",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "models/arc_qwen_1.5b_lora",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    logging_steps: int = 10,
    save_steps: int = 100,
    seed: int = 42
):
    """Fine-tune Qwen2.5-1.5B-Instruct with LoRA on ARC-AGI-2 without quantization."""
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available on this system! "
            "train.py is configured for remote GPU environments (Colab / Kaggle)."
        )

    logger.info(f"Loading tokenizer & model from '{model_name}' (no quantization)...")

    # 1. Tokenizer (Instruct checkpoint includes chat template natively)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Base Model (Full / Native Precision, No bitsandbytes / quantization)
    device_available = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if (device_available and torch.cuda.is_bf16_supported()) else (torch.float16 if device_available else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device_available else None,
        trust_remote_code=True,
    )

    # 3. LoRA Configuration (q/k/v/o/gate/up/down proj)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load Datasets
    logger.info(f"Loading training data from {train_file}...")
    data_files = {"train": train_file}
    if val_file and os.path.exists(val_file):
        data_files["validation"] = val_file
        logger.info(f"Loading validation data from {val_file}...")

    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets.get("validation", None)

    # 5. Format Prompts with Native Chat Template
    def formatting_prompts_func(example):
        output_texts = []
        if "messages" in example and isinstance(example["messages"][0], list):
            for msgs in example["messages"]:
                formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                output_texts.append(formatted)
        elif "messages" in example and isinstance(example["messages"], list):
            formatted = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            output_texts.append(formatted)
        elif "prompt" in example and "completion" in example:
            for p, c in zip(example["prompt"], example["completion"]):
                output_texts.append(f"{p}\n{c}{tokenizer.eos_token}")
        return output_texts

    # 6. Training Arguments (optim="adamw_torch", use_gradient_checkpointing=False)
    use_bf16 = device_available and torch.cuda.is_bf16_supported()
    use_fp16 = device_available and not use_bf16

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        evaluation_strategy="steps" if val_dataset is not None else "no",
        eval_steps=save_steps if val_dataset is not None else None,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch",                 # Explicitly adamw_torch as requested
        gradient_checkpointing=False,        # Explicitly False as requested
        weight_decay=0.01,
        seed=seed,
        report_to="none",
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=max_seq_length,
        dataset_text_field=None,
        args=training_args,
    )

    # 8. Train & Save
    logger.info("Starting fine-tuning...")
    trainer.train()

    logger.info(f"Saving LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuning completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B-Instruct on ARC-AGI-2 with LoRA & SFTTrainer.")
    parser.add_argument("--train-file", type=str, default="data/processed/arc_train.jsonl", help="Path to arc_train.jsonl")
    parser.add_argument("--val-file", type=str, default="data/processed/arc_val.jsonl", help="Path to arc_val.jsonl")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID.")
    parser.add_argument("--output-dir", type=str, default="models/arc_qwen_1.5b_lora", help="Output directory for LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    train(
        train_file=args.train_file,
        val_file=args.val_file if os.path.exists(args.val_file) else None,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
