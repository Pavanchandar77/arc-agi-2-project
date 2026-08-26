"""Bond-4B native-precision LoRA training (Transformers 5 / current TRL).

Foundation for the Bond experiment is Qwen/Qwen3.5-4B when launched via
scripts/train_bond_qwen35_4b.py or scripts/train_bond_lightning.py.
This module does not swap that id.

Architecture:
- Native fp16/bf16 load (no BitsAndBytes, not QLoRA)
- LoRA adapter (q/k/v/o/gate/up/down)
- TRL SFTTrainer + SFTConfig when available
- device_map="auto" is model-parallel shard placement, not torchrun DDP

Do not call the result a learned Bond checkpoint until adapter weights
are saved under the declared output_dir and reloaded.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hrps.hf_compat import (
    NATIVE_LORA,
    build_sft_config_kwargs,
    build_sft_trainer_kwargs,
    filter_to_signature,
    from_pretrained_dtype_kwargs,
    make_formatting_func,
    neutralize_incompatible_torchao,
    param_names,
    preview_formatted_example,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_sft_config_class():
    try:
        from trl import SFTConfig

        return SFTConfig
    except Exception:
        from transformers import TrainingArguments

        return TrainingArguments


def construct_sft_config(
    *,
    output_dir: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    max_seq_length: int,
    num_train_epochs: int,
    logging_steps: int,
    save_steps: int,
    has_eval: bool,
    use_fp16: bool,
    use_bf16: bool,
    seed: int,
    config_class: Any = None,
) -> Any:
    cls = config_class or _resolve_sft_config_class()
    params = param_names(cls)
    kwargs = build_sft_config_kwargs(
        params,
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        has_eval=has_eval,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        seed=seed,
    )
    kwargs = filter_to_signature(cls, kwargs)
    return cls(**kwargs)


def construct_sft_trainer(
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
    formatting_func: Any,
    trainer_class: Any = None,
) -> Any:
    if trainer_class is None:
        from trl import SFTTrainer

        trainer_class = SFTTrainer
    params = param_names(trainer_class)
    kwargs = build_sft_trainer_kwargs(
        params,
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
    )
    return trainer_class(**kwargs)


def train(
    train_file: str = "data/processed/arc_train.jsonl",
    val_file: Optional[str] = "data/processed/arc_val.jsonl",
    model_name: str = "Qwen/Qwen3.5-4B",
    output_dir: str = "models/bond_qwen35_4b",
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
    seed: int = 42,
):
    """Native-precision LoRA SFT. Public warmup_ratio stays 0.05 (5%)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torchao_state = neutralize_incompatible_torchao()
    logger.info("torchao_guard=%s", torchao_state)

    from peft import LoraConfig, get_peft_model

    n_gpu = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Bond-4B training is a remote GPU job. "
            "device_map='auto' is model-parallel placement, not torchrun DDP."
        )
    else:
        names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        logger.info(
            "CUDA ok n_gpu=%s names=%s. device_map='auto' shards the model "
            "across GPUs (model parallel), it does not launch two DDP processes.",
            n_gpu,
            names,
        )

    logger.info(
        "Loading tokenizer & model from '%s' (%s, no BitsAndBytes)",
        model_name,
        NATIVE_LORA,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_available = torch.cuda.is_available()
    load_dtype = (
        torch.bfloat16
        if (device_available and torch.cuda.is_bf16_supported())
        else (torch.float16 if device_available else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device_available else None,
        trust_remote_code=True,
        **from_pretrained_dtype_kwargs(load_dtype),
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading training data from %s", train_file)
    data_files = {"train": train_file}
    if val_file and os.path.exists(val_file):
        data_files["validation"] = val_file
        logger.info("Loading validation data from %s", val_file)

    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets.get("validation", None)

    formatting_prompts_func = make_formatting_func(tokenizer)
    sample = train_dataset[0]
    preview = preview_formatted_example(sample, tokenizer)
    logger.info(
        "formatted_example kind=%s chars=%s tokens=%s preview=%r",
        preview["kind"],
        preview["chars"],
        preview["n_tokens"],
        preview["preview"][:240],
    )
    if preview.get("looks_like_python_repr"):
        raise ValueError(
            "formatted SFT example looks like a raw Python repr; refusing to train. "
            "Expected a chat-template string from the messages field."
        )
    if preview.get("n_tokens") and preview["n_tokens"] > max_seq_length:
        logger.warning(
            "formatted example tokens=%s exceed max_length=%s; sequences will truncate",
            preview["n_tokens"],
            max_seq_length,
        )

    use_bf16 = device_available and torch.cuda.is_bf16_supported()
    use_fp16 = device_available and not use_bf16
    training_args = construct_sft_config(
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        has_eval=val_dataset is not None,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        seed=seed,
    )
    logger.info("sft_config_class=%s adapter=%s", type(training_args).__name__, NATIVE_LORA)

    trainer = construct_sft_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
    )

    logger.info("Starting native-precision LoRA fine-tuning...")
    trainer.train()

    logger.info("Saving LoRA adapter to %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    adapter_ok = any(Path(output_dir).glob("adapter_model*.safetensors")) or any(
        Path(output_dir).glob("adapter_model.bin")
    )
    logger.info("Fine-tuning completed. adapter_weights_present=%s", adapter_ok)
    if not adapter_ok:
        logger.warning("Adapter files were not found after save; do not call this a learned Bond checkpoint.")


def main():
    parser = argparse.ArgumentParser(
        description="Native-precision LoRA SFT. Bond-4B launcher pins Qwen/Qwen3.5-4B."
    )
    parser.add_argument("--train-file", type=str, default="artifacts/bond/train_scale/sft_actions.jsonl")
    parser.add_argument("--val-file", type=str, default="")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=str, default="models/bond_qwen35_4b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    val = args.val_file if args.val_file and os.path.exists(args.val_file) else None
    train(
        train_file=args.train_file,
        val_file=val,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
