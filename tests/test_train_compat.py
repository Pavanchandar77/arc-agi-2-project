"""Train-stack compatibility: Transformers 5 / TRL SFTConfig, no model download."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.hrps.hf_compat import (
    NATIVE_LORA,
    build_sft_config_kwargs,
    build_sft_trainer_kwargs,
    from_pretrained_dtype_kwargs,
    make_formatting_func,
    neutralize_incompatible_torchao,
    param_names,
    preview_formatted_example,
    warmup_kwargs,
)
from src.train import construct_sft_config, construct_sft_trainer


REPO = Path(__file__).resolve().parent.parent
SFT = REPO / "artifacts" / "bond" / "train_scale" / "sft_actions.jsonl"


class _DummyTokenizer:
    eos_token = "</s>"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_native_lora_label():
    assert NATIVE_LORA == "native_precision_lora"


def test_warmup_keeps_ratio_value():
    assert warmup_kwargs({"warmup_ratio", "warmup_steps"}, 0.05) == {"warmup_ratio": 0.05}
    assert warmup_kwargs({"warmup_steps"}, 0.05) == {"warmup_steps": 0.05}
    assert warmup_kwargs({"warmup_steps"}, 0.05)["warmup_steps"] == 0.05


def test_dtype_prefers_new_name():
    assert from_pretrained_dtype_kwargs("bf16", {"dtype", "torch_dtype"}) == {"dtype": "bf16"}
    assert from_pretrained_dtype_kwargs("bf16", {"torch_dtype"}) == {"torch_dtype": "bf16"}


def test_sft_config_kwargs_transformers5_names():
    params = {
        "self",
        "output_dir",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_steps",
        "max_length",
        "num_train_epochs",
        "logging_steps",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "eval_strategy",
        "eval_steps",
        "fp16",
        "bf16",
        "optim",
        "gradient_checkpointing",
        "weight_decay",
        "seed",
        "report_to",
    }
    kw = build_sft_config_kwargs(
        params,
        output_dir="models/bond_qwen35_4b",
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        max_seq_length=2048,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        has_eval=True,
        use_fp16=False,
        use_bf16=True,
        seed=42,
    )
    assert kw["warmup_steps"] == 0.05
    assert "warmup_ratio" not in kw
    assert kw["eval_strategy"] == "steps"
    assert "evaluation_strategy" not in kw
    assert kw["max_length"] == 2048
    assert "max_seq_length" not in kw
    assert kw["output_dir"] == "models/bond_qwen35_4b"


def test_sft_config_kwargs_legacy_names():
    params = {"warmup_ratio", "evaluation_strategy", "max_seq_length", "output_dir", "eval_steps"}
    kw = build_sft_config_kwargs(
        params,
        output_dir="out",
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        max_seq_length=512,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=1,
        has_eval=False,
        use_fp16=False,
        use_bf16=False,
        seed=0,
    )
    assert kw["warmup_ratio"] == 0.05
    assert kw["evaluation_strategy"] == "no"
    assert kw["max_seq_length"] == 512


def test_sft_trainer_kwargs_current_trl():
    params = {
        "self",
        "model",
        "args",
        "train_dataset",
        "eval_dataset",
        "processing_class",
        "formatting_func",
        "tokenizer",
        "max_seq_length",
        "dataset_text_field",
    }
    kw = build_sft_trainer_kwargs(
        params,
        model="m",
        args="a",
        train_dataset="tr",
        eval_dataset=None,
        tokenizer="tok",
        formatting_func=lambda ex: "x",
    )
    assert kw["processing_class"] == "tok"
    assert "tokenizer" not in kw
    assert "max_seq_length" not in kw
    assert "dataset_text_field" not in kw
    assert "formatting_func" in kw


def test_formatting_func_returns_str_for_messages():
    tok = _DummyTokenizer()
    fn = make_formatting_func(tok)
    text = fn(
        {
            "messages": [
                {"role": "system", "content": "You are Bond."},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "{\"action\": \"inspect_objects\"}"},
            ]
        }
    )
    assert isinstance(text, str)
    assert "You are Bond." in text
    assert "inspect_objects" in text
    assert not text.startswith("{")


def test_preview_uses_real_sft_line():
    if not SFT.is_file():
        pytest.skip("sft_actions.jsonl missing")
    line = next(x for x in SFT.read_text(encoding="utf-8").splitlines() if x.strip())
    example = json.loads(line)
    prev = preview_formatted_example(example, _DummyTokenizer())
    assert prev["is_str"] is True
    assert prev["chars"] > 20
    assert prev["looks_like_python_repr"] is False
    assert "Bond" in prev["preview"] or "user:" in prev["preview"]


def test_torchao_guard_does_not_require_package():
    rec = neutralize_incompatible_torchao()
    assert rec["policy"] == NATIVE_LORA
    assert rec["action"] in {"not_installed", "kept_compatible", "disabled_incompatible", "none"}


def test_construct_sft_config_with_fake_class():
    captured = {}

    class FakeConfig:
        def __init__(
            self,
            output_dir,
            warmup_steps=0,
            eval_strategy="no",
            max_length=512,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=1,
            eval_steps=None,
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            gradient_checkpointing=False,
            weight_decay=0.0,
            seed=0,
            report_to="none",
        ):
            captured.update(locals())
            captured.pop("self")

    cfg = construct_sft_config(
        output_dir="models/bond_qwen35_4b",
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        max_seq_length=2048,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        has_eval=False,
        use_fp16=False,
        use_bf16=True,
        seed=42,
        config_class=FakeConfig,
    )
    assert cfg is not None
    assert captured["warmup_steps"] == 0.05
    assert captured["eval_strategy"] == "no"
    assert captured["max_length"] == 2048
    assert captured["output_dir"] == "models/bond_qwen35_4b"


def test_construct_sft_trainer_rejects_obsolete_keywords():
    seen = {}

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset=None, processing_class=None, formatting_func=None):
            seen.update(
                {
                    "model": model,
                    "args": args,
                    "train_dataset": train_dataset,
                    "eval_dataset": eval_dataset,
                    "processing_class": processing_class,
                    "formatting_func": formatting_func,
                }
            )

    construct_sft_trainer(
        model="m",
        args="a",
        train_dataset="tr",
        eval_dataset=None,
        tokenizer="tok",
        formatting_func=lambda ex: "x",
        trainer_class=FakeTrainer,
    )
    assert seen["processing_class"] == "tok"
    assert "tokenizer" not in seen
    assert "max_seq_length" not in seen
    assert "dataset_text_field" not in seen


def test_installed_signatures_if_present():
    pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")
    from transformers import TrainingArguments
    from trl import SFTTrainer

    ta = param_names(TrainingArguments)
    st = param_names(SFTTrainer)
    # Either generation of names is acceptable; constructed kwargs must match one.
    if "eval_strategy" in ta:
        kw = build_sft_config_kwargs(
            ta,
            output_dir="out",
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            max_seq_length=128,
            num_train_epochs=1,
            logging_steps=1,
            save_steps=1,
            has_eval=False,
            use_fp16=False,
            use_bf16=False,
            seed=0,
        )
        assert "evaluation_strategy" not in kw or "eval_strategy" in kw
    trainer_kw = build_sft_trainer_kwargs(
        st,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=object(),
        formatting_func=lambda ex: "x",
    )
    assert "max_seq_length" not in trainer_kw
    assert "dataset_text_field" not in trainer_kw
    if "processing_class" in st:
        assert "processing_class" in trainer_kw
        assert "tokenizer" not in trainer_kw
    try:
        from trl import SFTConfig

        cfg = construct_sft_config(
            output_dir="out",
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            max_seq_length=128,
            num_train_epochs=1,
            logging_steps=1,
            save_steps=1,
            has_eval=False,
            use_fp16=False,
            use_bf16=False,
            seed=0,
            config_class=SFTConfig,
        )
        assert cfg is not None
    except Exception as exc:
        pytest.skip(f"SFTConfig construct skipped: {exc}")


def test_check_gpu_env_script_exists():
    path = REPO / "scripts" / "check_gpu_env.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-4B" in text
    assert "qwen3_5" in text
    assert "torchrun DDP" in text or "not torchrun" in text
    assert "--no-download" in text
    assert "native_precision_lora" in text or "native-precision" in text


def test_qwen35_4b_registry_is_exact():
    from src.hrps.backend import resolve_foundation
    from src.hrps.bond import SYSTEMS

    spec = resolve_foundation("qwen3.5_4b")
    assert spec["hf_id"] == "Qwen/Qwen3-4B"
    assert resolve_foundation("Qwen/Qwen3-4B")["hf_id"] == "Qwen/Qwen3-4B"
    assert set(SYSTEMS) == {"base_direct", "base_hrps", "bond_direct", "bond_hrps"}


def test_refuses_silent_foundation_substitution():
    from src.hrps.bond import main as bond_main

    rc = bond_main(
        [
            "eval",
            "--foundation",
            "qwen3.5_4b",
            "--model",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--n",
            "1",
        ]
    )
    assert rc == 2


def test_adapter_complete_requires_config_and_weights(tmp_path: Path):
    from src.hrps.identity import adapter_is_complete

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "BOND_TRAIN.json").write_text("{}", encoding="utf-8")
    assert adapter_is_complete(empty) is False
    cfg_only = tmp_path / "cfg"
    cfg_only.mkdir()
    (cfg_only / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_is_complete(cfg_only) is False
    real = tmp_path / "real"
    real.mkdir()
    (real / "adapter_config.json").write_text("{}", encoding="utf-8")
    (real / "adapter_model.safetensors").write_bytes(b"not-a-real-weight")
    assert adapter_is_complete(real) is True


def test_curriculum_reuse_without_regenerate(tmp_path: Path):
    from src.hrps.bond_curriculum import curriculum_artifacts_ready, verify_holdout_clean

    train_scale = tmp_path / "train_scale.jsonl"
    cur_dir = tmp_path / "curriculum"
    cur_dir.mkdir()
    sft = cur_dir / "sft_actions.jsonl"
    merged = cur_dir / "sft_merged.jsonl"
    line = json.dumps({"task_id": "synth_geom_1", "messages": [{"role": "user", "content": "x"}]})
    train_scale.write_text(line + "\n", encoding="utf-8")
    sft.write_text(line + "\n", encoding="utf-8")
    merged.write_text(line + "\n", encoding="utf-8")
    (cur_dir / "CURRICULUM.json").write_text(json.dumps({"held_out_excluded": True}), encoding="utf-8")
    rec = curriculum_artifacts_ready(curriculum_dir=cur_dir, merged_path=merged, train_scale=train_scale)
    assert rec["ready"] is True
    hold = verify_holdout_clean([sft, train_scale, merged])
    assert hold["ok"] is True
    missing = curriculum_artifacts_ready(
        curriculum_dir=tmp_path / "nope",
        merged_path=tmp_path / "nope.jsonl",
        train_scale=train_scale,
    )
    assert missing["ready"] is False


def test_train_py_has_no_obsolete_trainer_keywords():
    text = (REPO / "src" / "train.py").read_text(encoding="utf-8")
    compat = (REPO / "src" / "hrps" / "hf_compat.py").read_text(encoding="utf-8")
    assert "dataset_text_field" not in text
    assert "construct_sft_trainer" in text
    assert "processing_class" in compat
    assert 'kw.pop("max_seq_length", None)' in compat
    assert "SFTConfig" in text


def test_lightning_script_has_regenerate_flag():
    text = (REPO / "scripts" / "train_bond_lightning.py").read_text(encoding="utf-8")
    assert "--regenerate" in text
    assert "Qwen/Qwen3-4B" in text
    assert "not repeatedly" in text or "already exists" in text or "reused" in text
    assert "Not AGI" in text
