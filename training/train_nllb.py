"""
train_nllb.py
-------------
Fine-tune Meta's NLLB-200 (distilled 1.3B) on Kurdish Sorani → Swedish poetry
using LoRA (Parameter-Efficient Fine-Tuning).

Designed for a single GTX 3060 12 GB card.

Usage:
    python training/train_nllb.py --config training/config.yaml
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import evaluate
import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_list([{"ckb": p["ckb"], "swe": p["swe"]} for p in pairs])


# ── Tokenization ──────────────────────────────────────────────────────────────

def make_preprocess_fn(tokenizer, src_lang: str, tgt_lang: str, max_src: int, max_tgt: int):
    def preprocess(examples):
        tokenizer.src_lang = src_lang
        model_inputs = tokenizer(
            examples["ckb"],
            max_length=max_src,
            truncation=True,
            padding=False,
        )
        tokenizer.tgt_lang = tgt_lang
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["swe"],
                max_length=max_tgt,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


# ── Evaluation metric ─────────────────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    chrf_metric = evaluate.load("chrf")
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 in labels (padding) with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # chrF++ (best metric for morphologically rich languages)
        chrf_result = chrf_metric.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
            word_order=2,  # chrF++
        )

        # BLEU
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )

        return {
            "eval_chrf": round(chrf_result["score"], 4),
            "eval_bleu": round(bleu_result["score"], 4),
        }

    return compute_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    model_name = model_cfg["name"]
    src_lang = model_cfg["src_lang"]
    tgt_lang = model_cfg["tgt_lang"]
    use_4bit = model_cfg.get("use_4bit", False)

    print(f"Model : {model_name}")
    print(f"Pair  : {src_lang} → {tgt_lang}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Model ──────────────────────────────────────────────────────────────
    load_kwargs: dict = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)

    if train_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        # Required when gradient_checkpointing + PEFT
        model.enable_input_require_grads()

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    train_pairs = load_jsonl(data_cfg["train_file"])
    val_pairs = load_jsonl(data_cfg["val_file"])
    print(f"Train: {len(train_pairs):,}   Val: {len(val_pairs):,}")

    train_ds = build_dataset(train_pairs)
    val_ds = build_dataset(val_pairs)

    preprocess_fn = make_preprocess_fn(
        tokenizer,
        src_lang,
        tgt_lang,
        data_cfg["max_src_length"],
        data_cfg["max_tgt_length"],
    )

    train_ds = train_ds.map(preprocess_fn, batched=True, remove_columns=["ckb", "swe"])
    val_ds = val_ds.map(preprocess_fn, batched=True, remove_columns=["ckb", "swe"])

    # ── Training args ──────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        save_total_limit=train_cfg["save_total_limit"],
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg.get("report_to", "none"),
        predict_with_generate=True,
        seed=train_cfg.get("seed", 42),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    callbacks = []
    patience = train_cfg.get("early_stopping_patience", 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Save final adapter + tokenizer ────────────────────────────────────
    final_dir = Path(train_cfg["output_dir"]) / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to {final_dir}")


if __name__ == "__main__":
    main()
