"""
MIRA mini v1.0 — Training
Step2: Continued Pre-Training (Unsupervised)
Developed by: DeviceAlchemy LLC
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
"""

import argparse
import math
import os
import random

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="./data")
    p.add_argument("--output_dir",  default="./pretrain_output")
    p.add_argument("--max_seq_len", type=int,   default=384,
                   help="Token block length. 384 balances context vs. training cost at 671K scale.")
    p.add_argument("--max_steps",   type=int,   default=-1,
                   help="-1 = full epoch(s)")
    p.add_argument("--num_epochs",  type=int,   default=1,
                   help="1 full epoch is standard at this corpus volume.")
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="Per-device. Use 1 for RTX 3060 12 GB.")
    p.add_argument("--grad_accum",  type=int,   default=32,
                   help="Effective batch = batch_size × grad_accum = 32")
    p.add_argument("--lr",          type=float, default=1e-4,
                   help="Standard LR; sufficient data volume to avoid overfit risk.")
    p.add_argument("--sample",      type=float, default=0.30,
                   help="Fraction of corpus. 0.30 ≈ 200K abstracts — balances coverage vs. time.")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume_from", type=str,   default=None)
    return p.parse_args()


def load_model_4bit():
    print(f"Loading {BASE_MODEL} in 4-bit NF4…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache      = False
    model.config.pretraining_tp = 1
    return model


def prepare_dataset(data_dir, tokenizer, max_seq_len, sample_frac, seed):
    train_path = os.path.join(data_dir, "pretrain_train.txt")
    val_path   = os.path.join(data_dir, "pretrain_val.txt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"No training data at {train_path}\n"
            "Run: python MIRA_step1_prepare_pretrain.py --input /path/to/abstracts.txt"
        )

    raw = load_dataset("text", data_files={
        "train":      train_path,
        "validation": val_path,
    })

    if sample_frac < 1.0:
        n_train = int(len(raw["train"]) * sample_frac)
        n_val   = max(200, int(len(raw["validation"]) * sample_frac))
        n_val   = min(n_val, len(raw["validation"]))
        random.seed(seed)
        train_idx = random.sample(range(len(raw["train"])), n_train)
        val_idx   = random.sample(range(len(raw["validation"])), n_val)
        raw["train"]      = raw["train"].select(train_idx)
        raw["validation"] = raw["validation"].select(val_idx)
        print(f"  Subsampled to {sample_frac*100:.0f}%: "
              f"{len(raw['train']):,} train / {len(raw['validation']):,} val docs")
    else:
        print(f"  Full corpus: {len(raw['train']):,} train / {len(raw['validation']):,} val docs")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False, add_special_tokens=True)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concatenated["input_ids"]) // max_seq_len) * max_seq_len
        result = {
            k: [v[i: i + max_seq_len] for i in range(0, total, max_seq_len)]
            for k, v in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("  Tokenising and grouping into blocks…")
    tokenised = raw.map(
        tokenize_fn, batched=True, remove_columns=["text"],
        desc="Tokenising", num_proc=1,
    )
    grouped = tokenised.map(
        group_texts, batched=True,
        desc="Grouping", num_proc=1,
    )

    n_blocks = len(grouped["train"])
    print(f"  Train blocks ({max_seq_len} tokens): {n_blocks:,}")
    print(f"  Total train tokens: ~{n_blocks * max_seq_len / 1e6:.2f} M")
    return grouped, n_blocks


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_4bit()

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,        
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset, n_blocks = prepare_dataset(
        args.data_dir, tokenizer, args.max_seq_len, args.sample, args.seed
    )

    steps_per_epoch = math.ceil(n_blocks / (args.batch_size * args.grad_accum))
    
    eval_steps = max(100, steps_per_epoch // 10)
    save_steps = eval_steps * 2
    
    secs_per_step = 14
    total_steps   = steps_per_epoch * args.num_epochs if args.max_steps == -1 else args.max_steps
    est_hours     = (total_steps * secs_per_step) / 3600

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs if args.max_steps == -1 else 1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,         
        weight_decay=0.01,         
        max_grad_norm=1.0,
        bf16=False,
        fp16=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        resume_from_checkpoint=args.resume_from,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )

    print()
    print("=" * 60)
    print("Continued Pre-Training  —  Qwen2.5-1.5B-Instruct")
    print(f"  Objective       : causal LM (unsupervised, no labels)")
    print(f"  Corpus fraction : {args.sample*100:.0f}%  (~671K abstracts)")
    print(f"  Train tokens    : ~{n_blocks * args.max_seq_len / 1e6:.2f} M")
    print(f"  Seq length      : {args.max_seq_len} tokens")
    print(f"  Effective batch : {args.batch_size * args.grad_accum} sequences")
    print(f"  Steps/epoch     : ~{steps_per_epoch:,}")
    print(f"  Epochs          : {args.num_epochs}")
    print(f"  LoRA r/alpha    : {lora_cfg.r}/{lora_cfg.lora_alpha}")
    print(f"  Est. time       : ~{est_hours:.1f} h  "
          f"({'%.1f' % (est_hours/24)} days)")
    print("=" * 60)
    print()

    trainer.train(resume_from_checkpoint=args.resume_from)

    adapter_path = os.path.join(args.output_dir, "pretrain_adapter")
    print(f"\nSaving adapter → {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print("\nPre-training complete.")
    print(f"\nNext:")
    print(f"  python MIRA_step3_chat_finetune.py "
          f"--pretrain_adapter {adapter_path} "
          f"--abstracts ./raw/MASTER_merged.txt "
          f"--output_dir ./chat_output")


if __name__ == "__main__":
    main()
