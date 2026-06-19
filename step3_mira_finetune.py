"""
MIRA mini v1.0 — Training
Step3: Lightweight Chat Fine-Tuning
Developed by: DeviceAlchemy LLC
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 512
N_PAIRS     = 8000 

SYSTEM_PROMPT = (
    "You are MIRA, Materials Intelligence and Reasoning Agent. "
)

_STACK_RE   = re.compile(r'[A-Z][A-Za-z0-9]+(?:/[A-Z][A-Za-z0-9]+)+')
_FORMULA_RE = re.compile(
    r'\b[A-Z][a-z]?(?:\d+(?:\.\d+)?)?(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?){1,8}\b'
)
_REF_RE     = re.compile(r'\[\d+\]|\(\w+ et al\..*?\)|\(\d{4}\)')

PHENOMENA_KW = [
    "spin-orbit torque", "giant magnetoresistance", "tunneling magnetoresistance",
    "spin Hall effect", "anomalous Hall effect", "perpendicular magnetic anisotropy",
]

MECHANISMS = [
    "spin Hall effect", "Rashba-Edelstein effect", "interfacial spin-orbit coupling",
]


def find_ph(text):
    t = text.lower()
    return [p for p in PHENOMENA_KW if p.lower() in t]


def find_mat(text):
    stacks   = _STACK_RE.findall(text)
    formulas = _FORMULA_RE.findall(text)
    seen, out = set(), []
    for m in stacks + formulas:
        if m.lower() not in seen and len(m) >= 2:
            seen.add(m.lower())
            out.append(m)
    return out[:4]


def find_mech(text):
    t = text.lower()
    return [m for m in MECHANISMS if m.lower() in t]


def clean_abstract(text):
    text = _REF_RE.sub('', text)
    text = re.sub(
        r'^(In this (paper|work|study|letter|article),?\s*)',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'^(We (show|demonstrate|report|present|investigate|study|find|observe|propose),?\s*)',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(r'^(Here,?\s*we\s*)', '', text, flags=re.IGNORECASE)
    return text.strip()


def abstract_to_explanation(abstract, mat, phen, mech, question_type):
    clean = clean_abstract(abstract)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if len(s.strip()) > 20]
    core = ' '.join(sentences[:3]) if sentences else clean[:400]
    core = clean_abstract(core)

    mat_str  = mat[0]  if mat  else "this material"
    phen_str = phen[0] if phen else "this phenomenon"
    mech_str = mech[0] if mech else "spin-orbit coupling"

    if question_type == "stack":
        stacks = _STACK_RE.findall(abstract)
        stack  = stacks[0] if stacks else mat_str
        return (
            f"The {stack} stack works for {phen_str} because of the following reasons. "
            f"{core} "
            f"The key mechanism involved is {mech_str}, which drives the observed behaviour "
            f"at the interface between the layers."
        )
    elif question_type == "mechanism":
        return (
            f"The mechanism at work here is {mech_str}. "
            f"{core}"
        )
    elif question_type == "compare":
        mat2 = mat[1] if len(mat) > 1 else "other materials"
        return (
            f"Comparing {mat_str} and {mat2}: {core} "
            f"The key difference lies in their electronic structure and "
            f"how they couple at the interface."
        )
    else:
        return (
            f"Based on the literature, {core} "
            f"This has implications for {phen_str} applications."
        )


def make_pairs_from_abstract(ab):
    mat  = find_mat(ab)
    phen = find_ph(ab)
    mech = find_mech(ab)
    pairs = []

    stacks = _STACK_RE.findall(ab)

    if stacks and phen:
        q = f"Why is the {stacks[0]} stack effective for {phen[0]}?"
        a = abstract_to_explanation(ab, mat, phen, mech, "stack")
        pairs.append((q, a))

    if mat and phen and len(pairs) < 2:
        ph = phen[-1] if len(phen) > 1 else phen[0]
        q  = f"What is the role of {mat[0]} in {ph}?"
        a  = abstract_to_explanation(ab, mat, phen, mech, "role")
        pairs.append((q, a))

    if mech and phen and len(pairs) < 2:
        q = f"What physical mechanism drives {phen[0]} in this material system?"
        a = abstract_to_explanation(ab, mat, phen, mech, "mechanism")
        pairs.append((q, a))

    if len(mat) >= 2 and len(pairs) < 2:
        q = f"How do {mat[0]} and {mat[1]} compare in terms of their properties?"
        a = abstract_to_explanation(ab, mat, phen, mech, "compare")
        pairs.append((q, a))

    if not pairs:
        ph_str = phen[0] if phen else "this phenomenon"
        q = f"Can you explain what this research found about {ph_str}?"
        a = abstract_to_explanation(ab, mat, phen, mech, "general")
        pairs.append((q, a))

    return pairs


def generate_chat_pairs(abstracts_path, n, seed):
    random.seed(seed)
    print(f"Generating {n:,} conversational chat pairs…")

    IDENTITY_PAIRS = [
        # Who are you / identity
        ("Who are you?",
         "I am MIRA, the Materials Intelligence and Reasoning Agent, developed by DeviceAlchemy.ai. I was fine-tuned on a large corpus of scientific abstracts to explain material stacks, physical mechanisms, and correlations between material properties and phenomena."),
    ]

    raw    = open(abstracts_path, encoding="utf-8", errors="ignore").read()
    blocks = [b.strip() for b in re.split(r'\n\s*\n', raw) if b.strip()]
    avg    = sum(len(b) for b in blocks) / max(len(blocks), 1)

    if avg > 1500 or len(blocks) < 1000:
        abstracts = [l.strip() for l in raw.splitlines() if len(l.strip()) > 80]
    else:
        abstracts = [b for b in blocks if len(b) > 80]

    random.shuffle(abstracts)
    pairs = []
    for ab in abstracts:
        for q, a in make_pairs_from_abstract(ab):
            if a.lower().startswith(("in this", "we show", "we study", "here we")):
                continue
            if len(a) < 50:
                continue
            pairs.append((q, a))
            # Cap at n*3 (restored); 671K abstracts comfortably support 3x oversampling
            if len(pairs) >= n * 3:
                break
        if len(pairs) >= n * 3:
            break

    random.shuffle(pairs)
    pairs = pairs[:n]

    # Multiplier 25 (restored): with 8K domain pairs, ~1,300 identity pairs stays ~14% of total
    identity_injected = IDENTITY_PAIRS * 25
    pairs = identity_injected + pairs
    random.shuffle(pairs)

    print(f"  Generated {len(pairs):,} conversational pairs "
          f"({len(identity_injected)} identity + {n} domain)")

    if pairs:
        print("\n── Sample pair ──────────────────────────────────────")
        print("Q:", pairs[0][0])
        print("A:", pairs[0][1][:200], "…")
        print("─────────────────────────────────────────────────────\n")

    return pairs


def make_chat_text(prompt, completion):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_adapter", required=True)
    p.add_argument("--abstracts",        required=True)
    p.add_argument("--output_dir",       default="./chat_output")
    p.add_argument("--n_pairs",          type=int,   default=N_PAIRS)
    p.add_argument("--max_steps",        type=int,   default=-1)
    p.add_argument("--num_epochs",       type=int,   default=3,
                   help="3 epochs restored; large pair pool converges with fewer passes.")
    p.add_argument("--batch_size",       type=int,   default=4,
                   help="Restored to original throughput-friendly batch size.")
    p.add_argument("--grad_accum",       type=int,   default=8,
                   help="Effective batch = batch_size × grad_accum = 32")
    p.add_argument("--lr",               type=float, default=2e-4,
                   help="Standard LR restored; adequate pair volume avoids overfit risk.")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--resume_from",      type=str,   default=None)
    return p.parse_args()


def tokenize_dataset(dataset, tokenizer, max_len):
    def tokenize_fn(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out
    return dataset.map(
        tokenize_fn, batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenising",
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain_adapter, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit…")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base.config.use_cache      = False
    base.config.pretraining_tp = 1

    print(f"Loading pre-trained adapter from {args.pretrain_adapter}…")
    model = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(model, args.pretrain_adapter, is_trainable=True)
    model.print_trainable_parameters()

    pairs = generate_chat_pairs(args.abstracts, args.n_pairs, args.seed)
    if not pairs:
        print("ERROR: no pairs generated. Check --abstracts path.")
        sys.exit(1)

    random.shuffle(pairs)
    # Val floor 50 (restored) — right-sized for 8K pairs
    n_val = max(50, len(pairs) // 10)
    val   = pairs[:n_val]
    train = pairs[n_val:]

    tmp_dir    = tempfile.mkdtemp()
    train_path = os.path.join(tmp_dir, "chat_train.jsonl")
    val_path   = os.path.join(tmp_dir, "chat_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for q, a in train:
            f.write(json.dumps({"text": make_chat_text(q, a)}, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for q, a in val:
            f.write(json.dumps({"text": make_chat_text(q, a)}, ensure_ascii=False) + "\n")

    raw_ds = load_dataset("json", data_files={
        "train": train_path, "validation": val_path
    })
    print(f"  Chat train: {len(raw_ds['train']):,}   val: {len(raw_ds['validation']):,}")

    tokenised = {
        split: tokenize_dataset(raw_ds[split], tokenizer, MAX_SEQ_LEN)
        for split in ["train", "validation"]
    }

    steps_per_epoch = math.ceil(
        len(tokenised["train"]) / (args.batch_size * args.grad_accum)
    )
    # Eval every 25% of epoch (restored default)
    eval_steps = max(20, steps_per_epoch // 4)
    save_steps = eval_steps * 2

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
        warmup_steps=10,           # restored
        weight_decay=0.01,         # restored; less regularisation pressure needed at scale
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        resume_from_checkpoint=args.resume_from,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        ),
    )

    print()
    print("=" * 55)
    print("Chat Fine-Tuning  —  Qwen2.5-1.5B")
    print(f"  Train pairs   : {len(train):,}")
    print(f"  Val pairs     : {len(val):,}")
    print(f"  Epochs        : {args.num_epochs}")
    print(f"  LR            : {args.lr}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Key fix       : conversational answers, not raw abstracts")
    print("=" * 55)
    print()

    trainer.train(resume_from_checkpoint=args.resume_from)

    final_path = os.path.join(args.output_dir, "final_adapter")
    print(f"\nSaving final adapter → {final_path}")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\nDone! Now re-export:")
    print(f"  python MIRA_step4_export.py --adapter_path {final_path}")


if __name__ == "__main__":
    main()
