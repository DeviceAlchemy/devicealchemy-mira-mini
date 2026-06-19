"""
MIRA mini v1.0 — Training
Step1: Prepare Raw Text for Continued Pre-Training
Developed by: DeviceAlchemy LLC
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

_URL_RE       = re.compile(r'https?://\S+')
_DOI_RE       = re.compile(r'\b10\.\d{4,}/\S+')
_COPYRIGHT_RE = re.compile(r'©|copyright|\(c\)\s*\d{4}', re.IGNORECASE)
_MENU_RE      = re.compile(r'^(home|about|contact|login|register|search|menu|navigation)\s*$',
                            re.IGNORECASE)

MIN_ABSTRACT_LEN = 80


def clean(text: str) -> str:
    """Light cleaning — preserve scientific content, remove noise."""
    text = _URL_RE.sub('', text)
    text = _DOI_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_valid(text: str) -> bool:
    """Keep if it looks like scientific prose."""
    if len(text) < MIN_ABSTRACT_LEN:
        return False
    if _COPYRIGHT_RE.search(text):
        return False
    if _MENU_RE.match(text):
        return False
    if not re.search(r'[a-zA-Z]{4,}', text):
        return False
    return True


def read_abstracts(path: Path) -> list:
    size_mb = path.stat().st_size / 1e6
    print(f"Reading {path} ({size_mb:.1f} MB)…")

    raw    = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in re.split(r'\n\s*\n', raw) if b.strip()]
    avg    = sum(len(b) for b in blocks) / max(len(blocks), 1)

    if avg > 1500 or len(blocks) < 1000:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        print(f"  Format detected : one abstract per line")
        print(f"  Abstracts found : {len(lines):,}")
        return lines

    print(f"  Format detected : blank-line separated blocks")
    print(f"  Abstracts found : {len(blocks):,}  (avg {avg:.0f} chars)")
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,
                        help="Path to abstracts .txt file")
    parser.add_argument("--output_dir",  default="./data")
    parser.add_argument("--val_split",   type=float, default=0.02,
                        help="Validation fraction. 0.02 → ~13,400 val docs for 671K corpus.")
    parser.add_argument("--max_samples", type=int,   default=None,
                        help="Cap abstracts for quick tests")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    abstracts = read_abstracts(Path(args.input))

    if args.max_samples and len(abstracts) > args.max_samples:
        abstracts = random.sample(abstracts, args.max_samples)
        print(f"  Capped to       : {len(abstracts):,} (--max_samples)")

    print("Cleaning and filtering…")
    cleaned  = []
    skipped  = 0
    total    = len(abstracts)
    
    interval = max(1, total // 20)

    for i, ab in enumerate(abstracts):
        if i % interval == 0:
            pct = 100 * i // total
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"  [{bar}] {pct:3d}%  kept: {len(cleaned):,}", end="\r")

        c = clean(ab)
        if is_valid(c):
            cleaned.append(c)
        else:
            skipped += 1

    print(" " * 70, end="\r")
    print(f"  Kept    : {len(cleaned):,}")
    print(f"  Skipped : {skipped:,}  (too short / boilerplate)")

    if not cleaned:
        print("\nERROR: 0 abstracts passed the filter.")
        print("Check your input file with:  head -5 " + args.input)
        sys.exit(1)

    random.shuffle(cleaned)

    n_val = max(500, min(10_000, int(len(cleaned) * args.val_split)))
    n_val = min(n_val, len(cleaned) // 10)
    val   = cleaned[:n_val]
    train = cleaned[n_val:]

    train_path = os.path.join(args.output_dir, "pretrain_train.txt")
    val_path   = os.path.join(args.output_dir, "pretrain_val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val))

    total_chars  = sum(len(a) for a in cleaned)
    
    total_tokens = int(total_chars / 4.2)
    stats = {
        "total_abstracts":  len(cleaned),
        "train_abstracts":  len(train),
        "val_abstracts":    len(val),
        "total_chars":      total_chars,
        "approx_tokens":    total_tokens,
        "approx_tokens_M":  round(total_tokens / 1e6, 2),
    }
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("─" * 55)
    print(f"  Train abstracts : {len(train):,}")
    print(f"  Val abstracts   : {len(val):,}")
    print(f"  Total tokens    : ~{total_tokens/1e6:.2f} M")
    print(f"  Train file      : {train_path}")
    print(f"  Val file        : {val_path}")
    print("─" * 55)
    print()
    print("Sample (first abstract in train):")
    print("  " + train[0][:200] + "…")
    print()
    print("Next step:")
    print("  python MIRA_step2_pretrain.py --data_dir ./data --output_dir ./pretrain_output")


if __name__ == "__main__":
    main()
