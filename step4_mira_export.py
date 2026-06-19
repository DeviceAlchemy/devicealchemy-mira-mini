"""
MIRA mini v1.0 — Training
Step4: Merge Adapters and Export to GGUF for Ollama
Developed by: DeviceAlchemy LLC
Copyright (c) 2026
Code author: Shehrin Sayed, Ph.D.
"""

import argparse
import os
import subprocess
import sys
import torch
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """You are MIRA, Materials Intelligence and Reasoning Agent. You have been trained on a large corpus of scientific abstracts on condensed matter physics, device engineering, and materials science. You explain material stack predictions, physical mechanisms, and correlations between material properties and phenomena. Be precise, scientific, and concise."""

def merge(adapter_path, output_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Step 1/3 — Merging adapters into base model (CPU, ~5–10 min)…")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True).save_pretrained(output_dir)
    print(f"  ✓ Merged model → {output_dir}")
    return output_dir

def to_gguf(merged_dir, llama_cpp_dir, gguf_dir):
    lc     = Path(llama_cpp_dir)
    script = lc / "convert_hf_to_gguf.py"
    if not script.exists():
        script = lc / "convert.py"
    if not script.exists():
        print(f"ERROR: conversion script not found in {llama_cpp_dir}")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        sys.exit(1)

    os.makedirs(gguf_dir, exist_ok=True)
    f16 = os.path.join(gguf_dir, "mira-f16.gguf")

    print("Step 2/3 — Converting to GGUF F16…")
    r = subprocess.run([sys.executable, str(script),
                        merged_dir, "--outfile", f16, "--outtype", "f16"])
    if r.returncode != 0:
        print("ERROR: conversion failed"); sys.exit(1)
    print(f"  ✓ F16 GGUF → {f16}")
    return f16

def quantise(f16, llama_cpp_dir, gguf_dir):
    lc   = Path(llama_cpp_dir)
    # Check all possible locations including Windows .exe
    candidates = [
        lc / "build" / "bin" / "llama-quantize.exe",
        lc / "build" / "bin" / "llama-quantize",
        lc / "llama-quantize.exe",
        lc / "llama-quantize",
        lc / "quantize.exe",
        lc / "quantize",
    ]
    qbin = next((c for c in candidates if c.exists()), None)

    if qbin is None:
        print("ERROR: llama-quantize binary not found.")
        print("Expected locations checked:")
        for c in candidates:
            print(f"  {c}")
        print("\nDownload pre-built binaries from:")
        print("  https://github.com/ggerganov/llama.cpp/releases")
        print("and copy all .exe and .dll files into:")
        print(f"  {lc / 'build' / 'bin'}")
        sys.exit(1)

    q4 = os.path.join(gguf_dir, "mira-q4_k_m.gguf")
    print(f"Step 3/3 — Quantising to Q4_K_M (~1 GB for 1.5B model)…")
    print(f"  Using: {qbin}")
    r = subprocess.run([str(qbin), f16, q4, "Q4_K_M"])
    if r.returncode != 0:
        print("ERROR: quantisation failed"); sys.exit(1)
    os.remove(f16)
    size = os.path.getsize(q4) / 1e9
    print(f"  ✓ Q4_K_M GGUF → {q4}  ({size:.1f} GB)")
    return q4

def write_modelfile(q4, gguf_dir):
    mf_path = os.path.join(gguf_dir, "Modelfile")
    content = f"""FROM {os.path.abspath(q4)}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.5
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
"""
    with open(mf_path, "w") as f:
        f.write(content)
    print(f"  ✓ Modelfile → {mf_path}")
    return mf_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_path", required=True,
                   help="Path to final_adapter from step3_chat_finetune.py")
    p.add_argument("--output_dir",   default="./model_export")
    p.add_argument("--llama_cpp",    default="./llama.cpp")
    p.add_argument("--skip_gguf",    action="store_true",
                   help="Stop after merging, skip GGUF conversion")
    args = p.parse_args()

    merged_dir = os.path.join(args.output_dir, "merged_hf")
    gguf_dir   = os.path.join(args.output_dir, "gguf")

    merge(args.adapter_path, merged_dir)

    if args.skip_gguf:
        print(f"\nMerged HF model at: {merged_dir}")
        print("Skipped GGUF (--skip_gguf).")
        return

    f16 = to_gguf(merged_dir, args.llama_cpp, gguf_dir)
    q4  = quantise(f16, args.llama_cpp, gguf_dir)
    mf  = write_modelfile(q4, gguf_dir)

    print()
    print("=" * 55)
    print("All done!")
    print("=" * 55)
    print()
    print("Register with Ollama:")
    print(f"  ollama create mira -f {mf}")
    print()
    print("Test it:")
    print("  ollama run mira")
    print()
    print("Start the chat server:")
    print("  python inference/serve.py")
    print()
    print("Open in browser:")
    print("  http://localhost:5002/chat")


if __name__ == "__main__":
    main()
