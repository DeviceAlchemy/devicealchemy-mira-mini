# MIRA mini v1.0 — Materials Intelligence and Reasoning Agent

Training pipeline for MIRA, a Qwen2.5-1.5B-Instruct model fine-tuned via QLoRA on
materials science journal abstracts. Developed by DeviceAlchemy LLC.

The trained model (GGUF, quantized for Ollama) is hosted on Hugging Face:
**[link to your HF model repo here]**

## Pipeline

Run in order:

```bash
# 1. Clean and split raw abstracts into train/val
python step1_mira_prepare.py --input /path/to/abstracts.txt --output_dir ./data

# 2. Continued pre-training (unsupervised) on the abstracts
python step2_mira_train.py --data_dir ./data --output_dir ./pretrain_output

# 3. Chat fine-tuning (Q&A pairs generated from abstracts + identity pairs)
python step3_mira_finetune.py \
  --pretrain_adapter ./pretrain_output/pretrain_adapter \
  --abstracts /path/to/abstracts.txt \
  --output_dir ./chat_output

# 4. Merge adapters and export to GGUF for Ollama
python step4_mira_export.py --adapter_path ./chat_output/final_adapter --output_dir ./model_export
```

Step 4 requires [llama.cpp](https://github.com/ggerganov/llama.cpp) cloned locally for
GGUF conversion and quantization (pass `--llama_cpp /path/to/llama.cpp`).

## Setup

```bash
pip install -r requirements.txt
```

**Note:** `transformers` is pinned below 4.47.0. Newer versions introduced a breaking
change in tokenizer special-token handling that causes `AttributeError: 'list' object
has no attribute 'keys'` during llama.cpp GGUF conversion.

## Base model

[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Alibaba, Apache 2.0)

## Training data

Scientific paper abstracts from Gold Open Access journals (CC BY 4.0 or equivalent
permissive licenses), spanning condensed matter physics, materials science, electronic
devices, and related engineering domains.

## License

Apache 2.0

## Author

Shehrin Sayed, Ph.D. — [DeviceAlchemy.ai](https://devicealchemy.ai)
