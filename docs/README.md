# BtwGPT-1 by Open4o

A small (~150M parameters) Mixture of Experts (MoE) Transformer language model by **Open4o**, designed as a conversational AI friend.

## Architecture

- **Type**: MoE + Transformer (Mixtral-compatible)
- **Parameters**: ~150M total
- **Hidden size**: 768
- **Layers**: 16
- **Attention heads**: 12 (4 KV heads, GQA)
- **Experts**: 8 total, 2 active per token
- **Context**: 4096 tokens
- **Vocab**: 32,000 (French-focused BPE)

## Features

- Conversational AI designed to act as a friend
- Primary language: French
- Very few rules, natural interaction style
- GGUF-compatible for use in Ollama, LM Studio, llama.cpp, etc.
- Trained with PyTorch ROCm (AMD GPU support)

## Requirements

- Python 3.10+
- PyTorch 2.4+ with ROCm 6.2
- AMD GPU with ROCm support (tested on RX 7600 XT 16GB)

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

Place your training data in `data/train/`:
- `.txt` files: plain text, paragraph-separated
- `.jsonl` files: conversational format:
```json
{"conversations": [{"role": "user", "content": "Salut !"}, {"role": "assistant", "content": "Hey ! Comment tu vas ?"}]}
```

### 2. Train Tokenizer

```bash
python -c "from data.tokenizer import train_tokenizer; train_tokenizer('data/train/your_data.txt')"
```

### 3. Train Model

```bash
python train.py --config config/model_config.yaml
```

### 4. Export to HuggingFace Format

```bash
python export_hf.py --checkpoint checkpoints/btwgpt1_final.pt --output output/btwgpt1-hf
```

### 5. Convert to GGUF

```bash
git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py output/btwgpt1-hf --outtype f16
python llama.cpp/convert_hf_to_gguf.py output/btwgpt1-hf --outtype q8_0
```

### 6. Use in Ollama/LM Studio

The resulting `.gguf` file can be loaded directly in any GGUF-compatible application.

## Project Structure

```
BtwGPT-1/
├── config/model_config.yaml    # Model hyperparameters
├── model/
│   ├── components.py           # RMSNorm, RoPE
│   ├── moe.py                  # Mixture of Experts layer
│   └── transformer.py          # Full model architecture
├── data/
│   ├── dataset.py              # Data loading
│   └── tokenizer.py           # Tokenizer training/loading
├── train.py                    # Training script
├── export_hf.py                # HF export for GGUF conversion
└── docs/                       # Documentation
```

## License

Apache License 2.0 - Copyright 2025 Open4o. See [LICENSE](../LICENSE).
