# Changelog - BtwGPT-1

## [0.2.0] - 2025-06-15

### Added
- System prompt support (config/system_prompt.txt)
  - Default French "chill friend" personality
  - Auto-injected into conversations without explicit system message
- Evaluation loop (5% hold-out split, runs every eval_steps)
- Early stopping (configurable patience, saves best model)
- Perplexity tracking (train + eval)
- Gradient norm monitoring
- Tokens/sec throughput tracking
- MetricsLogger (saves to logs/training_metrics.json)
- Matplotlib visualization script (visualize.py)
  - Training & eval loss curves
  - Perplexity (log scale)
  - Learning rate schedule
  - Gradient norm over time
  - MoE router load balancing
  - Training throughput (tokens/sec)
  - Detailed loss comparison plot
  - Expert balance plot

## [0.1.0] - 2025-06-15

### Added
- Initial project structure
- MoE + Transformer model architecture (~150M params)
  - 16 layers, 768 hidden, 12 heads, 4 KV heads (GQA)
  - 8 experts, top-2 routing per token
  - RoPE, RMSNorm, SwiGLU activations
- Training pipeline with PyTorch ROCm support
  - Mixed precision (FP16)
  - Gradient accumulation
  - Cosine LR schedule with warmup
  - Load balancing auxiliary loss for MoE
- Data pipeline
  - SentencePiece BPE tokenizer (French-focused, 32K vocab)
  - Support for .txt and .jsonl training data
  - Mixtral-style chat template
- Export pipeline
  - HuggingFace format export (Mixtral-compatible)
  - Ready for GGUF conversion via llama.cpp
- Configuration (YAML-based)
- Documentation (README, TODO, CHANGELOG)
