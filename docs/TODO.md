# TODO - BtwGPT-1

## In Progress
- [ ] User provides training data (French conversational)
- [ ] Train tokenizer on user data

## Pending
- [ ] Train model on full dataset
- [ ] Evaluate model quality (perplexity, generation samples)
- [ ] Export to HuggingFace format
- [ ] Convert to GGUF (f16, q8_0, q4_k_m)
- [ ] Test in Ollama / LM Studio
- [ ] Upload to HuggingFace Hub
- [ ] Add English support (phase 2)

## Completed
- [x] Design model architecture (MoE + Transformer, ~150M params)
- [x] Implement model code (PyTorch ROCm)
- [x] Implement training pipeline
- [x] Implement GGUF export pipeline
- [x] Create tokenizer training code (SentencePiece BPE, French)
- [x] Create dataset loading code (text + JSONL support)
- [x] Add system prompt (French chill friend personality)
- [x] Add eval loop with early stopping
- [x] Add perplexity + gradient norm tracking
- [x] Add matplotlib visualization (visualize.py)
