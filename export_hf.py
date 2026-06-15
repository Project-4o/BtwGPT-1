"""
Export BtwGPT-1 to HuggingFace Format
=====================================
Converts a trained BtwGPT-1 checkpoint to HuggingFace-compatible format
so it can be converted to GGUF via llama.cpp's convert_hf_to_gguf.py.

Usage:
    python export_hf.py --checkpoint checkpoints/btwgpt1_final.pt --output output/btwgpt1-hf

Then convert to GGUF:
    python llama.cpp/convert_hf_to_gguf.py output/btwgpt1-hf --outtype f16
"""

import os
import json
import argparse
import shutil
import torch
from safetensors.torch import save_file

from model.transformer import BtwGPTForCausalLM


def remap_state_dict(state_dict: dict) -> dict:
    """
    Remap BtwGPT-1 state dict keys to Mixtral-compatible naming.
    This ensures convert_hf_to_gguf.py can process the model.
    """
    new_state_dict = {}

    key_mapping = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for old_key, param in state_dict.items():
        new_key = None

        if old_key in key_mapping:
            new_key = key_mapping[old_key]

        elif ".self_attn." in old_key:
            new_key = old_key.replace("model.layers.", "model.layers.")

        elif ".block_sparse_moe.gate." in old_key:
            new_key = old_key.replace("model.layers.", "model.layers.")

        elif ".block_sparse_moe.experts." in old_key:
            new_key = old_key.replace("model.layers.", "model.layers.")

        elif ".input_layernorm." in old_key:
            new_key = old_key

        elif ".post_attention_layernorm." in old_key:
            new_key = old_key

        else:
            new_key = old_key

        new_state_dict[new_key] = param

    return new_state_dict


def create_config_json(config: dict, output_dir: str):
    """Create HuggingFace-compatible config.json for Mixtral architecture."""
    model_cfg = config["model"]

    hf_config = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": model_cfg["hidden_size"],
        "intermediate_size": model_cfg["intermediate_size"],
        "num_hidden_layers": model_cfg["num_hidden_layers"],
        "num_attention_heads": model_cfg["num_attention_heads"],
        "num_key_value_heads": model_cfg["num_key_value_heads"],
        "max_position_embeddings": model_cfg["max_position_embeddings"],
        "vocab_size": model_cfg["vocab_size"],
        "rms_norm_eps": model_cfg["rms_norm_eps"],
        "rope_theta": model_cfg["rope_theta"],
        "num_local_experts": model_cfg["num_local_experts"],
        "num_experts_per_tok": model_cfg["num_experts_per_tok"],
        "hidden_act": model_cfg["hidden_act"],
        "tie_word_embeddings": model_cfg["tie_word_embeddings"],
        "initializer_range": model_cfg["initializer_range"],
        "torch_dtype": "float16",
        "sliding_window": None,
        "attention_dropout": 0.0,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
    }

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)


def create_tokenizer_config(output_dir: str):
    """Create tokenizer config files for HuggingFace."""
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 4096,
        "tokenizer_class": "LlamaTokenizer",
        "clean_up_tokenization_spaces": False,
    }

    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)


def export_to_hf(checkpoint_path: str, output_dir: str, tokenizer_model: str = None):
    """Export trained BtwGPT-1 model to HuggingFace format."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    print("Remapping state dict to Mixtral format...")
    hf_state_dict = remap_state_dict(state_dict)

    print("Saving model in safetensors format...")
    save_file(hf_state_dict, os.path.join(output_dir, "model.safetensors"))

    print("Creating config.json...")
    create_config_json(config, output_dir)

    print("Creating tokenizer config...")
    create_tokenizer_config(output_dir)

    if tokenizer_model and os.path.exists(tokenizer_model):
        print(f"Copying tokenizer model: {tokenizer_model}")
        shutil.copy(tokenizer_model, os.path.join(output_dir, "tokenizer.model"))

    print(f"\nExport complete! Model saved to: {output_dir}")
    print(f"\nTo convert to GGUF:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_dir} --outtype f16")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_dir} --outtype q8_0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BtwGPT-1 to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--output", type=str, default="output/btwgpt1-hf", help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/btwgpt.model", help="Tokenizer model path")
    args = parser.parse_args()

    export_to_hf(args.checkpoint, args.output, args.tokenizer)
