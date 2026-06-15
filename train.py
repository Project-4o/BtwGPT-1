"""
BtwGPT-1 Training Script
========================
Train the BtwGPT-1 MoE Transformer model on custom data using PyTorch ROCm.

Usage:
    python train.py --config config/model_config.yaml
"""

import os
import argparse
import json
import math
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model.transformer import BtwGPTForCausalLM
from data.dataset import BtwGPTDataset


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> BtwGPTForCausalLM:
    """Create BtwGPT-1 model from config."""
    model_cfg = config["model"]
    model = BtwGPTForCausalLM(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        intermediate_size=model_cfg["intermediate_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg["num_key_value_heads"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        rope_theta=model_cfg["rope_theta"],
        rms_norm_eps=model_cfg["rms_norm_eps"],
        num_local_experts=model_cfg["num_local_experts"],
        num_experts_per_tok=model_cfg["num_experts_per_tok"],
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MetricsLogger:
    """Log training metrics to JSON for matplotlib visualization."""

    def __init__(self, output_dir: str = "logs"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.metrics = {
            "steps": [],
            "train_loss": [],
            "eval_loss": [],
            "perplexity": [],
            "aux_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "epoch": [],
            "tokens_per_sec": [],
        }

    def log(self, step: int, **kwargs):
        """Log a metrics entry."""
        self.metrics["steps"].append(step)
        for key in self.metrics:
            if key == "steps":
                continue
            self.metrics[key].append(kwargs.get(key, None))

    def save(self):
        """Save metrics to JSON."""
        path = os.path.join(self.output_dir, "training_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)


@torch.no_grad()
def evaluate(model, eval_loader, device, fp16: bool = True) -> dict:
    """Run evaluation and compute loss + perplexity."""
    model.eval()
    total_loss = 0.0
    total_aux = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=fp16):
            outputs = model(input_ids=input_ids, labels=labels)

        mask = labels != -100
        num_tokens = mask.sum().item()
        total_loss += outputs["loss"].item() * num_tokens
        total_aux += outputs["aux_loss"].item()
        total_tokens += num_tokens
        num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    avg_aux = total_aux / max(num_batches, 1)

    model.train()
    return {"eval_loss": avg_loss, "perplexity": perplexity, "aux_loss": avg_aux}


def compute_grad_norm(model) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def train(config_path: str, resume_from: str = None):
    """Main training loop with eval, early stopping, and metrics logging."""
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    torch.manual_seed(train_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    print("\n--- Creating model ---")
    model = create_model(config)
    num_params = count_parameters(model)
    print(f"Model: {model_cfg['name']}")
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    model = model.to(device)

    print("\n--- Loading dataset ---")
    system_prompt_path = data_cfg.get("system_prompt", "config/system_prompt.txt")
    full_dataset = BtwGPTDataset(
        data_path=data_cfg["train_data"],
        tokenizer_path=data_cfg["tokenizer_model"],
        max_seq_length=data_cfg["max_seq_length"],
        split="train",
        system_prompt_path=system_prompt_path,
    )

    eval_size = max(1, int(len(full_dataset) * 0.05))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = random_split(
        full_dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(train_cfg["seed"]),
    )
    print(f"Training samples: {train_size:,}")
    print(f"Eval samples: {eval_size:,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    total_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * decay_ratio)) / 2.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    scaler = GradScaler(enabled=train_cfg["fp16"])
    metrics_logger = MetricsLogger(output_dir="logs")

    global_step = 0
    best_eval_loss = float("inf")
    patience_counter = 0
    early_stop_patience = train_cfg.get("early_stop_patience", 5)

    if resume_from and os.path.exists(resume_from):
        print(f"\n--- Resuming from checkpoint: {resume_from} ---")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint["global_step"]
        best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        print(f"Resumed at step {global_step}")

    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n--- Starting training ---")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Gradient accumulation: {train_cfg['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Max steps: {total_steps}")
    print(f"  FP16: {train_cfg['fp16']}")
    print(f"  Early stopping patience: {early_stop_patience} evals")
    print(f"  System prompt: {system_prompt_path}")
    print()

    model.train()
    accumulation_steps = train_cfg["gradient_accumulation_steps"]
    running_loss = 0.0
    running_aux_loss = 0.0
    running_grad_norm = 0.0
    log_interval = train_cfg["logging_steps"]
    tokens_processed = 0
    last_log_time = time.time()

    epoch = 0
    while global_step < total_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            batch_tokens = (labels != -100).sum().item()

            with autocast(enabled=train_cfg["fp16"]):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / accumulation_steps

            scaler.scale(loss).backward()

            running_loss += outputs["loss"].item()
            running_aux_loss += outputs["aux_loss"].item()
            tokens_processed += batch_tokens

            if (global_step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = compute_grad_norm(model)
                running_grad_norm += grad_norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            global_step += 1

            if global_step % log_interval == 0:
                elapsed = time.time() - last_log_time
                avg_loss = running_loss / log_interval
                avg_aux = running_aux_loss / log_interval
                avg_grad = running_grad_norm / max(log_interval // accumulation_steps, 1)
                lr = scheduler.get_last_lr()[0]
                tps = tokens_processed / max(elapsed, 1e-6)
                ppl = math.exp(min(avg_loss, 100))

                print(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {ppl:.1f} | "
                    f"Aux: {avg_aux:.4f} | "
                    f"Grad: {avg_grad:.2f} | "
                    f"LR: {lr:.2e} | "
                    f"Tok/s: {tps:.0f} | "
                    f"Epoch: {epoch}"
                )

                metrics_logger.log(
                    step=global_step,
                    train_loss=avg_loss,
                    eval_loss=None,
                    perplexity=ppl,
                    aux_loss=avg_aux,
                    learning_rate=lr,
                    grad_norm=avg_grad,
                    epoch=epoch,
                    tokens_per_sec=tps,
                )

                running_loss = 0.0
                running_aux_loss = 0.0
                running_grad_norm = 0.0
                tokens_processed = 0
                last_log_time = time.time()

            if global_step % train_cfg["eval_steps"] == 0:
                print("  Running evaluation...")
                eval_results = evaluate(model, eval_loader, device, train_cfg["fp16"])
                print(
                    f"  Eval Loss: {eval_results['eval_loss']:.4f} | "
                    f"Eval PPL: {eval_results['perplexity']:.1f}"
                )

                metrics_logger.log(
                    step=global_step,
                    train_loss=None,
                    eval_loss=eval_results["eval_loss"],
                    perplexity=eval_results["perplexity"],
                    aux_loss=eval_results["aux_loss"],
                    learning_rate=scheduler.get_last_lr()[0],
                    grad_norm=None,
                    epoch=epoch,
                    tokens_per_sec=None,
                )

                if eval_results["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_results["eval_loss"]
                    patience_counter = 0
                    best_path = "checkpoints/btwgpt1_best.pt"
                    torch.save(
                        {
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_eval_loss": best_eval_loss,
                            "config": config,
                        },
                        best_path,
                    )
                    print(f"  -> New best model saved! (loss={best_eval_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  -> No improvement ({patience_counter}/{early_stop_patience})")

                if patience_counter >= early_stop_patience:
                    print("\n--- Early stopping triggered! ---")
                    break

                metrics_logger.save()

            if global_step % train_cfg["save_steps"] == 0:
                save_path = f"checkpoints/btwgpt1_step{global_step}.pt"
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_eval_loss": best_eval_loss,
                        "config": config,
                    },
                    save_path,
                )
                print(f"  -> Checkpoint saved: {save_path}")

        if patience_counter >= early_stop_patience:
            break

    final_path = "checkpoints/btwgpt1_final.pt"
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "best_eval_loss": best_eval_loss,
            "config": config,
        },
        final_path,
    )
    metrics_logger.save()
    print(f"\n--- Training complete! ---")
    print(f"Final checkpoint: {final_path}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print(f"Total steps: {global_step}")
    print(f"Metrics saved to: logs/training_metrics.json")
    print(f"Run 'python visualize.py' to plot training curves.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BtwGPT-1")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args.config, args.resume)
