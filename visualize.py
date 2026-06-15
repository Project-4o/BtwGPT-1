"""
BtwGPT-1 Training Visualization
================================
Plot training metrics using matplotlib.

Usage:
    python visualize.py
    python visualize.py --metrics logs/training_metrics.json --output logs/plots/
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np


def load_metrics(metrics_path: str) -> dict:
    """Load training metrics from JSON."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_none(steps, values):
    """Filter out None values and return aligned arrays."""
    filtered_steps = []
    filtered_values = []
    for s, v in zip(steps, values):
        if v is not None:
            filtered_steps.append(s)
            filtered_values.append(v)
    return np.array(filtered_steps), np.array(filtered_values)


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_training_curves(metrics: dict, output_dir: str):
    """Generate all training visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    mplstyle.use("seaborn-v0_8-darkgrid")

    steps = metrics["steps"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("BtwGPT-1 Training Metrics", fontsize=16, fontweight="bold")

    # 1. Training Loss
    ax = axes[0, 0]
    s, v = filter_none(steps, metrics["train_loss"])
    if len(v) > 0:
        ax.plot(s, v, alpha=0.3, color="blue", label="Raw")
        ax.plot(s, smooth(v), color="blue", linewidth=2, label="Smoothed")
    s_eval, v_eval = filter_none(steps, metrics["eval_loss"])
    if len(v_eval) > 0:
        ax.plot(s_eval, v_eval, color="red", linewidth=2, marker="o", markersize=4, label="Eval")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Eval Loss")
    ax.legend()
    ax.set_ylim(bottom=0)

    # 2. Perplexity
    ax = axes[0, 1]
    s, v = filter_none(steps, metrics["perplexity"])
    if len(v) > 0:
        ax.plot(s, v, alpha=0.3, color="green")
        ax.plot(s, smooth(v), color="green", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity")
    ax.set_yscale("log")

    # 3. Learning Rate
    ax = axes[0, 2]
    s, v = filter_none(steps, metrics["learning_rate"])
    if len(v) > 0:
        ax.plot(s, v, color="orange", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")

    # 4. Gradient Norm
    ax = axes[1, 0]
    s, v = filter_none(steps, metrics["grad_norm"])
    if len(v) > 0:
        ax.plot(s, v, alpha=0.3, color="purple")
        ax.plot(s, smooth(v), color="purple", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Grad Norm (L2)")
    ax.set_title("Gradient Norm")

    # 5. Aux Loss (Router Load Balancing)
    ax = axes[1, 1]
    s, v = filter_none(steps, metrics["aux_loss"])
    if len(v) > 0:
        ax.plot(s, v, alpha=0.3, color="brown")
        ax.plot(s, smooth(v), color="brown", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Aux Loss")
    ax.set_title("MoE Router Aux Loss (Load Balancing)")

    # 6. Tokens per Second
    ax = axes[1, 2]
    s, v = filter_none(steps, metrics["tokens_per_sec"])
    if len(v) > 0:
        ax.plot(s, v, alpha=0.3, color="teal")
        ax.plot(s, smooth(v), color="teal", linewidth=2)
        ax.axhline(y=np.mean(v), color="red", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(v):.0f}")
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Training Throughput")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_overview.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_detail(metrics: dict, output_dir: str):
    """Detailed loss plot with train vs eval comparison."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    steps = metrics["steps"]
    s_train, v_train = filter_none(steps, metrics["train_loss"])
    s_eval, v_eval = filter_none(steps, metrics["eval_loss"])

    if len(v_train) > 0:
        ax.plot(s_train, v_train, alpha=0.2, color="blue")
        ax.plot(s_train, smooth(v_train, 0.95), color="blue", linewidth=2, label="Train Loss (smoothed)")

    if len(v_eval) > 0:
        ax.plot(s_eval, v_eval, color="red", linewidth=2, marker="o", markersize=5, label="Eval Loss")
        if len(v_eval) > 1:
            best_idx = np.argmin(v_eval)
            ax.axvline(x=s_eval[best_idx], color="green", linestyle="--", alpha=0.5,
                       label=f"Best eval: {v_eval[best_idx]:.4f} @ step {s_eval[best_idx]}")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("BtwGPT-1 — Train vs Eval Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, "loss_detail.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_expert_balance(metrics: dict, output_dir: str):
    """Plot MoE expert load balancing over time."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = metrics["steps"]
    s, v = filter_none(steps, metrics["aux_loss"])

    if len(v) > 0:
        ideal_balance = 1.0
        ax.plot(s, v, alpha=0.3, color="brown")
        ax.plot(s, smooth(v, 0.95), color="brown", linewidth=2, label="Router Aux Loss")
        ax.axhline(y=ideal_balance, color="green", linestyle="--", alpha=0.7,
                   label=f"Ideal balance = {ideal_balance}")
        ax.fill_between(s, smooth(v, 0.95), ideal_balance, alpha=0.1, color="red")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Aux Loss", fontsize=12)
    ax.set_title("MoE Expert Load Balancing", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, "expert_balance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main(metrics_path: str, output_dir: str):
    """Generate all plots."""
    print(f"Loading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)

    num_entries = len(metrics["steps"])
    print(f"Total metric entries: {num_entries}")

    if num_entries == 0:
        print("No metrics to plot. Train the model first!")
        return

    print(f"\nGenerating plots to: {output_dir}")
    plot_training_curves(metrics, output_dir)
    plot_loss_detail(metrics, output_dir)
    plot_expert_balance(metrics, output_dir)
    print("\nAll plots generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BtwGPT-1 training metrics")
    parser.add_argument("--metrics", type=str, default="logs/training_metrics.json", help="Path to metrics JSON")
    parser.add_argument("--output", type=str, default="logs/plots", help="Output directory for plots")
    args = parser.parse_args()

    main(args.metrics, args.output)
