import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(trainer):
    """
    Plot training and validation curves from a HuggingFace Trainer.

    Creates a 2x2 grid of plots showing:
    1. Training vs Validation loss over steps
    2. Validation loss over epochs
    3. F1 score over epochs
    4. Exact match over epochs

    Args:
        trainer: A trained HuggingFace Trainer object with log_history

    Returns:
        fig: The matplotlib figure object
    """
    # Extract clean aligned metrics (same as progress bar table)
    train_losses = {}  # step -> train_loss mapping
    eval_data = []  # list of eval entries

    for entry in trainer.state.log_history:
        step = entry.get("step")

        # Collect training losses (entries with 'loss' but no 'eval_loss')
        if "loss" in entry and "eval_loss" not in entry:
            train_losses[step] = entry["loss"]

        # Collect evaluation entries (only keep entries with epoch info)
        if "eval_loss" in entry and "epoch" in entry and entry["epoch"] is not None:
            eval_data.append(
                {
                    "step": step,
                    "epoch": entry["epoch"],
                    "eval_loss": entry["eval_loss"],
                    "eval_f1": entry["eval_f1"],
                    "eval_exact_match": entry["eval_exact_match"],
                }
            )

    # Convert to DataFrame
    metrics_df = pd.DataFrame(eval_data)

    # Add training loss by matching the step
    metrics_df["train_loss"] = metrics_df["step"].map(train_losses)

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progress: ELECTRA on SQuAD", fontsize=16, fontweight="bold")

    # Plot 1: Training vs Validation Loss over Steps (aligned)
    axes[0].plot(
        metrics_df["step"],
        metrics_df["train_loss"],
        marker="o",
        linestyle="-",
        color="#2E86AB",
        linewidth=2,
        markersize=6,
        label="Training Loss",
    )
    axes[0].plot(
        metrics_df["step"],
        metrics_df["eval_loss"],
        marker="s",
        linestyle="-",
        color="#A23B72",
        linewidth=2,
        markersize=6,
        label="Validation Loss",
    )
    axes[0].set_xlabel("Training Steps", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: F1 Score and Exact Match over Steps (both on same axis)
    axes[1].plot(
        metrics_df["step"],
        metrics_df["eval_f1"],
        marker="o",
        linestyle="-",
        color="#18A558",
        linewidth=2,
        markersize=8,
        label="F1 Score",
    )
    axes[1].plot(
        metrics_df["step"],
        metrics_df["eval_exact_match"],
        marker="s",
        linestyle="-",
        color="#F18F01",
        linewidth=2,
        markersize=8,
        label="Exact Match",
    )
    axes[1].set_xlabel("Training Steps", fontsize=12)
    axes[1].set_ylabel("Score (%)", fontsize=12)
    axes[1].set_title("F1 Score & Exact Match", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11, loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("📈 TRAINING SUMMARY")
    print("=" * 60)

    # Training loss summary (if available)
    if metrics_df["train_loss"].notna().any():
        train_loss_vals = metrics_df["train_loss"].dropna()
        print(f"Initial Training Loss: {train_loss_vals.iloc[0]:.4f}")
        print(f"Final Training Loss: {train_loss_vals.iloc[-1]:.4f}")
        loss_reduction = (
            (train_loss_vals.iloc[0] - train_loss_vals.iloc[-1])
            / train_loss_vals.iloc[0]
            * 100
        )
        print(f"Loss Reduction: {loss_reduction:.2f}%")

    print(f"\nBest F1 Score: {metrics_df['eval_f1'].max():.2f}%")
    print(f"Best Exact Match: {metrics_df['eval_exact_match'].max():.2f}%")
    print(f"Final Validation Loss: {metrics_df['eval_loss'].iloc[-1]:.4f}")

    print("\n" + "=" * 60)

    return fig
