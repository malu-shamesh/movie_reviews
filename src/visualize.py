"""Plotting helpers that save figures to an output directory instead of displaying inline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    ax.set_title("Sentiment Class Balance")
    _save(fig, output_dir, "class_balance.png")


def plot_confusion_matrix(matrix_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save(fig, output_dir, "confusion_matrix.png")
