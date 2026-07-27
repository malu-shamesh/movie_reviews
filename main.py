"""CLI entrypoint for the movie review sentiment classification pipeline.

Usage:
    python main.py --output-dir outputs --model-dir models/distilbert-imdb
    python main.py --sample-size 500 --epochs 1  # quick smoke test
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_reviews
from src.evaluate import evaluate_and_report
from src.model import save_model, train_model
from src.preprocessing import clean_reviews
from src.visualize import plot_class_balance, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Movie review sentiment classification pipeline")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to a local 'IMDB Dataset.csv'. If omitted, downloads via kagglehub.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/distilbert-imdb"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Optional row limit, useful for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reviews...")
    df = load_reviews(args.input)
    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=0)

    print("Generating exploratory plots...")
    plot_class_balance(df, args.output_dir)

    print(f"Cleaning {len(df)} reviews...")
    df = clean_reviews(df)

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    print(f"Fine-tuning for up to {args.epochs} epoch(s)...")
    trainer = train_model(train_df, test_df, output_dir=args.model_dir, epochs=args.epochs)
    save_model(trainer, args.model_dir)
    print(f"Saved model to {args.model_dir}")

    print("Evaluating on held-out test set...")
    report = evaluate_and_report(trainer, test_df, args.output_dir)
    print(f"Test accuracy: {report['accuracy']:.3f}")

    matrix_df = pd.read_csv(args.output_dir / "confusion_matrix.csv", index_col=0)
    plot_confusion_matrix(matrix_df, args.output_dir)
    print(f"Saved evaluation report and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
