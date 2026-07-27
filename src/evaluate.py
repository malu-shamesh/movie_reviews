"""Evaluate a trained model on held-out data and report metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Trainer

from src.model import tokenize_dataset

LABEL_NAMES = ["negative", "positive"]


def predict(trainer: Trainer, df: pd.DataFrame, text_column: str = "review") -> np.ndarray:
    """Return predicted label ids for each row in df."""
    dataset = tokenize_dataset(df, trainer.processing_class, text_column)
    logits = trainer.predict(dataset).predictions
    return np.argmax(logits, axis=-1)


def evaluate_and_report(
    trainer: Trainer, df: pd.DataFrame, output_dir: Path, text_column: str = "review"
) -> dict:
    """Run predictions on df, save a classification report and confusion matrix, return metrics."""
    predictions = predict(trainer, df, text_column)
    true_labels = df["label"].to_numpy()

    report = classification_report(
        true_labels, predictions, target_names=LABEL_NAMES, output_dict=True
    )
    matrix = confusion_matrix(true_labels, predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    return report
