"""Fine-tune a pretrained transformer for binary movie review sentiment classification."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256


def tokenize_dataset(df: pd.DataFrame, tokenizer, text_column: str) -> Dataset:
    dataset = Dataset.from_pandas(df[[text_column, "label"]].rename(columns={text_column: "text"}))
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


def train_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_dir: Path,
    text_column: str = "review",
    epochs: int = 3,
    batch_size: int = 16,
) -> Trainer:
    """Fine-tune DistilBERT on train_df, tracking eval_df metrics each epoch."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

    train_dataset = tokenize_dataset(train_df, tokenizer, text_column)
    eval_dataset = tokenize_dataset(eval_df, tokenizer, text_column)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    return trainer


def save_model(trainer: Trainer, output_dir: Path) -> None:
    trainer.save_model(str(output_dir))
    trainer.processing_class.save_pretrained(str(output_dir))
