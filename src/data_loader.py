"""Download and load the IMDB 50K movie reviews dataset from Kaggle."""

from pathlib import Path

import kagglehub
import pandas as pd

DATASET_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
CSV_FILENAME = "IMDB Dataset.csv"
LABEL_TO_ID = {"negative": 0, "positive": 1}


def download_dataset() -> Path:
    """Download the dataset via kagglehub and return the local directory path."""
    dataset_dir = kagglehub.dataset_download(DATASET_SLUG)
    return Path(dataset_dir)


def load_reviews(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the IMDB reviews CSV, dropping duplicates and mapping sentiment to a numeric label.

    If csv_path is not provided, downloads the dataset via kagglehub first.
    """
    if csv_path is None:
        dataset_dir = download_dataset()
        csv_path = dataset_dir / CSV_FILENAME

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates().dropna(subset=["review", "sentiment"])
    df["label"] = df["sentiment"].map(LABEL_TO_ID)
    return df
