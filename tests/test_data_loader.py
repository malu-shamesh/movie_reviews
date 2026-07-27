import pandas as pd

from src.data_loader import load_reviews


def test_load_reviews_dedupes_and_maps_labels(tmp_path):
    csv_path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {
            "review": ["Great movie", "Great movie", "Terrible film"],
            "sentiment": ["positive", "positive", "negative"],
        }
    ).to_csv(csv_path, index=False)

    df = load_reviews(csv_path)

    assert len(df) == 2
    assert set(df["label"]) == {0, 1}
