import pandas as pd

from src.preprocessing import clean_reviews, remove_html_tags


def test_remove_html_tags_strips_break_tags():
    assert remove_html_tags("Great movie.<br /><br />Loved it.") == "Great movie.Loved it."


def test_clean_reviews_cleans_the_review_column():
    df = pd.DataFrame({"review": ["<b>Amazing</b> film"]})
    result = clean_reviews(df)

    assert result.loc[0, "review"] == "Amazing film"
