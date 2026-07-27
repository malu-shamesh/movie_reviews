"""Review text cleaning: strip HTML tags.

The original notebook also lowercased text and stripped stopwords before BoW/TF-IDF
vectorization. A pretrained transformer's tokenizer already handles casing and
subword structure, and stopword removal actively discards information a language
model uses (e.g. negation words), so only HTML-tag stripping is kept here.
"""

import re

import pandas as pd

_HTML_TAG_PATTERN = re.compile(r"<.*?>")


def remove_html_tags(text: str) -> str:
    return _HTML_TAG_PATTERN.sub("", text)


def clean_reviews(df: pd.DataFrame, text_column: str = "review") -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].apply(remove_html_tags)
    return df
