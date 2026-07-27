# Movie Review Sentiment Classification

Classifies IMDB movie reviews as positive or negative. Originally a Kaggle notebook using bag-of-words/TF-IDF features with GaussianNB and RandomForest (capping around 85% accuracy); rewritten as a runnable Python pipeline that fine-tunes a pretrained transformer.

## Pipeline

1. **Data loading** ([src/data_loader.py](src/data_loader.py)) - downloads the [IMDB 50K movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) via `kagglehub`, or reads a local CSV, dropping duplicate reviews (the notebook found 17) and mapping sentiment to a numeric label.
2. **Preprocessing** ([src/preprocessing.py](src/preprocessing.py)) - strips HTML tags (`<br />` etc.) left over from the source review text. Lowercasing and stopword removal, which the notebook did for BoW/TF-IDF, are skipped here since a pretrained transformer's tokenizer handles casing itself, and stopword removal discards words (like negations) a language model relies on.
3. **Model** ([src/model.py](src/model.py)) - fine-tunes `distilbert-base-uncased` for binary sequence classification via Hugging Face `transformers`/`datasets`, tracking accuracy and F1 each epoch.
4. **Evaluation** ([src/evaluate.py](src/evaluate.py)) - runs the fine-tuned model on the held-out test set and saves a classification report and confusion matrix.
5. **Visualization** ([src/visualize.py](src/visualize.py)) - saves the sentiment class balance and confusion matrix as PNGs.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Kaggle API credentials are required for automatic dataset download - see [kagglehub's authentication docs](https://github.com/Kaggle/kagglehub#authenticate). Alternatively, download `IMDB Dataset.csv` manually and pass it with `--input`.

## Usage

```bash
# Full pipeline, downloads the dataset automatically
python main.py --output-dir outputs --model-dir models/distilbert-imdb

# Local CSV, capped sample and one epoch for a quick run
python main.py --input data/IMDB_Dataset.csv --sample-size 500 --epochs 1
```

Outputs land in `outputs/` (`classification_report.csv`, `confusion_matrix.csv`, plots), and the fine-tuned model in `--model-dir`.

## Tests

```bash
pytest
```
