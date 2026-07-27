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

## Why these changes

The original notebook vectorized reviews with bag-of-words or TF-IDF and classified with `GaussianNB` (63% accuracy) or `RandomForestClassifier` (85%). Both throw away word order and context - "not good" and "good" look similar to a bag-of-words model. The notebook's TF-IDF experiment also had a bug: after fitting `TfidfVectorizer` and transforming the data into `x_train_tfidf`/`x_test_tfidf`, the very next `RandomForestClassifier` cell trained on `x_train_bow`/`x_test_bow` instead - so the TF-IDF features were computed but never actually used, and the reported "TF-IDF" accuracy (0.7506) is identical to the earlier bigram-BoW result by coincidence of copy-pasted code, not because TF-IDF was evaluated.

Fine-tuning `distilbert-base-uncased` instead starts from a model already trained on a large text corpus and adapts it to this specific sentiment task, capturing word order and context that bag-of-words approaches can't - substantially higher accuracy is achievable on the same data.

**Verification note**: this network blocks access to huggingface.co, so the fine-tuning and evaluation modules (which need to download the pretrained model) could not be smoke-tested end-to-end here. The data loader and preprocessing modules were verified locally with passing tests. Verify a full run on unrestricted network/CI before relying on it.
