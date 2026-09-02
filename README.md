# Disaster Tweet Classifier

A simple, well-documented example project that uses TF‑IDF features with a Linear Support Vector Machine (LinearSVC) to classify whether a tweet describes a real disaster.

This repository contains data preprocessing, model training, and evaluation code used for the "Real or Not? NLP with Disaster Tweets" problem.

## Contents

- README.md — this file
- data/ — place `train.csv` and `test.csv` from Kaggle here
- src/ or notebooks/ — preprocessing, training, and evaluation code
- models/ — saved model and vectorizer artifacts (optional)

## Highlights

- Model: TF‑IDF vectorization + Linear SVM (LinearSVC)
- Task: Binary classification — does a tweet describe a real disaster?
- Reproducible steps for training and inference locally

## Dataset

This project uses the public Kaggle competition "Real or Not? NLP with Disaster Tweets":
https://www.kaggle.com/competitions/nlp-getting-started

My solution / walkthrough on Kaggle: https://www.kaggle.com/code/riadhhossain/disaster-tweet-classification-project

Download the CSV files (`train.csv`, `test.csv`) and place them in a `data/` directory at the repository root before running the scripts.

## Prerequisites

- Python 3.8+
- pip
- (recommended) virtualenv or venv

Install core dependencies:

pip install scikit-learn pandas numpy joblib nltk

If this repository includes a `requirements.txt`, prefer `pip install -r requirements.txt`.

## Quickstart — train and evaluate

1. Prepare data
   - Create a `data/` directory and put `train.csv` and `test.csv` inside.

2. Train the model
   - Example (adjust to the project's scripts):

python train.py --data-dir data/ --output models/linear_svm.joblib

3. Evaluate

python evaluate.py --data-dir data/ --model models/linear_svm.joblib

4. Predict on new text

```python
import joblib
model, vectorizer = joblib.load('models/linear_svm.joblib')  # adapt if saved differently
text = ["Forest fire near La Ronge Sask. Canada"]
X = vectorizer.transform(text)
print(model.predict(X))
```

Adjust file names and paths to match the repository's actual code.

## Evaluation & Results

Please run the included evaluation script (e.g., `evaluate.py`) to reproduce metrics. Example metrics to report:

- Accuracy
- Precision / Recall
- F1 score
- Confusion matrix

Example (replace with measured results):

- Cross-validated F1 score: 0.78
- TF‑IDF max_features: 20000
- LinearSVC C: 1.0

## Tips & next steps

- Improve preprocessing: lowercase, punctuation removal, URL/mention stripping, spelling normalization
- Try word n‑grams (1,2) and tune TF‑IDF (max_df, min_df, max_features)
- Compare LinearSVC to LogisticRegression or small neural networks
- Use cross‑validation and a proper training/validation split

## Contributing

Contributions are welcome. To contribute:

1. Open an issue describing the change or improvement.
2. Create a branch and submit a pull request.

## License

Add a `LICENSE` file to the repository (for example, MIT) to make the project's license explicit.

## Contact

If you have questions or find issues, please open an issue in this repository.
