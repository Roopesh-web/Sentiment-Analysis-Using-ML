"""
train.py - Model Training Script
=================================
Loads the dataset, preprocesses, extracts features, trains models, and saves the best one.

Usage:
    python train.py
    python train.py --dataset path/to/reviews.csv
    python train.py --method tfidf   (or bow)
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_dataframe
from feature_extraction import FeatureExtractor
from model import train_model, evaluate_model, print_metrics, save_model

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BASE_DIR, 'product_review_sentiment_dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def main():
    parser = argparse.ArgumentParser(description='Train Sentiment Analysis Models')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help='Path to the CSV dataset')
    parser.add_argument('--method', type=str, default='tfidf',
                        choices=['tfidf', 'bow'],
                        help='Feature extraction method')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing')
    args = parser.parse_args()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  PRODUCT REVIEW SENTIMENT ANALYSIS – TRAINING")
    print(f"{'═' * 60}")
    print(f"\n[1/5] Loading dataset: {args.dataset}")

    df = pd.read_csv(args.dataset)
    print(f"  → {len(df)} reviews loaded")
    print(f"  → Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    print(f"\n[2/5] Preprocessing text...")
    df = preprocess_dataframe(df, text_col='review_text', sentiment_col='sentiment')
    print(f"  → {len(df)} reviews after cleaning")

    # ── 3. Feature extraction ────────────────────────────────────────────────
    print(f"\n[3/5] Extracting features ({args.method.upper()})...")
    fe = FeatureExtractor(method=args.method)
    X = fe.fit_transform(df['cleaned_text'])
    y = df['sentiment'].values
    print(f"  → Feature matrix shape: {X.shape}")

    # ── 4. Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )
    print(f"  → Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── 5. Train & evaluate both models ──────────────────────────────────────
    print(f"\n[4/5] Training models...")
    model_names = ['naive_bayes', 'logistic_regression']
    results = {}

    for name in model_names:
        clf = train_model(X_train, y_train, model_name=name)
        metrics = evaluate_model(clf, X_test, y_test, model_name=name)
        results[name] = {'clf': clf, 'metrics': metrics}
        print_metrics(metrics)

    # ── 6. Save best model ───────────────────────────────────────────────────
    print(f"\n[5/5] Saving models...")

    # Determine best model by F1 score
    best_name = max(results, key=lambda n: results[n]['metrics']['f1'])
    best_clf = results[best_name]['clf']
    best_f1 = results[best_name]['metrics']['f1']

    # Save both models
    for name, data in results.items():
        save_model(data['clf'], name)

    # Save vectorizer
    fe.save()

    # Save best model indicator
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'best_model.txt'), 'w') as f:
        f.write(best_name)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═' * 60}")
    print(f"\n  Comparison Table:")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 65}")
    for name, data in results.items():
        m = data['metrics']
        marker = ' ★' if name == best_name else ''
        print(f"  {name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}{marker}")
    print(f"\n  ★ Best model: {best_name} (F1 = {best_f1:.4f})")
    print(f"  Models saved to: {MODELS_DIR}")
    print()


if __name__ == '__main__':
    main()
