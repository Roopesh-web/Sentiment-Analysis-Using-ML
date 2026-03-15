"""
model.py - Machine Learning Model Module
=========================================
Trains, evaluates, saves, and loads sentiment classification models.
"""

import os
import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocessing import preprocess_text
from feature_extraction import FeatureExtractor

# ── Default paths ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


# ═════════════════════════════════════════════════════════════════════════════
# Model factory
# ═════════════════════════════════════════════════════════════════════════════

def get_model(name: str = 'logistic_regression'):
    """
    Return an untrained sklearn classifier.

    Parameters
    ----------
    name : str
        'naive_bayes' or 'logistic_regression'
    """
    name = name.lower().replace(' ', '_')
    if name == 'naive_bayes':
        return MultinomialNB(alpha=1.0)
    elif name == 'logistic_regression':
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Use 'naive_bayes' or 'logistic_regression'.")


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, model_name: str = 'logistic_regression'):
    """Train and return a classifier."""
    clf = get_model(model_name)
    clf.fit(X_train, y_train)
    return clf


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(clf, X_test, y_test, model_name: str = '') -> dict:
    """
    Evaluate a trained classifier and return a metrics dictionary.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, classification_report,
                    confusion_matrix, predictions
    """
    y_pred = clf.predict(X_test)
    labels = sorted(list(set(y_test) | set(y_pred)))

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=labels),
        'labels': labels,
        'predictions': y_pred,
    }
    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print(f"\n{'═' * 60}")
    print(f"  Model: {metrics['model_name']}")
    print(f"{'═' * 60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"{'─' * 60}")
    print("  Classification Report:")
    print(metrics['classification_report'])
    print(f"  Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"{'═' * 60}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Save / Load
# ═════════════════════════════════════════════════════════════════════════════

def save_model(clf, model_name: str, filepath: str | None = None) -> str:
    """Persist trained model to disk."""
    if filepath is None:
        os.makedirs(MODELS_DIR, exist_ok=True)
        safe_name = model_name.lower().replace(' ', '_')
        filepath = os.path.join(MODELS_DIR, f'{safe_name}.joblib')
    joblib.dump(clf, filepath)
    print(f"[✓] Model saved → {filepath}")
    return filepath


def load_model(filepath: str):
    """Load a trained model from disk."""
    clf = joblib.load(filepath)
    print(f"[✓] Model loaded ← {filepath}")
    return clf


# ═════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ═════════════════════════════════════════════════════════════════════════════

def predict_single(review_text: str, clf, feature_extractor: FeatureExtractor) -> dict:
    """
    Predict sentiment for a single review.

    Returns
    -------
    dict with 'sentiment', 'confidence', and 'cleaned_text'.
    """
    cleaned = preprocess_text(review_text)
    X = feature_extractor.transform([cleaned])
    prediction = clf.predict(X)[0]

    # Confidence score
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))
        proba_dict = dict(zip(clf.classes_, [float(p) for p in proba]))
    else:
        confidence = None
        proba_dict = {}

    return {
        'sentiment': prediction,
        'confidence': confidence,
        'probabilities': proba_dict,
        'cleaned_text': cleaned,
    }


def predict_batch(reviews: list[str], clf, feature_extractor: FeatureExtractor) -> list[dict]:
    """Predict sentiment for a list of reviews."""
    return [predict_single(r, clf, feature_extractor) for r in reviews]
