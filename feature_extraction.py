"""
feature_extraction.py - Feature Extraction Module
==================================================
Converts preprocessed text into numerical feature vectors using TF-IDF or Bag-of-Words.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# ── Default paths ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class FeatureExtractor:
    """Wrapper around sklearn text vectorizers with save/load support."""

    def __init__(self, method: str = 'tfidf', max_features: int = 5000):
        """
        Parameters
        ----------
        method : str
            'tfidf' for TF-IDF or 'bow' for Bag-of-Words.
        max_features : int
            Maximum vocabulary size.
        """
        self.method = method.lower()
        self.max_features = max_features

        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),   # unigrams + bigrams
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )
        elif self.method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'tfidf' or 'bow'.")

    def fit_transform(self, texts):
        """Fit the vectorizer on `texts` and return the feature matrix."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """Transform new texts using the already-fitted vectorizer."""
        return self.vectorizer.transform(texts)

    def save(self, filepath: str | None = None):
        """Persist the fitted vectorizer to disk."""
        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, f'vectorizer_{self.method}.joblib')
        joblib.dump(self, filepath)
        print(f"[✓] Vectorizer saved → {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str) -> 'FeatureExtractor':
        """Load a previously saved FeatureExtractor."""
        obj = joblib.load(filepath)
        print(f"[✓] Vectorizer loaded ← {filepath}")
        return obj

    def get_feature_names(self):
        """Return the list of feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out()


if __name__ == '__main__':
    # Quick test
    sample_texts = [
        "product amazing works perfectly",
        "terrible quality broke day",
        "okay nothing special job",
        "love product great value",
        "worst purchase waste money",
    ]
    for m in ['tfidf', 'bow']:
        fe = FeatureExtractor(method=m)
        X = fe.fit_transform(sample_texts)
        print(f"\n{m.upper()} – shape: {X.shape}")
        print(f"  Top features: {fe.get_feature_names()[:10]}")
