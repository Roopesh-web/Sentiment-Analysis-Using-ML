"""
preprocessing.py - Text Preprocessing Module
=============================================
Handles all text cleaning and NLP preprocessing for the sentiment analysis pipeline.
"""

import re
import string
import nltk
import pandas as pd

# ── Download required NLTK data ──────────────────────────────────────────────
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Initialize NLP tools ─────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
# Keep sentiment-relevant words that NLTK removes by default
SENTIMENT_KEEPERS = {
    'not', 'no', 'nor', 'very', 'only', 'but', 'don', "don't",
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't", 'against', 'above',
    'below', 'few', 'more', 'most', 'too'
}
STOP_WORDS -= SENTIMENT_KEEPERS

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean a single review string:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Remove numbers
      5. Remove punctuation
      6. Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # URLs
    text = re.sub(r'<.*?>', '', text)                  # HTML
    text = re.sub(r'\d+', '', text)                    # numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # punctuation
    text = re.sub(r'\s+', ' ', text).strip()           # whitespace
    return text


def remove_stopwords(text: str) -> str:
    """Remove stopwords while keeping sentiment-relevant words."""
    tokens = word_tokenize(text)
    return ' '.join(t for t in tokens if t not in STOP_WORDS)


def lemmatize_text(text: str) -> str:
    """Lemmatize each token to its base form."""
    tokens = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(t) for t in tokens)


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for a single review string."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = 'review_text',
                         sentiment_col: str = 'sentiment') -> pd.DataFrame:
    """
    Preprocess an entire DataFrame of reviews.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `text_col`.
    text_col : str
        Column name for the review text.
    sentiment_col : str
        Column name for the sentiment label.

    Returns
    -------
    pd.DataFrame with added 'cleaned_text' column and NaN rows dropped.
    """
    df = df.copy()
    # Drop rows with missing text
    df.dropna(subset=[text_col], inplace=True)
    # Apply preprocessing
    df['cleaned_text'] = df[text_col].apply(preprocess_text)
    # Drop empty results
    df = df[df['cleaned_text'].str.strip().astype(bool)].reset_index(drop=True)
    return df


def map_rating_to_sentiment(rating: int) -> str:
    """Map a 1-5 star rating to a sentiment label."""
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'


if __name__ == '__main__':
    # Quick sanity check
    samples = [
        "This product is AMAZING! Best purchase ever!!!",
        "Terrible quality, broke after one day. Waste of money.",
        "It's okay, nothing special. Does the job.",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {preprocess_text(s)}")
        print()
