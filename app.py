"""
app.py - Streamlit Web Interface
=================================
Product Review Sentiment Analysis Dashboard with three pages:
  1. Analyze Review  – single review prediction
  2. Batch Analysis  – CSV upload with batch predictions
  3. Visualizations  – charts and word clouds
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO

# ── Project imports ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_text, preprocess_dataframe
from feature_extraction import FeatureExtractor
from model import (
    train_model, evaluate_model, save_model,
    load_model, predict_single, predict_batch,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASET_PATH = os.path.join(BASE_DIR, 'product_review_sentiment_dataset.csv')

# ═════════════════════════════════════════════════════════════════════════════
# Page config & custom CSS
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    .sentiment-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .sentiment-card:hover {
        transform: translateY(-2px);
    }

    .positive-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #00c853;
    }
    .negative-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #ff1744;
    }
    .neutral-card {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        border-left: 5px solid #ff9100;
    }

    .sentiment-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
    .sentiment-label { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
    .confidence-score { font-size: 1.1rem; color: #4a4a6a; margin-top: 0.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-label { font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }

    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    .stSidebar .stRadio label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Session state & model loading
# ═════════════════════════════════════════════════════════════════════════════

def get_available_models():
    """Check which trained models exist on disk."""
    models = {}
    for name in ['naive_bayes', 'logistic_regression']:
        path = os.path.join(MODELS_DIR, f'{name}.joblib')
        if os.path.exists(path):
            models[name] = path
    return models


def get_available_vectorizers():
    """Check which vectorizers exist on disk."""
    vecs = {}
    for method in ['tfidf', 'bow']:
        path = os.path.join(MODELS_DIR, f'vectorizer_{method}.joblib')
        if os.path.exists(path):
            vecs[method] = path
    return vecs


def auto_train():
    """Train models automatically if none exist."""
    with st.spinner("🔧 No trained models found. Training on the default dataset..."):
        df = pd.read_csv(DATASET_PATH)
        df = preprocess_dataframe(df, text_col='review_text', sentiment_col='sentiment')

        for method in ['tfidf']:
            fe = FeatureExtractor(method=method)
            X = fe.fit_transform(df['cleaned_text'])
            y = df['sentiment'].values

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            for model_name in ['naive_bayes', 'logistic_regression']:
                clf = train_model(X_train, y_train, model_name=model_name)
                save_model(clf, model_name)

            fe.save()

    st.success("✅ Models trained and saved successfully!")
    st.rerun()


@st.cache_resource
def load_cached_model(model_path):
    """Cache loaded models to avoid reloading on every interaction."""
    import joblib
    return joblib.load(model_path)


@st.cache_resource
def load_cached_vectorizer(vec_path):
    """Cache loaded vectorizer."""
    import joblib
    return joblib.load(vec_path)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "Choose a page:",
        ["🔍 Analyze Review", "📊 Batch Analysis", "📈 Visualizations"],
        index=0,
    )
    st.markdown("---")

    # Model selection
    available_models = get_available_models()
    available_vecs = get_available_vectorizers()

    if not available_models or not available_vecs:
        st.warning("⚠️ No trained models found.")
        if st.button("🚀 Train Models Now"):
            auto_train()
        st.stop()

    st.markdown("## ⚙️ Settings")

    model_display = {
        'naive_bayes': '🤖 Naive Bayes',
        'logistic_regression': '📐 Logistic Regression',
    }
    selected_model_name = st.selectbox(
        "Model:",
        list(available_models.keys()),
        format_func=lambda x: model_display.get(x, x),
    )

    vec_display = {'tfidf': '📊 TF-IDF', 'bow': '📦 Bag of Words'}
    selected_vec_method = st.selectbox(
        "Feature Extraction:",
        list(available_vecs.keys()),
        format_func=lambda x: vec_display.get(x, x),
    )

    # Load selected model and vectorizer
    clf = load_cached_model(available_models[selected_model_name])
    fe = load_cached_vectorizer(available_vecs[selected_vec_method])

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:0.8rem;'>"
        "Built with ❤️ using Streamlit & scikit-learn"
        "</p>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Header
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="main-header">
        <h1>🎯 Product Review Sentiment Analysis</h1>
        <p>AI-powered insights from customer reviews using NLP & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: Analyze Single Review
# ═════════════════════════════════════════════════════════════════════════════

if page == "🔍 Analyze Review":
    st.markdown("### 🔍 Analyze a Product Review")
    st.markdown("Enter a product review below and get an instant sentiment prediction.")

    review_input = st.text_area(
        "Type or paste your review here:",
        height=150,
        placeholder="e.g. 'This product is amazing! Battery life is incredible and the build quality is superb.'",
    )

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if analyze_btn and review_input.strip():
        with st.spinner("Analyzing sentiment..."):
            result = predict_single(review_input, clf, fe)

        sentiment = result['sentiment']
        confidence = result['confidence']
        probas = result['probabilities']

        # Sentiment display
        emoji_map = {'Positive': '😊', 'Negative': '😞', 'Neutral': '😐'}
        card_class = f"{sentiment.lower()}-card"
        emoji = emoji_map.get(sentiment, '🤔')

        st.markdown(
            f"""
            <div class="sentiment-card {card_class}">
                <div class="sentiment-emoji">{emoji}</div>
                <div class="sentiment-label">{sentiment}</div>
                <div class="confidence-score">Confidence: {confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability breakdown
        if probas:
            st.markdown("#### 📊 Probability Breakdown")
            prob_cols = st.columns(len(probas))
            colors = {'Positive': '#00c853', 'Negative': '#ff1744', 'Neutral': '#ff9100'}
            for i, (label, prob) in enumerate(sorted(probas.items())):
                with prob_cols[i]:
                    st.metric(label=f"{emoji_map.get(label, '')} {label}", value=f"{prob:.1%}")
                    st.progress(prob)

        # Show cleaned text
        with st.expander("🔧 Preprocessed Text"):
            st.code(result['cleaned_text'])

    elif analyze_btn:
        st.warning("⚠️ Please enter a review to analyze.")

    # Quick examples
    st.markdown("---")
    st.markdown("#### 💡 Try These Examples")
    examples = [
        ("😊", "Battery life of this phone is amazing. Best purchase I've ever made!"),
        ("😞", "Terrible quality, the product broke after just one day of use."),
        ("😐", "The product is okay for the price. Nothing special but does the job."),
    ]
    for emoji, example in examples:
        if st.button(f"{emoji} {example[:60]}...", key=f"ex_{hash(example)}"):
            st.session_state['example_review'] = example
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: Batch Analysis
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊 Batch Analysis":
    st.markdown("### 📊 Batch Sentiment Analysis")
    st.markdown("Upload a CSV file with a `review_text` column to analyze multiple reviews at once.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    # Option to use default dataset
    use_default = st.checkbox("📂 Use built-in sample dataset instead")

    if use_default:
        df = pd.read_csv(DATASET_PATH)
        st.info(f"Loaded {len(df)} reviews from the built-in dataset.")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.info(f"Loaded {len(df)} reviews from uploaded file.")
    else:
        df = None

    if df is not None:
        # Detect text column
        text_col = None
        for col_name in ['review_text', 'text', 'review', 'Review', 'Text', 'comment']:
            if col_name in df.columns:
                text_col = col_name
                break

        if text_col is None:
            st.error("❌ Could not find a text column. Please ensure your CSV has a column named `review_text`, `text`, or `review`.")
            st.stop()

        if st.button("🚀 Analyze All Reviews", type="primary"):
            with st.spinner(f"Analyzing {len(df)} reviews..."):
                results = predict_batch(df[text_col].astype(str).tolist(), clf, fe)

                df['predicted_sentiment'] = [r['sentiment'] for r in results]
                df['confidence'] = [r['confidence'] for r in results]

            # Summary metrics
            st.markdown("#### 📈 Summary")
            summary = df['predicted_sentiment'].value_counts()
            total = len(df)

            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Reviews", total)
            with metric_cols[1]:
                pos = summary.get('Positive', 0)
                st.metric("😊 Positive", f"{pos} ({pos/total:.0%})")
            with metric_cols[2]:
                neg = summary.get('Negative', 0)
                st.metric("😞 Negative", f"{neg} ({neg/total:.0%})")
            with metric_cols[3]:
                neu = summary.get('Neutral', 0)
                st.metric("😐 Neutral", f"{neu} ({neu/total:.0%})")

            # Results table
            st.markdown("#### 📋 Detailed Results")
            st.dataframe(
                df[[text_col, 'predicted_sentiment', 'confidence']].style.format(
                    {'confidence': '{:.1%}'}
                ),
                use_container_width=True,
                height=400,
            )

            # Download button
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )

            # Quick visualization
            st.markdown("#### 📊 Sentiment Distribution")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            colors_list = ['#00c853', '#ff1744', '#ff9100']
            # Pie chart
            summary.plot.pie(
                ax=axes[0],
                autopct='%1.1f%%',
                colors=colors_list[:len(summary)],
                startangle=90,
                textprops={'fontsize': 12},
            )
            axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('')

            # Bar chart
            summary.plot.bar(
                ax=axes[1],
                color=colors_list[:len(summary)],
                edgecolor='white',
            )
            axes[1].set_title('Review Count by Sentiment', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=0)

            for i, (label, val) in enumerate(summary.items()):
                axes[1].text(i, val + 0.5, str(val), ha='center', fontweight='bold', fontsize=12)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: Visualizations
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈 Visualizations":
    st.markdown("### 📈 Sentiment Visualizations")
    st.markdown("Visual insights from the product review dataset.")

    # Load and preprocess dataset
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df_processed = preprocess_dataframe(df, text_col='review_text', sentiment_col='sentiment')

        # Add predictions
        results = predict_batch(df['review_text'].astype(str).tolist(), clf, fe)
        df['predicted_sentiment'] = [r['sentiment'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]

        # ── Sentiment Distribution ────────────────────────────────────────
        st.markdown("#### 📊 Sentiment Distribution")
        summary = df['predicted_sentiment'].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#0e1117')

        colors_list = ['#00c853', '#ff1744', '#ff9100']

        # Pie chart
        wedges, texts, autotexts = axes[0].pie(
            summary.values,
            labels=summary.index,
            autopct='%1.1f%%',
            colors=colors_list[:len(summary)],
            startangle=90,
            textprops={'color': 'white', 'fontsize': 12},
            pctdistance=0.85,
            wedgeprops={'edgecolor': '#0e1117', 'linewidth': 2},
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='#0e1117')
        axes[0].add_artist(centre_circle)
        axes[0].set_title('Sentiment Distribution', fontsize=16, fontweight='bold', color='white')
        axes[0].set_facecolor('#0e1117')

        # Bar chart
        bars = axes[1].bar(
            summary.index, summary.values,
            color=colors_list[:len(summary)],
            edgecolor='white', linewidth=0.5,
            width=0.6,
        )
        axes[1].set_title('Review Count by Sentiment', fontsize=16, fontweight='bold', color='white')
        axes[1].set_facecolor('#0e1117')
        axes[1].tick_params(colors='white')
        axes[1].spines['bottom'].set_color('#333')
        axes[1].spines['left'].set_color('#333')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].set_ylabel('Count', color='white', fontsize=12)

        for bar, val in zip(bars, summary.values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha='center', color='white', fontweight='bold', fontsize=13,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Confidence Distribution ───────────────────────────────────────
        st.markdown("#### 📉 Confidence Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')

        for sentiment, color in zip(['Positive', 'Negative', 'Neutral'], colors_list):
            subset = df[df['predicted_sentiment'] == sentiment]['confidence']
            if len(subset) > 0:
                ax2.hist(subset, bins=20, alpha=0.7, label=sentiment, color=color, edgecolor='white', linewidth=0.5)

        ax2.set_xlabel('Confidence Score', color='white', fontsize=12)
        ax2.set_ylabel('Frequency', color='white', fontsize=12)
        ax2.set_title('Confidence Distribution by Sentiment', fontsize=16, fontweight='bold', color='white')
        ax2.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── Word Clouds ──────────────────────────────────────────────────
        st.markdown("#### ☁️ Word Clouds by Sentiment")
        wc_cols = st.columns(3)
        sentiments = ['Positive', 'Negative', 'Neutral']
        wc_colors = {
            'Positive': 'Greens',
            'Negative': 'Reds',
            'Neutral': 'Blues',
        }

        for i, sentiment in enumerate(sentiments):
            subset = df_processed[df_processed['sentiment'] == sentiment]['cleaned_text']
            text = ' '.join(subset.values)

            if text.strip():
                wc = WordCloud(
                    width=400, height=300,
                    background_color='#0e1117',
                    colormap=wc_colors[sentiment],
                    max_words=100,
                    contour_width=1,
                    contour_color='#333',
                ).generate(text)

                with wc_cols[i]:
                    fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                    fig_wc.patch.set_facecolor('#0e1117')
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis('off')
                    ax_wc.set_title(
                        f'{sentiments[i]}',
                        fontsize=14, fontweight='bold', color='white', pad=10,
                    )
                    plt.tight_layout()
                    st.pyplot(fig_wc)
                    plt.close()

        # ── Dataset Info ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📋 Dataset Overview")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Total Reviews", len(df))
        with info_cols[1]:
            st.metric("Unique Sentiments", df['sentiment'].nunique())
        with info_cols[2]:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

        st.dataframe(df.head(20), use_container_width=True)

    else:
        st.error(f"❌ Dataset not found at: {DATASET_PATH}")
        st.info("Please ensure the dataset file exists or train the model first.")
