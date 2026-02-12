

# 🧠 NLP Text Processing and Sentiment Analysis

An end-to-end Natural Language Processing project demonstrating **text preprocessing pipelines** and **sentiment analysis models** on real-world datasets — IMDB Movie Reviews and Amazon Software Reviews.

---

## 📌 Project Overview

This project is divided into two main phases:

**Phase 1 — Text Preprocessing:** Building robust text cleaning and normalization pipelines applied to two different datasets, covering everything from basic cleaning to advanced tokenization and lemmatization.

**Phase 2 — Sentiment Analysis:** Training and evaluating multiple machine learning classifiers using TF-IDF and CountVectorizer feature extraction techniques.

---

## 📁 Repository Structure

```
├── text_preprocessing_imdb.ipynb              # Text preprocessing on IMDB Movie Reviews
├── text_preprocessing_amazon_reviews.ipynb     # Text preprocessing on Amazon Software Reviews
├── sentiment_analysis_tfidf.ipynb             # Sentiment analysis using TF-IDF vectorization
├── sentiment_analysis_countvectorizer.ipynb   # Sentiment analysis using CountVectorizer
└── README.md
```

---

## 📊 Datasets Used

| Dataset | Source | Records Used | Task |
|---------|--------|--------------|------|
| IMDB Movie Reviews | [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 500 | Preprocessing + Sentiment Analysis |
| Amazon Software Reviews | [Amazon Review Data](https://nijianmo.github.io/amazon/index.html) | 500–10,000 (chunked) | Preprocessing + Sentiment Analysis |

---

## 🔧 Text Preprocessing Pipeline

Both preprocessing notebooks implement the following steps:

### Cleaning
- **Lowercasing** — Uniform text casing
- **HTML Tag Removal** — Regex-based stripping of HTML elements
- **URL Removal** — Removing hyperlinks from text
- **Punctuation Removal** — Using `string.punctuation` translation
- **Stopword Removal** — Filtering out common English stopwords via NLTK
- **Emoji Handling** — Removal or conversion to text using the `emoji` library

### Tokenization (4 methods compared)
- `str.split()` — Basic Python split
- **Regular Expressions** — Pattern-based tokenization
- **NLTK** — `word_tokenize` and `sent_tokenize`
- **spaCy** — Using `en_core_web_sm` language model

### Normalization
- **Stemming** — Porter, Snowball, Lancaster, and Regex-based stemmers compared
- **Lemmatization** — WordNet Lemmatizer and `pywsd` sentence-level lemmatization

---

## 🤖 Sentiment Analysis Models

### Feature Extraction Methods
- **TF-IDF Vectorizer** (`sentiment_analysis_tfidf.ipynb`)
- **CountVectorizer** (`sentiment_analysis_countvectorizer.ipynb`)

### Machine Learning Classifiers

| Model | Description |
|-------|-------------|
| **Multinomial Naive Bayes** | Probabilistic classifier well-suited for text classification |
| **Random Forest** | Ensemble of 500 decision trees (`n_estimators=500`) |
| **Gradient Boosting** | Boosted ensemble with 1000 estimators (`n_estimators=1000`) |
| **Logistic Regression** | Linear classifier with class weight handling |

### Evaluation
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Train/Test Split (80/20)

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| Language | Python 3.x |
| Data Handling | Pandas, NumPy |
| NLP | NLTK, spaCy, pywsd, emoji, re |
| ML / Feature Extraction | Scikit-learn (TF-IDF, CountVectorizer, LabelEncoder) |
| Models | RandomForestClassifier, MultinomialNB, GradientBoostingClassifier, LogisticRegression |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn nltk spacy pywsd emoji
python -m spacy download en_core_web_sm
```

### NLTK Downloads

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```

### Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Shaiksameenasulthana/NLP-Text-Processing-and-Sentiment-Analysis.git
   cd NLP-Text-Processing-and-Sentiment-Analysis
   ```
2. Open any notebook in Jupyter:
   ```bash
   jupyter notebook
   ```
3. Update the dataset file paths in the notebooks to match your local directory before running.

---

## 📈 Workflow Diagram

```
Raw Text Data
     │
     ▼
┌─────────────────────┐
│  Text Preprocessing  │
│  • Lowercasing       │
│  • HTML/URL Removal  │
│  • Punctuation       │
│  • Stopwords         │
│  • Emoji Handling    │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│   Tokenization       │
│  • Split / Regex     │
│  • NLTK / spaCy      │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│   Normalization      │
│  • Stemming          │
│  • Lemmatization     │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Feature Extraction  │
│  • TF-IDF            │
│  • CountVectorizer   │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│   ML Classification  │
│  • Naive Bayes       │
│  • Random Forest     │
│  • Gradient Boosting │
│  • Logistic Reg.     │
└────────┬────────────┘
         ▼
    Sentiment Output
   (Positive/Negative)
```

---

## 📝 Key Learnings

- Preprocessing significantly impacts model accuracy — clean data leads to better predictions.
- Lemmatization preserves word meaning better than stemming, though it is slower.
- TF-IDF generally outperforms raw CountVectorizer for sentiment tasks by weighing term importance.
- Gradient Boosting achieved the highest accuracy among the classifiers tested.

---

## 🙋‍♀️ Author

**Shaik Sameena Sulthana**

- GitHub: [@Shaiksameenasulthana](https://github.com/Shaiksameenasulthana)

---

## 📄 License

This project is open source and available for educational purposes.

---

⭐ If you found this project helpful, please consider giving it a star!
