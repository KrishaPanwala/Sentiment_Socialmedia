# 💬 Social Media Sentiment Analysis

An end-to-end NLP project that classifies social media posts into **Positive, Negative, and Neutral** sentiments using advanced text preprocessing, multi-source feature engineering, and XGBoost classification with SMOTE for class imbalance handling.

---

## 📊 Dataset

**Source:** Social media sentiment dataset with tweets, hashtags, engagement metrics, and country/platform metadata.

Place `sentimentdataset.csv` in the root directory. Key columns used:

| Column | Description |
|---|---|
| `Text` | Tweet/post content |
| `Hashtags` | Associated hashtags |
| `Sentiment` | Raw sentiment label |
| `Likes` / `Retweets` | Engagement metrics |
| `Country` | Country of origin |
| `Platform` | Social media platform |
| `Timestamp` | Post datetime |

---

## 🧠 Project Pipeline

```
Raw Social Media Data
    ↓
[ Text Cleaning ]
  - Remove URLs, mentions, special characters
  - Tokenization, lowercasing
  - Stopword removal (keeping negations: not, very, only...)
  - Lemmatization (WordNet)
    ↓
[ EDA & Visualization ]
  - Retweet/Likes distribution
  - Country-wise tweet distribution
  - Monthly tweet trends
  - WordCloud, Top 10 Hashtags
    ↓
[ Feature Engineering ]
  - TF-IDF (5000 features) on cleaned text
  - CountVectorizer n-gram (1-3) on hashtags
  - Temporal features (Year, Month, Day, Hour)
  - Engagement score (Likes + Retweets)
  - One-hot encoding for Country
    ↓
[ Class Imbalance → SMOTE ]
    ↓
[ XGBoost Classifier + RandomizedSearchCV ]
    ↓
[ Evaluation: Accuracy, Classification Report, Confusion Matrix ]
```

---

## ✨ What's Covered

- 🧹 **NLP Preprocessing** — URL removal, tokenization, smart stopword filtering (preserves negations), lemmatization
- ☁️ **WordCloud & Hashtag Analysis** — visual exploration of most common terms
- ⚖️ **Class Imbalance Handling** — SMOTE oversampling on training set
- 🔢 **Multi-Source Features** — TF-IDF text + hashtag n-grams + engagement + temporal + geographic
- 🎯 **Sentiment Mapping** — granular emotions mapped to Positive / Negative / Neutral
- 🤖 **XGBoost Multiclass Classification** — `multi:softmax` objective
- 🎛️ **Hyperparameter Tuning** — RandomizedSearchCV with cross-validation
- 📊 **Full Evaluation** — accuracy, precision, recall, F1-score, confusion matrix

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `NLTK` | Tokenization, stopwords, lemmatization |
| `Scikit-learn` | TF-IDF, CountVectorizer, train/test split, metrics |
| `imbalanced-learn` | SMOTE for class imbalance |
| `XGBoost` | Gradient boosting classifier |
| `WordCloud` | Text visualization |
| `Pandas` + `NumPy` | Data manipulation |
| `Matplotlib` + `Seaborn` | Visualization |

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── Sentiment.ipynb           # Main notebook
├── sentimentdataset.csv      # Dataset (not tracked in git)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place `sentimentdataset.csv` in the root directory.

### 4. Run the notebook
```bash
jupyter notebook Sentiment.ipynb
```

---

## 📦 requirements.txt

```
pandas
numpy
matplotlib
seaborn
nltk
wordcloud
scikit-learn
imbalanced-learn
xgboost
jupyter
```

---

## 📌 Key Concepts Demonstrated

- **NLP Text Preprocessing** — full pipeline from raw text to model-ready features
- **Smart Stopword Handling** — preserving sentiment-critical negation words
- **TF-IDF + N-gram Features** — converting text to numerical representations
- **Class Imbalance (SMOTE)** — synthetic oversampling on training data only
- **Multi-class Classification** — 3-class sentiment with XGBoost
- **Hyperparameter Optimization** — RandomizedSearchCV with stratified CV
- **Feature Engineering** — combining text, hashtag, engagement, and temporal signals

---
