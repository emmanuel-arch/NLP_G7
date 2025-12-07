# NLP Sentiment Classification Project

## Apple & Google Product Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Classification-orange.svg)

---

## 📋 Project Overview

This project develops an **NLP sentiment classification system** that automatically analyzes Twitter (X) data to identify sentiment toward Apple and Google products. Using advanced natural language processing techniques and machine learning models, we classify tweets into sentiment categories to help stakeholders understand public opinion and make data-driven decisions.

**Collaborators:**
Emmanuel Birgen

**Date:** December 2025

---

## 🎯 Business Understanding

### Problem Statement

Technology companies like Apple and Google need to monitor real-time public sentiment expressed on social media platforms. With millions of tweets generated daily, manual analysis is impractical and inefficient.

**Key Business Questions:**
- How are users reacting to new product launches or feature updates?
- Are negative emotions suddenly increasing around a particular product?
- Which products attract the most positive engagement?
- How does Apple's sentiment compare to Google's?

### Stakeholders

- **Primary:** Product and Marketing Teams at Apple and Google
- **Secondary:** 
  - Customer Experience Analysts
  - Social Media Managers
  - Competitor Intelligence Teams
  - Consumer Behavior Researchers

### Solution Value

Our automated sentiment classification model provides:
- ✅ Scalable, real-time sentiment monitoring
- ✅ Faster insights for strategic decision-making
- ✅ Early detection of sentiment shifts
- ✅ Data-driven product improvement prioritization
- ✅ Competitive benchmarking between brands

---

## 📊 Dataset Description

### Overview

- **Total Tweets:** 9,093 (after cleaning: 9,071)
- **Data Source:** [Data.world Twitter Dataset](https://query.data.world/s/3r3b3chhfpyo7545c4regquyxcmc34)
- **Features:**
  - `tweet_text` – Full content of the tweet
  - `emotion_in_tweet_is_directed_at` – Target brand/product (Apple, Google, etc.)
  - `is_there_an_emotion_directed_at_a_brand_or_product` – Sentiment label

### Data Characteristics

- **Unique Tweets:** 9,065 (high diversity)
- **Brands Covered:** Apple (iPhone, iPad, Mac), Google (Android, Nexus, Pixel)
- **Sentiment Distribution:**
  - No emotion toward brand/product
  - Positive emotion
  - Negative emotion
  - Neutral (can't tell)
- **Missing Data:** 63.8% in brand-target column (many generic tweets)
- **Duplicates:** 22 records (<0.25%)

---

## 🔧 Methodology

### 1. Data Preparation

#### Data Cleaning
- Removed duplicate tweets (22 records)
- Handled missing values by filtering tweets with clear brand targets
- Normalized column names for clarity

#### Text Preprocessing
- **Lowercase conversion** for consistency
- **URL removal** (http, www patterns)
- **Mention removal** (@username patterns)
- **Hashtag symbol removal** (kept the text)
- **Special character removal** (punctuation, numbers)
- **Stopword removal** (common English words)
- **Lemmatization** (word normalization)

#### Brand Classification
- Grouped products into two main categories:
  - **Apple:** iPhone, iPad, Mac, iOS
  - **Google:** Android, Nexus, Pixel

### 2. Exploratory Data Analysis (EDA)

#### Key Insights Discovered

1. **Sentiment Distribution**
   - Analyzed overall sentiment patterns
   - Compared sentiment across Apple vs. Google
   
2. **Text Length Analysis**
   - Average tweet length by sentiment
   - Distribution patterns of word counts

3. **Word Frequency Analysis**
   - Top positive sentiment words
   - Top negative sentiment words
   - Bigram analysis for context

4. **Product-Specific Insights**
   - Most positive product: Identified through sentiment counts
   - Most negative product: Highlighted areas of concern
   - Keyword analysis (battery, crash, lag, slow, overheat, bug)

5. **Visual Insights**
   - Word clouds for positive and negative sentiments
   - Sentiment distribution by company
   - Product comparison charts

### 3. Feature Engineering

Our comprehensive feature engineering approach includes:

#### A. TF-IDF Vectorization
- **Max Features:** 5,000
- **N-gram Range:** Unigrams and bigrams (1, 2)
- **Min/Max Document Frequency:** 2 and 0.95
- Captures word importance across the corpus

#### B. Numeric Features (19 features total)

**Basic Text Statistics:**
- Character count
- Word count
- Average word length

**Sentiment Indicators:**
- Exclamation mark count
- Question mark count
- Positive word count (good, great, excellent, etc.)
- Negative word count (bad, terrible, horrible, etc.)

**Social Media Signals:**
- Uppercase character count (intensity)
- Hashtag count
- Mention count (@username)

**Company Encoding:**
- Binary flag for Apple
- Binary flag for Google

**Advanced Linguistic Features:**
- Punctuation density
- Capital letter ratio
- Unique word ratio (vocabulary diversity)
- Repeated character patterns (e.g., "soooo")
- URL presence indicator
- Ellipsis count
- Average sentence length

#### C. Feature Scaling & Combination
- StandardScaler normalization for numeric features
- Combined TF-IDF (5,000 dims) + Numeric (19 dims) = **5,019 total features**

#### D. Target Encoding
- Label encoding for sentiment categories
- Mapping: Each sentiment label → numeric code

### 4. Train-Test Split
- **Split Ratio:** 80% training, 20% testing
- **Stratified Sampling:** Maintains class distribution
- **Random State:** 42 (for reproducibility)

---

## 🤖 Machine Learning Models

### Baseline Models

1. **Logistic Regression**
   - Linear classification approach
   - Fast training and inference
   - Good interpretability

2. **Multinomial Naive Bayes**
   - Probabilistic classifier
   - Works well with text data
   - Handles sparse features efficiently

3. **Random Forest**
   - Ensemble learning method
   - Handles non-linear relationships
   - Feature importance analysis

4. **Support Vector Machine (SVM)**
   - Powerful for high-dimensional data
   - Finds optimal decision boundaries
   - Kernel-based classification

### Advanced Model

5. **LSTM (Long Short-Term Memory)**
   - Deep learning approach
   - Captures sequential patterns
   - Handles long-range dependencies
   - Uses word embeddings

---

## 📈 Model Evaluation Metrics

Models are evaluated using:
- **Accuracy:** Overall correctness
- **Precision:** Positive prediction accuracy
- **Recall:** True positive detection rate
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed classification breakdown

---

## 📁 Project Structure

```
NLP_G7/
│
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
│
├── data/                         # Raw and cleaned datasets
│
├── images/                       # Visualizations (plots, wordclouds, charts)
│
├── Documents/                    # Additional project documents
│
├── notebook/
│   └── NLP_Analysis.ipynb        # Main analysis notebook
│
└── Power Point Group 7.pdf       # Final project presentation
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
tensorflow
wordcloud
```

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. Open and run the notebook:
   ```bash
   jupyter notebook notebook/NLP_Analysis.ipynb
   ```

---

## 📊 Key Results

### Feature Engineering Achievements

- **5,019 total engineered features** combining:
  - 5,000 TF-IDF features (text patterns)
  - 19 handcrafted numeric features (linguistic signals)
  
- **Comprehensive preprocessing pipeline**:
  - Text cleaning and normalization
  - Stopword removal and lemmatization
  - Brand classification and encoding
  
- **Rich feature set** capturing:
  - Semantic meaning (TF-IDF)
  - Emotional intensity (punctuation, capitals)
  - Social media patterns (hashtags, mentions)
  - Linguistic diversity (unique word ratios)

### Model Performance

*(Model results will be added after training and evaluation)*

---

## 🔮 Future Work

1. **Model Enhancement**
   - Hyperparameter tuning with GridSearchCV
   - Ensemble methods combining multiple models
   - Transformer-based models (BERT, RoBERTa)

2. **Feature Engineering**
   - Sentiment lexicons (VADER, TextBlob)
   - Part-of-speech tagging
   - Named entity recognition
   - Emoji sentiment analysis

3. **Production Deployment**
   - Real-time sentiment monitoring API
   - Dashboard for visualization
   - Automated alerting system

4. **Social Impact Application**
   - Adapt model for mental health monitoring
   - Depression detection from social media text
   - Early intervention systems

---

## 📝 Conclusions

This project successfully demonstrates the application of NLP and machine learning techniques to solve a real-world business problem. By automating sentiment analysis of social media data, we enable technology companies to:

- Monitor brand perception at scale
- Respond quickly to customer sentiment shifts
- Make data-driven product and marketing decisions
- Benchmark competitive positioning

The comprehensive feature engineering approach, combining TF-IDF with handcrafted linguistic features, provides models with rich information for accurate sentiment classification.

---

## 👥 Contributors

**Group 7 - Phase 4 Project**

---

## 📄 License

This project is for educational purposes as part of a data science curriculum.

---

## 🙏 Acknowledgments

- Data source: Data.world Twitter Dataset
- NLTK library for text processing tools
- Scikit-learn for machine learning implementations
- TensorFlow/Keras for deep learning capabilities

---

*Last Updated: December 2025*

