# SmartSpam – AI-Based Spam Classification System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Backend-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%203.10-green.svg)](https://www.nltk.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.03%25-brightgreen.svg)]()
[![F1-Score](https://img.shields.io/badge/Spam%20F1--Score-96.15%25-success.svg)]()

---

## 📖 1. Project Introduction

**SmartSpam** is a full-stack, end-to-end Machine Learning web application designed to accurately identify and filter unsolicited, fraudulent, and malicious electronic communications (SMS and email messages). Developed as an academic Natural Language Processing (NLP) project, SmartSpam combines linguistically informed text normalization, Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering, and a probability-calibrated Support Vector Machine classifier backed by a modern, responsive single-page web dashboard.

---

## 🎯 2. Problem Statement

With the exponential surge in mobile telecommunications and online banking, spam and phishing attacks have grown increasingly sophisticated. Unsolicited text messages frequently deceive recipients into surrendering financial credentials or downloading malware through deceptive urgency, monetary offers, and phone redirection.

Detecting spam in short messages presents specific machine learning challenges:
1. **Severe Class Imbalance:** Real-world communications contain vastly more legitimate messages than spam (~87% Ham vs. ~13% Spam).
2. **Text Sparsity:** Short text messages (SMS) have constrained character limits, limiting lexical context.
3. **High Cost of False Positives:** Misclassifying a critical legitimate message as spam can cause severe user disruption, necessitating models that maximize spam recall while maintaining near-zero false alarms.
4. **Adversarial Obfuscation:** Spammers deliberately inject irregular punctuation, numbers, and misspellings to evade rule-based keyword filters.

---

## 🏆 3. Project Objectives

1. Develop a high-performance NLP preprocessing pipeline that cleans noise while preserving critical semantic cues (e.g., negations, currency symbols, URLs, and phone numbers).
2. Compare multiple probabilistic and linear classification algorithms using leak-free stratified evaluation.
3. Implement probability calibration (Platt Scaling) to provide trustworthy confidence percentages.
4. Build a lightweight, scalable RESTful backend in Python Flask with persistent SQLite audit logging.
5. Create a professional, responsive AI SaaS dashboard with real-time classification, dynamic Chart.js analytics, and full audit controls.

---

## 💻 4. Technology Stack

| Layer | Technologies Used | Description |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5, CSS3, Modern JavaScript (ES6+), Chart.js | Responsive Single-Page Application (SPA) dashboard with animated confidence meters, KPI counters, and dark navy typography. |
| **Backend API** | Python 3.12, Flask 3.1, Flask-CORS, Werkzeug | RESTful API microservice exposing prediction, audit history, and aggregated statistics. |
| **NLP Pipeline** | NLTK (Natural Language Toolkit) | Tokenization, lowercasing, stopword removal, negation preservation, and Porter Stemming. |
| **Machine Learning** | Scikit-learn (LinearSVC, MultinomialNB, Logistic Regression) | TF-IDF feature extraction, maximum-margin classification, and Platt Scaling calibration. |
| **Database** | SQLite3 with parameterized SQL | Persistent local storage for classification history, confidence scores, and dashboard metrics. |
| **Serialization** | Python `pickle` (`.pkl`) & JSON | Model artifact serialization and evaluation metrics export. |

---

## 🏗️ 5. System Architecture

The following diagram illustrates the complete data lifecycle from client message submission to inference and visualization:

```text
[ Client Web Dashboard (HTML5 / CSS3 / Vanilla JS / Chart.js) ]
                             │
                  HTTP POST /api/predict
                             ▼
               [ Flask REST API Controller (backend/app.py) ]
                             │
     ┌───────────────────────┴────────────────────────┐
     ▼                                                ▼
[ Preprocessing Pipeline ]                   [ SQLite Database Layer ]
 - URL / Phone Normalization                  (backend/database.py)
 - Negation-Aware Stopwords                           │
 - NLTK Porter Stemming                               ▼
     │                                    [ smartspam.db: prediction_history ]
     ▼
[ TF-IDF Vectorizer (models/vectorizer.pkl) ]
     │ (Transforms text to 4,000 n-gram features)
     ▼
[ Calibrated LinearSVC (models/linearsvc_model.pkl) ]
     │
     ▼
[ Prediction Result + Calibrated Probability P(Spam | X) ]
     │
     └───────────────────────┬────────────────────────┘
                             ▼
              HTTP Response 201 Created (JSON)
                             ▼
            [ Live UI Animated Display & Dashboard Stats ]
```

---

## 📊 6. Dataset Description

The system is trained and benchmarked on the authoritative **SMS Spam Collection Dataset** from the UCI Machine Learning Repository:

- **Original Raw Records:** 5,574 messages.
- **Duplicate Records Removed:** 403 rows (~7.23% duplicates, e.g., common marketing broadcasts and generic greetings).
- **Zero-Token Symbol Filter:** 6 records (messages consisting exclusively of isolated emoticons or non-alphanumeric punctuation yielding zero TF-IDF tokens).
- **Clean Unique Corpus:** **5,165 messages**.
  - **Legitimate (Ham):** 4,512 messages (**87.36%**)
  - **Spam:** 653 messages (**12.64%**)

---

## 🔬 7. NLP Preprocessing Methodology

Text preprocessing in [`ml_pipeline/preprocess.py`](ml_pipeline/preprocess.py) normalizes unstructured natural language while retaining discriminative spam signals:

1. **Entity Token Replacement:**
   - URLs $\rightarrow$ `urltoken` (e.g., `http://bit.ly/claim` $\rightarrow$ `urltoken`)
   - Emails $\rightarrow$ `emailtoken`
   - Currency Symbols $\rightarrow$ `moneytoken` (e.g., `£`, `$`, `€`)
   - Phone Numbers / $\ge 5$-Digit Codes $\rightarrow$ `phonetoken` (e.g., `09061743806` $\rightarrow$ `phonetoken`)
   - Isolated Numbers $\rightarrow$ `numtoken`
2. **Case Normalization & Tokenization:** Converts characters to lowercase and applies NLTK's `word_tokenize`.
3. **Negation Preservation in Stopwords:** Standard NLP stopword filters remove *"not"*, *"no"*, and *"nor"*, reversing sentence polarity. SmartSpam explicitly protects negation words:
   $$\text{Stopwords} = \text{NLTK English Stopwords} \setminus \{\text{"not"}, \text{"no"}, \text{"nor"}, \text{"never"}, \text{"neither"}, \text{"none"}, \text{"n't"}, \text{"without"}, \dots\}$$
4. **Morphological Stemming:** Applies NLTK's `PorterStemmer` to conflate inflectional variants to their lexical roots (e.g., *winning* $\rightarrow$ *win*, *prizes* $\rightarrow$ *prize*).

---

## 📐 8. TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) converts normalized text tokens into numerical vectors:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Sublinear Term Frequency:** Employs logarithmic term scaling ($\text{TF} = 1 + \log(\text{tf})$) to dampen the dominance of repeated words within a single message.
- **Inverse Document Frequency:** Downweights ubiquitous words while magnifying rare, high-information terms:
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
- **N-Gram Range:** Extracts unigrams and bigrams (`ngram_range=(1, 2)`) to capture multi-word spam triggers (*"cash prize"*, *"claim code"*, *"call now"*).
- **Feature Space:** Constrained to the top 4,000 most predictive features (`max_features=4000`).
- **Data Leakage Prevention:** The vectorizer is fit strictly on the 80% training split (`X_train`) and applied as a transform on `X_test` and live user inputs.

---

## 🤖 9. Machine Learning Algorithms Compared

All candidate models were evaluated on the **exact same 80:20 stratified split** (Training: 4,132 samples; Held-out Test: 1,033 samples) using identical TF-IDF features:

| Algorithm | Test Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Training Time | False Positives | False Negatives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated LinearSVC** *(Selected)* | **99.03%** | 96.90% | **95.42%** | **96.15%** | 43.23 ms | 4 | **6** |
| **Multinomial Naive Bayes ($\alpha=0.2$)** | 98.35% | 97.50% | 89.31% | 93.23% | 1.75 ms | 3 | 14 |
| **Multinomial Naive Bayes ($\alpha=1.0$)** | 98.26% | **100.00%** | 86.26% | 92.62% | **1.24 ms** | **0** | 18 |
| **Logistic Regression ($C=1.0$)** | 97.97% | 97.41% | 86.26% | 91.50% | 16.44 ms | 3 | 18 |

---

## 🎯 10. Final Model Selection & Justification

**Support Vector Classifier (`LinearSVC`) with Platt Scaling Calibration** was selected as the production model for the following reasons:

1. **Superior Spam Recall (95.42% vs 89.31%):** In spam filtering, failing to detect malicious messages is dangerous. LinearSVC caught 125 out of 131 actual spam messages in the test set, missing only 6.
2. **Optimal F1-Score (96.15%):** Outperforms Naive Bayes and Logistic Regression in overall harmonic balance.
3. **Maximum Margin Optimization:** SVM finds the optimal hyperplane that maximizes geometric margin separation between spam and ham clusters in sparse 4,000-dimensional TF-IDF space.
4. **Probability Calibration via Platt Scaling:** Because standard SVM outputs uncalibrated hyperplane distances, we wrapped `LinearSVC` with `CalibratedClassifierCV(method='sigmoid', cv=5)` trained strictly on `X_train`. This produces calibrated, trustworthy confidence probabilities without distorting the decision boundary.

---

## 📈 11. Evaluation Metrics and Confusion Matrix

### Held-Out Test Set Results ($N = 1,033$)
- **Overall Accuracy:** **99.03%**
- **Spam Precision:** **96.90%**
- **Spam Recall:** **95.42%**
- **Spam F1-Score:** **96.15%**
- **Ham Recall:** **99.56%** (898 out of 902 legitimate messages correctly approved)

### Confusion Matrix
```text
                       Predicted Legitimate (Ham)    Predicted Spam
Actual Legitimate:                898                      4           (Total: 902)
Actual Spam:                       6                     125           (Total: 131)
```

---

## 📁 12. Project Folder Structure

```text
SmartSpam/
├── backend/
│   ├── __init__.py                # Backend package initialization
│   ├── app.py                     # Flask web server and RESTful route controller
│   ├── config.py                  # Dynamic configuration and path resolution
│   ├── database.py                # Parameterized SQLite data access layer
│   ├── predictor.py               # Calibrated inference engine
│   ├── requirements.txt           # Python environment dependencies
│   ├── smartspam.db               # SQLite database (prediction history and stats)
│   ├── test_api.py                # Automated 37-point API test suite
│   ├── test_edge_cases.py         # 10-point robustness and edge case test suite
│   ├── test_database_persistence.py# Server restart persistence test suite
│   └── test_live_server.py        # Live HTTP network integration test suite
│
├── frontend/
│   ├── templates/
│   │   └── index.html             # Single-page application dashboard template
│   └── static/
│       ├── css/
│       │   └── style.css          # Modern AI SaaS responsive styling
│       └── js/
│           ├── app.js             # SPA routing, fetch integration, DOM logic
│           └── charts.js          # Chart.js dynamic distribution visualization
│
├── dataset/
│   ├── raw/
│   │   ├── SMSSpamCollection      # Original UCI dataset file
│   │   ├── readme                 # Original UCI dataset documentation
│   │   └── spam.csv               # Raw dataset with standard headers
│   └── processed/
│       └── cleaned_data.csv       # Preprocessed and normalized dataset
│
├── ml_pipeline/
│   ├── __init__.py                # Pipeline package initialization
│   ├── preprocess.py              # NLP cleaning, tokenization, stemming
│   ├── train.py                   # Model training and artifact generation
│   ├── evaluate.py                # Comprehensive metrics evaluation
│   └── audit_and_compare.py       # Multi-algorithm benchmark suite
│
├── models/
│   ├── linearsvc_model.pkl        # Production Calibrated LinearSVC model
│   ├── linearsvc_metrics.json     # LinearSVC test set metrics
│   ├── spam_model.pkl             # Preserved Multinomial Naive Bayes model
│   ├── vectorizer.pkl             # Fitted TF-IDF Vectorizer
│   ├── metrics.json               # Naive Bayes evaluation metrics
│   └── model_comparison.json      # Comparative benchmark results
│
├── .gitignore                     # Git ignore rules (venv, cache, temporary files)
└── README.md                      # Academic documentation and setup manual
```

---

## ⚙️ 13. Installation Instructions

### Prerequisites
- Python 3.10 or higher
- Git
- Modern web browser (Chrome, Edge, Firefox, Safari)

### Step 1: Clone or Navigate to Project
```bash
cd SmartSpam
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
.\venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r backend/requirements.txt
```

---

## 🚀 14. Execution Instructions

### Running the Web Application
```bash
python backend/app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

### Running the Automated Test Suites
```bash
# 1. Complete API regression suite (37 tests)
python backend/test_api.py

# 2. Comprehensive edge case suite (10 tests)
python backend/test_edge_cases.py

# 3. Server restart and database persistence suite
python backend/test_database_persistence.py

# 4. Multi-algorithm ML benchmark audit
python ml_pipeline/audit_and_compare.py
```

---

## 📡 15. API Endpoint Documentation

| Method | Endpoint | Description | Request Payload | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves main UI dashboard | None | `200 OK` (HTML) |
| `GET` | `/api/health` | Service health status | None | `200 OK` (JSON) |
| `POST` | `/api/predict` | Classifies message and stores result | `{"message": "string"}` | `201 Created` (JSON with label, confidence, record ID) |
| `GET` | `/api/history` | Fetches classification audit log | Query: `?limit=50&offset=0` | `200 OK` (JSON list of predictions) |
| `GET` | `/api/stats` | Fetches aggregate dashboard metrics | None | `200 OK` (JSON totals, spam %, model metrics) |
| `DELETE`| `/api/history/<id>` | Deletes specific history record | None | `200 OK` or `404 Not Found` (JSON) |

### Sample Prediction Request
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "WINNER! You have won a £1,000 cash prize. Call 09061743806 to claim now!"}'
```

### Sample Prediction Response
```json
{
  "success": true,
  "id": 14,
  "message": "WINNER! You have won a £1,000 cash prize. Call 09061743806 to claim now!",
  "classification": "Spam",
  "is_spam": true,
  "confidence": 100.0,
  "probabilities": {
    "spam": 1.0,
    "ham": 0.0
  },
  "cleaned_text": "winner moneytoken numtoken numtoken cash prize call phonetoken claim",
  "model_name": "Calibrated LinearSVC",
  "timestamp": "2026-11-13T16:20:00.000000+00:00"
}
```

---

## 🖼️ 16. Dashboard User Interface

The application features 5 primary dashboard views:
1. **Overview Dashboard:** Live KPI stat cards (Total Analyzed, Spam Detected, Legitimate Messages, Spam Rate), Chart.js doughnut distribution, and recent classification feed.
2. **Spam Detector:** Interactive text analysis suite with character counting, 5 one-click presets, progress meter, confidence display, and normalized NLP token breakdown.
3. **Audit History:** Real-time searchable history table with status badges and deletion controls.
4. **Analytics & Benchmarks:** Visual confusion matrix, model metrics, and comparative evaluation table.
5. **About Project:** Theory, formulas, and architecture documentation.

---

## 🔮 17. Future Enhancements

1. **Transformer-Based Embeddings:** Integrate distilled contextual models (e.g., DistilBERT or RoBERTa) to capture deep bidirectional semantic nuances in obfuscated text.
2. **Multilingual Classification:** Extend tokenization to support multilingual spam detection (e.g., Spanish, Hindi, French).
3. **User Feedback Loop:** Allow users to flag false positives and false negatives to generate active learning datasets for periodic retraining.
4. **Email Header & Metadata Parsing:** Expand beyond message body text to analyze SPF, DKIM, and IP reputation signals.

---

## 📜 18. Academic Integrity & License
This project is developed for educational and academic research purposes in Natural Language Processing and Applied Machine Learning. Released under the MIT License.
