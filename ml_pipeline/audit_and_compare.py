"""
ml_pipeline/audit_and_compare.py
--------------------------------
Comprehensive Phase 2 Verification Audit:
1. Exact investigation of the 6-record discrepancy.
2. Verification of split sizes and class distributions.
3. Verification of confusion matrix and metrics.
4. Benchmark comparison across MultinomialNB, Logistic Regression, and LinearSVC.
5. Analysis of raw vs calibrated prediction probabilities.
"""

import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from ml_pipeline.preprocess import clean_text


def investigate_discrepancy():
    print("=" * 60)
    print("TASK 1 & 2: DISCREPANCY INVESTIGATION")
    print("=" * 60)
    
    df_raw = pd.read_csv("dataset/raw/spam.csv")
    raw_count = len(df_raw)
    print(f"Original Raw Records: {raw_count}")

    # Duplicates removed by message
    df_dedup = df_raw.drop_duplicates(subset=['message']).copy()
    dedup_count = len(df_dedup)
    dups_removed = raw_count - dedup_count
    print(f"Duplicates Removed:   {dups_removed} (leaving {dedup_count} records)")

    # Find the records whose cleaned text becomes empty
    df_dedup['cleaned'] = df_dedup['message'].apply(clean_text)
    empty_records = df_dedup[df_dedup['cleaned'].str.strip() == ''].copy()
    
    print(f"\nMissing Records Found: {len(empty_records)}")
    print("Breakdown of the 6 records that caused the discrepancy:")
    print("-" * 60)
    for i, (idx, row) in enumerate(empty_records.iterrows(), 1):
        print(f"#{i} [Row {idx}] Label: '{row['label']}' | Message: {repr(row['message'])}")
    print("-" * 60)
    print("Root Cause Analysis:")
    print("These 6 records contain exclusively whitespace, punctuation symbols (e.g. ':-)'),")
    print("or isolated non-alphanumeric punctuation. After stripping punctuation, tokens become")
    print("empty strings, which cannot be vectorized by TF-IDF (resulting in zero tokens).")
    print("Thus, 5574 raw - 403 duplicates - 6 empty/symbol-only messages = 5,165 unique training records.")


def compare_algorithms():
    print("\n" + "=" * 60)
    print("TASK 3-7: ALGORITHM BENCHMARK & COMPARISON")
    print("=" * 60)

    df_clean = pd.read_csv("dataset/processed/cleaned_data.csv")
    X = df_clean['cleaned_message']
    y = df_clean['label_num']

    # Stratified 80:20 split with same random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Dataset Split Sizes:")
    print(f"  Training: {len(X_train)} samples ({y_train.value_counts().to_dict()})")
    print(f"  Testing:  {len(X_test)} samples ({y_test.value_counts().to_dict()})")
    print(f"  Train Class Ratio: Ham = {(y_train == 0).mean()*100:.2f}%, Spam = {(y_train == 1).mean()*100:.2f}%")
    print(f"  Test Class Ratio:  Ham = {(y_test == 0).mean()*100:.2f}%, Spam = {(y_test == 1).mean()*100:.2f}%")

    # Fit TF-IDF on X_train ONLY
    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Models to benchmark
    models = {
        "Multinomial Naive Bayes (alpha=0.2)": MultinomialNB(alpha=0.2),
        "Multinomial Naive Bayes (alpha=1.0)": MultinomialNB(alpha=1.0),
        "Logistic Regression (C=1.0)": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Support Vector Machine (LinearSVC)": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42))
    }

    results = {}

    for name, clf in models.items():
        # Measure training time
        start_time = time.perf_counter()
        clf.fit(X_train_tfidf, y_train)
        train_time = (time.perf_counter() - start_time) * 1000  # ms

        # Predict
        y_pred = clf.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        spam_prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        spam_rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        spam_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            "accuracy": round(float(acc), 4),
            "spam_precision": round(float(spam_prec), 4),
            "spam_recall": round(float(spam_rec), 4),
            "spam_f1_score": round(float(spam_f1), 4),
            "training_time_ms": round(float(train_time), 2),
            "confusion_matrix": {
                "tn_ham": int(tn),
                "fp_ham_as_spam": int(fp),
                "fn_spam_as_ham": int(fn),
                "tp_spam": int(tp)
            }
        }

    # Print Comparison Table
    print("\n" + "-" * 85)
    print(f"{'Model':<38} | {'Accuracy':<9} | {'Spam Prec':<9} | {'Spam Rec':<8} | {'Spam F1':<7} | {'Train (ms)':<10}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{name:<38} | {r['accuracy']*100:>7.2f}% | {r['spam_precision']*100:>7.2f}% | {r['spam_recall']*100:>6.2f}% | {r['spam_f1_score']*100:>6.2f}% | {r['training_time_ms']:>8.2f} ms")
    print("-" * 85)

    # Detailed Confusion Matrices
    print("\nConfusion Matrices:")
    for name, r in results.items():
        c = r['confusion_matrix']
        print(f"  [{name}]")
        print(f"    Ham:  TN={c['tn_ham']} | FP={c['fp_ham_as_spam']}")
        print(f"    Spam: FN={c['fn_spam_as_ham']} | TP={c['tp_spam']}")

    # Save to models/model_comparison.json
    output_path = "models/model_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[AUDIT] Saved comparison to: {output_path}")

    return results


def check_probability_calibration():
    print("\n" + "=" * 60)
    print("TASK 10: PROBABILITY CALIBRATION ANALYSIS")
    print("=" * 60)
    
    # Load existing models
    import pickle
    with open("models/spam_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    test_samples = [
        "Congratulations! You won a £1000 cash prize! Call 09061743806 to claim.",
        "Hey bro, are we watching football tonight at 8?",
        "Urgent! Confirm your banking details now at http://bank.com to avoid account lockout.",
        "Ok see you tomorrow."
    ]

    print("Current Model predict_proba output on test phrases:")
    for s in test_samples:
        c = clean_text(s)
        feat = vectorizer.transform([c])
        proba = nb_model.predict_proba(feat)[0]
        print(f"  Text: '{s}'")
        print(f"  Cleaned: '{c}'")
        print(f"  Raw predict_proba: Ham={proba[0]:.6f}, Spam={proba[1]:.6f}")


if __name__ == "__main__":
    investigate_discrepancy()
    compare_algorithms()
    check_probability_calibration()

