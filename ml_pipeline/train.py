"""
ml_pipeline/train.py
--------------------
Complete Model Training Pipeline for SmartSpam.
Executes:
1. Dataset loading & deduplication
2. NLP preprocessing (via ml_pipeline.preprocess)
3. Stratified 80:20 train-test split
4. TF-IDF feature extraction (fitted on train only to prevent leakage)
5. Multinomial Naive Bayes classifier training
6. Comprehensive evaluation (via ml_pipeline.evaluate)
7. Artifact serialization to models/
8. Smoke testing on sample messages
"""

import os
import sys
from pathlib import Path
import pickle
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from ml_pipeline.preprocess import clean_text, load_and_preprocess_dataset
from ml_pipeline.evaluate import evaluate_model, save_metrics


def run_training_pipeline(
    raw_data_path: str = "dataset/raw/spam.csv",
    processed_data_path: str = "dataset/processed/cleaned_data.csv",
    models_dir: str = "models",
    random_state: int = 42
):
    print("=" * 60)
    print("      SMARTSPAM: ML TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load and Preprocess Data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("\n[STEP 1/5] Loading & Preprocessing Dataset...")
    df = load_and_preprocess_dataset(
        raw_path=raw_data_path,
        output_path=processed_data_path
    )

    X = df['cleaned_message']
    y = df['label_num']

    # 2. Stratified 80:20 Train/Test Split
    print("\n[STEP 2/5] Performing Stratified 80:20 Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    print(f"Total Unique Samples: {len(df)}")
    print(f"Training Set Size:   {len(X_train)} samples ({y_train.value_counts().to_dict()} [0=Ham, 1=Spam])")
    print(f"Testing Set Size:    {len(X_test)} samples ({y_test.value_counts().to_dict()} [0=Ham, 1=Spam])")
    print(f"Training Class Ratio: Spam = {y_train.mean()*100:.2f}%, Ham = {(1-y_train.mean())*100:.2f}%")
    print(f"Testing Class Ratio:  Spam = {y_test.mean()*100:.2f}%, Ham = {(1-y_test.mean())*100:.2f}%")

    # 3. TF-IDF Feature Extraction (Fit on X_train ONLY)
    print("\n[STEP 3/5] Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),  # unigrams & bigrams
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF Vocabulary Size: {len(vectorizer.vocabulary_)} features")
    print(f"X_train_tfidf shape:    {X_train_tfidf.shape}")
    print(f"X_test_tfidf shape:     {X_test_tfidf.shape}")

    # 4. Train Multinomial Naive Bayes Classifier
    print("\n[STEP 4/5] Training Multinomial Naive Bayes Model...")
    # alpha=0.2 provides high sensitivity for spam detection while maintaining near 100% precision
    classifier = MultinomialNB(alpha=0.2)
    classifier.fit(X_train_tfidf, y_train)
    print("Model training completed successfully.")

    # 5. Evaluate Model
    print("\n[STEP 5/5] Evaluating Model Performance on Test Data...")
    y_pred = classifier.predict(X_test_tfidf)
    y_proba = classifier.predict_proba(X_test_tfidf)

    metrics = evaluate_model(y_test, y_pred, y_proba, print_report=True)

    # 6. Save Artifacts
    print("\n[SAVING ARTIFACTS]")
    model_path = os.path.join(models_dir, "spam_model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")

    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)
    print(f"Saved Trained Model:       {model_path}")

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved Fitted Vectorizer:   {vectorizer_path}")

    save_metrics(metrics, metrics_path)

    # 7. Verification: Test sample predictions
    print("\n" + "=" * 60)
    print("          SAMPLE PREDICTION VERIFICATION")
    print("=" * 60)

    test_samples = [
        "Congratulations! You've won a $1,000 Walmart Gift Card. Click here to claim your code: http://bit.ly/claim-prize or call 9876543210 immediately!",
        "Urgent! Your mobile number has been selected for a £500 cash bonus. Text WIN to 80085 now to claim. T&Cs apply.",
        "Hey, are we still meeting for lunch at 1 PM today? Let me know.",
        "Can you send me the lecture notes for yesterday's machine learning class? Thanks!",
        "Important notification: Your account has been suspended. Please verify your password at http://secure-login.bank.com to restore access."
    ]

    for i, msg in enumerate(test_samples, 1):
        cleaned = clean_text(msg)
        features = vectorizer.transform([cleaned])
        pred_label = classifier.predict(features)[0]
        pred_proba = classifier.predict_proba(features)[0]
        confidence = pred_proba[pred_label] * 100
        label_str = "SPAM" if pred_label == 1 else "HAM (Legitimate)"

        print(f"\n[Sample {i}]")
        print(f"  Input:      {msg}")
        print(f"  Cleaned:    {cleaned}")
        print(f"  Prediction: {label_str}")
        print(f"  Confidence: {confidence:.2f}% (P(Spam)={pred_proba[1]:.4f}, P(Ham)={pred_proba[0]:.4f})")

    print("\n" + "=" * 60)
    print("      PHASE 2 PIPELINE EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
