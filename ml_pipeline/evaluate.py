"""
ml_pipeline/evaluate.py
-----------------------
Model evaluation module for SmartSpam.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix,
and full classification report with specific focus on Spam class (imbalance metrics).
"""

import json
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred, y_proba=None, print_report: bool = True) -> Dict[str, Any]:
    """
    Evaluates classifier predictions against ground truth labels.
    Class 0 = Ham, Class 1 = Spam.
    
    Returns a dictionary of all calculated metrics.
    """
    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Class 1 (Spam) specific metrics (Crucial for imbalanced classification)
    spam_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    spam_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spam_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Class 0 (Ham) specific metrics
    ham_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    ham_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    ham_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Weighted and Macro averages
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Classification report as dict and text
    report_dict = classification_report(y_true, y_pred, target_names=['Ham', 'Spam'], output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=['Ham', 'Spam'])

    metrics = {
        "accuracy": round(float(acc), 4),
        "spam_precision": round(float(spam_precision), 4),
        "spam_recall": round(float(spam_recall), 4),
        "spam_f1": round(float(spam_f1), 4),
        "ham_precision": round(float(ham_precision), 4),
        "ham_recall": round(float(ham_recall), 4),
        "ham_f1": round(float(ham_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": {
            "true_negatives_ham": int(tn),
            "false_positives_ham_as_spam": int(fp),
            "false_negatives_spam_as_ham": int(fn),
            "true_positives_spam": int(tp),
            "matrix": cm.tolist()
        },
        "classification_report": report_dict
    }

    if print_report:
        print("\n" + "="*50)
        print("          MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy:        {metrics['accuracy'] * 100:.2f}%")
        print(f"Spam Precision:          {metrics['spam_precision'] * 100:.2f}%  (Low false alarm rate)")
        print(f"Spam Recall:             {metrics['spam_recall'] * 100:.2f}%  (Spam detection rate)")
        print(f"Spam F1-Score:           {metrics['spam_f1'] * 100:.2f}%")
        print(f"Ham Precision:           {metrics['ham_precision'] * 100:.2f}%")
        print(f"Ham Recall:              {metrics['ham_recall'] * 100:.2f}%")
        print(f"Weighted F1-Score:       {metrics['weighted_f1'] * 100:.2f}%")
        print("-" * 50)
        print("CONFUSION MATRIX:")
        print(f"               Predicted Ham    Predicted Spam")
        print(f"Actual Ham:        {tn:<16} {fp:<16} (Total: {tn+fp})")
        print(f"Actual Spam:       {fn:<16} {tp:<16} (Total: {fn+tp})")
        print("-" * 50)
        print("CLASSIFICATION REPORT:\n")
        print(report_text)
        print("="*50 + "\n")

    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: str = "models/metrics.json"):
    """Saves evaluation metrics dictionary to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[EVALUATE] Saved evaluation metrics to: {filepath}")

