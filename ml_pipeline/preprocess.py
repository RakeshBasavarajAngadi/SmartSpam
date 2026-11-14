"""
ml_pipeline/preprocess.py
-------------------------
NLP Preprocessing module for SmartSpam.
Includes functions for text normalization, cleaning, tokenization,
stopword removal (with negation preservation), stemming, and dataset cleaning.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# English stopwords with negations explicitly preserved
NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none', "n't",
    'without', 'barely', 'hardly', 'scarcely'
}
DEFAULT_STOPWORDS = set(stopwords.words('english'))
FILTERED_STOPWORDS = DEFAULT_STOPWORDS - NEGATION_WORDS

# Regex patterns for normalization
URL_REGEX = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,5}\b|\b\d{5,}\b')
CURRENCY_REGEX = re.compile(r'[$£€₹¥]')
NUMERIC_REGEX = re.compile(r'\b\d+\b')


def clean_text(text: str) -> str:
    """
    Cleans and normalizes a single text string for NLP feature extraction.
    
    Steps:
      1. Normalize URLs -> 'urltoken'
      2. Normalize emails -> 'emailtoken'
      3. Normalize currency symbols -> 'moneytoken'
      4. Normalize phone numbers / long numeric sequences -> 'phonetoken'
      5. Normalize remaining standalone numbers -> 'numtoken'
      6. Convert to lowercase
      7. Tokenize using NLTK word_tokenize
      8. Filter non-alphanumeric punctuation and non-negation stopwords
      9. Apply Porter Stemming
      10. Re-join tokens into a normalized string
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Normalize URLs
    text = URL_REGEX.sub(' urltoken ', text)

    # 2. Normalize emails
    text = EMAIL_REGEX.sub(' emailtoken ', text)

    # 3. Normalize currencies
    text = CURRENCY_REGEX.sub(' moneytoken ', text)

    # 4. Normalize phone numbers / 5+ digit numbers
    text = PHONE_REGEX.sub(' phonetoken ', text)

    # 5. Normalize remaining numbers
    text = NUMERIC_REGEX.sub(' numtoken ', text)

    # 6. Lowercase
    text = text.lower()

    # 7. Tokenize
    tokens = nltk.word_tokenize(text)

    # 8 & 9. Filter punctuation & stopwords, apply Porter Stemming
    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalnum() and token not in FILTERED_STOPWORDS
    ]

    return " ".join(cleaned_tokens)


def load_and_preprocess_dataset(
    raw_path: str = "dataset/raw/spam.csv",
    output_path: str = "dataset/processed/cleaned_data.csv"
) -> pd.DataFrame:
    """
    Loads the raw dataset, documents & removes duplicates,
    applies NLP preprocessing, and exports the processed dataset.
    Leaves original raw dataset unchanged.
    """
    df = pd.read_csv(raw_path)
    initial_count = len(df)
    
    # Check and remove duplicate rows
    duplicates = df.duplicated().sum()
    df_clean = df.drop_duplicates(subset=['message']).copy()
    final_count = len(df_clean)
    removed_count = initial_count - final_count

    print(f"[PREPROCESS] Initial records: {initial_count}")
    print(f"[PREPROCESS] Duplicate messages removed: {removed_count} ({removed_count / initial_count * 100:.2f}%)")
    print(f"[PREPROCESS] Unique records remaining: {final_count}")

    # Map labels: 'ham' -> 0, 'spam' -> 1
    df_clean['label_num'] = df_clean['label'].map({'ham': 0, 'spam': 1})
    
    # Apply text preprocessing
    print("[PREPROCESS] Applying text normalization and NLTK pipeline...")
    df_clean['cleaned_message'] = df_clean['message'].apply(clean_text)

    # Remove any empty rows after cleaning (if message was purely discarded punctuation)
    df_clean = df_clean[df_clean['cleaned_message'].str.strip() != ''].copy()
    print(f"[PREPROCESS] Final clean records: {len(df_clean)}")

    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"[PREPROCESS] Saved cleaned dataset to: {output_path}")

    return df_clean


if __name__ == "__main__":
    import os
    print("Testing preprocessing module...")
    sample = "WINNER!! As a valued customer you have won a £1,000 cash prize! Call 09061743806 to claim or visit http://win-now.com"
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))

    sample_neg = "No, I am not interested in your loan offer. Do not call me again!"
    print("Original:", sample_neg)
    print("Cleaned: ", clean_text(sample_neg))

