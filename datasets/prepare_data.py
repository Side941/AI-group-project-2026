"""
datasets/prepare_data.py
=========================
Splits Suicide_Detection.csv into train/test with cleaning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from src.config import (
    SUICIDE_RAW_DATA_PATH,
    SUICIDE_TRAIN_PATH,
    SUICIDE_TEST_PATH,
    TEXT_COL,
    LABEL_COL,
)

MIN_CHARS = 50
MAX_CHARS = 5000


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    """Prepare suicide detection dataset."""
    print(f"\n{'='*60}")
    print("Preparing: Suicide Detection Dataset")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(SUICIDE_RAW_DATA_PATH)
    
    # Keep only text and label columns
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df.columns = ['text', 'label']
    
    print(f"Original rows: {len(df):,}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # Filter by length
    before = len(df)
    df = df[df['text'].str.len() >= MIN_CHARS]
    df = df[df['text'].str.len() <= MAX_CHARS]
    print(f"After length filter ({MIN_CHARS}-{MAX_CHARS} chars): {len(df):,} rows "
          f"(removed {before - len(df):,})")
    
    # Drop empty texts
    before = len(df)
    df = df[df['text'].str.len() > 0]
    print(f"After removing empty: {len(df):,} rows (removed {before - len(df):,})")
    
    print(f"\nNew class distribution:\n{df['label'].value_counts()}")
    
    # Stratified split: 80% train, 20% test
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    train_df.to_csv(SUICIDE_TRAIN_PATH, index=False)
    test_df.to_csv(SUICIDE_TEST_PATH, index=False)
    
    print(f"\n✅ Train: {len(train_df):,} rows → {SUICIDE_TRAIN_PATH}")
    print(f"✅ Test:  {len(test_df):,} rows → {SUICIDE_TEST_PATH}")
    print(f"\nTrain class distribution:\n{train_df['label'].value_counts()}")
    print(f"\nTest class distribution:\n{test_df['label'].value_counts()}")


if __name__ == "__main__":
    main()