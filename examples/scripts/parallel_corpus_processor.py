#!/usr/bin/env python3
"""
Parallel Corpus Processor for K'Cho Training Data

This script processes aligned K'Cho-English parallel corpora to create
properly formatted training data for machine learning applications.

Features:
- Handles CSV files with quoted fields correctly
- Creates train/validation/test splits
- Exports in multiple formats (CSV, TXT)
- Maintains sentence alignment integrity
- Provides detailed processing statistics
"""

import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from kcho.collocation import CollocationExtractor, AssociationMeasure
from kcho.evaluation import load_gold_standard, evaluate_ranking
from kcho.kcho_system import KchoSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_corpus(file_path):
    """Loads the raw parallel corpus from a CSV file."""
    logging.info(f"Loading raw corpus from {file_path}...")
    df = pd.read_csv(file_path, quoting=csv.QUOTE_MINIMAL)
    logging.info(f"Loaded {len(df)} rows.")
    return df

def process_corpus_for_training(df, output_dir="training_data", test_size=0.1, validation_size=0.1, random_state=42):
    """
    Processes the DataFrame to create training, validation, and test splits,
    and exports them to CSV and plain text files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out rows with empty K'Cho or English text
    initial_rows = len(df)
    df.dropna(subset=['Text_Cho', 'Text_English'], inplace=True)
    df = df[df['Text_Cho'].str.strip() != '']
    df = df[df['Text_English'].str.strip() != '']
    
    valid_rows = len(df)
    skipped_rows = initial_rows - valid_rows
    success_rate = (valid_rows / initial_rows) * 100 if initial_rows > 0 else 0

    logging.info(f"Processing complete:")
    logging.info(f"   Valid pairs: {valid_rows}")
    logging.info(f"   Skipped pairs: {skipped_rows}")
    logging.info(f"   Success rate: {success_rate:.1f}%")

    # Create combined 'source' and 'target' columns for easier splitting
    df['source'] = df['Text_Cho']
    df['target'] = df['Text_English']

    # Split into train, validation, and test sets
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=validation_size/(1-test_size), random_state=random_state)

    logging.info(f"\nCreating Training Splits...")
    logging.info(f"   Total pairs: {len(df)}")
    logging.info(f"   Train: {len(train_df)} pairs ({len(train_df)/len(df):.1%})")
    logging.info(f"   Validation: {len(val_df)} pairs ({len(val_df)/len(df):.1%})")
    logging.info(f"   Test: {len(test_df)} pairs ({len(test_df)/len(df):.1%})")

    splits = {'train': train_df, 'validation': val_df, 'test': test_df}

    logging.info(f"\nExporting Training Data to {output_dir}...")
    for name, split_df in splits.items():
        # Export to CSV
        csv_path = os.path.join(output_dir, f"{name}.csv")
        split_df[['source', 'target', 'Book', 'Chapter', 'Verse_Key']].to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logging.info(f"     ✅ {name}: {len(split_df)} pairs exported")

        # Export to plain text (K'Cho)
        kcho_path = os.path.join(output_dir, f"{name}.kcho")
        split_df['source'].to_csv(kcho_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep='\n')

        # Export to plain text (English)
        eng_path = os.path.join(output_dir, f"{name}.eng")
        split_df['target'].to_csv(eng_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep='\n')
    
    return splits

def verify_alignment(output_dir, num_samples=3):
    """Verifies alignment of exported training data."""
    logging.info(f"\nVerifying Alignment...")
    logging.info(f"First {num_samples} training pairs:")
    
    train_csv_path = os.path.join(output_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        logging.error(f"Train CSV not found at {train_csv_path}")
        return

    with open(train_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            logging.info(f"\nPair {i+1}:")
            logging.info(f"  Book: {row['Book']}, Chapter: {row['Chapter']}, Verse: {row['Verse_Key']}")
            logging.info(f"  K'Cho: {row['source'][:50]}...")
            logging.info(f"  English: {row['target'][:50]}...")
            logging.info(f"  Lengths: K'Cho={len(row['source'].split())}, English={len(row['target'].split())}")

def main():
    logging.info("🚀 Parallel Corpus Processor")
    logging.info("=" * 50)

    corpus_path = "data/parallel_corpora/aligned_cho_english.csv"
    output_dir = "training_data"

    # Step 1: Verify initial alignment
    logging.info("\n🔍 Verifying Parallel Sentence Alignment...")
    logging.info("First 5 rows:")
    df_sample = pd.read_csv(corpus_path, nrows=5, quoting=csv.QUOTE_MINIMAL)
    for i, row in df_sample.iterrows():
        logging.info(f"\nRow {i+1}:")
        logging.info(f"  Book: {row['Book']}")
        logging.info(f"  Chapter: {row['Chapter']}")
        logging.info(f"  Verse: {row['Verse_Key']}")
        logging.info(f"  K'Cho: {row['Text_Cho'][:50]}...")
        logging.info(f"  English: {row['Text_English'][:50]}...")

    # Step 2: Process the full corpus
    df = load_raw_corpus(corpus_path)
    logging.info("\n🔧 Processing Corpus with Fixed Alignment...")
    splits = process_corpus_for_training(df, output_dir=output_dir)

    # Step 3: Verify alignment of exported data
    verify_alignment(output_dir)

    logging.info("\n" + "=" * 50)
    logging.info("✅ Processing complete: Fixed alignment and exported training data.")

if __name__ == "__main__":
    main()