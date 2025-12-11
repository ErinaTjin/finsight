# load_both_datasets.py
import pandas as pd
import numpy as np
from datasets import load_dataset
from datetime import datetime
import os
import sys
import subprocess

print("=" * 60)
print("FINANCIAL DATASETS LOADER - FNSPID + NIFTY")
print("=" * 60)

# ============================================
# 1. LOAD FNSPID DATASET (from CSV files)
# ============================================
print("\n📥 LOADING FNSPID DATASET...")
print("-" * 40)

# Check what FNSPID files you have
print("Available FNSPID files:")
for file in os.listdir('.'):
    if 'nasdaq' in file.lower() or '.csv' in file:
        print(f"  • {file}")

# Load the main FNSPID news file
try:
    # Try different possible filenames
    fnspid_files = [f for f in os.listdir('.') if 'nasdaq' in f.lower() and f.endswith('.csv')]
    
    if not fnspid_files:
        print("❌ No FNSPID CSV file found! Make sure you downloaded:")
        print("   wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv")
        df_finspid = None
    else:
        fnspid_file = fnspid_files[0]
        print(f"📄 Loading: {fnspid_file}")
        df_finspid = pd.read_csv(fnspid_file)
        
        print(f"✅ Successfully loaded FNSPID")
        print(f"   • Shape: {df_finspid.shape[0]} rows, {df_finspid.shape[1]} columns")
        print(f"   • Columns: {list(df_finspid.columns)}")
        print(f"   • First few rows:")
        print(df_finspid.head(3))
        
except Exception as e:
    print(f"❌ Error loading FNSPID: {e}")
    df_finspid = None

# ============================================
# 2. LOAD NIFTY DATASET (from Hugging Face)
# ============================================
print("\n\n📥 LOADING NIFTY DATASET FROM HUGGING FACE...")
print("-" * 40)

try:
    # Load using Hugging Face datasets
    print("Loading dataset: raeidsaqur/NIFTY")
    dataset = load_dataset("raeidsaqur/NIFTY")
    
    # Convert to pandas
    df_nifty = pd.DataFrame(dataset['train'])
    
    print(f"✅ Successfully loaded NIFTY")
    print(f"   • Shape: {df_nifty.shape[0]} rows, {df_nifty.shape[1]} columns")
    print(f"   • Columns: {list(df_nifty.columns)}")
    print(f"   • First few rows:")
    print(df_nifty.head(3))
    
except Exception as e:
    print(f"❌ Error loading NIFTY: {e}")
    df_nifty = None

##########################
# 3. analyze_datasets    #
##########################
import json

print("🔍 DETAILED DATASET ANALYSIS")
print("=" * 60)

print("\n📊 FNSPID DATASET STRUCTURE:")
print(f"Shape: {df_finspid.shape}")
print("\nColumns and Data Types:")
for col in df_finspid.columns:
    print(f"  • {col}: {df_finspid[col].dtype}")
    print(f"    Sample: {str(df_finspid[col].iloc[0])[:80]}..." if len(str(df_finspid[col].iloc[0])) > 80 else f"    Sample: {df_finspid[col].iloc[0]}")

print(f"\nMissing Values:")
missing = df_finspid.isnull().sum()
print(missing[missing > 0])

print("\n📊 NIFTY DATASET STRUCTURE:")
print(f"Shape: {df_nifty.shape}")
print("\nColumns and Data Types:")
for col in df_nifty.columns:
    print(f"  • {col}: {df_nifty[col].dtype}")
    sample_val = df_nifty[col].iloc[0]
    if isinstance(sample_val, str) and len(sample_val) > 80:
        sample_val = sample_val[:80] + "..."
    print(f"    Sample: {sample_val}")

# Save the analysis
analysis = {
    'fnspid_columns': list(df_finspid.columns),
    'nifty_columns': list(df_nifty.columns),
    'fnspid_shape': df_finspid.shape,
    'nifty_shape': df_nifty.shape,
    'unification_plan': {
        'target_column': 'fnspid_source',
        'ticker': 'nifty_source',
        'headline': 'map_logic',
        'timestamp': 'need_to_create',
        'price_change': 'fnspid: price_change, nifty: pct_change',
        'sentiment_label': 'need_to_create_from_price_change',
        'source': "'FNSPID' or 'NIFTY'"
    }
}

with open('dataset_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f"\n✅ Analysis saved to 'dataset_analysis.json'")
