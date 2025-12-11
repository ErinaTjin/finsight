# unified_preprocessor.py
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import re
from typing import Dict, List, Optional

class FinancialDataPreprocessor:
    def __init__(self):
        print("🔧 Financial Data Preprocessor Initialized")
        
    def preprocess_fnspid(self, df_finspid: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess FNSPID dataset
        """
        print("\n" + "="*50)
        print("PREPROCESSING FNSPID DATASET")
        print("="*50)
        
        # Create a copy to avoid modifying original
        df_finspid_clean = df_finspid.copy()
        
        # Clean column names
        df_finspid_clean.columns = [col.lower().strip().replace(' ', '_') for col in df_finspid_clean.columns]
        
        # Map to standard columns
        fnspid_mapping = {
            'article_title': 'headline',
            'article': 'content',
            'stock_symbol': 'ticker',
            'date': 'timestamp',
            'lsa_summary': 'summary'  # Using LSA summary as default
        }
        
        # Apply mapping if columns exist
        for orig_col, new_col in fnspid_mapping.items():
            if orig_col in df_finspid_clean.columns:
                df_finspid_clean[new_col] = df_finspid_clean[orig_col]
        
        # Create required columns if they don't exist
        if 'headline' not in df_finspid_clean.columns:
            if 'article_title' in df_finspid_clean.columns:
                df_finspid_clean['headline'] = df_finspid_clean['article_title']
            else:
                # Create from content
                df_finspid_clean['headline'] = df_finspid_clean.get('content', '').str[:200]
        
        if 'ticker' not in df_finspid_clean.columns:
            if 'stock_symbol' in df_finspid_clean.columns:
                df_finspid_clean['ticker'] = df_finspid_clean['stock_symbol']
            else:
                df_finspid_clean['ticker'] = 'UNKNOWN'
        
        if 'timestamp' not in df_finspid_clean.columns:
            if 'date' in df_finspid_clean.columns:
                df_finspid_clean['timestamp'] = df_finspid_clean['date']
            else:
                df_finspid_clean['timestamp'] = pd.NaT
        
        if 'content' not in df_finspid_clean.columns:
            if 'article' in df_finspid_clean.columns:
                df_finspid_clean['content'] = df_finspid_clean['article']
            else:
                df_finspid_clean['content'] = df_finspid_clean['headline']
        
        if 'summary' not in df_finspid_clean.columns:
            # Use content truncated as summary
            df_finspid_clean['summary'] = df_finspid_clean['content'].str[:500]
        
        # Clean text data
        text_columns = ['headline', 'content', 'summary']
        for col in text_columns:
            if col in df_finspid_clean.columns:
                df_finspid_clean[col] = df_finspid_clean[col].astype(str).str.strip()
                # Remove extra whitespace
                df_finspid_clean[col] = df_finspid_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        # Clean ticker symbols
        if 'ticker' in df_finspid_clean.columns:
            df_finspid_clean['ticker'] = df_finspid_clean['ticker'].astype(str).str.upper().str.strip()
        
        # Parse timestamps
        if 'timestamp' in df_finspid_clean.columns:
            df_finspid_clean['timestamp'] = pd.to_datetime(
                df_finspid_clean['timestamp'], 
                errors='coerce',
                utc=True
            )
        
        # Add metadata
        df_finspid_clean['source'] = 'FNSPID'
        df_finspid_clean['has_price_data'] = False  # Will be updated later
        
        # Select and order columns
        standard_columns = [
            'ticker', 'headline', 'content', 'summary', 'timestamp',
            'source', 'has_price_data'
        ]
        
        # Keep original columns for reference
        for col in standard_columns:
            if col not in df_finspid_clean.columns:
                df_finspid_clean[col] = None
        
        # Reorder
        df_finspid_clean = df_finspid_clean[standard_columns + 
                                          [c for c in df_finspid_clean.columns if c not in standard_columns]]
        
        print(f"✅ FNSPID preprocessed: {len(df_finspid_clean)} rows")
        print(f"📊 Columns: {list(df_finspid_clean.columns)}")
        
        return df_finspid_clean
    
    def preprocess_nifty(self, df_nifty: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess NIFTY dataset
        """
        print("\n" + "="*50)
        print("PREPROCESSING NIFTY DATASET")
        print("="*50)
        
        df_nifty_clean = df_nifty.copy()
        
        # Clean column names
        df_nifty_clean.columns = [col.lower().strip() for col in df_nifty_clean.columns]
        
        # Map to standard columns
        nifty_mapping = {
            'news': 'headline',
            'date': 'timestamp',
            'pct_change': 'price_change',
            'label': 'sentiment_label'  # Keep original label for reference
        }
        
        # Apply mapping
        for orig_col, new_col in nifty_mapping.items():
            if orig_col in df_nifty_clean.columns:
                df_nifty_clean[new_col] = df_nifty_clean[orig_col]
        
        # Create content column (combine headlines)
        if 'headline' in df_nifty_clean.columns:
            df_nifty_clean['content'] = df_nifty_clean['headline']
        
        # Create summary (truncated content)
        if 'content' in df_nifty_clean.columns:
            df_nifty_clean['summary'] = df_nifty_clean['content'].str[:500]
        
        # Set ticker - NIFTY is primarily about S&P 500 (SPY)
        df_nifty_clean['ticker'] = 'SPY'
        
        # Parse dates - NIFTY dates are in format like '6/1/2010'
        if 'timestamp' in df_nifty_clean.columns:
            df_nifty_clean['timestamp'] = pd.to_datetime(
                df_nifty_clean['timestamp'],
                format='%m/%d/%Y',
                errors='coerce',
                utc=True
            )
        
        # Clean text data
        text_columns = ['headline', 'content', 'summary']
        for col in text_columns:
            if col in df_nifty_clean.columns:
                df_nifty_clean[col] = df_nifty_clean[col].astype(str).str.strip()
                # Handle multiline headlines (they seem to have \n separators)
                df_nifty_clean[col] = df_nifty_clean[col].str.replace('\n', ' | ', regex=False)
                # Remove extra whitespace
                df_nifty_clean[col] = df_nifty_clean[col].str.replace(r'\s+', ' ', regex=True)
        
        # Add metadata
        df_nifty_clean['source'] = 'NIFTY'
        df_nifty_clean['has_price_data'] = True  # NIFTY has price change data
        
        # Select and order columns
        standard_columns = [
            'ticker', 'headline', 'content', 'summary', 'timestamp',
            'price_change', 'sentiment_label', 'source', 'has_price_data'
        ]
        
        # Keep original columns for reference
        for col in standard_columns:
            if col not in df_nifty_clean.columns:
                df_nifty_clean[col] = None
        
        # Reorder
        df_nifty_clean = df_nifty_clean[standard_columns + 
                                       [c for c in df_nifty_clean.columns if c not in standard_columns]]
        
        print(f"✅ NIFTY preprocessed: {len(df_nifty_clean)} rows")
        print(f"📊 Columns: {list(df_nifty_clean.columns)}")
        
        return df_nifty_clean
    
    def combine_datasets(self, df_finspid: pd.DataFrame, df_nifty: pd.DataFrame) -> pd.DataFrame:
        """
        Combine both datasets into a unified format
        """
        print("\n" + "="*50)
        print("COMBINING DATASETS")
        print("="*50)
        
        # Ensure both have required columns
        required_columns = [
            'ticker', 'headline', 'content', 'summary', 'timestamp',
            'source', 'has_price_data'
        ]
        
        # Add missing columns to FNSPID
        for col in required_columns:
            if col not in df_finspid.columns:
                df_finspid[col] = None
        
        # Add missing columns to NIFTY (if any)
        for col in required_columns:
            if col not in df_nifty.columns:
                df_nifty[col] = None
        
        # Add NIFTY-specific columns to FNSPID
        if 'price_change' not in df_finspid.columns:
            df_finspid['price_change'] = None
        if 'sentiment_label' not in df_finspid.columns:
            df_finspid['sentiment_label'] = None
        
        # Add FNSPID-specific columns to NIFTY (if any)
        if 'url' in df_finspid.columns and 'url' not in df_nifty.columns:
            df_nifty['url'] = None
        
        # Get all unique columns
        all_columns = list(set(df_finspid.columns.tolist() + df_nifty.columns.tolist()))
        
        # Add missing columns to each dataframe
        for col in all_columns:
            if col not in df_finspid.columns:
                df_finspid[col] = None
            if col not in df_nifty.columns:
                df_nifty[col] = None
        
        # Combine
        combined_df = pd.concat([df_finspid, df_nifty], ignore_index=True, sort=False)
        
        print(f"⚠️ Skipping timestamp sort - dataset too large: {len(combined_df):,} rows")
        print("⚠️ Sorting not required for model training")
        print(f"✅ Combined: {len(df_finspid)} FNSPID + {len(df_nifty)} NIFTY = {len(combined_df)} total rows")
        print(f"📊 Final columns: {list(combined_df.columns)}")
        
        # Show sample
        print("\n📋 Sample of combined data:")
        print(combined_df[['ticker', 'headline', 'timestamp', 'source', 'price_change']].head())
        
        return combined_df
    
    def analyze_datasets(self, df_finspid: pd.DataFrame, df_nifty: pd.DataFrame):
        """
        Analyze both datasets
        """
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        analysis = {
            'fnspid': {
                'total_rows': len(df_finspid),
                'columns': df_finspid.columns.tolist(),
                'tickers': df_finspid['ticker'].nunique() if 'ticker' in df_finspid.columns else 0,
                'date_range': None,
                'missing_values': df_finspid.isnull().sum().to_dict()
            },
            'nifty': {
                'total_rows': len(df_nifty),
                'columns': df_nifty.columns.tolist(),
                'tickers': df_nifty['ticker'].nunique() if 'ticker' in df_nifty.columns else 0,
                'date_range': None,
                'missing_values': df_nifty.isnull().sum().to_dict()
            }
        }
        
        # Date ranges
        if 'timestamp' in df_finspid.columns:
            analysis['fnspid']['date_range'] = {
                'min': df_finspid['timestamp'].min(),
                'max': df_finspid['timestamp'].max()
            }
        
        if 'timestamp' in df_nifty.columns:
            analysis['nifty']['date_range'] = {
                'min': df_nifty['timestamp'].min(),
                'max': df_nifty['timestamp'].max()
            }
        
        # Print analysis
        print("\n📊 FNSPID Analysis:")
        print(f"  • Rows: {analysis['fnspid']['total_rows']}")
        print(f"  • Unique tickers: {analysis['fnspid']['tickers']}")
        if analysis['fnspid']['date_range']:
            print(f"  • Date range: {analysis['fnspid']['date_range']['min']} to {analysis['fnspid']['date_range']['max']}")
        
        print("\n📊 NIFTY Analysis:")
        print(f"  • Rows: {analysis['nifty']['total_rows']}")
        print(f"  • Unique tickers: {analysis['nifty']['tickers']}")
        if analysis['nifty']['date_range']:
            print(f"  • Date range: {analysis['nifty']['date_range']['min']} to {analysis['nifty']['date_range']['max']}")
        
        # Save analysis
        import json
        with open('dataset_analysis_detailed.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Analysis saved to: dataset_analysis_detailed.json")
        
        return analysis
    
    def save_processed_data(self, combined_df: pd.DataFrame, output_dir: str = 'processed_data'):
        """
        Save processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined data
        combined_path = os.path.join(output_dir, 'combined_financial_news.csv')
        combined_df.to_csv(combined_path, index=False)
        
        # Save by source
        if 'source' in combined_df.columns:
            for source in combined_df['source'].unique():
                source_df = combined_df[combined_df['source'] == source]
                source_path = os.path.join(output_dir, f'{source.lower()}_processed.csv')
                source_df.to_csv(source_path, index=False)
                print(f"💾 Saved {source}: {len(source_df)} rows to {source_path}")
        
        # Save sample
        sample_path = os.path.join(output_dir, 'sample_100_rows.csv')
        combined_df.head(100).to_csv(sample_path, index=False)
        
        print(f"💾 Combined data saved to: {combined_path}")
        print(f"💾 Sample saved to: {sample_path}")
        
        return combined_path
    
    def run_preprocessing_pipeline(self, df_finspid: pd.DataFrame, df_nifty: pd.DataFrame):
        """
        Complete preprocessing pipeline
        """
        print("🚀 STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Analyze datasets
            self.analyze_datasets(df_finspid, df_nifty)
            
            # Step 2: Preprocess FNSPID
            fnspid_processed = self.preprocess_fnspid(df_finspid)
            
            # Step 3: Preprocess NIFTY
            nifty_processed = self.preprocess_nifty(df_nifty)
            
            # Step 4: Combine
            combined = self.combine_datasets(fnspid_processed, nifty_processed)
            
            # Step 5: Save
            output_path = self.save_processed_data(combined)
            
            print("\n" + "="*60)
            print("🎉 PREPROCESSING COMPLETE!")
            print("="*60)
            print(f"\n📊 Total processed rows: {len(combined)}")
            print(f"💾 Data saved to: {output_path}")
            
            return combined
            
        except Exception as e:
            print(f"\n❌ Error in preprocessing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


# Quick test function
def test_preprocessing():
    """Test the preprocessing with sample data"""
    # Create sample data based on your description
    sample_fnspid = pd.DataFrame({
        'Unnamed: 0': [0.0],
        'Date': ['2023-12-16 23:00:00 UTC'],
        'Article_title': ['Interesting A Put And Call Options For August ...'],
        'Stock_symbol': ['A'],
        'Article': ['Investors in Agilent Technologies, Inc. (Symbol: A) saw new options become available today...'],
        'Lsa_summary': ['Because the $125.00 strike represents an approximate...']
    })
    
    sample_nifty = pd.DataFrame({
        'id': ['nifty_0'],
        'date': ['6/1/2010'],
        'news': ['China Officials Likely Knew of Bad Milk\nSony CEO on Strategy...'],
        'pct_change': [0.0042],
        'label': ['Neutral']
    })
    
    preprocessor = FinancialDataPreprocessor()
    result = preprocessor.run_preprocessing_pipeline(sample_fnspid, sample_nifty)
    
    return result


if __name__ == "__main__":
    # Test the preprocessing
    print("Testing preprocessing pipeline...")
    test_preprocessing()