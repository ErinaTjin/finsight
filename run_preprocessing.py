# run_preprocessing.py
import pandas as pd

print("="*60)
print("ENHANCED FINANCIAL DATA PREPROCESSING PIPELINE")
print("="*60)

# First, load the datasets
print("\n📥 Loading datasets...")
try:
    # Import your existing loader
    from load_both_datasets import df_finspid, df_nifty
    
    print(f"✅ Loaded FNSPID: {df_finspid.shape if df_finspid is not None else 'None'}")
    print(f"✅ Loaded NIFTY: {df_nifty.shape if df_nifty is not None else 'None'}")

    print("\n📅 Filtering datasets for 2024 and 2025 only...")
    
    def filter_by_year(df, date_column):
        """Filter dataframe to only include 2024 and 2025 data"""
        if df is None or date_column not in df.columns:
            return df
        
        # Convert to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=True)
        
        # Filter for 2024 and 2025
        df_filtered = df[df[date_column] >= '2024-01-01'].copy()
        
        initial_count = len(df)
        filtered_count = len(df_filtered)
        removed_count = initial_count - filtered_count
        
        print(f"   - Filtered: {filtered_count} rows remaining ({removed_count} rows removed)")
        print(f"   - Date range: {df_filtered[date_column].min()} to {df_filtered[date_column].max()}")
        
        return df_filtered
    
    # Apply filtering to both datasets
    if df_finspid is not None:
        # Determine the date column name in FNSPID
        fnspid_date_col = None
        possible_date_cols = ['date', 'Date', 'timestamp', 'Timestamp', 'time', 'Time']
        
        for col in possible_date_cols:
            if col in df_finspid.columns:
                fnspid_date_col = col
                break
        
        if fnspid_date_col:
            df_finspid = filter_by_year(df_finspid, fnspid_date_col)
        else:
            print("⚠️ Could not find date column in FNSPID dataset")
    
    if df_nifty is not None:
        # Determine the date column name in NIFTY
        nifty_date_col = None
        possible_date_cols = ['date', 'Date', 'timestamp', 'Timestamp', 'time', 'Time']
        
        for col in possible_date_cols:
            if col in df_nifty.columns:
                nifty_date_col = col
                break
        
        if nifty_date_col:
            df_nifty = filter_by_year(df_nifty, nifty_date_col)
        else:
            print("⚠️ Could not find date column in NIFTY dataset")
    # ============================================
    
    # Show column names
    if df_finspid is not None:
        print(f"\n📋 FNSPID columns: {list(df_finspid.columns)}")
    if df_nifty is not None:
        print(f"📋 NIFTY columns: {list(df_nifty.columns)}")
    
except ImportError:
    print("❌ Could not import load_both_datasets")
    print("Please make sure load_both_datasets.py is in the same folder")
    exit(1)
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    exit(1)

# Now run the enhanced preprocessing
print("\n🚀 Starting enhanced preprocessing pipeline...")
try:
    from unified_preprocessor import FinancialDataPreprocessor
    
    # Configuration for the preprocessing
    config = {
        'deduplicate': True,
        'align_prices': True,  # Set to False for faster testing
        'create_sentiment_labels': True,
        'ner_entity_mapping': False,  # Set to True if transformers is installed
        'price_window_hours': 24,
        'sentiment_thresholds': (-0.01, 0.01),
        'min_news_length': 20,
        'max_news_length': 10000
    }
    
    # Create preprocessor and run
    preprocessor = FinancialDataPreprocessor(config)
    combined_result, task_datasets = preprocessor.run_preprocessing_pipeline(df_finspid, df_nifty)
    
    if combined_result is not None:
        print("\n✅ Pipeline completed successfully!")
        print("\n📁 Check the 'processed_data' folder for results:")
        print("   • combined_financial_news.csv - All combined data")
        print("   • fnspid_processed.csv - Just FNSPID data")
        print("   • nifty_processed.csv - Just NIFTY data")
        print("   • summarization_dataset.csv - For summarization training")
        print("   • sentiment_dataset.csv - For sentiment classification")
        print("   • signal_dataset.csv - For signal prediction training")
        print("   • sample_100_rows.csv - Sample for inspection")
        print("   • preprocessing_config.json - Configuration used")
        
        print("\n💡 Key Features Implemented:")
        print("   1. ✅ Text cleaning and normalization")
        print("   2. ✅ Deduplication of news items")
        print("   3. ✅ Alignment with stock price movements (FNSPID)")
        print("   4. ✅ Ground-truth sentiment labels from price changes")
        print("   5. ✅ Entity recognition and ticker mapping (optional)")
        print("   6. ✅ Unified schema for both datasets")
        print("   7. ✅ Task-specific datasets for different training objectives")
        
        print("\n📊 Dataset Statistics:")
        print(f"   Total rows: {len(combined_result)}")
        print(f"   With price data: {combined_result['has_price_data'].sum()}")
        print(f"   Sentiment distribution:")
        print(f"     - Bullish: {len(combined_result[combined_result['sentiment_label'] == 'bullish'])}")
        print(f"     - Neutral: {len(combined_result[combined_result['sentiment_label'] == 'neutral'])}")
        print(f"     - Bearish: {len(combined_result[combined_result['sentiment_label'] == 'bearish'])}")
    
except ImportError as e:
    print(f"❌ Could not import preprocessing module: {e}")
    print("Make sure unified_preprocessor.py is in the same folder")
except Exception as e:
    print(f"❌ Error in pipeline: {e}")
    import traceback
    traceback.print_exc()