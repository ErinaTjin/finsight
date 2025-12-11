# run_preprocessing.py
print("="*60)
print("FINANCIAL DATA PREPROCESSING PIPELINE")
print("="*60)

# First, load the datasets
print("\n📥 Loading datasets...")
try:
    # Import your existing loader
    from load_both_datasets import df_finspid, df_nifty
    
    print(f"✅ Loaded FNSPID: {df_finspid.shape if df_finspid is not None else 'None'}")
    print(f"✅ Loaded NIFTY: {df_nifty.shape if df_nifty is not None else 'None'}")
    
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

# Now run the corrected preprocessing
print("\n🚀 Starting corrected preprocessing pipeline...")
try:
    from unified_preprocessor import FinancialDataPreprocessor
    
    # Create preprocessor and run
    preprocessor = FinancialDataPreprocessor()
    result = preprocessor.run_preprocessing_pipeline(df_finspid, df_nifty)
    
    if result is not None:
        print("\n✅ Pipeline completed successfully!")
        print("\n📁 Check the 'processed_data' folder for results:")
        print("   • combined_financial_news.csv - All combined data")
        print("   • fnspid_processed.csv - Just FNSPID data")
        print("   • nifty_processed.csv - Just NIFTY data")
        print("   • sample_100_rows.csv - Sample for inspection")
        
        print("\n💡 Next steps:")
        print("   1. Inspect the processed_data/ folder")
        print("   2. For model training, focus on columns: ticker, headline/content, timestamp, price_change")
        print("   3. Use NIFTY data for supervised sentiment analysis (has price_change)")
        print("   4. Use FNSPID for additional training data")
    
except ImportError as e:
    print(f"❌ Could not import preprocessing module: {e}")
    print("Make sure unified_preprocessor_corrected.py is in the same folder")
except Exception as e:
    print(f"❌ Error in pipeline: {e}")
    import traceback
    traceback.print_exc()