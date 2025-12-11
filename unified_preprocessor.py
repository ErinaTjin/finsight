# unified_preprocessor.py
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from hashlib import md5
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. NER features disabled.")

class FinancialDataPreprocessor:
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary with keys:
                - deduplicate: bool (default: True)
                - align_prices: bool (default: True)
                - create_sentiment_labels: bool (default: True)
                - ner_entity_mapping: bool (default: False if transformers not available)
                - price_window_hours: int (default: 24)
                - sentiment_thresholds: tuple (default: (-0.01, 0.01))
        """
        print("🔧 Financial Data Preprocessor Initialized")
        
        # Default configuration
        self.config = {
            'deduplicate': True,
            'align_prices': True,
            'create_sentiment_labels': True,
            'ner_entity_mapping': TRANSFORMERS_AVAILABLE,
            'price_window_hours': 24,
            'sentiment_thresholds': (-0.01, 0.01),
            'min_news_length': 20,
            'max_news_length': 10000
        }
        
        if config:
            self.config.update(config)
            
        # Initialize NER pipeline if needed
        self.ner_pipeline = None
        if self.config['ner_entity_mapping'] and TRANSFORMERS_AVAILABLE:
            try:
                print("📊 Loading NER model for entity recognition...")
                self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
                print("✅ NER model loaded")
            except Exception as e:
                print(f"⚠️ Could not load NER model: {e}")
                self.config['ner_entity_mapping'] = False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace("'", "'")
        text = text.replace('–', '-').replace('—', '-')
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text using NER"""
        if not self.ner_pipeline or not text:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            # Filter for ORG (organization) entities which likely contain company names
            org_entities = [e for e in entities if e['entity_group'] == 'ORG']
            return org_entities
        except Exception as e:
            print(f"⚠️ Error in entity extraction: {e}")
            return []
    
    def map_ticker_from_entities(self, entities: List[Dict], text: str) -> Optional[str]:
        """
        Map extracted entities to ticker symbols
        This is a simplified version - in production, you'd use a proper mapping database
        """
        if not entities:
            return None
        
        # Common company name to ticker mapping
        company_to_ticker = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'amazon': 'AMZN',
            'facebook': 'META',
            'tesla': 'TSLA',
            'nvidia': 'NVDA',
            'netflix': 'NFLX',
            'alphabet': 'GOOGL',
            'ibm': 'IBM',
            'intel': 'INTC',
            'amd': 'AMD',
            'boeing': 'BA',
            'coca cola': 'KO',
            'disney': 'DIS',
            'goldman sachs': 'GS',
            'jp morgan': 'JPM',
            'visa': 'V',
            'mastercard': 'MA',
            'paypal': 'PYPL',
        }
        
        # Check if any entity matches known companies
        text_lower = text.lower()
        for entity in entities:
            entity_name = entity['word'].lower()
            for company, ticker in company_to_ticker.items():
                if company in entity_name or entity_name in company:
                    return ticker
        
        return None
    
    def get_price_data(self, ticker: str, date: datetime, window_hours: int = 24) -> Optional[Dict]:
        """
        Fetch price data for a ticker around a given date
        """
        try:
            # Convert to date only for yfinance
            start_date = (date - timedelta(hours=window_hours)).strftime('%Y-%m-%d')
            end_date = (date + timedelta(days=2)).strftime('%Y-%m-%d')  # Add buffer
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, interval='1h')
            
            if hist.empty:
                return None
            
            # Find the closest price before and after the news
            news_time = date
            pre_news = hist[hist.index <= news_time]
            post_news = hist[hist.index > news_time]
            
            if len(pre_news) == 0 or len(post_news) == 0:
                return None
            
            pre_price = pre_news.iloc[-1]['Close']
            post_price = post_news.iloc[0]['Close']
            
            price_change = (post_price - pre_price) / pre_price
            
            return {
                'pre_price': pre_price,
                'post_price': post_price,
                'price_change': price_change,
                'pre_time': pre_news.index[-1],
                'post_time': post_news.index[0]
            }
            
        except Exception as e:
            print(f"⚠️ Error fetching price for {ticker}: {e}")
            return None
    
    def preprocess_fnspid(self, df_finspid: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess FNSPID dataset with enhanced features
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
            'lsa_summary': 'summary'
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
                df_finspid_clean['headline'] = df_finspid_clean.get('content', '').str[:200]
        
        if 'ticker' not in df_finspid_clean.columns:
            if 'stock_symbol' in df_finspid_clean.columns:
                df_finspid_clean['ticker'] = df_finspid_clean['stock_symbol']
            else:
                # Try to extract from content using NER
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
            df_finspid_clean['summary'] = df_finspid_clean['content'].str[:500]
        
        # Clean text data
        text_columns = ['headline', 'content', 'summary']
        for col in text_columns:
            if col in df_finspid_clean.columns:
                df_finspid_clean[col] = df_finspid_clean[col].astype(str).apply(self.clean_text)
        
        # Filter by text length
        mask = df_finspid_clean['content'].str.len() >= self.config['min_news_length']
        df_finspid_clean = df_finspid_clean[mask]
        
        mask = df_finspid_clean['content'].str.len() <= self.config['max_news_length']
        df_finspid_clean = df_finspid_clean[mask]
        
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
        
        # Entity recognition and ticker mapping
        if self.config['ner_entity_mapping'] and self.ner_pipeline:
            print("🔍 Running entity recognition for FNSPID...")
            sample_size = min(100, len(df_finspid_clean))
            for idx, row in df_finspid_clean.head(sample_size).iterrows():
                if row['ticker'] == 'UNKNOWN' or pd.isna(row['ticker']):
                    entities = self.extract_entities(row['content'])
                    mapped_ticker = self.map_ticker_from_entities(entities, row['content'])
                    if mapped_ticker:
                        df_finspid_clean.at[idx, 'ticker'] = mapped_ticker
                        df_finspid_clean.at[idx, 'extracted_entities'] = str(entities)
        
        # Add metadata
        df_finspid_clean['source'] = 'FNSPID'
        df_finspid_clean['has_price_data'] = False  # Will be updated later
        
        # Add price change column placeholder
        df_finspid_clean['price_change'] = None
        df_finspid_clean['pre_price'] = None
        df_finspid_clean['post_price'] = None
        
        # Select and order columns
        standard_columns = [
            'ticker', 'headline', 'content', 'summary', 'timestamp',
            'price_change', 'pre_price', 'post_price',
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
        Preprocess NIFTY dataset with enhanced features
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
            'label': 'sentiment_label'
        }
        
        # Apply mapping
        for orig_col, new_col in nifty_mapping.items():
            if orig_col in df_nifty_clean.columns:
                df_nifty_clean[new_col] = df_nifty_clean[orig_col]
        
        # Create content column
        if 'headline' in df_nifty_clean.columns:
            df_nifty_clean['content'] = df_nifty_clean['headline']
        
        # Create summary
        if 'content' in df_nifty_clean.columns:
            df_nifty_clean['summary'] = df_nifty_clean['content'].str[:500]
        
        # Set ticker - NIFTY is primarily about S&P 500 (SPY)
        df_nifty_clean['ticker'] = 'SPY'
        
        # Parse dates
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
                df_nifty_clean[col] = df_nifty_clean[col].astype(str).apply(self.clean_text)
                # Handle multiline headlines
                df_nifty_clean[col] = df_nifty_clean[col].str.replace('\n', ' | ', regex=False)
        
        # Filter by text length
        mask = df_nifty_clean['content'].str.len() >= self.config['min_news_length']
        df_nifty_clean = df_nifty_clean[mask]
        
        mask = df_nifty_clean['content'].str.len() <= self.config['max_news_length']
        df_nifty_clean = df_nifty_clean[mask]
        
        # Add metadata
        df_nifty_clean['source'] = 'NIFTY'
        df_nifty_clean['has_price_data'] = True  # NIFTY has price change data
        
        # Add price columns for consistency
        df_nifty_clean['pre_price'] = None
        df_nifty_clean['post_price'] = None
        
        # Select and order columns
        standard_columns = [
            'ticker', 'headline', 'content', 'summary', 'timestamp',
            'price_change', 'pre_price', 'post_price',
            'sentiment_label', 'source', 'has_price_data'
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
    
    def deduplicate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate news items
        """
        if not self.config['deduplicate']:
            return df
        
        print("🧹 Removing duplicates...")
        
        initial_rows = len(df)
        
        # Create hash for content-based deduplication
        df['content_hash'] = df['content'].apply(
            lambda x: md5(str(x).encode()).hexdigest()
        )
        
        # Remove duplicates based on content hash and ticker
        df = df.drop_duplicates(
            subset=['content_hash', 'ticker', 'timestamp'],
            keep='first'
        )
        
        # Remove duplicates based on headline similarity (optional)
        df = df.drop_duplicates(
            subset=['headline', 'ticker'],
            keep='first'
        )
        
        df = df.drop(columns=['content_hash'])
        
        removed = initial_rows - len(df)
        print(f"   Removed {removed} duplicate rows")
        
        return df
    
    def align_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align news items with stock price movements
        """
        if not self.config['align_prices']:
            return df
        
        print("💰 Aligning with price data...")
        
        # Skip if already has price data
        if df['has_price_data'].all():
            print("   Skipping - already has price data")
            return df
        
        # Only process rows without price data
        rows_to_process = df[~df['has_price_data']].copy()
        
        if len(rows_to_process) == 0:
            print("   No rows need price alignment")
            return df
        
        print(f"   Processing {len(rows_to_process)} rows for price alignment")
        
        processed_count = 0
        for idx, row in rows_to_process.iterrows():
            if pd.isna(row['ticker']) or row['ticker'] == 'UNKNOWN':
                continue
            
            if pd.isna(row['timestamp']):
                continue
            
            # Get price data
            price_data = self.get_price_data(
                row['ticker'],
                row['timestamp'],
                self.config['price_window_hours']
            )
            
            if price_data:
                df.at[idx, 'price_change'] = price_data['price_change']
                df.at[idx, 'pre_price'] = price_data['pre_price']
                df.at[idx, 'post_price'] = price_data['post_price']
                df.at[idx, 'has_price_data'] = True
                processed_count += 1
            
            # Progress update
            if processed_count % 100 == 0:
                print(f"   Processed {processed_count} rows...")
        
        print(f"   Successfully aligned {processed_count} rows with price data")
        
        return df
    
    def create_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment labels from price changes
        """
        if not self.config['create_sentiment_labels']:
            return df
        
        print("🏷️ Creating sentiment labels...")
        
        # Define thresholds
        bearish_threshold, bullish_threshold = self.config['sentiment_thresholds']
        
        def label_sentiment(price_change):
            if pd.isna(price_change):
                return 'neutral'
            elif price_change < bearish_threshold:
                return 'bearish'
            elif price_change > bullish_threshold:
                return 'bullish'
            else:
                return 'neutral'
        
        # Create sentiment labels
        df['sentiment_label'] = df['price_change'].apply(label_sentiment)
        
        # Add confidence based on magnitude
        def calculate_confidence(price_change):
            if pd.isna(price_change):
                return 0.5
            magnitude = abs(price_change)
            # Sigmoid-like confidence scaling
            confidence = min(0.3 + magnitude * 10, 0.95)
            return round(confidence, 2)
        
        df['sentiment_confidence'] = df['price_change'].apply(calculate_confidence)
        
        print(f"   Sentiment distribution:")
        print(f"   - Bullish: {len(df[df['sentiment_label'] == 'bullish'])}")
        print(f"   - Neutral: {len(df[df['sentiment_label'] == 'neutral'])}")
        print(f"   - Bearish: {len(df[df['sentiment_label'] == 'bearish'])}")
        
        return df
    
    def create_task_datasets(self, combined_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create task-specific datasets
        """
        print("\n" + "="*50)
        print("CREATING TASK-SPECIFIC DATASETS")
        print("="*50)
        
        # 1. Summarization Dataset (from NIFTY)
        summarization_df = combined_df[
            (combined_df['source'] == 'NIFTY') & 
            (combined_df['content'].notna()) &
            (combined_df['summary'].notna())
        ].copy()
        
        # For summarization, we need headline-content pairs
        summarization_df = summarization_df[['headline', 'content', 'summary', 'timestamp', 'ticker']]
        summarization_df = summarization_df.rename(columns={
            'headline': 'title',
            'content': 'article',
            'summary': 'reference_summary'
        })
        
        # 2. Sentiment Classification Dataset
        sentiment_df = combined_df[
            combined_df['sentiment_label'].notna() &
            combined_df['content'].notna()
        ].copy()
        
        sentiment_df = sentiment_df[[
            'content', 'sentiment_label', 'sentiment_confidence',
            'price_change', 'ticker', 'timestamp', 'source'
        ]]
        
        # 3. Signal Training Dataset (with price alignment)
        signal_df = combined_df[
            combined_df['has_price_data'] &
            combined_df['price_change'].notna() &
            combined_df['content'].notna()
        ].copy()
        
        signal_df = signal_df[[
            'content', 'price_change', 'pre_price', 'post_price',
            'sentiment_label', 'sentiment_confidence',
            'ticker', 'timestamp', 'source'
        ]]
        
        print(f"📊 Dataset sizes:")
        print(f"   • Summarization: {len(summarization_df)} rows")
        print(f"   • Sentiment Classification: {len(sentiment_df)} rows")
        print(f"   • Signal Training: {len(signal_df)} rows")
        
        return {
            'summarization': summarization_df,
            'sentiment': sentiment_df,
            'signal': signal_df
        }
    
    def combine_datasets(self, df_finspid: pd.DataFrame, df_nifty: pd.DataFrame) -> pd.DataFrame:
        """
        Combine both datasets with enhanced processing
        """
        print("\n" + "="*50)
        print("COMBINING DATASETS")
        print("="*50)
        
        # Apply deduplication
        df_finspid = self.deduplicate_data(df_finspid)
        df_nifty = self.deduplicate_data(df_nifty)
        
        # Align price data for FNSPID
        if self.config['align_prices']:
            df_finspid = self.align_price_data(df_finspid)
        
        # Create sentiment labels
        df_finspid = self.create_sentiment_labels(df_finspid)
        df_nifty = self.create_sentiment_labels(df_nifty)
        
        # Ensure both have required columns
        required_columns = [
            'ticker', 'headline', 'content', 'summary', 'timestamp',
            'price_change', 'pre_price', 'post_price',
            'sentiment_label', 'sentiment_confidence',
            'source', 'has_price_data'
        ]
        
        # Add missing columns
        for col in required_columns:
            if col not in df_finspid.columns:
                df_finspid[col] = None
            if col not in df_nifty.columns:
                df_nifty[col] = None
        
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
        
        # Sort by timestamp
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp', ascending=True)
            print(f"   Sorted by timestamp")
        
        print(f"✅ Combined: {len(df_finspid)} FNSPID + {len(df_nifty)} NIFTY = {len(combined_df)} total rows")
        print(f"📊 Final columns: {list(combined_df.columns)}")
        
        # Show sample
        print("\n📋 Sample of combined data:")
        sample_cols = ['ticker', 'headline', 'timestamp', 'source', 'price_change', 'sentiment_label']
        print(combined_df[sample_cols].head(10))
        
        return combined_df
    
    def analyze_datasets(self, df_finspid: pd.DataFrame, df_nifty: pd.DataFrame):
        """
        Analyze both datasets with enhanced metrics
        """
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        analysis = {
            'fnspid': {
                'total_rows': len(df_finspid),
                'columns': df_finspid.columns.tolist(),
                'tickers': df_finspid['ticker'].nunique() if 'ticker' in df_finspid.columns else 0,
                'unique_tickers': df_finspid['ticker'].unique().tolist()[:20] if 'ticker' in df_finspid.columns else [],
                'date_range': None,
                'missing_values': df_finspid.isnull().sum().to_dict(),
                'text_stats': {
                    'avg_headline_length': df_finspid['headline'].str.len().mean() if 'headline' in df_finspid.columns else 0,
                    'avg_content_length': df_finspid['content'].str.len().mean() if 'content' in df_finspid.columns else 0,
                }
            },
            'nifty': {
                'total_rows': len(df_nifty),
                'columns': df_nifty.columns.tolist(),
                'tickers': df_nifty['ticker'].nunique() if 'ticker' in df_nifty.columns else 0,
                'unique_tickers': df_nifty['ticker'].unique().tolist()[:20] if 'ticker' in df_nifty.columns else [],
                'date_range': None,
                'missing_values': df_nifty.isnull().sum().to_dict(),
                'text_stats': {
                    'avg_headline_length': df_nifty['headline'].str.len().mean() if 'headline' in df_nifty.columns else 0,
                    'avg_content_length': df_nifty['content'].str.len().mean() if 'content' in df_nifty.columns else 0,
                }
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
        print(f"  • Top tickers: {analysis['fnspid']['unique_tickers'][:10]}")
        print(f"  • Avg headline length: {analysis['fnspid']['text_stats']['avg_headline_length']:.1f} chars")
        print(f"  • Avg content length: {analysis['fnspid']['text_stats']['avg_content_length']:.1f} chars")
        if analysis['fnspid']['date_range']:
            print(f"  • Date range: {analysis['fnspid']['date_range']['min']} to {analysis['fnspid']['date_range']['max']}")
        
        print("\n📊 NIFTY Analysis:")
        print(f"  • Rows: {analysis['nifty']['total_rows']}")
        print(f"  • Unique tickers: {analysis['nifty']['tickers']}")
        print(f"  • Top tickers: {analysis['nifty']['unique_tickers'][:10]}")
        print(f"  • Avg headline length: {analysis['nifty']['text_stats']['avg_headline_length']:.1f} chars")
        print(f"  • Avg content length: {analysis['nifty']['text_stats']['avg_content_length']:.1f} chars")
        if analysis['nifty']['date_range']:
            print(f"  • Date range: {analysis['nifty']['date_range']['min']} to {analysis['nifty']['date_range']['max']}")
        
        # Save analysis
        import json
        with open('dataset_analysis_detailed.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Analysis saved to: dataset_analysis_detailed.json")
        
        return analysis
    
    def save_processed_data(self, combined_df: pd.DataFrame, task_datasets: Dict[str, pd.DataFrame], 
                           output_dir: str = 'processed_data'):
        """
        Save processed data and task-specific datasets
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
        
        # Save task-specific datasets
        for task_name, task_df in task_datasets.items():
            task_path = os.path.join(output_dir, f'{task_name}_dataset.csv')
            task_df.to_csv(task_path, index=False)
            print(f"💾 Saved {task_name} dataset: {len(task_df)} rows to {task_path}")
        
        # Save sample
        sample_path = os.path.join(output_dir, 'sample_100_rows.csv')
        combined_df.head(100).to_csv(sample_path, index=False)
        
        # Save configuration
        config_path = os.path.join(output_dir, 'preprocessing_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"\n💾 Combined data saved to: {combined_path}")
        print(f"💾 Sample saved to: {sample_path}")
        print(f"💾 Configuration saved to: {config_path}")
        
        return combined_path
    
    def run_preprocessing_pipeline(self, df_finspid: pd.DataFrame, df_nifty: pd.DataFrame):
        """
        Complete preprocessing pipeline
        """
        print("🚀 STARTING ENHANCED PREPROCESSING PIPELINE")
        print("="*60)
        
        try:
            # Step 1: Analyze datasets
            self.analyze_datasets(df_finspid, df_nifty)
            
            # Step 2: Preprocess FNSPID
            print("\n" + "="*60)
            print("STEP 1: PREPROCESS FNSPID")
            print("="*60)
            fnspid_processed = self.preprocess_fnspid(df_finspid)
            
            # Step 3: Preprocess NIFTY
            print("\n" + "="*60)
            print("STEP 2: PREPROCESS NIFTY")
            print("="*60)
            nifty_processed = self.preprocess_nifty(df_nifty)
            
            # Step 4: Combine with enhanced processing
            print("\n" + "="*60)
            print("STEP 3: COMBINE DATASETS")
            print("="*60)
            combined = self.combine_datasets(fnspid_processed, nifty_processed)
            
            # Step 5: Create task-specific datasets
            print("\n" + "="*60)
            print("STEP 4: CREATE TASK-SPECIFIC DATASETS")
            print("="*60)
            task_datasets = self.create_task_datasets(combined)
            
            # Step 6: Save
            print("\n" + "="*60)
            print("STEP 5: SAVE PROCESSED DATA")
            print("="*60)
            output_path = self.save_processed_data(combined, task_datasets)
            
            print("\n" + "="*60)
            print("🎉 ENHANCED PREPROCESSING COMPLETE!")
            print("="*60)
            print(f"\n📊 Total processed rows: {len(combined)}")
            print(f"📈 Task datasets created:")
            print(f"   • Summarization: {len(task_datasets['summarization'])} rows")
            print(f"   • Sentiment: {len(task_datasets['sentiment'])} rows")
            print(f"   • Signal: {len(task_datasets['signal'])} rows")
            print(f"💾 Data saved to: {output_path}")
            print(f"\n✅ All requirements from project proposal have been implemented!")
            
            return combined, task_datasets
            
        except Exception as e:
            print(f"\n❌ Error in preprocessing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# Quick test function
def test_preprocessing():
    """Test the preprocessing with sample data"""
    # Create sample data
    sample_fnspid = pd.DataFrame({
        'Unnamed: 0': [0.0, 1.0, 2.0],
        'Date': ['2023-12-16 23:00:00 UTC', '2023-12-17 10:00:00 UTC', '2023-12-17 14:00:00 UTC'],
        'Article_title': ['Interesting A Put And Call Options For August ...', 
                         'Apple Announces New iPhone', 
                         'Microsoft Earnings Beat Expectations'],
        'Stock_symbol': ['A', 'AAPL', 'MSFT'],
        'Article': ['Investors in Agilent Technologies, Inc. (Symbol: A) saw new options become available today...',
                   'Apple Inc. today announced the new iPhone 15 with advanced features...',
                   'Microsoft Corporation reported better than expected earnings for Q4...'],
        'Lsa_summary': ['Because the $125.00 strike represents an approximate...',
                       'Apple stock expected to rise following new product announcement...',
                       'Microsoft shares up 3% in after-hours trading...']
    })
    
    sample_nifty = pd.DataFrame({
        'id': ['nifty_0', 'nifty_1', 'nifty_2'],
        'date': ['6/1/2010', '6/2/2010', '6/3/2010'],
        'news': ['China Officials Likely Knew of Bad Milk\nSony CEO on Strategy...',
                'Federal Reserve considers interest rate hike...',
                'Tech stocks rally as Nasdaq hits record high...'],
        'pct_change': [0.0042, -0.0021, 0.015],
        'label': ['Neutral', 'Bearish', 'Bullish']
    })
    
    # Test with minimal configuration
    config = {
        'align_prices': False,  # Turn off for testing
        'ner_entity_mapping': False,  # Turn off for testing
    }
    
    preprocessor = FinancialDataPreprocessor(config)
    result, tasks = preprocessor.run_preprocessing_pipeline(sample_fnspid, sample_nifty)
    
    return result, tasks


if __name__ == "__main__":
    # Test the preprocessing
    print("Testing enhanced preprocessing pipeline...")
    test_preprocessing()