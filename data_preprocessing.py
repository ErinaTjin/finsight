import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from typing import Tuple, Dict
import yaml

class DataPreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the CSV file"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def prepare_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for sentiment classification"""
        # Use available sentiment labels or create from price_change
        if 'sentiment_label' in df.columns:
            df['clean_sentiment'] = df['sentiment_label'].map({
                'positive': 2,
                'neutral': 1,
                'negative': 0
            })
        elif 'price_change' in df.columns:
            # Create sentiment labels from price change
            df['clean_sentiment'] = pd.cut(
                df['price_change'],
                bins=[-np.inf, -0.01, 0.01, np.inf],
                labels=[0, 1, 2]  # negative, neutral, positive
            )
        else:
            raise ValueError("No sentiment labels or price_change column found")
        
        # Preprocess text
        text_col = self.config['data']['text_column']
        df['clean_text'] = df[text_col].apply(self.preprocess_text)
        
        return df[['clean_text', 'clean_sentiment']].dropna()
    
    def prepare_summarization_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for summarization"""
        # Use headline as target, content as source
        df['clean_content'] = df['content'].apply(self.preprocess_text)
        df['clean_headline'] = df['headline'].apply(self.preprocess_text)
        
        return df[['clean_content', 'clean_headline']].dropna()
    
    def split_data(self, df: pd.DataFrame, task: str = 'sentiment') -> Dict:
        """Split data into train/val/test"""
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=42,
            stratify=df['clean_sentiment'] if task == 'sentiment' else None
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['clean_sentiment'] if task == 'sentiment' else None
        )
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }