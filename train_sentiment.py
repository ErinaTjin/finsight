import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.sentiment_classifier import SentimentTrainer
import pandas as pd

def main():
    # Load and preprocess data
    preprocessor = DataPreprocessor("config.yaml")
    
    # Load your CSV file
    df = preprocessor.load_data("data/raw/combined_financial_news.csv")
    
    # Prepare sentiment data
    sentiment_df = preprocessor.prepare_sentiment_data(df)
    
    # Split data
    splits = preprocessor.split_data(sentiment_df, task='sentiment')
    
    # Initialize trainer
    trainer = SentimentTrainer(model_name="yiyanghkust/finbert-tone")
    
    # Preprocess datasets
    train_dataset = trainer.preprocess_data(splits['train'])
    val_dataset = trainer.preprocess_data(splits['val'])
    
    # Train
    trainer.train(train_dataset, val_dataset, output_dir="./models/sentiment")
    
    print("Sentiment model training complete!")

if __name__ == "__main__":
    main()