"""
Test evaluation metrics with sample data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.evaluation_metrics import FinSightEvaluator
import pandas as pd
import numpy as np

def create_test_data():
    """Create test data for evaluation"""
    
    # Summarization test data
    summarization_data = {
        'references': [
            "Apple Inc. reported quarterly earnings that beat analyst expectations.",
            "The Federal Reserve announced it would maintain current interest rates.",
            "Tesla shares surged after the company announced new battery technology."
        ],
        'candidates': [
            "Apple earnings exceeded expectations this quarter.",
            "Fed keeps interest rates unchanged for now.",
            "Tesla stock rose on battery news announcement."
        ]
    }
    
    # Sentiment test data
    sentiment_data = {
        'true_labels': ['bullish', 'bearish', 'neutral', 'bullish', 'neutral'],
        'predicted_labels': ['bullish', 'neutral', 'neutral', 'bullish', 'bearish'],
        'labels': ['bullish', 'bearish', 'neutral']
    }
    
    # Correlation test data
    correlation_data = {
        'sentiment_scores': [0.8, 0.6, 0.3, -0.2, -0.5, 0.7, 0.4],
        'price_changes': [0.02, 0.01, 0.005, -0.01, -0.02, 0.015, 0.008]
    }
    
    return summarization_data, sentiment_data, correlation_data

def test_comprehensive_evaluation():
    """Test comprehensive evaluation pipeline"""
    print("🧪 Testing comprehensive evaluation...")
    
    evaluator = FinSightEvaluator()
    
    # Create test data
    summarization_data, sentiment_data, correlation_data = create_test_data()
    
    # Run evaluation
    report = evaluator.run_comprehensive_evaluation(
        summarization_data=summarization_data,
        sentiment_data=sentiment_data,
        correlation_data=correlation_data
    )
    
    # Print results
    print("\n📊 EVALUATION RESULTS:")
    print("="*50)
    
    if report['summarization']:
        print("\n📝 SUMMARIZATION METRICS:")
        rouge = report['summarization']
        print(f"  ROUGE-1: {rouge.get('rouge_1_mean', 0):.3f}")
        print(f"  ROUGE-2: {rouge.get('rouge_2_mean', 0):.3f}")
        print(f"  ROUGE-L: {rouge.get('rouge_l_mean', 0):.3f}")
    
    if report['sentiment']:
        print("\n📈 SENTIMENT METRICS:")
        sent = report['sentiment']
        print(f"  Accuracy: {sent.get('accuracy', 0):.3f}")
        print(f"  F1 Score: {sent.get('f1_score', 0):.3f}")
        print(f"  Precision: {sent.get('precision', 0):.3f}")
        print(f"  Recall: {sent.get('recall', 0):.3f}")
    
    if report['correlation']:
        print("\n📊 CORRELATION METRICS:")
        corr = report['correlation']
        print(f"  Pearson r: {corr.get('pearson_r', 0):.3f}")
        print(f"  Spearman r: {corr.get('spearman_r', 0):.3f}")
        print(f"  Interpretation: {corr.get('interpretation', 'N/A')}")
    
    print(f"\n✅ Evaluation report saved to: evaluation_report.json")
    return report

def test_with_your_data():
    """Test with your actual preprocessed data"""
    print("\n📊 Testing with your preprocessed data...")
    
    try:
        # Load your preprocessed data
        data_path = "processed_data/combined_financial_news.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            print(f"📁 Loaded data: {df.shape}")
            
            # Check if we have price change data (for correlation)
            if 'price_change' in df.columns and 'sentiment_label' in df.columns:
                # Create sample correlation data
                sentiment_scores = []
                price_changes = []
                
                # Convert sentiment labels to scores
                for idx, row in df.head(1000).iterrows():  # Use first 1000 rows
                    sentiment = str(row.get('sentiment_label', '')).lower()
                    price_change = row.get('price_change')
                    
                    if sentiment == 'bullish':
                        sentiment_scores.append(1.0)
                    elif sentiment == 'bearish':
                        sentiment_scores.append(-1.0)
                    elif sentiment == 'neutral':
                        sentiment_scores.append(0.0)
                    else:
                        continue
                    
                    if pd.notna(price_change):
                        price_changes.append(float(price_change))
                
                # Test correlation
                if len(sentiment_scores) > 10:
                    evaluator = FinSightEvaluator()
                    corr_result = evaluator.evaluate_correlation(sentiment_scores[:len(price_changes)], price_changes)
                    
                    print(f"\n📈 Real Data Correlation:")
                    print(f"  Pearson r: {corr_result.get('pearson_r', 0):.3f}")
                    print(f"  n_samples: {corr_result.get('n_valid_pairs', 0)}")
            
            return df
            
        else:
            print(f"⚠️ No processed data found at: {data_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("EVALUATION METRICS TESTING")
    print("="*60)
    
    # Test with synthetic data
    report = test_comprehensive_evaluation()
    
    # Test with your actual data
    print("\n" + "="*60)
    your_data = test_with_your_data()