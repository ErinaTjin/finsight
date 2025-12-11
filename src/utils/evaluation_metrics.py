"""
Evaluation metrics for FinSight project
ROUGE for summarization, F1 for sentiment, Pearson for correlation
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinSightEvaluator:
    """
    Comprehensive evaluation metrics for FinSight project
    """
    
    def __init__(self):
        print("📊 FinSight Evaluator Initialized")
        
    # ===================== SUMMARIZATION METRICS =====================
    
    def calculate_rouge_simple(self, reference: str, candidate: str) -> Dict:
        """
        Simple ROUGE implementation (for Week 1 testing)
        Will be replaced with proper ROUGE library in Week 2
        """
        # Tokenize into words
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        
        # Calculate overlap
        overlap = len(ref_words.intersection(cand_words))
        
        if len(ref_words) == 0:
            rouge_1 = 0.0
        else:
            rouge_1 = overlap / len(ref_words)
        
        if len(cand_words) == 0:
            rouge_2 = 0.0
        else:
            rouge_2 = overlap / len(cand_words)
        
        rouge_l = 2 * rouge_1 * rouge_2 / (rouge_1 + rouge_2 + 1e-8)
        
        return {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'overlap_words': overlap,
            'reference_length': len(ref_words),
            'candidate_length': len(cand_words)
        }
    
    def evaluate_summarization_batch(self, references: List[str], candidates: List[str]) -> Dict:
        """
        Evaluate summarization on a batch of examples
        """
        if len(references) != len(candidates):
            raise ValueError(f"Mismatched lengths: references={len(references)}, candidates={len(candidates)}")
        
        results = []
        for ref, cand in zip(references, candidates):
            rouge_score = self.calculate_rouge_simple(ref, cand)
            results.append(rouge_score)
        
        # Aggregate results
        aggregated = {
            'rouge_1_mean': np.mean([r['rouge_1'] for r in results]),
            'rouge_2_mean': np.mean([r['rouge_2'] for r in results]),
            'rouge_l_mean': np.mean([r['rouge_l'] for r in results]),
            'rouge_1_std': np.std([r['rouge_1'] for r in results]),
            'n_samples': len(results),
            'individual_scores': results
        }
        
        return aggregated
    
    # ===================== SENTIMENT METRICS =====================
    
    def evaluate_sentiment(self, y_true: List, y_pred: List, labels: Optional[List] = None) -> Dict:
        """
        Evaluate sentiment classification results
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Mismatched lengths: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        # Convert to appropriate format
        y_true = [str(label).lower() for label in y_true]
        y_pred = [str(label).lower() for label in y_pred]
        
        # Get unique labels
        if labels is None:
            labels = sorted(set(y_true + y_pred))
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_metrics = {}
        for label in labels:
            # Binary classification for this label
            y_true_binary = [1 if y == label else 0 for y in y_true]
            y_pred_binary = [1 if y == label else 0 for y in y_pred]
            
            try:
                prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1_class = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                support = sum(y_true_binary)
                
                per_class_metrics[label] = {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1_class,
                    'support': support
                }
            except:
                per_class_metrics[label] = {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0}
        
        # Confusion matrix (simplified)
        confusion = {}
        for true_label in labels:
            confusion[true_label] = {}
            for pred_label in labels:
                count = sum(1 for t, p in zip(y_true, y_pred) if t == true_label and p == pred_label)
                confusion[true_label][pred_label] = count
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class': per_class_metrics,
            'confusion_matrix': confusion,
            'n_samples': len(y_true)
        }
    
    # ===================== CORRELATION METRICS =====================
    
    def evaluate_correlation(self, sentiment_scores: List[float], price_changes: List[float]) -> Dict:
        """
        Evaluate correlation between sentiment scores and price changes
        """
        if len(sentiment_scores) != len(price_changes):
            raise ValueError(f"Mismatched lengths: sentiment={len(sentiment_scores)}, price={len(price_changes)}")
        
        # Remove NaN values
        valid_pairs = [(s, p) for s, p in zip(sentiment_scores, price_changes) 
                      if not (np.isnan(s) or np.isnan(p))]
        
        if len(valid_pairs) < 2:
            return {
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'spearman_r': np.nan,
                'spearman_p': np.nan,
                'n_valid_pairs': len(valid_pairs),
                'message': 'Insufficient valid data points'
            }
        
        s_scores, p_changes = zip(*valid_pairs)
        
        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = pearsonr(s_scores, p_changes)
        
        # Spearman correlation (monotonic relationship)
        spearman_r, spearman_p = spearmanr(s_scores, p_changes)
        
        # Additional statistics
        sentiment_mean = np.mean(s_scores)
        sentiment_std = np.std(s_scores)
        price_mean = np.mean(p_changes)
        price_std = np.std(p_changes)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'sentiment_mean': sentiment_mean,
            'sentiment_std': sentiment_std,
            'price_mean': price_mean,
            'price_std': price_std,
            'n_valid_pairs': len(valid_pairs),
            'interpretation': self._interpret_correlation(pearson_r)
        }
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "No correlation"
        elif abs_r < 0.3:
            return "Weak correlation"
        elif abs_r < 0.5:
            return "Moderate correlation"
        elif abs_r < 0.7:
            return "Strong correlation"
        else:
            return "Very strong correlation"
    
    def rolling_correlation(self, sentiment_series: pd.Series, price_series: pd.Series, 
                           window: int = 30) -> pd.Series:
        """
        Calculate rolling correlation over time
        """
        # Align series and remove NaN
        df = pd.DataFrame({
            'sentiment': sentiment_series,
            'price': price_series
        }).dropna()
        
        # Calculate rolling correlation
        correlations = df['sentiment'].rolling(window=window).corr(df['price'])
        
        return correlations
    
    # ===================== COMPREHENSIVE EVALUATION =====================
    
    def run_comprehensive_evaluation(self, 
                                   summarization_data: Optional[Dict] = None,
                                   sentiment_data: Optional[Dict] = None,
                                   correlation_data: Optional[Dict] = None) -> Dict:
        """
        Run all evaluation metrics and generate report
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'summarization': {},
            'sentiment': {},
            'correlation': {}
        }
        
        # Summarization evaluation
        if summarization_data:
            logger.info("Running summarization evaluation...")
            refs = summarization_data.get('references', [])
            cands = summarization_data.get('candidates', [])
            if refs and cands:
                report['summarization'] = self.evaluate_summarization_batch(refs, cands)
        
        # Sentiment evaluation
        if sentiment_data:
            logger.info("Running sentiment evaluation...")
            y_true = sentiment_data.get('true_labels', [])
            y_pred = sentiment_data.get('predicted_labels', [])
            labels = sentiment_data.get('labels', ['bullish', 'bearish', 'neutral'])
            if y_true and y_pred:
                report['sentiment'] = self.evaluate_sentiment(y_true, y_pred, labels)
        
        # Correlation evaluation
        if correlation_data:
            logger.info("Running correlation evaluation...")
            sentiments = correlation_data.get('sentiment_scores', [])
            prices = correlation_data.get('price_changes', [])
            if sentiments and prices:
                report['correlation'] = self.evaluate_correlation(sentiments, prices)
        
        # Save report
        self.save_evaluation_report(report)
        
        return report
    
    def save_evaluation_report(self, report: Dict, output_path: str = 'evaluation_report.json'):
        """Save evaluation report to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return obj
            
            json.dump(report, f, indent=2, default=convert_types)
        
        logger.info(f"✅ Evaluation report saved to: {output_path}")
        return output_path
    
    def load_evaluation_report(self, report_path: str = 'evaluation_report.json') -> Dict:
        """Load evaluation report from JSON file"""
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        return report


# Singleton instance for easy access
evaluator = FinSightEvaluator()


# ===================== TEST FUNCTIONS =====================

def test_summarization_metrics():
    """Test summarization evaluation metrics"""
    print("🧪 Testing summarization metrics...")
    
    references = [
        "Apple reported record earnings for the fourth quarter",
        "The Federal Reserve increased interest rates by 0.25%",
        "Tesla stock surged after announcing new battery technology"
    ]
    
    candidates = [
        "Apple had great fourth quarter earnings",
        "Fed raised interest rates slightly",
        "Tesla stock rose on battery news"
    ]
    
    results = evaluator.evaluate_summarization_batch(references, candidates)
    print(f"✅ ROUGE Scores: {results['rouge_1_mean']:.3f}, {results['rouge_2_mean']:.3f}")
    return results


def test_sentiment_metrics():
    """Test sentiment evaluation metrics"""
    print("🧪 Testing sentiment metrics...")
    
    true_labels = ['bullish', 'bearish', 'neutral', 'bullish', 'neutral']
    pred_labels = ['bullish', 'neutral', 'neutral', 'bullish', 'bearish']
    
    results = evaluator.evaluate_sentiment(true_labels, pred_labels)
    print(f"✅ F1 Score: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
    return results


def test_correlation_metrics():
    """Test correlation evaluation metrics"""
    print("🧪 Testing correlation metrics...")
    
    sentiment_scores = [0.8, 0.6, 0.3, -0.2, -0.5]
    price_changes = [0.02, 0.01, 0.005, -0.01, -0.02]
    
    results = evaluator.evaluate_correlation(sentiment_scores, price_changes)
    print(f"✅ Pearson r: {results['pearson_r']:.3f}, p: {results['pearson_p']:.3f}")
    return results


def run_all_tests():
    """Run all metric tests"""
    print("="*60)
    print("RUNNING EVALUATION METRICS TESTS")
    print("="*60)
    
    results = {}
    
    try:
        results['summarization'] = test_summarization_metrics()
        results['sentiment'] = test_sentiment_metrics()
        results['correlation'] = test_correlation_metrics()
        
        # Save combined results
        evaluator.save_evaluation_report({
            'test_results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }, 'test_evaluation_report.json')
        
        print("\n✅ All tests completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run tests
    run_all_tests()