# Evaluation Metrics Definition - FinSight

## 1. Summarization Quality (ROUGE Metrics)

### Metrics:
- **ROUGE-1**: Unigram overlap between reference and generated summary
- **ROUGE-2**: Bigram overlap between reference and generated summary  
- **ROUGE-L**: Longest common subsequence overlap

### Baseline Targets (Week 2):
- ROUGE-1 > 0.30
- ROUGE-2 > 0.15
- ROUGE-L > 0.25

### Data: NIFTY Dataset
- Reference: Original news headlines
- Candidate: Model-generated summaries

## 2. Sentiment Classification (F1 Score)

### Metrics:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class performance

### Baseline Targets (Week 2):
- Overall F1 > 0.60
- Per-class F1 > 0.55
- Accuracy > 0.65

### Data: FNSPID Dataset (with price direction labels)
- Labels: Bullish (price increase), Bearish (price decrease), Neutral

## 3. Price-Sentiment Correlation (Pearson/Spearman)

### Metrics:
- **Pearson Correlation (r)**: Linear relationship strength
- **Spearman Correlation (ρ)**: Monotonic relationship strength  
- **Rolling Correlation**: Time-varying relationship
- **Statistical Significance (p-value)**: p < 0.05

### Baseline Targets (Week 3):
- Pearson |r| > 0.20
- Spearman |ρ| > 0.15
- Statistically significant (p < 0.05)

### Data: FNSPID + NIFTY combined
- Sentiment scores from model predictions
- Actual price changes from market data

## 4. Implementation Timeline

### Week 1 (Setup):
- ✅ Define metrics in code
- ✅ Create evaluation module
- ✅ Set baseline targets

### Week 2 (Initial Results):
- Implement ROUGE for summarization
- Calculate F1 for sentiment classification
- Establish initial baselines

### Week 3 (Correlation Analysis):
- Implement Pearson/Spearman correlation
- Calculate rolling correlations
- Validate signal effectiveness

### Week 4 (Optimization):
- Improve metrics through model tuning
- Validate with out-of-sample data
- Final performance reporting

## 5. Files Structure
src/utils/evaluation_metrics.py # Main evaluation module
tests/test_evaluation.py # Unit tests
docs/evaluation_metrics.md # Documentation
evaluation_results/ # Output directory


## 6. Dependencies
- rouge-score (for ROUGE metrics)
- scikit-learn (for F1, precision, recall)
- scipy (for Pearson/Spearman correlation)
- pandas/numpy (data manipulation)