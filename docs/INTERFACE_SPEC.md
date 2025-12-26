# Interface Specification (Locked)

This document defines the frozen data and API interfaces for the FinSight system.
Once locked, these interfaces should not be changed without team agreement.

---

## Summarisation Output (erina)

Source: summarization_model.py

Fields:
- ticker (string): Stock ticker
- timestamp (datetime, UTC): News publish time
- original_headline (string): Raw headline
- summary (string): LLM-generated summary
- summary_confidence (float): Confidence score (0–1)

---

## Sentiment Output (erina)

Source: sentiment_classifier.py

Fields:
- ticker (string)
- timestamp (datetime, UTC)
- sentiment_score (float): Range [-1, 1]
- sentiment_label (string): negative | neutral | positive
- probability (float): Model confidence

---

## Signal Analytics Output (shi ying)

Fields:
- ticker (string)
- date (date)
- rolling_sentiment (float)
- window_days (integer)

Correlation Metrics:
- pearson (float)
- spearman (float)
- p_value (float)

---

## API Contracts (zelda)

POST /summarise  
Input: raw news text  
Output: summary, confidence  

POST /sentiment  
Input: raw or summarised text  
Output: sentiment score, label, probability  

GET /signal  
Input: ticker, window  
Output: rolling sentiment index and correlations
