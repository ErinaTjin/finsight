from fastapi import APIRouter
from .schemas import (
    TextRequest,
    SummariseResponse,
    SentimentResponse,
    SignalResponse,
)

from summarization_model import summarize_text
from sentiment_classifier import predict_sentiment

router = APIRouter()



@router.post("/summarise", response_model=SummariseResponse)
def summarise_news(request: TextRequest):
    summary, confidence = summarize_text(request.text)
    return {
        "summary": summary,
        "confidence": confidence
    }


@router.post("/sentiment", response_model=SentimentResponse)
def analyse_sentiment(request: TextRequest):
    score, label, prob = predict_sentiment(request.text)
    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "probability": prob
    }


@router.get("/signal", response_model=SignalResponse)
def get_signal(ticker: str, window_days: int = 7):
    # TEMP dummy response (real analytics wired in Step 4)
    return {
        "ticker": ticker,
        "window_days": window_days,
        "rolling_sentiment": [0.1, 0.2, 0.3],
        "pearson": 0.42,
        "spearman": 0.38
    }
