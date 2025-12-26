from fastapi import APIRouter
from .schemas import (
    TextRequest,
    SummariseResponse,
    SentimentResponse,
    SignalResponse,
)

router = APIRouter()


@router.post("/summarise", response_model=SummariseResponse)
def summarise_news(request: TextRequest):
    # TEMP dummy response (real model testing later)
    return {
        "summary": "This is a placeholder summary.",
        "confidence": 0.99
    }


@router.post("/sentiment", response_model=SentimentResponse)
def analyse_sentiment(request: TextRequest):
    # TEMP dummy response (real model wired in Step 4)
    return {
        "sentiment_score": 0.5,
        "sentiment_label": "positive",
        "probability": 0.9
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
