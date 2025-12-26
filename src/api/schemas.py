from pydantic import BaseModel


# -------- Requests --------

class TextRequest(BaseModel):
    text: str


# -------- Responses --------

class SummariseResponse(BaseModel):
    summary: str
    confidence: float


class SentimentResponse(BaseModel):
    sentiment_score: float
    sentiment_label: str
    probability: float


class SignalResponse(BaseModel):
    ticker: str
    window_days: int
    rolling_sentiment: list
    pearson: float
    spearman: float
