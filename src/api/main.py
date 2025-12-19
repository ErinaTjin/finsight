"""
FinSight FastAPI Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FinSight API",
    description="LLM-Based Real-Time Market Intelligence Co-Pilot",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class NewsItem(BaseModel):
    headline: str
    content: str
    ticker: str
    timestamp: datetime
    source: str

class SentimentRequest(BaseModel):
    text: str
    ticker: Optional[str] = None
    model: str = "finbert"

class SentimentResponse(BaseModel):
    sentiment: str  # bullish/bearish/neutral
    confidence: float
    score: float
    model: str

class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150

class SummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float

class SignalRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class SignalResponse(BaseModel):
    ticker: str
    sentiment_trend: List[float]
    dates: List[str]
    current_sentiment: str
    recommendation: str

# Health check endpoints
@app.get("/")
async def root():
    return {
        "message": "FinSight API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": [
            "/docs - API documentation",
            "/health - Health check",
            "/summarize - News summarization",
            "/sentiment - Sentiment analysis",
            "/signal/{ticker} - Market signals"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "finsight-api"
    }

# Placeholder endpoints (to be implemented in Week 2)
@app.post("/summarize", response_model=SummaryResponse)
async def summarize_news(request: SummaryRequest):
    """Summarize financial news"""
    logger.info(f"Summarizing text of length {len(request.text)}")
    
    # Placeholder - will be replaced with actual LLM in Week 2
    summary = request.text[:min(request.max_length, len(request.text))]
    
    return SummaryResponse(
        summary=summary,
        original_length=len(request.text),
        summary_length=len(summary),
        compression_ratio=len(summary) / len(request.text) if request.text else 0
    )

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of financial text"""
    logger.info(f"Analyzing sentiment for ticker: {request.ticker}")
    
    # Placeholder - will be replaced with FinBERT in Week 2
    return SentimentResponse(
        sentiment="neutral",
        confidence=0.75,
        score=0.0,
        model=request.model
    )

@app.get("/signal/{ticker}", response_model=SignalResponse)
async def get_signal(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get sentiment signals for a ticker"""
    logger.info(f"Getting signals for {ticker}")
    
    # Placeholder - will be replaced with actual analytics in Week 2
    return SignalResponse(
        ticker=ticker,
        sentiment_trend=[0.1, 0.2, 0.15, 0.3, 0.25],
        dates=["2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04", "2023-12-05"],
        current_sentiment="neutral",
        recommendation="hold"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import pipeline
import yaml

app = FastAPI(title="Financial News Analysis API")

# Load models (lazy loading)
sentiment_pipeline = None
summarization_pipeline = None

class SentimentRequest(BaseModel):
    text: str
    model_name: Optional[str] = "finbert"

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 128
    min_length: Optional[int] = 30

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global sentiment_pipeline, summarization_pipeline
    
    # Load sentiment model
    try:
        sentiment_pipeline = pipeline(
            "text-classification",
            model="./models/sentiment",
            tokenizer="./models/sentiment"
        )
    except:
        print("Warning: Sentiment model not found")
    
    # Load summarization model
    try:
        summarization_pipeline = pipeline(
            "summarization",
            model="./models/summarization",
            tokenizer="./models/summarization"
        )
    except:
        print("Warning: Summarization model not found")

@app.get("/")
async def root():
    return {"message": "Financial News Analysis API"}

@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Sentiment model not loaded")
    
    try:
        result = sentiment_pipeline(request.text)[0]
        return {
            "text": request.text,
            "sentiment": result['label'],
            "confidence": result['score']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    if summarization_pipeline is None:
        raise HTTPException(status_code=503, detail="Summarization model not loaded")
    
    try:
        result = summarization_pipeline(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=False
        )[0]['summary_text']
        
        return {
            "original_text": request.text[:200] + "...",
            "summary": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))