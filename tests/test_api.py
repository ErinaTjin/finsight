"""
Test FastAPI endpoints
"""
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "FinSight" in response.json()["message"]

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_summarize():
    """Test summarization endpoint"""
    response = client.post("/summarize", json={
        "text": "Apple reported record earnings this quarter.",
        "max_length": 50
    })
    assert response.status_code == 200
    assert "summary" in response.json()