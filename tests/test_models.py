"""
Test model loading
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.load_models import ModelManager

def test_model_manager():
    """Test ModelManager initialization"""
    manager = ModelManager(device="cpu")
    assert manager.device == "cpu"
    assert isinstance(manager.pipelines, dict)

def test_finbert_loading():
    """Test FinBERT loading"""
    manager = ModelManager(device="cpu")
    pipeline = manager.load_finbert()
    
    if pipeline is not None:
        # Test with sample text
        result = pipeline("Stock market rises today")
        assert isinstance(result, list)
        assert len(result) > 0