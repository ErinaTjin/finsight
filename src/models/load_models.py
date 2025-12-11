"""
Load and initialize baseline LLM models for FinSight
"""
import logging
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipelines = {}
        logger.info(f"Initializing ModelManager on device: {device}")
    
    def load_finbert(self):
        """Load FinBERT for sentiment analysis"""
        try:
            logger.info("Loading FinBERT model...")
            
            # Create sentiment analysis pipeline
            finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if self.device == "cuda" else -1
            )
            
            self.pipelines["finbert"] = finbert_pipeline
            logger.info("FinBERT loaded successfully")
            return finbert_pipeline
            
        except Exception as e:
            logger.error(f"Error loading FinBERT: {e}")
            return None
    
    def load_summarization_model(self, model_name="facebook/bart-large-cnn"):
        """Load BART model for summarization"""
        try:
            logger.info(f"Loading summarization model: {model_name}")
            
            summarization_pipeline = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            
            self.pipelines["summarization"] = summarization_pipeline
            logger.info("Summarization model loaded successfully")
            return summarization_pipeline
            
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            return None
    
    def get_model(self, model_name):
        """Get a loaded model by name"""
        return self.pipelines.get(model_name)
    
    def list_loaded_models(self):
        """List all loaded models"""
        return list(self.pipelines.keys())

# Singleton instance
model_manager = ModelManager()

def init_models():
    """Initialize all baseline models"""
    logger.info("Initializing baseline models...")
    
    # Load FinBERT for Week 2 sentiment analysis
    model_manager.load_finbert()
    
    # Load BART for summarization
    model_manager.load_summarization_model()
    
    logger.info(f"Models loaded: {model_manager.list_loaded_models()}")
    return model_manager