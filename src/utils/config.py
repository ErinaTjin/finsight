"""
Configuration management for FinSight
"""
import os
from dotenv import load_dotenv
from typing import Optional, List
import json

# Load environment variables
load_dotenv()

class Settings:
    """Application settings for FinSight"""
    
    def __init__(self):
        # ===================== API SETTINGS =====================
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.API_VERSION = "1.0.0"
        self.API_TITLE = "FinSight API"
        self.API_DESCRIPTION = "LLM-Based Real-Time Market Intelligence Co-Pilot"
        
        # ===================== MODEL SETTINGS =====================
        self.DEFAULT_SENTIMENT_MODEL = os.getenv("DEFAULT_SENTIMENT_MODEL", "ProsusAI/finbert")
        self.DEFAULT_SUMMARIZATION_MODEL = os.getenv("DEFAULT_SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
        self.DEFAULT_EMBEDDINGS_MODEL = os.getenv("DEFAULT_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Model cache and performance
        self.MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "model_cache")
        self.USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
        self.MODEL_DEVICE = "cuda" if self.USE_GPU else "cpu"
        self.MAX_MODEL_WORKERS = int(os.getenv("MAX_MODEL_WORKERS", "2"))
        
        # ===================== FINANCIAL APIs =====================
        self.FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
        self.NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # API rate limits
        self.FINNHUB_RATE_LIMIT = int(os.getenv("FINNHUB_RATE_LIMIT", "60"))  # calls per minute
        self.NEWSAPI_RATE_LIMIT = int(os.getenv("NEWSAPI_RATE_LIMIT", "100"))  # calls per day
        self.ALPHA_VANTAGE_RATE_LIMIT = int(os.getenv("ALPHA_VANTAGE_RATE_LIMIT", "5"))  # calls per minute
        
        # ===================== LLM APIs (Optional) =====================
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        
        # ===================== DATA PATHS =====================
        self.DATA_DIR = os.getenv("DATA_DIR", "data")
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, "raw")
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, "processed")
        self.PROCESSED_DATA_FILE = os.path.join(self.PROCESSED_DATA_DIR, "combined_financial_news.csv")
        
        # Dataset paths
        self.FNSPID_DATA_PATH = os.getenv("FNSPID_DATA_PATH", "nasdaq_external_data.csv")
        self.NIFTY_DATA_PATH = os.getenv("NIFTY_DATA_PATH", "nifty_dataset")
        
        # ===================== EVALUATION SETTINGS =====================
        self.EVALUATION_OUTPUT_DIR = "evaluation_results"
        
        # Summarization Evaluation
        self.ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
        self.SUMMARIZATION_METRICS = ['rouge', 'bleu', 'bertscore']
        
        # Sentiment Evaluation
        self.SENTIMENT_LABELS = ['bullish', 'bearish', 'neutral']
        self.SENTIMENT_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']
        
        # Correlation Evaluation
        self.CORRELATION_WINDOW = int(os.getenv("CORRELATION_WINDOW", "30"))  # days for rolling correlation
        self.CORRELATION_METRICS = ['pearson', 'spearman']
        
        # Baseline Thresholds (Week 2-3)
        self.BASELINE_ROUGE_1 = float(os.getenv("BASELINE_ROUGE_1", "0.3"))
        self.BASELINE_F1 = float(os.getenv("BASELINE_F1", "0.6"))
        self.BASELINE_CORRELATION = float(os.getenv("BASELINE_CORRELATION", "0.2"))
        
        # ===================== PREPROCESSING SETTINGS =====================
        self.PREPROCESS_CHUNK_SIZE = int(os.getenv("PREPROCESS_CHUNK_SIZE", "100000"))
        self.MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "512"))
        self.MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "10"))
        
        # Text cleaning
        self.REMOVE_URLS = os.getenv("REMOVE_URLS", "True").lower() == "true"
        self.REMOVE_SPECIAL_CHARS = os.getenv("REMOVE_SPECIAL_CHARS", "True").lower() == "true"
        self.LOWER_CASE = os.getenv("LOWER_CASE", "True").lower() == "true"
        
        # ===================== LOGGING SETTINGS =====================
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "finsight.log")
        self.LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # ===================== SECURITY SETTINGS =====================
        self.CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))
        self.API_KEY = os.getenv("API_KEY")
        self.REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "False").lower() == "true"
        
        # ===================== CACHE SETTINGS =====================
        self.USE_REDIS = os.getenv("USE_REDIS", "False").lower() == "true"
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # seconds
        
        # ===================== FRONTEND SETTINGS =====================
        self.FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "7860"))
        self.FRONTEND_THEME = os.getenv("FRONTEND_THEME", "dark")
        self.ENABLE_GRADIO = os.getenv("ENABLE_GRADIO", "True").lower() == "true"
        
        # ===================== DEPLOYMENT SETTINGS =====================
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "finsight:latest")
        self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    
    def validate(self):
        """Validate configuration settings"""
        issues = []
        
        # Check required directories
        required_dirs = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_CACHE_DIR,
            self.EVALUATION_OUTPUT_DIR
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                issues.append(f"Directory does not exist: {dir_path}")
        
        # Check API keys if features are enabled
        if self.FINNHUB_API_KEY in [None, "", "your_finnhub_key_here"]:
            issues.append("FINNHUB_API_KEY not set or using default value")
        
        if self.NEWSAPI_API_KEY in [None, "", "your_newsapi_key_here"]:
            issues.append("NEWSAPI_API_KEY not set or using default value")
        
        # Validate port numbers
        if not (1 <= self.API_PORT <= 65535):
            issues.append(f"Invalid API_PORT: {self.API_PORT}")
        
        if not (1 <= self.FRONTEND_PORT <= 65535):
            issues.append(f"Invalid FRONTEND_PORT: {self.FRONTEND_PORT}")
        
        # Validate model paths
        if not self.DEFAULT_SENTIMENT_MODEL:
            issues.append("DEFAULT_SENTIMENT_MODEL not set")
        
        if not self.DEFAULT_SUMMARIZATION_MODEL:
            issues.append("DEFAULT_SUMMARIZATION_MODEL not set")
        
        return issues
    
    def setup_directories(self):
        """Create all required directories"""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_CACHE_DIR,
            self.EVALUATION_OUTPUT_DIR,
            f"{self.EVALUATION_OUTPUT_DIR}/summarization",
            f"{self.EVALUATION_OUTPUT_DIR}/sentiment",
            f"{self.EVALUATION_OUTPUT_DIR}/correlation",
            f"{self.EVALUATION_OUTPUT_DIR}/baseline",
            f"{self.EVALUATION_OUTPUT_DIR}/final",
            "logs",
            "cache",
            "reports"
        ]
        
        created = []
        for dir_path in directories:
            try:
                os.makedirs(dir_path, exist_ok=True)
                created.append(dir_path)
            except Exception as e:
                print(f"Warning: Could not create directory {dir_path}: {e}")
        
        return created
    
    def get_summary(self):
        """Get configuration summary (excluding sensitive data)"""
        return {
            "api": {
                "host": self.API_HOST,
                "port": self.API_PORT,
                "debug": self.DEBUG,
                "version": self.API_VERSION
            },
            "models": {
                "sentiment": self.DEFAULT_SENTIMENT_MODEL,
                "summarization": self.DEFAULT_SUMMARIZATION_MODEL,
                "device": self.MODEL_DEVICE
            },
            "data": {
                "data_dir": self.DATA_DIR,
                "processed_dir": self.PROCESSED_DATA_DIR
            },
            "evaluation": {
                "output_dir": self.EVALUATION_OUTPUT_DIR,
                "baseline_rouge": self.BASELINE_ROUGE_1,
                "baseline_f1": self.BASELINE_F1,
                "baseline_correlation": self.BASELINE_CORRELATION
            },
            "environment": self.ENVIRONMENT
        }
    
    def save_config_summary(self, output_path: str = "config_summary.json"):
        """Save non-sensitive configuration summary to file"""
        summary = self.get_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return output_path


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance"""
    return settings


def init_config():
    """Initialize configuration - create directories, validate settings"""
    print("🔧 Initializing FinSight Configuration...")
    
    # Setup directories
    created = settings.setup_directories()
    print(f"📁 Created directories: {len(created)}")
    
    # Validate configuration
    issues = settings.validate()
    
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"  • {issue}")
        
        if len(issues) > 3:
            print("\n❌ Critical configuration issues. Please fix before proceeding.")
            return False
        else:
            print("\n⚠️ Minor configuration issues. Continuing with defaults...")
    
    # Save configuration summary
    summary_path = settings.save_config_summary()
    print(f"📄 Configuration summary saved to: {summary_path}")
    
    print(f"✅ Configuration initialized successfully!")
    print(f"   Environment: {settings.ENVIRONMENT}")
    print(f"   API: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"   Models: {settings.DEFAULT_SENTIMENT_MODEL}, {settings.DEFAULT_SUMMARIZATION_MODEL}")
    
    return True


# Test the configuration
if __name__ == "__main__":
    print("="*60)
    print("FinSight Configuration Test")
    print("="*60)
    
    success = init_config()
    
    if success:
        print("\n📊 Configuration Summary:")
        summary = settings.get_summary()
        for category, values in summary.items():
            print(f"\n{category.upper()}:")
            for key, value in values.items():
                print(f"  {key}: {value}")
    else:
        print("\n❌ Configuration initialization failed")