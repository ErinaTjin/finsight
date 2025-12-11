# FinSight: LLM-Based Real-Time Market Intelligence Co-Pilot

## 🚀 Week 1: Development Environment Setup Complete!

### What's Set Up:
✅ **FastAPI Backend** with REST endpoints  
✅ **LLM Libraries** (Transformers, FinBERT, BART)  
✅ **Virtual Environment** with all dependencies  
✅ **Git Repository** with proper structure  
✅ **Data Preprocessing Pipeline** (FNSPID + NIFTY)  
✅ **Test Suite** for API and models  

### 📁 Project Structure
finsight-project/
├── src/
│ ├── api/ # FastAPI backend (main.py)
│ ├── models/ # LLM models (load_models.py)
│ ├── utils/ # Utilities (config.py, preprocessor)
│ └── data/ # Data loading
├── processed_data/ # Preprocessed datasets
├── tests/ # Unit tests
├── notebooks/ # Jupyter notebooks
└── scripts/ # Run scripts


### 🛠️ Quick Start

#### 1. Activate Virtual Environment
```bash
.\venv\Scripts\Activate.ps1

#### 2. Run the API
python run_api.py

#### 3. Test Models
python test_models.py

#### 4. Access APIs
API Docs: http://localhost:8000/docs

Health Check: http://localhost:8000/health

# API Endpoints
GET / - Welcome message
GET /health - Health check
POST /summarize - News summarization (Week 2)
POST /sentiment - Sentiment analysis (Week 2)
GET /data/stats - View processed data stats

# Test suite
# Run all tests
pytest tests/
# Run specific tests
pytest tests/test_api.py
pytest tests/test_models.py