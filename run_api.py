"""
Run the FinSight API
"""
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    print("🚀 Starting FinSight API...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🏥 Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )