"""
Test model loading
"""
from src.models.load_models import init_models

print("🧪 Testing model loading...")
models = init_models()

print(f"\n✅ Loaded models: {models.list_loaded_models()}")

# Test FinBERT if loaded
if "finbert" in models.pipelines:
    print("\n📊 Testing FinBERT with sample texts:")
    test_results = models.test_finbert()
    
    if test_results:
        for result in test_results:
            print(f"\nText: {result['text']}")
            print(f"Result: {result['result']}")