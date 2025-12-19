import subprocess
import time

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}")
    print('='*50)
    
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Success!")
        if result.stdout:
            print("Output:", result.stdout[:500])
    else:
        print("Error:", result.stderr)
    
    return result.returncode

def main():
    # Step 1: Train sentiment model
    print("\n1. Training Sentiment Model...")
    run_script("scripts/train_sentiment.py")
    
    # Step 2: Train summarization model
    print("\n2. Training Summarization Model...")
    run_script("scripts/train_summarization.py")
    
    # Step 3: Start API
    print("\n3. Starting FastAPI Server...")
    print("API will be available at http://localhost:8000")
    print("Press Ctrl+C to stop")
    
    # Run API in background
    api_process = subprocess.Popen(
        ['uvicorn', 'src.api.main:app', '--host', '0.0.0.0', '--port', '8000', '--reload']
    )
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_process.terminate()

if __name__ == "__main__":
    main()