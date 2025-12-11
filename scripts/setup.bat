@echo off
echo ========================================
echo    FinSight Windows Setup Script
echo ========================================

echo Step 1: Creating virtual environment...
python -m venv venv

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Installing dependencies...
pip install --upgrade pip
pip install fastapi uvicorn pandas numpy transformers torch gradio

echo Step 4: Copying environment file...
if not exist .env copy .env.example .env

echo.
echo ========================================
echo    Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Update .env file with your API keys
echo 2. Run: uvicorn src.api.main:app --reload
echo 3. Open: http://localhost:8000/docs
echo.
pause