# AI Personal Finance Advisor - Setup Script
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AI Personal Finance Advisor - Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check location
if (-not (Test-Path "backend/app.py")) {
    Write-Host "ERROR: Run this from project root!" -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

Write-Host "[2/3] Installing dependencies..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

Write-Host "[3/3] Starting server..." -ForegroundColor Yellow
Set-Location backend
python app.py
