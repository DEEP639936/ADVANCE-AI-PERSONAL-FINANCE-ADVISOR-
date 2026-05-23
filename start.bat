@echo off
title AI Personal Finance Advisor v2.0 - INR Edition
echo.
echo  =========================================
echo    AI PERSONAL FINANCE ADVISOR v2.0
echo    INR Edition  ^|  ML-Powered
echo  -----------------------------------------
echo    Demo Login:  demo / demo123
echo    URL:  http://127.0.0.1:5000
echo  =========================================
echo.

echo [*] Installing dependencies...
pip install Flask Flask-CORS Flask-SQLAlchemy bcrypt PyJWT scikit-learn pandas numpy -q

echo [*] Starting server...
cd backend
python app.py
pause
