@echo off
echo ================================================
echo   Crypto ML Forecasting Dashboard
echo   http://localhost:8080
echo ================================================
cd /d "%~dp0"
python -W ignore start_app.py
pause
