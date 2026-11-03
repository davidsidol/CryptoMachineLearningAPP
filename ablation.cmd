@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe ablation.py > logs\ablation_run.log 2>&1
