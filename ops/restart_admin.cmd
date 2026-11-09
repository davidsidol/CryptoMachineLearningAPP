@echo off
powershell -NoProfile -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\Apps\CryptoML\restart_app.ps1'"
