@echo off
powershell -NoProfile -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\Apps\CryptoML\ops\apply_fixes.ps1'"
