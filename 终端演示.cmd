@echo off
cd /d "%~dp0"
title DocuMind CLI demo
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\work\.venv-documind\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo   Uploading sample docs and asking questions (API must be running)...
echo.
"%PY%" scripts\demo.py
echo.
pause