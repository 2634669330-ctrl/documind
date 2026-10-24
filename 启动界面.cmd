@echo off
cd /d "%~dp0"
title DocuMind UI  -  http://localhost:8501
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\work\.venv-documind\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo   Starting web UI, browser will open http://localhost:8501
echo   Make sure the API server window is already running.
echo.

"%PY%" -m streamlit run frontend\app.py --server.port 8501
echo.
pause