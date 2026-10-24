@echo off
cd /d "%~dp0"
title DocuMind API  -  http://127.0.0.1:8000
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\work\.venv-documind\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo  ============================================================
echo   DocuMind API server  (keep this window open)
echo.
echo   Swagger UI : http://127.0.0.1:8000/docs
echo   Health     : http://127.0.0.1:8000/health
echo   Stop       : Ctrl + C
echo  ============================================================
echo.

"%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
echo.
echo  server stopped.
pause