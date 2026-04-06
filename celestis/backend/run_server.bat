@echo off
title Celestis AI v2 — Backend Server
color 0A
echo.
echo  ███████╗███████╗██╗     ███████╗███████╗████████╗██╗███████╗
echo  ██╔════╝██╔════╝██║     ██╔════╝██╔════╝╚══██╔══╝██║██╔════╝
echo  ██║     █████╗  ██║     █████╗  ███████╗   ██║   ██║███████╗
echo  ██║     ██╔══╝  ██║     ██╔══╝  ╚════██║   ██║   ██║╚════██║
echo  ╚██████╗███████╗███████╗███████╗███████║   ██║   ██║███████║
echo   ╚═════╝╚══════╝╚══════╝╚══════╝╚══════╝   ╚═╝   ╚═╝╚══════╝
echo      AI v2 Backend  ^|  Real-Time Solar Storm Intelligence
echo.

cd /d %~dp0

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo [2/3] Dependencies ready.
echo [3/3] Starting FastAPI server on http://localhost:8000
echo.
echo  Docs available at: http://localhost:8000/docs
echo  Live data at:      http://localhost:8000/api/live
echo.
echo  Keep this window open while using the dashboard.
echo  Press Ctrl+C to stop the server.
echo.

python -m uvicorn api:app --host 127.0.0.1 --port 8001 --reload

pause
