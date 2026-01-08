@echo off
title Video Translate - Web Tools Server
cd /d "%~dp0"
echo ============================================
echo   Video Translate - Web Tools Server
echo ============================================
echo.
echo Starting server (auto port: 8765-8774)
echo Browser will open automatically
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.
python video_rename_server.py
if errorlevel 1 (
    echo.
    echo [Error] Server failed to start
    echo Please make sure Python is installed
    echo.
    pause
)
