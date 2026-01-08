@echo off
title Video Rename Server
cd /d "%~dp0"
echo Starting Video Rename Server...
echo.
python video_rename_server.py
if errorlevel 1 (
    echo.
    echo [Error] Server failed to start
    echo.
    pause
)
