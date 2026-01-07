@echo off
title Video Translate Studio
cd /d "%~dp0"
python -m gui.main
if errorlevel 1 (
    echo.
    echo [Error] GUI failed to start
    echo Please install customtkinter:
    echo   pip install customtkinter
    echo.
    pause
)
