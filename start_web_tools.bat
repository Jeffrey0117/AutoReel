@echo off
title Video Translate - Web Tools Server
cd /d "%~dp0"
echo ============================================
echo   Video Translate - Web Tools Server
echo ============================================
echo.
echo Starting server on http://localhost:8765
echo.
echo Available tools:
echo   - Video Rename: http://localhost:8765/video_rename.html
echo   - Subtitle Position: http://localhost:8765/subtitle_position_editor.html
echo   - Subtitle Editor: http://localhost:8765/subtitle_editor.html
echo   - IG Caption: http://localhost:8765/ig_caption_editor.html
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
