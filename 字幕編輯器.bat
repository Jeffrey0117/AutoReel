@echo off
chcp 65001 >nul
echo ========================================
echo   字幕編輯器
echo ========================================
echo.
echo 啟動中...會自動開啟瀏覽器
echo 按 Ctrl+C 可停止伺服器
echo.

python subtitle_position_server.py
