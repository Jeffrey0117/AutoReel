@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: 設定編碼
set PYTHONIOENCODING=utf-8

:: 設定 DeepSeek API Key
:: 請先在系統環境變數中設定 DEEPSEEK_API_KEY
if not defined DEEPSEEK_API_KEY (
    echo [錯誤] 請先設定 DEEPSEEK_API_KEY 環境變數
    echo 設定方式: set DEEPSEEK_API_KEY=your-api-key-here
    echo.
    pause
    exit /b 1
)

echo =====================================
echo   翻譯影片工作流程
echo =====================================
echo.
echo 功能: 自動語音識別 + 翻譯 + 生成剪映字幕
echo 翻譯: DeepSeek API
echo.

python translate_video.py --batch --pipeline

echo.
pause
