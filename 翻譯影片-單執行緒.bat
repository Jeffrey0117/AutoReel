@echo off
chcp 65001 >nul
echo ========================================
echo   影片翻譯工具 (單執行緒)
echo ========================================
echo.
echo 將處理 videos/translate_raw 資料夾中的影片
echo.

python translate_video.py --batch

echo.
echo ========================================
echo 處理完成！
echo ========================================
pause
