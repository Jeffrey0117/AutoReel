@echo off
chcp 65001 >nul
echo ========================================
echo   影片翻譯工具 (多執行緒加速)
echo ========================================
echo.
echo 將以 2 個執行緒並行處理影片
echo.

python translate_video.py --batch --parallel --workers 2

echo.
echo ========================================
echo 處理完成！
echo ========================================
pause
