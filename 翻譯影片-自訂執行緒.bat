@echo off
chcp 65001 >nul
echo ========================================
echo   影片翻譯工具 (自訂執行緒數量)
echo ========================================
echo.
echo 建議值：
echo   - 有 GPU: 1-2
echo   - 純 CPU: 2-4
echo   - API 很穩: 可以更高
echo.
set /p workers="請輸入執行緒數量 (預設 2): "
if "%workers%"=="" set workers=2
echo.
echo 使用 %workers% 個執行緒處理...
echo.

python translate_video.py --batch --parallel --workers %workers%

echo.
echo ========================================
echo 處理完成！
echo ========================================
pause
