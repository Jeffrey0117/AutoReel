@echo off
REM =============================================================================
REM Video Translation Project - Installation Script
REM =============================================================================
REM This script installs all dependencies for the video translation project
REM Run this script in a Command Prompt with administrator privileges if needed
REM =============================================================================

echo.
echo ============================================================
echo   Video Translation Project - Installation Script
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10 or later from https://python.org
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

REM Ask user about GPU support
echo ============================================================
echo   GPU Support Selection
echo ============================================================
echo.
echo Do you have an NVIDIA GPU and want CUDA acceleration?
echo   1. Yes - Install with CUDA 12.1 support (Recommended for RTX 30/40 series)
echo   2. Yes - Install with CUDA 11.8 support (For older GPUs/systems)
echo   3. No  - Install CPU-only version
echo.
set /p GPU_CHOICE="Enter your choice (1/2/3): "

echo.
echo ============================================================
echo   Installing PyTorch
echo ============================================================
echo.

if "%GPU_CHOICE%"=="1" (
    echo [INFO] Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%GPU_CHOICE%"=="2" (
    echo [INFO] Installing PyTorch with CUDA 11.8 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo [INFO] Installing PyTorch CPU-only version...
    pip install torch torchvision torchaudio
)

if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Installing Project Dependencies
echo ============================================================
echo.

REM Install faster-whisper and other dependencies
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Verifying Installation
echo ============================================================
echo.

REM Verify faster-whisper installation
python -c "import faster_whisper; print('[OK] faster-whisper version:', faster_whisper.__version__)" 2>nul
if errorlevel 1 (
    echo [WARNING] faster-whisper not installed correctly
) else (
    echo [OK] faster-whisper installed successfully
)

REM Verify torch installation
python -c "import torch; print('[OK] PyTorch version:', torch.__version__); print('[OK] CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [WARNING] PyTorch not installed correctly
)

REM Verify googletrans installation
python -c "from googletrans import Translator; print('[OK] googletrans installed')" 2>nul
if errorlevel 1 (
    echo [WARNING] googletrans not installed correctly
)

echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Make sure FFmpeg is installed and in your PATH
echo      (winget install ffmpeg)
echo   2. Edit translation_config.json to set your preferences
echo   3. Run: python translate_video.py --help
echo.
pause
