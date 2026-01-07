# =============================================================================
# Video Translation Project - PowerShell Installation Script
# =============================================================================
# This script installs all dependencies for the video translation project
# Run: powershell -ExecutionPolicy Bypass -File install.ps1
# =============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Video Translation Project - Installation Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[INFO] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10 or later from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  GPU Support Selection" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Do you have an NVIDIA GPU and want CUDA acceleration?"
Write-Host "  1. Yes - Install with CUDA 12.1 support (Recommended for RTX 30/40 series)"
Write-Host "  2. Yes - Install with CUDA 11.8 support (For older GPUs/systems)"
Write-Host "  3. No  - Install CPU-only version"
Write-Host ""

$gpuChoice = Read-Host "Enter your choice (1/2/3)"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Installing PyTorch" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

switch ($gpuChoice) {
    "1" {
        Write-Host "[INFO] Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
    "2" {
        Write-Host "[INFO] Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    }
    default {
        Write-Host "[INFO] Installing PyTorch CPU-only version..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install PyTorch" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Installing Project Dependencies" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Verifying Installation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verify installations
try {
    $result = python -c "import faster_whisper; print(faster_whisper.__version__)" 2>&1
    Write-Host "[OK] faster-whisper version: $result" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] faster-whisper not installed correctly" -ForegroundColor Yellow
}

try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>&1
    Write-Host "[OK] PyTorch version: $torchVersion" -ForegroundColor Green
    Write-Host "[OK] CUDA available: $cudaAvailable" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] PyTorch not installed correctly" -ForegroundColor Yellow
}

try {
    python -c "from googletrans import Translator" 2>&1
    Write-Host "[OK] googletrans installed" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] googletrans not installed correctly" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Make sure FFmpeg is installed and in your PATH"
Write-Host "     (winget install ffmpeg)"
Write-Host "  2. Edit translation_config.json to set your preferences"
Write-Host "  3. Run: python translate_video.py --help"
Write-Host ""
Read-Host "Press Enter to exit"
