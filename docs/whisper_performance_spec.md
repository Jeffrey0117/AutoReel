# Whisper 效能優化規格書

## 概述

本文檔定義 video-translate-project 中 Whisper 語音識別模組的效能優化方案，目標是將轉錄速度提升 **4-8 倍**。

---

## 現況分析

### 當前配置 (`translation_config.json`)

```json
{
  "whisper": {
    "model": "base",
    "language": "en",
    "device": "auto",
    "max_words_per_segment": 8
  }
}
```

### 效能瓶頸

| 瓶頸類型 | 說明 | 影響程度 |
|---------|------|---------|
| 模型載入 | 每次處理重新載入模型 | 中 |
| 推論速度 | openai-whisper 未優化 | **高** |
| GPU 利用率 | 未使用 INT8/FP16 量化 | 高 |
| 批次處理 | 逐檔處理，無並行 | 中 |
| I/O 等待 | 翻譯 API 等待時間 | 中 |

### 基準效能 (估算)

| 影片長度 | 當前耗時 (CPU) | 當前耗時 (GPU) |
|---------|---------------|---------------|
| 1 分鐘 | ~60 秒 | ~15 秒 |
| 5 分鐘 | ~300 秒 | ~75 秒 |
| 10 分鐘 | ~600 秒 | ~150 秒 |

---

## 優化方案

### 方案一：faster-whisper 遷移 (推薦)

#### 技術原理

`faster-whisper` 使用 **CTranslate2** 引擎，相比 openai-whisper：

- **4x 更快** 的推論速度
- **2x 更少** 的記憶體使用
- 支援 **INT8 量化** (CPU/GPU)
- 支援 **批次處理**

#### 效能對比

| 指標 | openai-whisper | faster-whisper | 提升 |
|------|---------------|----------------|------|
| 推論速度 | 1x | 4-8x | 🚀 |
| VRAM 使用 | 100% | ~50% | ✅ |
| CPU 效能 | 慢 | INT8 加速 | ✅ |
| 首次載入 | 慢 | 快 | ✅ |

#### API 對比

**現有代碼 (openai-whisper):**
```python
import whisper
model = whisper.load_model("base", device="cuda")
result = model.transcribe(
    video_path,
    language="en",
    task="transcribe",
    verbose=False,
    word_timestamps=True
)
```

**新代碼 (faster-whisper):**
```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "base",
    device="cuda",
    compute_type="float16"  # 或 "int8" for CPU
)

segments, info = model.transcribe(
    video_path,
    language="en",
    task="transcribe",
    word_timestamps=True,
    vad_filter=True,  # 語音活動偵測，跳過靜音
    vad_parameters=dict(min_silence_duration_ms=500)
)

# 注意：segments 是 generator，需要迭代
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

#### 安裝需求

```bash
# 移除舊版
pip uninstall openai-whisper

# 安裝 faster-whisper
pip install faster-whisper

# GPU 支援 (需要 cuDNN 和 cuBLAS)
# Windows: 自動從 PyPI 下載 CUDA 庫
# Linux: 需要安裝 CUDA toolkit
```

#### compute_type 選項

| compute_type | 設備 | 速度 | 精度 | VRAM |
|-------------|------|------|------|------|
| `float32` | CPU/GPU | 最慢 | 最高 | 高 |
| `float16` | GPU | 快 | 高 | 中 |
| `int8_float16` | GPU | 更快 | 中 | 低 |
| `int8` | CPU/GPU | 最快 | 中 | 最低 |

**推薦配置：**
- GPU: `float16` 或 `int8_float16`
- CPU: `int8`

---

### 方案二：GPU 優化

#### 2.1 確保 CUDA 啟用

```python
# 檢測腳本
import torch

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

#### 2.2 Windows CUDA 安裝

```bash
# 安裝 CUDA 版本 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 驗證安裝
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2.3 GPU 記憶體管理

```python
import torch

# 清理 GPU 記憶體
torch.cuda.empty_cache()

# 設定記憶體分配器
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

#### 2.4 模型選擇 vs VRAM

| 模型 | 參數量 | VRAM 需求 | 速度 | 準確度 |
|------|-------|----------|------|-------|
| tiny | 39M | ~1 GB | 最快 | 低 |
| base | 74M | ~1.5 GB | 快 | 中 |
| small | 244M | ~2.5 GB | 中 | 高 |
| medium | 769M | ~5 GB | 慢 | 更高 |
| large-v3 | 1550M | ~10 GB | 最慢 | 最高 |

---

### 方案三：並行處理架構

#### 3.1 處理流程分析

```
┌─────────────────────────────────────────────────────────────┐
│                      處理流程 (當前)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Video 1: [===轉錄===][==翻譯==][=草稿=]                      │
│  Video 2:                       [===轉錄===][==翻譯==][=草稿=]│
│  Video 3:                                            [===... │
│                                                              │
│  時間軸 ───────────────────────────────────────────────────> │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      處理流程 (優化後)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  轉錄 (GPU): [V1][V2][V3]  ← 循序 (GPU 資源獨佔)              │
│  翻譯 (API): [V1][V2][V3]  ← 並行 (I/O bound)                │
│  草稿 (I/O): [V1][V2][V3]  ← 並行 (I/O bound)                │
│                                                              │
│  Pipeline:                                                   │
│  V1: [轉錄]                                                  │
│  V2:       [轉錄]                                            │
│  V1:       [翻譯]                                            │
│  V3:             [轉錄]                                      │
│  V2:             [翻譯]                                      │
│  V1:             [草稿]                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2 Pipeline 架構

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from dataclasses import dataclass
from typing import Optional
import threading

@dataclass
class VideoTask:
    video_path: str
    subtitles: Optional[list] = None
    translated: Optional[list] = None
    draft_path: Optional[str] = None
    status: str = "pending"

class TranscriptionPipeline:
    """
    三階段 Pipeline 架構：
    1. 轉錄 (GPU-bound): 循序處理，避免 GPU 競爭
    2. 翻譯 (I/O-bound): 並行處理，最大化 API 吞吐量
    3. 草稿生成 (I/O-bound): 並行處理
    """

    def __init__(self, config: dict):
        self.config = config
        self.transcribe_queue = Queue()
        self.translate_queue = Queue()
        self.draft_queue = Queue()

        # 翻譯和草稿生成可並行
        self.translate_workers = config.get("parallel", {}).get("translate_workers", 4)
        self.draft_workers = config.get("parallel", {}).get("draft_workers", 2)

    async def process_batch(self, video_paths: list) -> list:
        """Pipeline 批次處理"""

        tasks = [VideoTask(path) for path in video_paths]

        # Stage 1: 轉錄 (循序)
        for task in tasks:
            task.subtitles = await self._transcribe(task.video_path)
            task.status = "transcribed"
            self.translate_queue.put(task)

        # Stage 2: 翻譯 (並行)
        with ThreadPoolExecutor(max_workers=self.translate_workers) as executor:
            translate_futures = []
            while not self.translate_queue.empty():
                task = self.translate_queue.get()
                future = executor.submit(self._translate_sync, task)
                translate_futures.append(future)

            for future in translate_futures:
                task = future.result()
                task.status = "translated"
                self.draft_queue.put(task)

        # Stage 3: 草稿生成 (並行)
        with ThreadPoolExecutor(max_workers=self.draft_workers) as executor:
            draft_futures = []
            while not self.draft_queue.empty():
                task = self.draft_queue.get()
                future = executor.submit(self._generate_draft_sync, task)
                draft_futures.append(future)

            results = [future.result() for future in draft_futures]

        return results
```

#### 3.3 並行配置

```json
{
  "parallel": {
    "enabled": true,
    "mode": "pipeline",
    "transcribe_workers": 1,
    "translate_workers": 4,
    "draft_workers": 2,
    "max_concurrent_videos": 8
  }
}
```

#### 3.4 工作類型分析

| 工作類型 | 資源瓶頸 | 並行策略 | 建議 Workers |
|---------|---------|---------|-------------|
| 轉錄 | GPU/CPU | 循序 | 1 |
| 翻譯 | Network I/O | 高度並行 | 4-8 |
| 草稿生成 | Disk I/O | 中度並行 | 2-4 |

---

## 實作規格

### 配置結構更新

```json
{
  "whisper": {
    "engine": "faster-whisper",
    "model": "base",
    "language": "en",
    "device": "auto",
    "compute_type": "float16",
    "max_words_per_segment": 8,
    "vad_filter": true,
    "vad_parameters": {
      "min_silence_duration_ms": 500,
      "speech_pad_ms": 400
    }
  },
  "parallel": {
    "enabled": true,
    "mode": "pipeline",
    "translate_workers": 4,
    "draft_workers": 2
  },
  "performance": {
    "cache_model": true,
    "batch_size": 16,
    "prefetch_videos": 2
  }
}
```

### 模組更新

#### `subtitle_generator.py` 修改

```python
class SubtitleGenerator:
    """支援 openai-whisper 和 faster-whisper 雙引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.engine = config.get("whisper", {}).get("engine", "openai-whisper")
        self.model = None

    def _load_model(self):
        """延遲載入模型"""
        if self.model is not None:
            return self.model

        whisper_config = self.config.get("whisper", {})
        model_name = whisper_config.get("model", "base")
        device = self._get_device()

        if self.engine == "faster-whisper":
            from faster_whisper import WhisperModel
            compute_type = whisper_config.get("compute_type", "float16")
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        else:
            import whisper
            self.model = whisper.load_model(model_name, device=device)

        return self.model

    def _get_device(self) -> str:
        """自動偵測最佳設備"""
        device = self.config.get("whisper", {}).get("device", "auto")
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def transcribe(self, video_path: str) -> list:
        """轉錄影片"""
        model = self._load_model()
        whisper_config = self.config.get("whisper", {})

        if self.engine == "faster-whisper":
            return self._transcribe_faster(model, video_path, whisper_config)
        else:
            return self._transcribe_openai(model, video_path, whisper_config)

    def _transcribe_faster(self, model, video_path: str, config: dict) -> list:
        """使用 faster-whisper 轉錄"""
        segments, info = model.transcribe(
            video_path,
            language=config.get("language", "en"),
            task="transcribe",
            word_timestamps=True,
            vad_filter=config.get("vad_filter", True),
            vad_parameters=config.get("vad_parameters", {})
        )

        entries = []
        for segment in segments:
            # 處理 word-level timestamps
            words = segment.words or []
            entry = SubtitleEntry(
                index=len(entries) + 1,
                start_time=segment.start,
                end_time=segment.end,
                text_original=segment.text.strip(),
                text_translated=""
            )
            entries.append(entry)

        return self._split_long_segments(entries, config.get("max_words_per_segment", 8))
```

---

## 效能預期

### 優化後效能估算

| 影片長度 | 當前耗時 | 優化後 (GPU) | 優化後 (CPU) |
|---------|---------|-------------|-------------|
| 1 分鐘 | ~60 秒 | ~5 秒 | ~15 秒 |
| 5 分鐘 | ~300 秒 | ~20 秒 | ~60 秒 |
| 10 分鐘 | ~600 秒 | ~40 秒 | ~120 秒 |

### 批次處理效能 (10 個 5 分鐘影片)

| 模式 | 耗時 | 說明 |
|------|------|------|
| 當前循序 | ~50 分鐘 | 每個影片 5 分鐘 |
| Pipeline 並行 | ~15 分鐘 | 轉錄循序 + 翻譯/草稿並行 |
| faster-whisper + Pipeline | ~5 分鐘 | 全面優化 |

---

## 實作階段

### Phase 1: faster-whisper 整合
- [ ] 安裝 faster-whisper 依賴
- [ ] 更新 SubtitleGenerator 支援雙引擎
- [ ] 更新配置檔格式
- [ ] 效能基準測試

### Phase 2: GPU 優化
- [ ] CUDA 環境檢測腳本
- [ ] compute_type 自動選擇
- [ ] VRAM 監控和管理
- [ ] 回退機制 (GPU 失敗時用 CPU)

### Phase 3: Pipeline 並行
- [ ] TranscriptionPipeline 類實作
- [ ] Queue-based 工作分派
- [ ] 進度追蹤和回報
- [ ] 錯誤處理和重試

### Phase 4: 測試與調優
- [ ] 效能基準測試腳本
- [ ] 不同硬體配置測試
- [ ] 最佳參數調優
- [ ] 文檔更新

---

## 相容性考量

### 向後相容

```json
{
  "whisper": {
    "engine": "openai-whisper"  // 保持原有行為
  }
}
```

### 漸進式遷移

1. 先部署 faster-whisper，設為可選
2. 收集效能數據
3. 確認穩定後設為預設
4. 棄用 openai-whisper 支援

---

## 依賴更新

```txt
# requirements_translation.txt (更新)

# 語音識別 (二選一)
# openai-whisper>=20231117  # 舊版，保留相容
faster-whisper>=1.0.0        # 新版，推薦

# GPU 支援
torch>=2.0.0
# Windows: pip install torch --index-url https://download.pytorch.org/whl/cu121

# 翻譯 API
openai>=1.0.0
```

---

## 附錄

### A. 效能測試腳本

```python
# benchmark_whisper.py
import time
from pathlib import Path

def benchmark_whisper(video_path: str, engine: str = "faster-whisper"):
    """效能基準測試"""
    from subtitle_generator import SubtitleGenerator

    config = {
        "whisper": {
            "engine": engine,
            "model": "base",
            "device": "auto",
            "compute_type": "float16"
        }
    }

    generator = SubtitleGenerator(config)

    # 預熱
    print("Warming up...")
    _ = generator.transcribe(video_path)

    # 正式測試
    print("Benchmarking...")
    times = []
    for i in range(3):
        start = time.time()
        _ = generator.transcribe(video_path)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    avg_time = sum(times) / len(times)
    print(f"\nAverage: {avg_time:.2f}s")
    return avg_time

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    benchmark_whisper(video)
```

### B. GPU 檢測腳本

```python
# check_gpu.py
def check_gpu_support():
    """檢測 GPU 支援狀態"""
    print("=" * 50)
    print("GPU Support Check")
    print("=" * 50)

    # PyTorch CUDA
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
            print(f"Compute capability: {props.major}.{props.minor}")
    except ImportError:
        print("PyTorch not installed")

    # CTranslate2 (faster-whisper backend)
    try:
        import ctranslate2
        print(f"\nCTranslate2: {ctranslate2.__version__}")
        print(f"CUDA support: {ctranslate2.get_cuda_device_count() > 0}")
    except ImportError:
        print("\nCTranslate2 not installed")

    print("=" * 50)

if __name__ == "__main__":
    check_gpu_support()
```

---

*文件版本: v1.0*
*建立日期: 2024-12-30*
*作者: Claude Code*
