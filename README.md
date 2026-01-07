# Video Translate Project

全自動化視頻翻譯字幕系統，將英文影片自動轉錄、翻譯成中文，並生成剪映（JianyingPro）編輯草稿。

## 功能特色

- **高精度語音識別**：使用 faster-whisper 進行英文語音轉文字，支援 GPU 加速
- **智能翻譯**：整合 DeepSeek API 進行英翻中翻譯，支援並行批次處理
- **剪映整合**：自動生成剪映專業版（JianyingPro）可匯入的字幕草稿
- **三種執行模式**：單執行緒、多執行緒、Pipeline 模式，適應不同場景需求
- **批次處理**：支援多檔案批次轉錄與翻譯
- **Web UI 編輯器**：提供網頁介面進行字幕位置調整與 IG 文案編輯
- **自動字幕樣式**：支援自訂字體、顏色、描邊、陰影等樣式設定
- **VAD 過濾**：語音活動偵測，有效過濾靜音片段

## 系統需求

| 項目 | 需求 |
|------|------|
| 作業系統 | Windows 10 / Windows 11 |
| Python | 3.13 或以上 |
| GPU（選用） | NVIDIA 顯示卡，支援 CUDA 12.x |
| 記憶體 | 建議 8GB 以上（使用 large 模型建議 16GB） |
| 硬碟空間 | 至少 5GB（含模型檔案） |

## 安裝步驟

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 安裝 CUDA（GPU 加速，選用）

如需使用 GPU 加速：
1. 前往 [NVIDIA CUDA 下載頁面](https://developer.nvidia.com/cuda-downloads)
2. 下載並安裝 CUDA 12.x 版本
3. 確認安裝成功：`nvcc --version`

### 3. 設定 API Key

設定 DeepSeek API Key 環境變數：

```cmd
set DEEPSEEK_API_KEY=your-api-key-here
```

或在系統環境變數中永久設定 `DEEPSEEK_API_KEY`。

## 快速開始

### 處理單一影片

```bash
python translate_video.py video.mp4
```

### 批次處理

將影片放入 `videos/translate_raw/` 資料夾，然後執行：

```bash
# 使用 Pipeline 模式（推薦）
python translate_video.py --batch --pipeline

# 強制重新處理
python translate_video.py --batch --pipeline --force
```

### 使用批次檔

```bash
翻譯影片-單執行緒.bat    # 單執行緒模式
翻譯影片-多執行緒.bat    # 多執行緒模式
翻譯影片-自訂執行緒.bat  # 自訂執行緒數
```

## 處理流程

```
英文影片 → Whisper 語音識別 → DeepSeek 翻譯 → 生成剪映草稿
```

輸出檔案：
- `subtitles/{影片名}_en.srt` - 英文字幕
- `subtitles/{影片名}_zh.srt` - 中文字幕
- `subtitles/{影片名}.json` - 完整字幕資料
- 剪映草稿資料夾

## 配置說明

編輯 `translation_config.json` 進行設定：

### Whisper 設定

| 參數 | 說明 | 選項 |
|------|------|------|
| model | 模型大小 | tiny, base, small, medium, large-v3 |
| device | 運算裝置 | auto, cuda, cpu |
| engine | 引擎類型 | faster-whisper, openai-whisper |
| compute_type | 運算精度 | float16, int8 |

### 翻譯設定

| 參數 | 說明 |
|------|------|
| provider | 翻譯服務（deepseek, openai, google） |
| api_key_env | API Key 環境變數名稱 |
| target_lang | 目標語言（zh-TW） |

### 並行處理設定

| 參數 | 說明 |
|------|------|
| mode | 執行模式（sequential, parallel, pipeline） |
| translate_workers | 翻譯執行緒數（預設 4） |
| draft_workers | 草稿生成執行緒數（預設 2） |

## 執行模式

### 單執行緒模式
循序執行，穩定性高，適合測試。

### 多執行緒模式
多檔案並行處理，適合大量短影片。

### Pipeline 模式（推薦）
```
影片 1: [轉錄] → [翻譯] → [生成草稿]
影片 2:        [轉錄] → [翻譯] → [生成草稿]
影片 3:               [轉錄] → [翻譯] → ...
```
充分利用 GPU 與 CPU 資源，效率最高。

## 目錄結構

```
video-translate-project/
├── translate_video.py         # 主程式
├── subtitle_generator.py      # 字幕生成模組
├── translation_config.json    # 配置檔案
├── subtitle_position_server.py # 字幕編輯伺服器
├── translate_editor_server.py  # 翻譯編輯器 API
├── pyJianYingDraft/           # 剪映草稿生成模組
├── videos/translate_raw/      # 待處理影片
├── subtitles/                 # 輸出字幕
├── 翻譯專案/                   # 剪映模板
└── docs/                      # 文檔
```

## Web UI 編輯器

### 字幕位置編輯器

```bash
python subtitle_position_server.py
```
開啟瀏覽器訪問 `http://localhost:8766`

### 翻譯編輯器

```bash
python translate_editor_server.py
```

## 常見問題

### CUDA out of memory？
- 改用較小模型（base 或 small）
- 將 `compute_type` 改為 `int8`
- 設定 `device` 為 `cpu`

### 轉錄速度很慢？
- 確認 GPU 正常運作：`python check_gpu.py`
- 檢查 `device` 設定

### 翻譯結果不準確？
- 確認 `DEEPSEEK_API_KEY` 設定正確
- 調整 `max_words_per_segment` 參數

### 剪映無法匯入草稿？
- 確認剪映版本為專業版（JianyingPro）
- 重新啟動剪映讓它重新掃描草稿目錄

## 工具程式

| 程式 | 用途 |
|------|------|
| check_gpu.py | GPU 診斷工具 |
| benchmark_whisper.py | Whisper 效能測試 |
| test_transcription.py | 轉錄測試 |

## 技術棧

- **Python** 3.13+
- **faster-whisper** - CTranslate2 優化的 Whisper
- **DeepSeek API** - 翻譯服務
- **pyJianYingDraft** - 剪映草稿生成

## 授權

本專案僅供學習與個人使用。
