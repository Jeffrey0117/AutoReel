# 三項功能規格報告

## 問題 1：標題顏色隨機問題

### 現狀分析
- 標題顏色應該從 `random_bg_colors` 陣列隨機選擇
- 用戶反映顏色似乎總是橘色，懷疑隨機功能有問題

### 可能原因
1. **Python random 種子問題**：每次執行 random 可能使用相同種子
2. **陣列順序問題**：橘色 `#ff6600` 在陣列中可能有較高機率被選中
3. **快取問題**：剪映可能快取了舊的顏色值

### 解決方案
1. 檢查 `_update_template_texts()` 中的 `random.choice()` 邏輯
2. 增加更多顏色選項，確保分佈均勻
3. 加入日誌輸出，顯示每次選中的顏色
4. 考慮使用 `random.seed(time.time())` 確保隨機性

### 設定檔位置
- `translation_config.json` → `title_style.random_bg_colors`

---

## 問題 2：字幕批量取代 UI

### 需求描述
- 翻譯後的字幕可能有重複錯字
- 需要一個 UI 介面快速進行「尋找並取代」操作
- 類似文字編輯器的 Find & Replace 功能

### 功能規格
1. **輸入欄位**
   - 尋找文字 (Find)
   - 取代文字 (Replace)

2. **操作按鈕**
   - 「預覽」：顯示會被取代的字幕數量
   - 「全部取代」：執行取代
   - 「取代並儲存」：取代後儲存到草稿

3. **顯示區域**
   - 顯示所有字幕清單（可編輯）
   - 標記匹配的文字
   - 顯示取代前後對比

4. **進階功能**
   - 支援正則表達式 (可選)
   - 支援大小寫敏感切換
   - 批量取代歷史記錄

### 技術實現
- 擴展現有的 `subtitle_position_editor.html`
- 或建立新的 `subtitle_editor.html`
- 新增 API endpoints:
  - `GET /api/subtitles?draft=xxx` - 取得字幕列表
  - `POST /api/subtitles/replace` - 執行取代
  - `POST /api/subtitles/save` - 儲存修改

### 檔案影響
- `subtitle_position_server.py` - 新增 API
- `subtitle_editor.html` - 新 UI 頁面（或擴展現有）

---

## 問題 3：多執行緒批量處理

### 現狀分析
- 目前 `batch_process()` 是單執行緒，逐一處理影片
- 對於大量影片，處理時間很長

### 瓶頸分析
1. **Whisper 語音識別**：CPU/GPU 密集，約 1-5 分鐘/影片
2. **DeepSeek 翻譯**：網路 I/O，受 API 限制
3. **草稿生成**：I/O 密集，相對快速

### 解決方案

#### 方案 A：多進程處理 (multiprocessing)
```python
from multiprocessing import Pool

def batch_process_parallel(self, video_folder, max_workers=4):
    with Pool(max_workers) as pool:
        pool.map(self.process_video, video_files)
```
- **優點**：充分利用多核 CPU
- **缺點**：Whisper 可能有 GPU 競爭問題

#### 方案 B：非同步處理 (asyncio)
```python
import asyncio

async def batch_process_async(self, video_folder, max_workers=4):
    semaphore = asyncio.Semaphore(max_workers)
    tasks = [self.process_video_async(f, semaphore) for f in video_files]
    await asyncio.gather(*tasks)
```
- **優點**：適合 I/O 密集型任務
- **缺點**：需要重構為 async 函數

#### 方案 C：分階段並行
1. **階段 1**：並行執行 Whisper（受 GPU 限制，建議 1-2 並行）
2. **階段 2**：並行執行翻譯（可以 4-8 並行，受 API 限制）
3. **階段 3**：並行生成草稿（可以高並行）

### 推薦方案
- **方案 C（分階段並行）**最適合
- Whisper 用 GPU 時建議單執行緒
- 翻譯 API 可以並行但要注意 rate limit
- 加入進度條顯示（tqdm）

### 設定選項
```json
{
  "parallel": {
    "enabled": true,
    "whisper_workers": 1,
    "translate_workers": 4,
    "draft_workers": 8
  }
}
```

---

## Agent 分工

| Agent | 任務 | 預計工作 |
|-------|------|----------|
| Agent 1 | 標題顏色修復 | 檢查並修復 random 邏輯，加入日誌 |
| Agent 2 | 字幕編輯 UI | 建立取代功能的 HTML + API |
| Agent 3 | 多執行緒處理 | 實現分階段並行處理邏輯 |

---

## 優先級建議

1. **高優先**：問題 1（標題顏色）- 快速修復
2. **中優先**：問題 2（字幕編輯）- 實用功能
3. **低優先**：問題 3（多執行緒）- 效能優化，複雜度高
