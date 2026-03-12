from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import os

# Load .env from project root
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

from models import init_db
from api.routes import router
from api.youtube_routes import router as youtube_router
from api.ig_ytdlp_routes import router as ig_ytdlp_router
from api.translate_routes import router as translate_router
from api.caption_routes import router as caption_router
from api.publish_routes import router as publish_router
from api.draft_routes import router as draft_router
from api.websocket import manager
from services.downloader import download_service
from services.translate_service import translate_service
from services.caption_service import caption_service
from services.ig_publisher import ig_publisher
from services.telegram_bot import telegram_bot_service
from services.telegram_notifications import TelegramNotifier

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時初始化資料庫
    init_db()

    loop = asyncio.get_running_loop()

    # 設定 translate_service 的 event loop 和 WebSocket manager
    translate_service.set_event_loop(loop)
    translate_service.set_ws_manager(manager)

    # 設定 caption_service 的 event loop 和 WebSocket manager
    caption_service.set_event_loop(loop)
    caption_service.set_ws_manager(manager)

    # 設定 ig_publisher 的 event loop 和 WebSocket manager
    ig_publisher.set_event_loop(loop)
    ig_publisher.set_ws_manager(manager)

    # Telegram Bot
    print(f"[main] TELEGRAM_BOT_TOKEN={'SET' if os.environ.get('TELEGRAM_BOT_TOKEN') else 'NOT SET'}")
    telegram_bot_service.set_event_loop(loop)
    telegram_bot_service.set_ws_manager(manager)
    await telegram_bot_service.start()
    print(f"[main] Telegram bot running: {telegram_bot_service._is_running}")

    if telegram_bot_service._is_running:
        notifier = TelegramNotifier(telegram_bot_service)
        telegram_bot_service.notifier = notifier
        manager.add_listener(notifier.on_broadcast)

    yield
    # 關閉時清理
    await telegram_bot_service.stop()
    if download_service.driver:
        download_service.driver.quit()
    # 關閉 IG 發文瀏覽器
    await ig_publisher.close()


app = FastAPI(
    title="AutoReels API",
    description="AutoReels - 翻譯 + 下載服務",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 設定 - 允許本地開發來源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源（開發用）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "autoreel"}


# 註冊 API 路由
app.include_router(router)
app.include_router(youtube_router)
app.include_router(ig_ytdlp_router)  # IG yt-dlp 備案路由
app.include_router(translate_router)  # 翻譯 API 路由
app.include_router(caption_router)   # IG 文案 API 路由
app.include_router(publish_router)   # IG 發文 API 路由
app.include_router(draft_router)    # 剪映草稿操作 API 路由

# MCP Server — 自動把所有 FastAPI routes 變 MCP tools
try:
    from fastapi_mcp import FastApiMCP
    mcp = FastApiMCP(
        app,
        name="AutoReel",
        description="Video translation + JianYing draft automation",
    )
    mcp.mount()
except ImportError:
    pass  # fastapi-mcp not installed

# 靜態檔案 - CSS
styles_dir = PROJECT_ROOT / "styles"
if styles_dir.exists():
    app.mount("/styles", StaticFiles(directory=str(styles_dir)), name="styles")


# WebSocket 端點
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 可以處理來自前端的訊息
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# 下載控制端點
@app.post("/api/download/start")
async def start_download():
    """開始下載佇列"""
    status = download_service.get_status()
    if status["is_running"]:
        return {"success": False, "message": "Already running", **status}

    # 在背景執行下載
    asyncio.create_task(download_service.start_downloads())
    return {"success": True, "message": "Download started", "status": "running"}


@app.post("/api/download/stop")
async def stop_download():
    """停止下載"""
    result = await download_service.stop_downloads()
    return result


@app.get("/api/download/status")
async def download_status():
    """取得下載服務詳細狀態"""
    return download_service.get_status()


# --- 系統工具 ---

@app.post("/api/system/open-folder")
async def open_folder(body: dict):
    """用系統檔案管理器開啟資料夾"""
    import subprocess
    import platform

    folder_path = body.get("path", "")
    if not folder_path:
        return {"success": False, "error": "No path provided"}

    # 將相對路徑轉為絕對路徑
    folder = Path(folder_path)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder_path

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    try:
        system = platform.system()
        if system == "Windows":
            subprocess.Popen(["explorer", str(folder)])
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", str(folder)])
        else:  # Linux
            subprocess.Popen(["xdg-open", str(folder)])
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- 靜態 HTML 頁面 ---

@app.get("/")
async def serve_index():
    """Serve app.html as the main page."""
    html_file = PROJECT_ROOT / "app.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")
    return {"error": "app.html not found"}


@app.get("/{name}.html")
async def serve_html(name: str):
    """Serve any .html file from project root."""
    html_file = PROJECT_ROOT / f"{name}.html"
    if html_file.exists():
        return FileResponse(str(html_file), media_type="text/html")
    return {"error": f"{name}.html not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
