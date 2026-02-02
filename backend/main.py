from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

from models import init_db
from api.routes import router
from api.youtube_routes import router as youtube_router
from api.ig_ytdlp_routes import router as ig_ytdlp_router
from api.translate_routes import router as translate_router
from api.caption_routes import router as caption_router
from api.websocket import manager
from services.downloader import download_service
from services.translate_service import translate_service
from services.caption_service import caption_service

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時初始化資料庫
    init_db()

    # 設定 translate_service 的 event loop 和 WebSocket manager
    translate_service.set_event_loop(asyncio.get_running_loop())
    translate_service.set_ws_manager(manager)

    # 設定 caption_service 的 event loop 和 WebSocket manager
    caption_service.set_event_loop(asyncio.get_running_loop())
    caption_service.set_ws_manager(manager)

    yield
    # 關閉時清理
    if download_service.driver:
        download_service.driver.quit()


app = FastAPI(
    title="AutoReels API",
    description="AutoReels - 翻譯 + 下載服務",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 設定 - 允許本地開發來源
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://localhost:5501",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:5501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊 API 路由
app.include_router(router)
app.include_router(youtube_router)
app.include_router(ig_ytdlp_router)  # IG yt-dlp 備案路由
app.include_router(translate_router)  # 翻譯 API 路由
app.include_router(caption_router)   # IG 文案 API 路由

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
