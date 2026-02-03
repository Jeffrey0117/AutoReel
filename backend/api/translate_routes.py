"""
Translate API Routes - handles video upload, translation pipeline, URL download,
results listing, and configuration management.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from services.translate_service import translate_service

router = APIRouter(prefix="/api/translate", tags=["translate"])


class StartRequest(BaseModel):
    force: bool = False


class UrlDownloadRequest(BaseModel):
    url: str
    auto_translate: bool = True


class FullProcessRequest(BaseModel):
    url: str
    auto_caption: bool = True


class RenameRequest(BaseModel):
    new_name: str


class ConfigUpdate(BaseModel):
    config: dict


# --- Video Management ---

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept drag-and-drop uploaded video files."""
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''

    if ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    content = await file.read()
    dest = translate_service.save_uploaded_file(file.filename, content)

    return {
        "success": True,
        "filename": file.filename,
        "size": len(content),
        "path": str(dest)
    }


@router.get("/videos")
async def list_videos():
    """List videos in translate_raw with their status."""
    return translate_service.list_videos()


@router.post("/videos/{filename}/rename")
async def rename_video(filename: str, body: RenameRequest):
    """Rename a video in translate_raw."""
    result = translate_service.rename_video(filename, body.new_name)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/videos/{filename}")
async def delete_video(filename: str):
    """Delete a video from translate_raw."""
    if translate_service.delete_video(filename):
        return {"success": True, "filename": filename}
    raise HTTPException(404, f"Video not found: {filename}")


# --- Translation Pipeline ---

@router.post("/start")
async def start_pipeline(body: Optional[StartRequest] = None):
    """Start the translation pipeline (background thread)."""
    force = body.force if body else False

    if translate_service.is_running:
        return {"success": False, "message": "Pipeline already running"}

    started = translate_service.start_pipeline(force=force)
    return {"success": started, "message": "Pipeline started" if started else "Failed to start"}


@router.post("/start/{filename}")
async def start_single(filename: str, body: Optional[StartRequest] = None):
    """Start translation for a single video file."""
    force = body.force if body else False

    if translate_service.is_running:
        return {"success": False, "message": "Pipeline already running"}

    started = translate_service.start_single(filename, force=force)
    if not started:
        return {"success": False, "message": "Video not found or failed to start"}
    return {"success": True, "message": f"Started translating {filename}"}


@router.get("/status")
async def pipeline_status():
    """Get current pipeline status."""
    return translate_service.get_status()


# --- URL Download ---

@router.post("/download-url")
async def download_url(body: UrlDownloadRequest):
    """Download video from URL (yt-dlp), optionally auto-start translation."""
    if translate_service.is_downloading:
        return {"success": False, "message": "Already downloading"}

    started = translate_service.start_url_download(
        url=body.url,
        auto_translate=body.auto_translate
    )
    return {"success": started, "message": "Download started" if started else "Failed to start download"}


@router.post("/process-url")
async def process_url(body: FullProcessRequest):
    """Complete process from URL: Download → Translate → Generate Caption."""
    if translate_service.is_downloading or translate_service.is_running:
        return {"success": False, "message": "Already processing"}

    started = translate_service.start_full_process(
        url=body.url,
        auto_caption=body.auto_caption
    )
    return {"success": started, "message": "Processing started" if started else "Failed to start"}


# --- Results ---

@router.get("/results")
async def list_results():
    """List completed translation results."""
    return translate_service.list_results()


@router.get("/results/{name}/srt")
async def download_srt(name: str):
    """Download SRT file for a given video name."""
    srt_path = translate_service.get_srt_path(name)
    if not srt_path:
        raise HTTPException(404, f"SRT not found for: {name}")

    return FileResponse(
        path=str(srt_path),
        filename=srt_path.name,
        media_type="application/x-subrip"
    )


# --- Configuration ---

@router.get("/config")
async def get_config():
    """Read translation_config.json."""
    return translate_service.get_config()


@router.put("/config")
async def update_config(body: ConfigUpdate):
    """Update translation_config.json."""
    updated = translate_service.update_config(body.config)
    return {"success": True, "config": updated}
