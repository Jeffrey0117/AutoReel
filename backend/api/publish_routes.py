"""
Publish API Routes — IG scheduling endpoints.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from services.ig_publisher import ig_publisher

router = APIRouter(prefix="/api/publish", tags=["publish"])


class CreateScheduleRequest(BaseModel):
    filename: str
    caption: str
    scheduled_at: str  # ISO format


@router.get("/videos")
async def list_videos():
    """List all video files in the videos directory."""
    return ig_publisher.list_videos()


@router.get("/thumbnail/{filename:path}")
async def get_thumbnail(filename: str):
    """Return a thumbnail image for the given video file."""
    thumb = ig_publisher.get_thumbnail_path(filename)
    if not thumb:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(thumb), media_type="image/jpeg")


@router.get("/schedules")
async def list_schedules():
    """List all schedules."""
    return ig_publisher.get_schedules()


@router.post("/schedules")
async def create_schedule(req: CreateScheduleRequest):
    """Create a new publish schedule."""
    video_path = ig_publisher.VIDEOS_DIR / req.filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return ig_publisher.create_schedule(req.filename, req.caption, req.scheduled_at)


@router.post("/schedules/{schedule_id}/retry")
async def retry_schedule(schedule_id: str):
    """Retry a failed schedule (reset and re-run immediately)."""
    entry = ig_publisher.retry_schedule(schedule_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Schedule not found or not retryable")
    return entry


@router.delete("/schedules/{schedule_id}")
async def cancel_schedule(schedule_id: str):
    """Cancel a pending schedule."""
    ok = ig_publisher.cancel_schedule(schedule_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Schedule not found or not cancellable")
    return {"success": True}
