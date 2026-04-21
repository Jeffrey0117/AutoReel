"""
IG Publisher Service — APScheduler-based scheduling engine.
Manages IG post schedules with JSON persistence, subprocess execution,
and WebSocket broadcasting for real-time status updates.
"""

import asyncio
import json
import os
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


class IgPublisher:
    DATA_FILE = PROJECT_ROOT / "data" / "ig_schedules.json"
    VIDEOS_DIR = Path(os.path.expanduser("~")) / "Videos"
    THUMBNAIL_DIR = PROJECT_ROOT / "data" / "thumbnails"
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_manager = None
        self._scheduler: Optional[BackgroundScheduler] = None
        self._schedules: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    # --- Lifecycle ---

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def set_ws_manager(self, manager):
        self._ws_manager = manager

    def start(self):
        """Start scheduler and restore pending jobs."""
        self._scheduler = BackgroundScheduler()
        self._scheduler.start()
        self._load_data()
        self._restore_pending_jobs()
        print(f"[ig_publisher] started, {len(self._schedules)} schedules loaded")

    def shutdown(self):
        """Shutdown scheduler gracefully."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            print("[ig_publisher] scheduler shut down")

    async def close(self):
        """Async cleanup (called from lifespan)."""
        pass

    # --- Public API ---

    def list_videos(self) -> List[Dict[str, Any]]:
        """Scan VIDEOS_DIR for video files."""
        if not self.VIDEOS_DIR.exists():
            return []

        videos = []
        for f in sorted(self.VIDEOS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file() and f.suffix.lower() in self.VIDEO_EXTENSIONS:
                stat = f.stat()
                videos.append({
                    "filename": f.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 1),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
        return videos

    def create_schedule(self, filename: str, caption: str, scheduled_at: str) -> Dict[str, Any]:
        """Create a new schedule and register APScheduler job."""
        schedule_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        run_time = datetime.fromisoformat(scheduled_at)

        entry = {
            "id": schedule_id,
            "filename": filename,
            "caption": caption,
            "scheduled_at": scheduled_at,
            "status": "pending",
            "created_at": now,
            "result": None,
            "error": None,
        }

        with self._lock:
            self._schedules.append(entry)
            self._save_data()

        self._add_job(schedule_id, run_time)
        self._broadcast({"type": "ig_publish_status", "data": entry})
        return entry

    def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel/dismiss a pending or failed schedule."""
        with self._lock:
            entry = self._find_schedule(schedule_id)
            if not entry or entry["status"] not in ("pending", "failed"):
                return False

            entry["status"] = "cancelled"
            self._save_data()

        try:
            self._scheduler.remove_job(schedule_id)
        except Exception:
            pass

        self._broadcast({"type": "ig_publish_status", "data": entry})
        return True

    def retry_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Reset a failed schedule and re-run it immediately."""
        with self._lock:
            entry = self._find_schedule(schedule_id)
            if not entry or entry["status"] != "failed":
                return None
            entry["status"] = "pending"
            entry["error"] = None
            entry["result"] = None
            entry["scheduled_at"] = datetime.now().isoformat()
            self._save_data()

        self._add_job(schedule_id, datetime.now())
        self._broadcast({"type": "ig_publish_status", "data": entry})
        return entry

    def get_schedules(self) -> List[Dict[str, Any]]:
        """Return all schedules (excluding cancelled)."""
        with self._lock:
            return [
                s for s in self._schedules
                if s["status"] != "cancelled"
            ]

    def get_thumbnail_path(self, filename: str) -> Optional[Path]:
        """Generate thumbnail using ffmpeg, cached in data/thumbnails/."""
        self.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = filename.replace(" ", "_").rsplit(".", 1)[0] + ".jpg"
        thumb_path = self.THUMBNAIL_DIR / safe_name
        video_path = self.VIDEOS_DIR / filename

        if thumb_path.exists():
            return thumb_path

        if not video_path.exists():
            return None

        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", str(video_path),
                    "-ss", "00:00:01",
                    "-vframes", "1",
                    "-vf", "scale=320:-1",
                    "-q:v", "5",
                    str(thumb_path),
                ],
                capture_output=True,
                timeout=15,
            )
            if thumb_path.exists():
                return thumb_path
        except Exception as e:
            print(f"[ig_publisher] thumbnail failed for {filename}: {e}")

        return None

    # --- Internal: APScheduler callback ---

    def _execute_post(self, schedule_id: str):
        """APScheduler callback: hand off to ig_helper (Tampermonkey + pyautogui bridge)."""
        with self._lock:
            entry = self._find_schedule(schedule_id)
            if not entry or entry["status"] != "pending":
                return
            entry["status"] = "posting"
            self._save_data()

        self._broadcast({"type": "ig_publish_status", "data": entry})

        video_path = str(self.VIDEOS_DIR / entry["filename"])

        try:
            # Lazy import — avoids circular dep at module load
            from services.ig_helper import ig_helper

            result = ig_helper.submit_job(
                schedule_id=schedule_id,
                video_path=video_path,
                caption=entry["caption"],
                timeout_s=600,
            )

            if result.get("success"):
                with self._lock:
                    entry["status"] = "done"
                    entry["result"] = result
                    self._save_data()
                self._broadcast({"type": "ig_publish_done", "data": entry})
            else:
                error_msg = result.get("error") or "Unknown error"
                with self._lock:
                    entry["status"] = "failed"
                    entry["error"] = error_msg
                    self._save_data()
                self._broadcast({"type": "ig_publish_failed", "data": entry})

        except Exception as e:
            with self._lock:
                entry["status"] = "failed"
                entry["error"] = str(e)
                self._save_data()
            self._broadcast({"type": "ig_publish_failed", "data": entry})

    # --- Internal helpers ---

    def _add_job(self, schedule_id: str, run_time: datetime):
        """Register an APScheduler job."""
        self._scheduler.add_job(
            self._execute_post,
            trigger="date",
            run_date=run_time,
            args=[schedule_id],
            id=schedule_id,
            replace_existing=True,
            misfire_grace_time=None,  # Always run, even if late
        )

    def _restore_pending_jobs(self):
        """On startup, restore pending schedules: expired → run now, future → re-schedule."""
        now = datetime.now()
        for entry in self._schedules:
            if entry["status"] != "pending":
                continue
            run_time = datetime.fromisoformat(entry["scheduled_at"])
            if run_time <= now:
                # Expired while server was down — run immediately
                self._add_job(entry["id"], now)
            else:
                self._add_job(entry["id"], run_time)

    def _find_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Find schedule entry by id (caller must hold lock or be in locked context)."""
        for s in self._schedules:
            if s["id"] == schedule_id:
                return s
        return None

    def _broadcast(self, message: dict):
        """Thread-safe WebSocket broadcast."""
        if self._loop and self._ws_manager:
            asyncio.run_coroutine_threadsafe(
                self._ws_manager.broadcast(message),
                self._loop,
            )

    def _load_data(self):
        """Load schedules from JSON file."""
        if self.DATA_FILE.exists():
            try:
                with open(self.DATA_FILE, "r", encoding="utf-8") as f:
                    self._schedules = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[ig_publisher] failed to load data: {e}")
                self._schedules = []
        else:
            self._schedules = []

    def _save_data(self):
        """Save schedules to JSON file (caller must hold lock)."""
        self.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self._schedules, f, ensure_ascii=False, indent=2)


ig_publisher = IgPublisher()
