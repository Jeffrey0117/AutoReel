"""
Translate Service - wraps TranslationWorkflow + TranscriptionPipeline
for web API usage with WebSocket progress reporting.
"""

import os
import sys
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path so we can import translate_video
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TranslateService:
    """Service layer for translation pipeline, bridging sync pipeline to async WebSocket."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.videos_folder = self.project_root / "videos" / "translate_raw"
        self.subtitles_folder = self.project_root / "subtitles"
        self.config_path = self.project_root / "translation_config.json"

        # Execution state
        self.is_running = False
        self.is_downloading = False
        self.current_task_id: Optional[str] = None

        # Async event loop reference (set by route setup)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_manager = None

        # Thread lock
        self._lock = threading.Lock()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def set_ws_manager(self, manager):
        self._ws_manager = manager

    def _broadcast(self, message: dict):
        """Thread-safe broadcast to WebSocket clients."""
        if self._loop and self._ws_manager:
            asyncio.run_coroutine_threadsafe(
                self._ws_manager.broadcast(message),
                self._loop
            )

    def _progress_callback(self, event: str, data: dict):
        """Callback invoked from pipeline threads, pushes to WebSocket."""
        self._broadcast({
            "type": f"translate_{event}",
            "data": data
        })

    def get_config(self) -> dict:
        """Read translation_config.json."""
        import json
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def update_config(self, updates: dict) -> dict:
        """Merge updates into translation_config.json."""
        import json
        config = self.get_config()
        config.update(updates)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config

    def list_videos(self) -> List[dict]:
        """List videos in translate_raw folder with their processing status."""
        self.videos_folder.mkdir(parents=True, exist_ok=True)

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        videos = []

        for f in sorted(self.videos_folder.iterdir()):
            if f.suffix.lower() in video_extensions and f.is_file():
                video_name = f.stem
                # Check processing status
                srt_zh = self.subtitles_folder / f"{video_name}_zh.srt"
                subtitle_json = self.subtitles_folder / f"{video_name}.json"

                status = "pending"
                if srt_zh.exists() and subtitle_json.exists():
                    status = "done"
                elif self.is_running:
                    status = "processing"

                videos.append({
                    "filename": f.name,
                    "name": video_name,
                    "size": f.stat().st_size,
                    "status": status,
                })

        return videos

    def list_results(self) -> List[dict]:
        """List completed translation results (SRT files + draft names)."""
        self.subtitles_folder.mkdir(parents=True, exist_ok=True)

        results = []
        config = self.get_config()
        output_prefix = config.get("output", {}).get("output_prefix", "翻譯專案_")

        for srt_file in sorted(self.subtitles_folder.glob("*_zh.srt")):
            video_name = srt_file.stem.replace("_zh", "")
            draft_name = f"{output_prefix}{video_name}"
            results.append({
                "video_name": video_name,
                "srt_path": str(srt_file),
                "srt_filename": srt_file.name,
                "draft_name": draft_name,
            })

        return results

    def get_srt_path(self, name: str) -> Optional[Path]:
        """Get SRT file path for a given video name."""
        srt_file = self.subtitles_folder / f"{name}_zh.srt"
        if srt_file.exists():
            return srt_file
        return None

    def save_uploaded_file(self, filename: str, content: bytes) -> Path:
        """Save uploaded video file to translate_raw folder."""
        self.videos_folder.mkdir(parents=True, exist_ok=True)
        dest = self.videos_folder / filename
        with open(dest, 'wb') as f:
            f.write(content)
        return dest

    def delete_video(self, filename: str) -> bool:
        """Delete a video from translate_raw folder."""
        target = self.videos_folder / filename
        if target.exists():
            target.unlink()
            return True
        return False

    def rename_video(self, filename: str, new_name: str) -> Dict[str, Any]:
        """Rename a video in translate_raw folder.

        Args:
            filename: Current filename (e.g. 'video1.mp4')
            new_name: New name without extension (e.g. 'My Video Title')

        Returns:
            Dict with success status and new filename.
        """
        new_name = new_name.strip()
        if not new_name:
            return {"success": False, "error": "名稱不能為空"}

        invalid_chars = '<>:"/\\|?*'
        if any(c in new_name for c in invalid_chars):
            return {"success": False, "error": f"檔名不能包含 {invalid_chars}"}

        old_path = self.videos_folder / filename
        if not old_path.exists():
            return {"success": False, "error": "檔案不存在"}

        extension = old_path.suffix
        new_filename = new_name + extension
        new_path = old_path.parent / new_filename

        if old_path == new_path:
            return {"success": False, "error": "檔名沒有變更"}

        if new_path.exists():
            return {"success": False, "error": "已有同名檔案"}

        old_path.rename(new_path)
        return {
            "success": True,
            "old_filename": filename,
            "new_filename": new_filename,
        }

    def start_single(self, filename: str, force: bool = False):
        """Start translation for a single video file."""
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True

        video_path = self.videos_folder / filename
        if not video_path.exists():
            with self._lock:
                self.is_running = False
            return False

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(force, str(video_path)),
            daemon=True,
            name="translate-single"
        )
        thread.start()
        return True

    def start_pipeline(self, force: bool = False):
        """Start the translation pipeline in a background thread."""
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(force,),
            daemon=True,
            name="translate-pipeline"
        )
        thread.start()
        return True

    def _run_pipeline(self, force: bool = False, single_video_path: str = None):
        """Run the pipeline (called from background thread)."""
        try:
            from translate_video import TranslationWorkflow, TranscriptionPipeline

            workflow = TranslationWorkflow(str(self.config_path))
            config = workflow.config.get("parallel", {})

            if single_video_path:
                video_files = [Path(single_video_path)]
            else:
                # Discover video files
                video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
                video_files = []
                for ext in video_extensions:
                    video_files.extend(self.videos_folder.glob(ext))

            if not video_files:
                self._broadcast({
                    "type": "translate_pipeline_done",
                    "data": {"stats": {}, "success_count": 0, "failed_count": 0, "message": "No videos found"}
                })
                return

            video_paths = [str(f) for f in video_files]

            if len(video_paths) > 1:
                # Pipeline mode for multiple videos
                pipeline = TranscriptionPipeline(
                    workflow, config,
                    progress_callback=self._progress_callback
                )
                pipeline.process(video_paths, force=force)
            else:
                # Single video - use process_video directly with manual callbacks
                video_path = video_paths[0]
                video_name = Path(video_path).stem
                result = None

                self._progress_callback("pipeline_start", {
                    "total": 1,
                    "translate_workers": 1,
                    "draft_workers": 1
                })
                self._progress_callback("transcribe_start", {
                    "video": video_name, "total": 1, "completed": 0
                })

                try:
                    result = workflow.process_video(video_path, force=force)
                    if result:
                        self._progress_callback("video_complete", {
                            "video": video_name,
                            "draft": result,
                            "elapsed": 0,
                            "stats": {"total": 1, "completed": 1, "failed": 0}
                        })
                    else:
                        self._progress_callback("video_error", {
                            "video": video_name,
                            "error": "Processing returned None",
                            "stats": {"total": 1, "completed": 0, "failed": 1}
                        })
                except Exception as e:
                    self._progress_callback("video_error", {
                        "video": video_name,
                        "error": str(e),
                        "stats": {"total": 1, "completed": 0, "failed": 1}
                    })

                succeeded = result is not None
                self._progress_callback("pipeline_done", {
                    "stats": {"total": 1, "completed": 1 if succeeded else 0, "failed": 0 if succeeded else 1},
                    "success_count": 1 if succeeded else 0,
                    "failed_count": 0 if succeeded else 1
                })

        except Exception as e:
            self._broadcast({
                "type": "translate_pipeline_done",
                "data": {
                    "stats": {},
                    "success_count": 0,
                    "failed_count": 0,
                    "error": str(e)
                }
            })
        finally:
            with self._lock:
                self.is_running = False

    def start_url_download(self, url: str, auto_translate: bool = True):
        """Download video from URL using yt-dlp, then optionally start translation."""
        with self._lock:
            if self.is_downloading:
                return False
            self.is_downloading = True

        thread = threading.Thread(
            target=self._run_url_download,
            args=(url, auto_translate),
            daemon=True,
            name="translate-url-download"
        )
        thread.start()
        return True

    def start_full_process(self, url: str, auto_caption: bool = True):
        """Complete process: Download → Translate → Generate Caption."""
        with self._lock:
            if self.is_downloading or self.is_running:
                return False
            self.is_downloading = True

        thread = threading.Thread(
            target=self._run_full_process,
            args=(url, auto_caption),
            daemon=True,
            name="translate-full-process"
        )
        thread.start()
        return True

    def _run_full_process(self, url: str, auto_caption: bool = True):
        """Run complete process: Download → Translate → Auto-rename → Generate Caption."""
        downloaded_file = None
        try:
            # Step 1: Download
            self._broadcast({
                "type": "translate_download_progress",
                "data": {"status": "starting", "url": url, "progress": 0, "step": "下載中..."}
            })

            downloaded_file = self._download_video(url)

            if not downloaded_file:
                self._broadcast({
                    "type": "translate_download_progress",
                    "data": {"status": "failed", "url": url, "error": "下載失敗"}
                })
                return

            self._broadcast({
                "type": "translate_download_progress",
                "data": {"status": "completed", "url": url, "filename": downloaded_file.name, "progress": 100}
            })

            # Mark download done, start translation
            with self._lock:
                self.is_downloading = False
                self.is_running = True

            # Step 2: Translate
            video_name = downloaded_file.stem
            self._broadcast({
                "type": "translate_pipeline_start",
                "data": {"total": 1, "step": "翻譯中..."}
            })

            result = self._translate_single_video(str(downloaded_file))

            # Step 2.5: Auto-rename based on subtitle content
            if result:
                self._broadcast({
                    "type": "translate_rename_start",
                    "data": {"video": video_name, "step": "生成標題中..."}
                })
                new_name = self._generate_title_from_srt(video_name)
                if new_name and new_name != video_name:
                    renamed = self._rename_video_and_subtitles(video_name, new_name)
                    if renamed:
                        self._broadcast({
                            "type": "translate_renamed",
                            "data": {"old_name": video_name, "new_name": new_name}
                        })
                        video_name = new_name

            # Step 3: Generate Caption (if translation succeeded and auto_caption enabled)
            caption_path = None
            if result and auto_caption:
                self._broadcast({
                    "type": "translate_caption_start",
                    "data": {"video": video_name, "step": "生成文案中..."}
                })
                caption_path = self._generate_caption(video_name)

            # Build result
            srt_path = str(self.subtitles_folder / f"{video_name}_zh.srt")
            results = [{
                "video": video_name,
                "srt_path": srt_path if Path(srt_path).exists() else None,
                "caption_path": caption_path
            }]

            self._broadcast({
                "type": "translate_pipeline_done",
                "data": {
                    "success_count": 1 if result else 0,
                    "failed_count": 0 if result else 1,
                    "results": results,
                    "auto_caption": False  # Already done
                }
            })

        except Exception as e:
            self._broadcast({
                "type": "translate_pipeline_done",
                "data": {
                    "success_count": 0,
                    "failed_count": 1,
                    "error": str(e),
                    "results": []
                }
            })
        finally:
            with self._lock:
                self.is_downloading = False
                self.is_running = False

    def _download_video(self, url: str) -> Optional[Path]:
        """Download video using yt-dlp, return downloaded file path."""
        self.videos_folder.mkdir(parents=True, exist_ok=True)

        output_template = str(self.videos_folder / "%(title)s.%(ext)s")
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--no-playlist",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_template,
            "--newline",
            "--print", "after_move:filepath",
            url
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        downloaded_path = None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue

            # Parse progress
            if '[download]' in line and '%' in line:
                try:
                    pct_str = line.split('%')[0].split()[-1]
                    pct = float(pct_str)
                    self._broadcast({
                        "type": "translate_download_progress",
                        "data": {"status": "downloading", "url": url, "progress": pct}
                    })
                except (ValueError, IndexError):
                    pass
            elif '[Merger]' in line:
                self._broadcast({
                    "type": "translate_download_progress",
                    "data": {"status": "processing", "url": url, "progress": 95}
                })
            # Capture final file path
            elif line.endswith('.mp4') and self.videos_folder.name in line:
                downloaded_path = Path(line)

        process.wait()

        if process.returncode == 0 and downloaded_path and downloaded_path.exists():
            return downloaded_path

        # Fallback: find most recently modified mp4
        mp4_files = list(self.videos_folder.glob("*.mp4"))
        if mp4_files:
            return max(mp4_files, key=lambda f: f.stat().st_mtime)

        return None

    def _translate_single_video(self, video_path: str) -> Optional[str]:
        """Translate a single video, return draft name if successful."""
        try:
            from translate_video import TranslationWorkflow

            workflow = TranslationWorkflow(str(self.config_path))
            video_name = Path(video_path).stem

            self._progress_callback("transcribe_start", {
                "video": video_name, "total": 1, "completed": 0
            })

            result = workflow.process_video(video_path, force=False)

            if result:
                self._progress_callback("video_complete", {
                    "video": video_name,
                    "draft": result,
                    "elapsed": 0,
                    "stats": {"total": 1, "completed": 1, "failed": 0}
                })
                return result
            else:
                self._progress_callback("video_error", {
                    "video": video_name,
                    "error": "Processing returned None"
                })
                return None

        except Exception as e:
            self._progress_callback("video_error", {
                "video": Path(video_path).stem,
                "error": str(e)
            })
            return None

    def _generate_caption(self, video_name: str) -> Optional[str]:
        """Generate IG caption for translated video."""
        try:
            from services.caption_service import caption_service

            # Find the SRT file
            srt_path = self.subtitles_folder / f"{video_name}_zh.srt"
            if not srt_path.exists():
                self._broadcast({
                    "type": "translate_caption_error",
                    "data": {"video": video_name, "error": "SRT 檔案不存在"}
                })
                return None

            # Read SRT content and extract text
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            # Parse SRT to extract text lines
            import re
            text_lines = []
            for block in srt_content.strip().split('\n\n'):
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    text_lines.append(' '.join(lines[2:]))
            subtitles_text = '\n'.join(text_lines)

            if not subtitles_text:
                self._broadcast({
                    "type": "translate_caption_error",
                    "data": {"video": video_name, "error": "無法從 SRT 提取字幕"}
                })
                return None

            # Generate caption using caption service's internal method
            config = caption_service._read_config()
            examples = config.get("ig_caption", {}).get("examples", [])

            if not examples:
                self._broadcast({
                    "type": "translate_caption_error",
                    "data": {"video": video_name, "error": "請先新增至少一個風格範例"}
                })
                return None

            caption = caption_service._call_deepseek_api(subtitles_text, examples, config)

            if not caption:
                self._broadcast({
                    "type": "translate_caption_error",
                    "data": {"video": video_name, "error": "API 呼叫失敗"}
                })
                return None

            # Save caption
            output_path = self.subtitles_folder / f"{video_name}_ig_caption.txt"
            output_path.write_text(caption, encoding="utf-8")

            self._broadcast({
                "type": "translate_caption_done",
                "data": {"video": video_name, "path": str(output_path)}
            })

            return str(output_path)

        except Exception as e:
            self._broadcast({
                "type": "translate_caption_error",
                "data": {"video": video_name, "error": str(e)}
            })
            return None

    def _generate_title_from_srt(self, video_name: str) -> Optional[str]:
        """Generate a short title from SRT content using AI."""
        try:
            import requests
            import json

            srt_path = self.subtitles_folder / f"{video_name}_zh.srt"
            if not srt_path.exists():
                return None

            # Read and parse SRT
            with open(srt_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            text_lines = []
            for block in srt_content.strip().split('\n\n'):
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    text_lines.append(' '.join(lines[2:]))

            subtitles_text = '\n'.join(text_lines[:20])  # Only use first 20 lines
            if not subtitles_text:
                return None

            # Get API config
            config = self.get_config()
            translation_config = config.get("translation", {})
            api_key = translation_config.get("api_key") or os.environ.get(
                translation_config.get("api_key_env", "DEEPSEEK_API_KEY")
            )
            base_url = translation_config.get("base_url", "https://api.deepseek.com")
            model = translation_config.get("model", "deepseek-chat")

            if not api_key:
                return None

            prompt = f"""根據以下影片字幕內容，生成一個簡短的中文標題（10-20字）。
標題要能概括影片主題，吸引人點擊。
只輸出標題本身，不要加引號或其他說明。

字幕內容：
{subtitles_text}"""

            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                title = result["choices"][0]["message"]["content"].strip()
                # Clean up title - remove quotes, newlines, invalid chars
                title = title.strip('"\'""')
                title = title.replace('\n', ' ').strip()
                invalid_chars = '<>:"/\\|?*'
                for char in invalid_chars:
                    title = title.replace(char, '')
                # Limit length
                if len(title) > 50:
                    title = title[:50]
                return title if title else None

            return None

        except Exception as e:
            self._broadcast({
                "type": "translate_rename_error",
                "data": {"video": video_name, "error": str(e)}
            })
            return None

    def _rename_video_and_subtitles(self, old_name: str, new_name: str) -> bool:
        """Rename video file and all related subtitle files."""
        try:
            # Find and rename video file
            video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
            video_renamed = False

            for ext in video_extensions:
                old_video = self.videos_folder / f"{old_name}{ext}"
                if old_video.exists():
                    new_video = self.videos_folder / f"{new_name}{ext}"
                    old_video.rename(new_video)
                    video_renamed = True
                    break

            # Rename subtitle files
            subtitle_patterns = [
                (f"{old_name}.json", f"{new_name}.json"),
                (f"{old_name}.srt", f"{new_name}.srt"),
                (f"{old_name}_zh.srt", f"{new_name}_zh.srt"),
                (f"{old_name}_en.srt", f"{new_name}_en.srt"),
            ]

            for old_pattern, new_pattern in subtitle_patterns:
                old_path = self.subtitles_folder / old_pattern
                if old_path.exists():
                    new_path = self.subtitles_folder / new_pattern
                    old_path.rename(new_path)

            return video_renamed

        except Exception as e:
            self._broadcast({
                "type": "translate_rename_error",
                "data": {"old_name": old_name, "new_name": new_name, "error": str(e)}
            })
            return False

    def _run_url_download(self, url: str, auto_translate: bool = True):
        """Download from URL using yt-dlp (called from background thread)."""
        try:
            self.videos_folder.mkdir(parents=True, exist_ok=True)

            self._broadcast({
                "type": "translate_download_progress",
                "data": {"status": "starting", "url": url, "progress": 0}
            })

            # Use yt-dlp to download
            output_template = str(self.videos_folder / "%(title)s.%(ext)s")
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--no-playlist",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", output_template,
                "--newline",
                url
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            filename = None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                # Parse yt-dlp output for progress
                if '[download]' in line:
                    if 'Destination:' in line:
                        filename = line.split('Destination:')[-1].strip()
                        self._broadcast({
                            "type": "translate_download_progress",
                            "data": {"status": "downloading", "url": url, "filename": Path(filename).name, "progress": 0}
                        })
                    elif '%' in line:
                        try:
                            pct_str = line.split('%')[0].split()[-1]
                            pct = float(pct_str)
                            self._broadcast({
                                "type": "translate_download_progress",
                                "data": {"status": "downloading", "url": url, "progress": pct}
                            })
                        except (ValueError, IndexError):
                            pass
                    elif 'has already been downloaded' in line:
                        filename = line.split('[download]')[1].split('has already')[0].strip()
                        self._broadcast({
                            "type": "translate_download_progress",
                            "data": {"status": "already_exists", "url": url, "filename": Path(filename).name, "progress": 100}
                        })
                elif '[Merger]' in line or '[ExtractAudio]' in line:
                    self._broadcast({
                        "type": "translate_download_progress",
                        "data": {"status": "processing", "url": url, "progress": 95}
                    })

            process.wait()

            if process.returncode == 0:
                self._broadcast({
                    "type": "translate_download_progress",
                    "data": {"status": "completed", "url": url, "progress": 100}
                })
            else:
                self._broadcast({
                    "type": "translate_download_progress",
                    "data": {"status": "failed", "url": url, "error": f"yt-dlp exited with code {process.returncode}"}
                })

        except Exception as e:
            self._broadcast({
                "type": "translate_download_progress",
                "data": {"status": "failed", "url": url, "error": str(e)}
            })
        finally:
            with self._lock:
                self.is_downloading = False

    def get_status(self) -> dict:
        """Get current service status."""
        return {
            "is_running": self.is_running,
            "is_downloading": self.is_downloading,
        }


# Singleton instance
translate_service = TranslateService()
