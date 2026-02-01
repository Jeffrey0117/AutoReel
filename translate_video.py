#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻譯影片工作流程 - 主程式
自動語音識別 + 翻譯 + 生成剪映字幕草稿
"""

import os
import sys
import glob as glob_module

# Windows 設定 UTF-8 編碼
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Windows: 自動尋找並加入 ffmpeg 路徑
if sys.platform == 'win32':
    # 檢查 ffmpeg 是否已在 PATH
    ffmpeg_in_path = any(
        os.path.exists(os.path.join(p, 'ffmpeg.exe'))
        for p in os.environ.get('PATH', '').split(os.pathsep)
    )
    if not ffmpeg_in_path:
        # 嘗試找到 WinGet 安裝的 ffmpeg
        winget_ffmpeg = glob_module.glob(
            os.path.expanduser('~/AppData/Local/Microsoft/WinGet/Packages/*/ffmpeg-*/bin')
        )
        if winget_ffmpeg:
            os.environ['PATH'] = winget_ffmpeg[0] + os.pathsep + os.environ.get('PATH', '')

import json
import shutil
import glob
import uuid
import copy
import random
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress tracker
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None, position=None, leave=True):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"   [{self.desc}] {self.n}/{self.total}")
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

from subtitle_generator import SubtitleGenerator, SubtitleEntry


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = auto()
    TRANSCRIBING = auto()
    TRANSLATING = auto()
    GENERATING_DRAFT = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class PipelineTask:
    """Represents a single video processing task in the pipeline"""
    video_path: str
    video_name: str
    status: TaskStatus = TaskStatus.PENDING
    entries: Optional[List[SubtitleEntry]] = None
    draft_name: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __hash__(self):
        return hash(self.video_path)

    def elapsed_time(self) -> float:
        """Return elapsed time in seconds"""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at


class TranscriptionPipeline:
    """
    Pipeline for parallel video translation processing.

    Architecture:
    - Transcription: Sequential (GPU-bound, one at a time)
    - Translation: Parallel (I/O-bound, multiple workers)
    - Draft Generation: Parallel (I/O-bound, multiple workers)

    Flow:
    Video 1: transcribe -> translate -> generate draft
    Video 2:             transcribe -> translate -> generate draft
    Video 3:                         transcribe -> translate -> generate draft

    While Video 1 is translating, Video 2 can start transcribing.
    """

    def __init__(self, workflow: 'TranslationWorkflow', config: dict,
                 progress_callback: Optional[Callable[[str, dict], None]] = None):
        """
        Initialize the pipeline.

        Args:
            workflow: The TranslationWorkflow instance
            config: Pipeline configuration dict
            progress_callback: Optional callback(event_type, data) for progress reporting.
                When provided, tqdm progress bars are skipped.
        """
        self.workflow = workflow
        self.config = config
        self.progress_callback = progress_callback

        # Worker configuration
        self.translate_workers = config.get("translate_workers", 4)
        self.draft_workers = config.get("draft_workers", 2)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        # Queues for pipeline stages
        self.transcription_queue: Queue[PipelineTask] = Queue()
        self.translation_queue: Queue[PipelineTask] = Queue()
        self.draft_queue: Queue[PipelineTask] = Queue()

        # Task tracking
        self.tasks: Dict[str, PipelineTask] = {}
        self.lock = threading.Lock()

        # Progress tracking
        self.progress_bars: Dict[str, Any] = {}

        # Control flags
        self._stop_event = threading.Event()
        self._transcription_done = threading.Event()
        self._translation_done = threading.Event()

        # Statistics
        self.stats = {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "transcribed": 0,
            "translated": 0,
            "drafts_generated": 0
        }

    def _notify(self, event: str, data: dict = None):
        """Send progress notification via callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(event, data or {})
            except Exception:
                pass

    def _create_progress_bars(self, total: int) -> Dict[str, Any]:
        """Create progress bars for each stage"""
        if TQDM_AVAILABLE:
            return {
                "transcribe": tqdm(total=total, desc="Transcribing", unit="video", position=0, leave=True),
                "translate": tqdm(total=total, desc="Translating", unit="video", position=1, leave=True),
                "draft": tqdm(total=total, desc="Generating", unit="draft", position=2, leave=True),
                "overall": tqdm(total=total, desc="Overall", unit="video", position=3, leave=True)
            }
        else:
            return {
                "transcribe": tqdm(total=total, desc="Transcribing"),
                "translate": tqdm(total=total, desc="Translating"),
                "draft": tqdm(total=total, desc="Generating"),
                "overall": tqdm(total=total, desc="Overall")
            }

    def _close_progress_bars(self):
        """Close all progress bars"""
        for bar in self.progress_bars.values():
            bar.close()

    def add_video(self, video_path: str) -> PipelineTask:
        """Add a video to the pipeline queue"""
        video_path = str(Path(video_path).resolve())
        video_name = Path(video_path).stem

        task = PipelineTask(
            video_path=video_path,
            video_name=video_name
        )

        with self.lock:
            self.tasks[video_path] = task
            self.stats["total"] += 1

        self.transcription_queue.put(task)
        return task

    def _transcribe_worker(self, force: bool = False):
        """
        Worker for transcription (runs sequentially - GPU bound).

        This worker processes videos one at a time to avoid GPU contention.
        """
        while not self._stop_event.is_set():
            try:
                task = self.transcription_queue.get(timeout=1.0)
            except Empty:
                # Check if all videos have been added and queue is empty
                if self.transcription_queue.empty():
                    break
                continue

            if task is None:  # Poison pill
                break

            task.status = TaskStatus.TRANSCRIBING
            task.started_at = time.time()

            try:
                video_name = task.video_name
                subtitle_json = self.workflow.subtitle_folder / f"{video_name}.json"

                self._notify("transcribe_start", {
                    "video": video_name,
                    "total": self.stats["total"],
                    "completed": self.stats["transcribed"]
                })

                # Check if we should skip transcription
                if not force and subtitle_json.exists():
                    print(f"\n[Pipeline] Loading existing subtitles for: {video_name}")
                    task.entries = self.workflow.subtitle_gen.load_from_json(str(subtitle_json))
                else:
                    print(f"\n[Pipeline] Transcribing: {video_name}")
                    task.entries = self.workflow.subtitle_gen.transcribe(task.video_path)

                    # Save original subtitles
                    self.workflow.subtitle_gen.export_srt(
                        task.entries,
                        str(self.workflow.subtitle_folder / f"{video_name}_en.srt"),
                        use_translated=False
                    )

                with self.lock:
                    self.stats["transcribed"] += 1

                if self.progress_bars:
                    self.progress_bars["transcribe"].update(1)
                    self.progress_bars["transcribe"].set_postfix(current=video_name)

                self._notify("transcribe_done", {
                    "video": video_name,
                    "total": self.stats["total"],
                    "completed": self.stats["transcribed"]
                })

                # Move to translation queue
                self.translation_queue.put(task)

            except Exception as e:
                self._handle_task_error(task, f"Transcription failed: {str(e)}")

            finally:
                self.transcription_queue.task_done()

        # Signal transcription is complete
        self._transcription_done.set()
        # Add poison pills for translation workers
        for _ in range(self.translate_workers):
            self.translation_queue.put(None)

    def _translate_single(self, task: PipelineTask) -> bool:
        """Translate a single task with retry logic"""
        self._notify("translate_start", {
            "video": task.video_name,
            "total": self.stats["total"],
            "completed": self.stats["translated"]
        })

        # 檢查是否已經翻譯過（避免浪費 API 額度）
        already_translated = all(
            entry.text_translated and entry.text_translated.strip()
            for entry in task.entries
        ) if task.entries else False

        if already_translated:
            print(f"\n[Pipeline] Skip translation (already done): {task.video_name}")
            # 直接保存已有的翻譯結果
            self.workflow.subtitle_gen.export_srt(
                task.entries,
                str(self.workflow.subtitle_folder / f"{task.video_name}_zh.srt"),
                use_translated=True
            )
            subtitle_json = self.workflow.subtitle_folder / f"{task.video_name}.json"
            self.workflow.subtitle_gen.export_json(task.entries, str(subtitle_json))
            return True

        for attempt in range(self.max_retries):
            try:
                task.status = TaskStatus.TRANSLATING
                print(f"\n[Pipeline] Translating: {task.video_name} (attempt {attempt + 1})")

                task.entries = self.workflow.subtitle_gen.translate_entries(task.entries)

                # Save translated subtitles
                self.workflow.subtitle_gen.export_srt(
                    task.entries,
                    str(self.workflow.subtitle_folder / f"{task.video_name}_zh.srt"),
                    use_translated=True
                )

                # Save JSON
                subtitle_json = self.workflow.subtitle_folder / f"{task.video_name}.json"
                self.workflow.subtitle_gen.export_json(task.entries, str(subtitle_json))

                return True

            except Exception as e:
                task.retry_count += 1
                if attempt < self.max_retries - 1:
                    print(f"   [Retry] Translation failed for {task.video_name}, retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise e

        return False

    def _translation_worker(self):
        """Worker for translation (parallel - I/O bound)"""
        while not self._stop_event.is_set():
            try:
                task = self.translation_queue.get(timeout=1.0)
            except Empty:
                # Check if transcription is done and queue is empty
                if self._transcription_done.is_set() and self.translation_queue.empty():
                    break
                continue

            if task is None:  # Poison pill
                self.translation_queue.task_done()
                break

            try:
                if self._translate_single(task):
                    with self.lock:
                        self.stats["translated"] += 1

                    if self.progress_bars:
                        self.progress_bars["translate"].update(1)
                        self.progress_bars["translate"].set_postfix(current=task.video_name)

                    self._notify("translate_done", {
                        "video": task.video_name,
                        "total": self.stats["total"],
                        "completed": self.stats["translated"]
                    })

                    # Move to draft generation queue
                    self.draft_queue.put(task)

            except Exception as e:
                self._handle_task_error(task, f"Translation failed: {str(e)}")

            finally:
                self.translation_queue.task_done()

        # Signal translation is done (only when all workers are done)
        with self.lock:
            if hasattr(self, '_translation_workers_done'):
                self._translation_workers_done += 1
            else:
                self._translation_workers_done = 1

            if self._translation_workers_done >= self.translate_workers:
                self._translation_done.set()
                # Add poison pills for draft workers
                for _ in range(self.draft_workers):
                    self.draft_queue.put(None)

    def _generate_draft_single(self, task: PipelineTask, force: bool = False) -> Optional[str]:
        """Generate draft for a single task with retry logic"""
        self._notify("draft_start", {
            "video": task.video_name,
            "total": self.stats["total"],
            "completed": self.stats["drafts_generated"]
        })

        for attempt in range(self.max_retries):
            try:
                task.status = TaskStatus.GENERATING_DRAFT
                print(f"\n[Pipeline] Generating draft: {task.video_name} (attempt {attempt + 1})")

                video_name = task.video_name
                output_name = f"{self.workflow.output_prefix}{video_name}"

                # Check if draft exists
                output_folder = self.workflow.jianying_draft_root / output_name
                if output_folder.exists() and not force:
                    print(f"   [Skip] Draft already exists: {output_name}")
                    return output_name

                # Load template
                template_data = self.workflow._load_template()
                if not template_data:
                    raise Exception("Failed to load template")

                # Deep copy template
                draft_data = copy.deepcopy(template_data)

                # Clean template texts
                draft_data = self.workflow._clean_template_texts(draft_data, keep_count=2)

                # Replace video
                draft_data = self.workflow._replace_video_in_draft(draft_data, task.video_path)

                # Get video duration
                video_duration = draft_data.get("duration", 0)

                # Update template texts
                draft_data = self.workflow._update_template_texts(draft_data, video_name, video_duration)

                # Add subtitles
                draft_data = self.workflow._add_subtitles_to_draft(draft_data, task.entries, template_data)

                # Save draft
                if output_folder.exists():
                    def remove_readonly(func, path, excinfo):
                        import stat
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    try:
                        shutil.rmtree(output_folder, onerror=remove_readonly)
                    except Exception as e:
                        print(f"   [Warning] Cannot fully delete old folder: {e}")

                output_folder.mkdir(parents=True, exist_ok=True)

                # Write draft_content.json
                with open(output_folder / "draft_content.json", 'w', encoding='utf-8') as f:
                    json.dump(draft_data, f, ensure_ascii=False)

                # Copy other template files
                template_folder = self.workflow.jianying_draft_root / self.workflow.template_name
                for file in ["draft_meta_info.json", "draft_settings"]:
                    src = template_folder / file
                    if src.exists():
                        shutil.copy(src, output_folder / file)

                return output_name

            except Exception as e:
                task.retry_count += 1
                if attempt < self.max_retries - 1:
                    print(f"   [Retry] Draft generation failed for {task.video_name}, retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e

        return None

    def _draft_worker(self, force: bool = False):
        """Worker for draft generation (parallel - I/O bound)"""
        while not self._stop_event.is_set():
            try:
                task = self.draft_queue.get(timeout=1.0)
            except Empty:
                # Check if translation is done and queue is empty
                if self._translation_done.is_set() and self.draft_queue.empty():
                    break
                continue

            if task is None:  # Poison pill
                self.draft_queue.task_done()
                break

            try:
                draft_name = self._generate_draft_single(task, force)

                if draft_name:
                    task.draft_name = draft_name
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()

                    with self.lock:
                        self.stats["drafts_generated"] += 1
                        self.stats["completed"] += 1

                    if self.progress_bars:
                        self.progress_bars["draft"].update(1)
                        self.progress_bars["overall"].update(1)
                        elapsed = task.elapsed_time()
                        self.progress_bars["overall"].set_postfix(
                            completed=task.video_name,
                            time=f"{elapsed:.1f}s"
                        )

                    self._notify("draft_done", {
                        "video": task.video_name,
                        "total": self.stats["total"],
                        "completed": self.stats["drafts_generated"]
                    })
                    self._notify("video_complete", {
                        "video": task.video_name,
                        "draft": draft_name,
                        "elapsed": task.elapsed_time(),
                        "stats": dict(self.stats)
                    })

                    print(f"\n[Pipeline] Completed: {task.video_name} ({task.elapsed_time():.1f}s)")

            except Exception as e:
                self._handle_task_error(task, f"Draft generation failed: {str(e)}")

            finally:
                self.draft_queue.task_done()

    def _handle_task_error(self, task: PipelineTask, error_msg: str):
        """Handle task error"""
        task.status = TaskStatus.FAILED
        task.error = error_msg
        task.completed_at = time.time()

        with self.lock:
            self.stats["failed"] += 1

        if self.progress_bars:
            self.progress_bars["overall"].update(1)

        self._notify("video_error", {
            "video": task.video_name,
            "error": error_msg,
            "stats": dict(self.stats)
        })

        print(f"\n[Pipeline] FAILED: {task.video_name}")
        print(f"   Error: {error_msg}")
        import traceback
        traceback.print_exc()

    def process(self, video_files: List[str], force: bool = False) -> Dict[str, Any]:
        """
        Process multiple videos through the pipeline.

        Args:
            video_files: List of video file paths
            force: Force reprocessing even if drafts exist

        Returns:
            Dictionary with processing results and statistics
        """
        if not video_files:
            return {"success": [], "failed": [], "stats": self.stats}

        total = len(video_files)
        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting parallel processing")
        print(f"   Videos: {total}")
        print(f"   Translate workers: {self.translate_workers}")
        print(f"   Draft workers: {self.draft_workers}")
        print(f"{'='*60}\n")

        self._notify("pipeline_start", {
            "total": total,
            "translate_workers": self.translate_workers,
            "draft_workers": self.draft_workers
        })

        # Create progress bars (skip when using callback)
        if not self.progress_callback:
            self.progress_bars = self._create_progress_bars(total)

        # Add all videos to the queue
        for video_path in video_files:
            self.add_video(video_path)

        # Create worker threads
        threads = []

        # Transcription worker (single thread - GPU bound)
        transcribe_thread = threading.Thread(
            target=self._transcribe_worker,
            args=(force,),
            name="transcribe-worker"
        )
        threads.append(transcribe_thread)

        # Translation workers (multiple threads - I/O bound)
        for i in range(self.translate_workers):
            translate_thread = threading.Thread(
                target=self._translation_worker,
                name=f"translate-worker-{i}"
            )
            threads.append(translate_thread)

        # Draft generation workers (multiple threads - I/O bound)
        for i in range(self.draft_workers):
            draft_thread = threading.Thread(
                target=self._draft_worker,
                args=(force,),
                name=f"draft-worker-{i}"
            )
            threads.append(draft_thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Close progress bars
        self._close_progress_bars()

        # Collect results
        success = []
        failed = []

        for task in self.tasks.values():
            if task.status == TaskStatus.COMPLETED:
                success.append({
                    "video": task.video_name,
                    "draft": task.draft_name,
                    "time": task.elapsed_time()
                })
            else:
                failed.append({
                    "video": task.video_name,
                    "error": task.error,
                    "status": task.status.name
                })

        # Print summary
        print(f"\n{'='*60}")
        print(f"[Pipeline] Processing complete!")
        print(f"   Total: {self.stats['total']}")
        print(f"   Completed: {self.stats['completed']}")
        print(f"   Failed: {self.stats['failed']}")
        print(f"   Transcribed: {self.stats['transcribed']}")
        print(f"   Translated: {self.stats['translated']}")
        print(f"   Drafts generated: {self.stats['drafts_generated']}")
        print(f"{'='*60}\n")

        self._notify("pipeline_done", {
            "stats": dict(self.stats),
            "success_count": len(success),
            "failed_count": len(failed)
        })

        # 如果有失敗的影片，生成失敗清單
        if failed:
            failed_file = self.workflow.subtitle_folder / "failed_videos.txt"
            with open(failed_file, 'w', encoding='utf-8') as f:
                f.write(f"# 失敗影片清單 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 總共 {len(failed)} 個影片處理失敗\n\n")
                for item in failed:
                    f.write(f"{item['video']}\n")
                    f.write(f"  狀態: {item['status']}\n")
                    f.write(f"  錯誤: {item['error']}\n\n")

            print(f"[Warning] {len(failed)} 個影片處理失敗！")
            print(f"   失敗清單: {failed_file}")
            print(f"\n   失敗的影片:")
            for item in failed:
                err_msg = str(item['error'] or 'Unknown error')[:50]
                print(f"   - {item['video']}: {err_msg}...")
            print(f"\n   重跑失敗影片:")
            print(f"   python translate_video.py --batch --pipeline --force")

        return {
            "success": success,
            "failed": failed,
            "stats": self.stats
        }

    def stop(self):
        """Stop the pipeline gracefully"""
        self._stop_event.set()


class TranslationWorkflow:
    """翻譯影片工作流程"""

    def __init__(self, config_path: str = "translation_config.json"):
        self.config = self._load_config(config_path)
        self.subtitle_gen = SubtitleGenerator(config_path)

        # 路徑設定
        self.project_root = Path(__file__).parent
        self.jianying_draft_root = self._get_jianying_draft_root()
        self.template_name = self.config.get("output", {}).get("template_name", "翻譯專案")
        self.output_prefix = self.config.get("output", {}).get("output_prefix", "翻譯專案_")
        self.subtitle_folder = self.project_root / self.config.get("output", {}).get("subtitle_folder", "subtitles")

        # 確保字幕資料夾存在
        self.subtitle_folder.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """載入設定檔"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _get_jianying_draft_root(self) -> Path:
        """取得剪映草稿根路徑"""
        # 優先從 config.json 讀取
        config_path = self.project_root / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return Path(config.get("jianying_draft_folder", ""))
            except:
                pass

        username = os.environ.get("USERNAME") or os.getlogin()
        return Path(rf"C:\Users\{username}\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft")

    def _load_template(self) -> Optional[dict]:
        """載入翻譯專案模板 - 每次都從本地強制複製到剪映草稿夾"""
        local_template_folder = self.project_root / self.template_name
        local_template_file = local_template_folder / "draft_content.json"
        dest_folder = self.jianying_draft_root / self.template_name

        # 檢查本地模板是否存在
        if not local_template_file.exists():
            print(f"[Error] 找不到本地模板: {local_template_folder}")
            return None

        # 強制從本地複製到剪映草稿夾（每次都覆蓋）
        print(f"[Template] 同步模板到剪映草稿夾...")
        if dest_folder.exists():
            shutil.rmtree(dest_folder, ignore_errors=True)
        shutil.copytree(local_template_folder, dest_folder)

        template_path = dest_folder / "draft_content.json"
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _hex_to_rgb(self, hex_color: str) -> list:
        """將 HEX 顏色轉換為 RGB (0-1 範圍)"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return [r, g, b]

    def _remove_punctuation(self, text: str) -> str:
        """移除標點符號"""
        import re
        # 移除中英文標點符號
        punctuation = r'[，。！？、；：""''（）【】《》\.,!?\-\'\";:\(\)\[\]<>]'
        return re.sub(punctuation, '', text)

    def _auto_line_break(self, text: str, max_chars: int = 20) -> str:
        """自動換行 - 每 max_chars 個字元插入換行"""
        if len(text) <= max_chars:
            return text

        # 中文字幕自動換行
        lines = []
        current_line = ""
        for char in text:
            current_line += char
            if len(current_line) >= max_chars:
                lines.append(current_line)
                current_line = ""
        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    def _create_subtitle_material(self, entry: SubtitleEntry, template_style: dict) -> dict:
        """創建單個字幕素材 - 從模板提取樣式值"""
        material_id = str(uuid.uuid4()).upper()

        # 使用翻譯後的文字，如果沒有則用原文
        text = entry.text_translated if entry.text_translated else entry.text_original

        # 移除標點符號
        text = self._remove_punctuation(text)

        # 自動換行處理
        max_chars = self.config.get("subtitle_style", {}).get("max_chars_per_line", 18)
        text = self._auto_line_break(text, max_chars)

        # 從模板提取樣式值
        font_size = template_style.get("font_size", 8.0)
        font_resource_id = template_style.get("font_resource_id", "")
        font_path = template_style.get("font_path", "")
        border_width = template_style.get("border_width", 0.1166)
        border_color = template_style.get("border_color", "#000000")
        bold_width = template_style.get("bold_width", 0.008)
        shadow_alpha = template_style.get("shadow_alpha", 0.618)
        shadow_distance = template_style.get("shadow_distance", 5.0)
        shadow_smoothing = template_style.get("shadow_smoothing", 0.45)
        line_max_width = template_style.get("line_max_width", 0.82)
        fonts = template_style.get("fonts", [])
        # 背景設定
        background_alpha = template_style.get("background_alpha", 0.64)
        background_style = template_style.get("background_style", 1)
        background_color = template_style.get("background_color", "#000000")

        # 描邊設定
        stroke_width = template_style.get("stroke_width", 0.173)
        stroke_color = template_style.get("stroke_color", "#000000")
        stroke_rgb = self._hex_to_rgb(stroke_color)

        # 文字顏色設定
        text_color = template_style.get("text_color", "#FFFFFF")
        text_color_random = template_style.get("text_color_random", False)
        text_color_options = template_style.get("text_color_options", ["#FFFFFF"])

        # 如果啟用隨機顏色
        if text_color_random and text_color_options:
            text_color = random.choice(text_color_options)

        text_rgb = self._hex_to_rgb(text_color)

        # 建立 content JSON（包含 strokes 描邊）
        style_entry = {
            "fill": {"content": {"solid": {"color": text_rgb}}},
            "font": {
                "id": font_resource_id,
                "path": font_path
            },
            "strokes": [
                {
                    "content": {"solid": {"color": stroke_rgb}},
                    "width": stroke_width
                }
            ],
            "size": font_size,
            "useLetterColor": True,
            "range": [0, len(text)]
        }

        content_data = {
            "text": text,
            "styles": [style_entry]
        }

        return {
            "id": material_id,
            "type": "text",
            "content": json.dumps(content_data, ensure_ascii=False),
            "font_size": font_size,
            "text_color": text_color,
            "background_color": background_color,
            "background_alpha": background_alpha,
            "background_style": background_style,
            "alignment": 1,
            "add_type": 0,
            "check_flag": 63,
            "combo_info": {"text_templates": []},
            "border_alpha": 1.0,
            "border_color": border_color,
            "border_width": border_width,
            "bold_width": bold_width,
            "font_category_id": "",
            "font_category_name": "",
            "font_id": "",
            "font_name": "",
            "font_path": font_path,
            "font_resource_id": font_resource_id,
            "font_source_platform": 0,
            "font_team_id": "",
            "font_title": "none",
            "font_url": "",
            "fonts": fonts,
            "force_apply_line_max_width": True,
            "global_alpha": 1.0,
            "has_shadow": True,
            "is_rich_text": False,
            "italic_degree": 0,
            "ktv_color": "",
            "language": "",
            "layer_weight": 1,
            "letter_spacing": 0.0,
            "line_feed": 1,
            "line_max_width": line_max_width,
            "line_spacing": 0.02,
            "multi_language_current": "none",
            "name": "",
            "preset_category": "",
            "preset_category_id": "",
            "preset_has_set_alignment": False,
            "preset_id": "",
            "preset_index": 0,
            "preset_name": "",
            "recognize_task_id": "",
            "recognize_type": 0,
            "relevance_segment": [],
            "shadow_alpha": shadow_alpha,
            "shadow_angle": -45.0,
            "shadow_color": "#000000",
            "shadow_distance": shadow_distance,
            "shadow_point": {"x": 0.636, "y": -0.636},
            "shadow_smoothing": shadow_smoothing,
            "shape_clip_x": False,
            "shape_clip_y": False,
            "style_name": "",
            "sub_type": 0,
            "subtitle_keywords": None,
            "text_alpha": 1.0,
            "text_size": 30,
            "tts_auto_update": False,
            "typesetting": 0,
            "underline": False,
            "underline_offset": 0.22,
            "underline_width": 0.05,
            "use_effect_default_color": True,
            "words": {"end_time": [], "start_time": [], "text": []}
        }

    def _create_subtitle_segment(self, entry: SubtitleEntry, material_id: str,
                                  style: dict, render_index: int) -> dict:
        """創建字幕軌道片段"""
        return {
            "caption_info": None,
            "cartoon": False,
            "clip": {
                "alpha": 1.0,
                "flip": {"horizontal": False, "vertical": False},
                "rotation": 0.0,
                "scale": {"x": 1.0, "y": 1.0},
                "transform": {"x": 0.0, "y": style.get("position_y", -0.75)}
            },
            "common_keyframes": [],
            "enable_adjust": False,
            "enable_color_correct_adjust": False,
            "enable_color_curves": True,
            "enable_color_match_adjust": False,
            "enable_color_wheels": True,
            "enable_lut": False,
            "enable_smart_color_adjust": False,
            "extra_material_refs": [],
            "group_id": "",
            "hdr_settings": None,
            "id": str(uuid.uuid4()).upper(),
            "intensifies_audio": False,
            "is_placeholder": False,
            "is_tone_modify": False,
            "keyframe_refs": [],
            "last_nonzero_volume": 1.0,
            "material_id": material_id,
            "render_index": render_index,
            "responsive_layout": {
                "enable": False,
                "horizontal_pos_layout": 0,
                "size_layout": 0,
                "target_follow": "",
                "vertical_pos_layout": 0
            },
            "reverse": False,
            "source_timerange": None,
            "speed": 1.0,
            "target_timerange": {
                "duration": entry.duration_us,
                "start": entry.start_time_us
            },
            "template_id": "",
            "template_scene": "default",
            "track_attribute": 0,
            "track_render_index": 0,
            "uniform_scale": {"on": True, "value": 1.0},
            "visible": True,
            "volume": 1.0
        }

    def _add_subtitles_to_draft(self, draft_data: dict,
                                 entries: List[SubtitleEntry],
                                 template_data: dict = None) -> dict:
        """將字幕添加到草稿 - 使用設定檔的樣式"""
        style = self.config.get("subtitle_style", {})

        # 創建字幕素材和片段
        subtitle_materials = []
        subtitle_segments = []

        base_render_index = 20000  # 字幕在較高層級

        for i, entry in enumerate(entries):
            # 創建素材
            material = self._create_subtitle_material(entry, style)
            subtitle_materials.append(material)

            # 創建片段
            segment = self._create_subtitle_segment(
                entry, material["id"], style, base_render_index + i
            )
            subtitle_segments.append(segment)

        # 添加素材到 materials.texts
        if "texts" not in draft_data["materials"]:
            draft_data["materials"]["texts"] = []
        draft_data["materials"]["texts"].extend(subtitle_materials)

        # 創建新的字幕軌道
        subtitle_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()).upper(),
            "is_default_name": True,
            "name": "字幕軌道",
            "segments": subtitle_segments,
            "type": "text"
        }

        # 添加軌道
        draft_data["tracks"].append(subtitle_track)

        print(f"   [OK] 添加了 {len(entries)} 條字幕")
        return draft_data

    def _generate_ig_caption(self, video_name: str, entries: List[SubtitleEntry]):
        """生成 IG 文案"""
        import requests

        # 檢查是否有範例
        ig_config = self.config.get("ig_caption", {})
        examples = ig_config.get("examples", [])

        if not examples:
            # 沒有範例就跳過
            return

        print("\n[IG] Step 4: 生成 IG 文案")

        # 取得翻譯後的字幕內容
        subtitles = [entry.translated_text or entry.text for entry in entries]
        video_content = "\n".join(subtitles)

        # 取得 API 設定
        translation_config = self.config.get("translation", {})
        api_key = os.environ.get(translation_config.get("api_key_env", "DEEPSEEK_API_KEY"))
        base_url = translation_config.get("base_url", "https://api.deepseek.com")
        model = translation_config.get("model", "deepseek-chat")

        if not api_key:
            print("   [Skip] 找不到 API Key，跳過文案生成")
            return

        # 準備範例（最多 3 個）
        examples_text = "\n\n---\n\n".join(examples[:3])

        prompt = f"""你是一位專業的社群媒體文案寫手。請根據以下影片內容，用我的風格寫一篇 Instagram 文案。

## 我的文案風格範例：
{examples_text}

## 影片內容（字幕）：
{video_content}

## 要求：
1. 模仿我的語氣和風格（口語化、有趣、帶點幽默）
2. 開頭要吸引眼球
3. 用「-」分隔段落
4. 結尾加上相關的 hashtag（5-10個）
5. 總長度適合 IG 貼文（不要太長）
6. 繁體中文

請直接輸出文案，不要加任何解釋："""

        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                caption = result["choices"][0]["message"]["content"].strip()

                # 儲存文案
                caption_file = self.subtitle_folder / f"{video_name}_ig_caption.txt"
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)

                print(f"   [OK] IG 文案已儲存: {caption_file}")
                # 顯示前 100 字
                preview = caption[:100] + "..." if len(caption) > 100 else caption
                print(f"   預覽: {preview}")
            else:
                print(f"   [Error] API 錯誤: {response.status_code}")

        except Exception as e:
            print(f"   [Error] 生成失敗: {e}")

    def _create_title_material(self, title_text: str, duration_us: int) -> dict:
        """創建頂部標題素材 - 使用影片檔名和隨機背景顏色"""
        material_id = str(uuid.uuid4()).upper()
        title_style = self.config.get("title_style", {})

        # 隨機選擇背景顏色
        bg_colors = title_style.get("random_bg_colors", ["#ff0000", "#00ff00", "#0066ff"])

        # 日誌：顯示可選顏色列表
        print(f"   [Color] 可選顏色列表: {bg_colors}")

        # 隨機選擇背景顏色
        bg_color = random.choice(bg_colors)

        # 日誌：顯示選中的顏色
        print(f"   [Color] 隨機選中顏色: {bg_color}")

        font_size = title_style.get("font_size", 5.0)
        font_resource_id = title_style.get("font_resource_id", "7265534770845585980")
        border_width = title_style.get("border_width", 0.0365)
        border_color = title_style.get("border_color", "#666666")
        shadow_alpha = title_style.get("shadow_alpha", 0.9)
        background_alpha = title_style.get("background_alpha", 1.0)

        # 建立 content JSON
        text_rgb = self._hex_to_rgb("#FFFFFF")
        style_entry = {
            "fill": {"content": {"solid": {"color": text_rgb}}},
            "font": {
                "id": font_resource_id,
                "path": ""
            },
            "size": font_size,
            "range": [0, len(title_text)]
        }

        content_data = {
            "text": title_text,
            "styles": [style_entry]
        }

        return {
            "id": material_id,
            "type": "text",
            "content": json.dumps(content_data, ensure_ascii=False),
            "font_size": font_size,
            "text_color": "#FFFFFF",
            "background_color": bg_color,
            "background_alpha": background_alpha,
            "background_style": 1,
            "alignment": 1,
            "add_type": 0,
            "check_flag": 63,
            "combo_info": {"text_templates": []},
            "border_alpha": 1.0,
            "border_color": border_color,
            "border_width": border_width,
            "bold_width": 0.0,
            "font_category_id": "",
            "font_category_name": "",
            "font_id": "",
            "font_name": "",
            "font_path": "",
            "font_resource_id": font_resource_id,
            "font_source_platform": 0,
            "font_team_id": "",
            "font_title": "none",
            "font_url": "",
            "fonts": [],
            "global_alpha": 1.0,
            "has_shadow": True,
            "is_rich_text": False,
            "italic_degree": 0,
            "ktv_color": "",
            "language": "",
            "layer_weight": 1,
            "letter_spacing": 0.0,
            "line_feed": 1,
            "line_max_width": 0.82,
            "line_spacing": 0.02,
            "force_apply_line_max_width": True,
            "multi_language_current": "none",
            "name": "",
            "preset_category": "",
            "preset_category_id": "",
            "preset_has_set_alignment": False,
            "preset_id": "",
            "preset_index": 0,
            "preset_name": "",
            "recognize_task_id": "",
            "recognize_type": 0,
            "relevance_segment": [],
            "shadow_alpha": shadow_alpha,
            "shadow_angle": -45.0,
            "shadow_color": "#000000",
            "shadow_distance": 5.0,
            "shadow_point": {"x": 0.636, "y": -0.636},
            "shadow_smoothing": 0.45,
            "shape_clip_x": False,
            "shape_clip_y": False,
            "style_name": "",
            "sub_type": 0,
            "subtitle_keywords": None,
            "text_alpha": 1.0,
            "text_size": 30,
            "tts_auto_update": False,
            "typesetting": 0,
            "underline": False,
            "underline_offset": 0.22,
            "underline_width": 0.05,
            "use_effect_default_color": True,
            "words": {"end_time": [], "start_time": [], "text": []}
        }

    def _add_title_to_draft(self, draft_data: dict, video_name: str, duration_us: int) -> dict:
        """添加頂部標題（使用影片檔名）"""
        title_style = self.config.get("title_style", {})
        position_y = title_style.get("position_y", 0.8)

        # 創建標題素材
        title_material = self._create_title_material(video_name, duration_us)

        # 添加到素材列表
        if "texts" not in draft_data["materials"]:
            draft_data["materials"]["texts"] = []
        draft_data["materials"]["texts"].append(title_material)

        # 創建標題片段
        title_segment = {
            "caption_info": None,
            "cartoon": False,
            "clip": {
                "alpha": 1.0,
                "flip": {"horizontal": False, "vertical": False},
                "rotation": 0.0,
                "scale": {"x": 1.0, "y": 1.0},
                "transform": {"x": 0.0, "y": position_y}
            },
            "common_keyframes": [],
            "enable_adjust": False,
            "enable_color_correct_adjust": False,
            "enable_color_curves": True,
            "enable_color_match_adjust": False,
            "enable_color_wheels": True,
            "enable_lut": False,
            "enable_smart_color_adjust": False,
            "extra_material_refs": [],
            "group_id": "",
            "hdr_settings": None,
            "id": str(uuid.uuid4()).upper(),
            "intensifies_audio": False,
            "is_placeholder": False,
            "is_tone_modify": False,
            "keyframe_refs": [],
            "last_nonzero_volume": 1.0,
            "material_id": title_material["id"],
            "render_index": 30000,
            "responsive_layout": {
                "enable": False,
                "horizontal_pos_layout": 0,
                "size_layout": 0,
                "target_follow": "",
                "vertical_pos_layout": 0
            },
            "reverse": False,
            "source_timerange": None,
            "speed": 1.0,
            "target_timerange": {
                "duration": duration_us,
                "start": 0
            },
            "template_id": "",
            "template_scene": "default",
            "track_attribute": 0,
            "track_render_index": 0,
            "uniform_scale": {"on": True, "value": 1.0},
            "visible": True,
            "volume": 1.0
        }

        # 創建標題軌道
        title_track = {
            "attribute": 0,
            "flag": 0,
            "id": str(uuid.uuid4()).upper(),
            "is_default_name": True,
            "name": "標題軌道",
            "segments": [title_segment],
            "type": "text"
        }

        draft_data["tracks"].append(title_track)
        print(f"   [Title] 添加頂部標題: {video_name}")
        return draft_data

    def _clean_template_texts(self, draft_data: dict, keep_count: int = 2) -> dict:
        """清理模板中的字幕文字，但保留前 N 個模板文字（標題、@html_cat 等）"""
        texts = draft_data.get("materials", {}).get("texts", [])

        if len(texts) > keep_count:
            # 保留前 keep_count 個，刪除其餘（英文字幕）
            kept_text_ids = set(t.get("id") for t in texts[:keep_count])
            draft_data["materials"]["texts"] = texts[:keep_count]
            print(f"   [Clean] 保留 {keep_count} 個模板文字，清除 {len(texts) - keep_count} 個字幕")

            # 清理軌道中引用被刪除素材的片段
            for track in draft_data.get("tracks", []):
                if track.get("type") == "text":
                    original_count = len(track.get("segments", []))
                    track["segments"] = [
                        seg for seg in track.get("segments", [])
                        if seg.get("material_id") in kept_text_ids
                    ]
                    removed = original_count - len(track["segments"])
                    if removed > 0:
                        print(f"   [Clean] 從軌道移除 {removed} 個字幕片段")

        return draft_data

    def _update_template_texts(self, draft_data: dict, video_name: str, duration_us: int) -> dict:
        """更新模板文字：第一個改為影片檔名，所有保留的文字更新時長"""
        texts = draft_data.get("materials", {}).get("texts", [])

        # 收集需要更新時長的素材 ID
        text_ids_to_update = set()

        for i, text_material in enumerate(texts):
            text_ids_to_update.add(text_material.get("id"))

            if i == 0:
                # 第一個文字改為影片檔名 + 隨機背景顏色
                title_style = self.config.get("title_style", {})
                bg_colors = title_style.get("random_bg_colors", ["#ff0000", "#00ff00", "#0066ff"])

                # 日誌：顯示可選顏色列表
                print(f"   [Color] 可選顏色列表: {bg_colors}")

                # 隨機選擇背景顏色
                bg_color = random.choice(bg_colors)

                # 日誌：顯示選中的顏色
                print(f"   [Color] 隨機選中顏色: {bg_color}")

                # 更新背景顏色
                text_material["background_color"] = bg_color

                # 更新文字內容
                try:
                    content = json.loads(text_material.get("content", "{}"))
                    old_text = content.get("text", "")
                    content["text"] = video_name
                    # 更新 styles 的 range
                    if "styles" in content:
                        for style in content["styles"]:
                            if "range" in style:
                                style["range"] = [0, len(video_name)]
                    text_material["content"] = json.dumps(content, ensure_ascii=False)
                    print(f"   [Title] 標題: {old_text} -> {video_name} (背景: {bg_color})")
                except:
                    pass

        # 更新軌道片段的時長
        for track in draft_data.get("tracks", []):
            if track.get("type") == "text":
                for segment in track.get("segments", []):
                    if segment.get("material_id") in text_ids_to_update:
                        if "target_timerange" in segment:
                            segment["target_timerange"]["duration"] = duration_us
                            segment["target_timerange"]["start"] = 0

        return draft_data

    def _replace_video_in_draft(self, draft_data: dict, video_path: str) -> dict:
        """替換草稿中的影片 - 使用與面相專案相同的邏輯"""
        import pyJianYingDraft as pjy

        # 確保使用絕對路徑
        video_path = os.path.abspath(video_path)

        # 創建新影片素材
        new_video = pjy.VideoMaterial(video_path)
        print(f"   [Video] 新影片路徑: {new_video.path}")
        print(f"   [Video] 新影片素材 ID: {new_video.material_id}")
        print(f"   [Video] 影片時長: {new_video.duration/1000000:.2f} 秒")

        # 分析原有素材，分類處理
        original_videos = draft_data.get("materials", {}).get("videos", [])
        preserved_materials = []  # 保留的素材（圖片等）
        replaced_material_ids = set()  # 被替換的影片素材 ID

        for material in original_videos:
            material_type = material.get("type", "unknown")
            material_id = material.get("id", "")

            if material_type == "video":
                # 這是真正的影片素材，需要替換
                replaced_material_ids.add(material_id)
                print(f"   [Video] 將替換影片素材: {os.path.basename(material.get('path', ''))} (id={material_id})")
            else:
                # 這是圖片或其他素材，保留
                preserved_materials.append(material)
                print(f"   [Image] 保留素材: {os.path.basename(material.get('path', ''))} (type={material_type})")

        # 如果找到需要替換的影片素材，基於第一個創建新素材
        if replaced_material_ids:
            # 找到第一個影片素材作為模板
            video_template = None
            for material in original_videos:
                if material.get("type") == "video":
                    video_template = copy.deepcopy(material)
                    break

            if video_template:
                # 更新為新影片的屬性，使用 VideoMaterial 的絕對路徑
                video_template.update({
                    "id": new_video.material_id,
                    "path": new_video.path,
                    "duration": new_video.duration,
                    "material_id": new_video.material_id,
                    "width": new_video.width,
                    "height": new_video.height,
                    "material_name": new_video.material_name,
                })

                # 新素材列表 = 保留的素材 + 新影片
                draft_data["materials"]["videos"] = preserved_materials + [video_template]
        else:
            # 沒有找到影片素材，使用 export_json 生成完整素材
            print("   [Warning] 模板中沒有找到影片素材，直接添加")
            draft_data["materials"]["videos"].append(new_video.export_json())

        # 更新軌道片段的 material_id
        segments_updated = 0
        for track in draft_data.get("tracks", []):
            if track.get("type") != "video":
                continue

            for segment in track.get("segments", []):
                old_material_id = segment.get("material_id", "")

                # 只替換指向原影片素材的片段
                if old_material_id in replaced_material_ids:
                    segment["material_id"] = new_video.material_id

                    # 更新時間範圍
                    if "target_timerange" in segment:
                        segment["target_timerange"]["duration"] = new_video.duration
                    if "source_timerange" in segment:
                        segment["source_timerange"]["duration"] = new_video.duration

                    segments_updated += 1

        print(f"   [OK] 更新了 {segments_updated} 個軌道片段")

        # 更新草稿總時長
        draft_data["duration"] = new_video.duration

        return draft_data

    def process_video(self, video_path: str,
                      skip_transcribe: bool = False,
                      skip_translate: bool = False,
                      force: bool = False) -> Optional[str]:
        """
        處理單個影片

        Args:
            video_path: 影片路徑
            skip_transcribe: 跳過語音識別（使用現有字幕檔）
            skip_translate: 跳過翻譯
            force: 強制重新處理（即使草稿已存在）

        Returns:
            生成的草稿名稱，失敗返回 None
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"[Error] 找不到影片: {video_path}")
            return None

        video_name = video_path.stem
        output_name = f"{self.output_prefix}{video_name}"

        # 檢查草稿是否已存在
        output_folder = self.jianying_draft_root / output_name
        if output_folder.exists() and not force:
            print(f"\n[Skip] 已存在草稿: {output_name}（使用 --force 強制重新處理）")
            return output_name

        print(f"\n{'='*60}")
        print(f"[Video] 處理影片: {video_name}")
        print(f"{'='*60}")

        # Step 1: 語音識別
        subtitle_json = self.subtitle_folder / f"{video_name}.json"

        if skip_transcribe and subtitle_json.exists():
            print("[File] 載入現有字幕檔...")
            entries = self.subtitle_gen.load_from_json(str(subtitle_json))
        else:
            print("[Audio] Step 1: 語音識別 (Whisper)")
            entries = self.subtitle_gen.transcribe(str(video_path))

            # 儲存原始字幕
            self.subtitle_gen.export_srt(
                entries,
                str(self.subtitle_folder / f"{video_name}_en.srt"),
                use_translated=False
            )

        # Step 2: 翻譯
        if not skip_translate:
            # 檢查是否已經翻譯過（避免浪費 API 額度）
            already_translated = all(
                entry.text_translated and entry.text_translated.strip()
                for entry in entries
            ) if entries else False

            if already_translated:
                print("[Skip] 字幕已翻譯過，跳過翻譯步驟")
            else:
                print("[Web] Step 2: 翻譯字幕")
                entries = self.subtitle_gen.translate_entries(entries)

            # 儲存翻譯後的字幕
            self.subtitle_gen.export_srt(
                entries,
                str(self.subtitle_folder / f"{video_name}_zh.srt"),
                use_translated=True
            )

        # 儲存完整字幕資料
        self.subtitle_gen.export_json(entries, str(subtitle_json))

        # Step 3: 生成剪映草稿
        print("[Note] Step 3: 生成剪映草稿")

        template_data = self._load_template()
        if not template_data:
            return None

        # 深度複製模板
        draft_data = copy.deepcopy(template_data)

        # 清理模板中的英文字幕，保留前 2 個模板文字（標題、@html_cat）
        draft_data = self._clean_template_texts(draft_data, keep_count=2)

        # 替換影片
        draft_data = self._replace_video_in_draft(draft_data, str(video_path))

        # 取得影片時長
        video_duration = draft_data.get("duration", 0)

        # 更新模板文字（標題改為影片檔名 + 隨機背景，更新時長）
        draft_data = self._update_template_texts(draft_data, video_name, video_duration)

        # 添加翻譯後的字幕（使用原始模板的字幕樣式）
        draft_data = self._add_subtitles_to_draft(draft_data, entries, template_data)

        # 儲存草稿
        output_folder = self.jianying_draft_root / output_name
        if output_folder.exists():
            # Windows 上剪映會創建 .backup 資料夾，可能無法刪除
            # 改用忽略錯誤的方式刪除
            def remove_readonly(func, path, excinfo):
                import stat
                os.chmod(path, stat.S_IWRITE)
                func(path)
            try:
                shutil.rmtree(output_folder, onerror=remove_readonly)
            except Exception as e:
                print(f"   [Warning] 無法完全刪除舊資料夾，將覆蓋檔案: {e}")
        output_folder.mkdir(parents=True, exist_ok=True)

        # 寫入 draft_content.json
        with open(output_folder / "draft_content.json", 'w', encoding='utf-8') as f:
            json.dump(draft_data, f, ensure_ascii=False)

        # 複製其他模板檔案
        template_folder = self.jianying_draft_root / self.template_name
        for file in ["draft_meta_info.json", "draft_settings"]:
            src = template_folder / file
            if src.exists():
                shutil.copy(src, output_folder / file)

        print(f"\n[OK] 草稿生成完成: {output_name}")
        print(f"   字幕數量: {len(entries)}")
        print(f"   位置: {output_folder}")

        return output_name

    def batch_process(self, video_folder: str = None, force: bool = False,
                      parallel: bool = None, max_workers: int = None,
                      pipeline_mode: bool = None):
        """批量處理影片

        Args:
            video_folder: 影片資料夾路徑
            force: 強制重新處理
            parallel: 是否啟用並行處理（None 表示使用設定檔）
            max_workers: 最大並行數量（None 表示使用設定檔）
            pipeline_mode: 是否使用 pipeline 模式（None 表示使用設定檔）
        """
        if video_folder is None:
            # 使用翻譯專案專屬的影片資料夾
            video_folder = self.config.get("input", {}).get("videos_folder", "videos/translate_raw")

        video_folder = Path(video_folder)
        if not video_folder.exists():
            print(f"[Error] 找不到影片資料夾: {video_folder}")
            return

        # 讀取並行處理設定
        parallel_config = self.config.get("parallel", {})
        if parallel is None:
            parallel = parallel_config.get("enabled", False)
        if max_workers is None:
            max_workers = parallel_config.get("max_workers", 2)

        # 檢查 pipeline 模式設定
        if pipeline_mode is None:
            config_mode = parallel_config.get("mode", "sequential")
            pipeline_mode = config_mode == "pipeline"

        print(f"[Video] 批量處理翻譯影片")
        print(f"[Folder] 影片資料夾: {video_folder}")
        if force:
            print("[Mode] 強制模式：會重新處理所有影片")
        else:
            print("[Mode] 一般模式：跳過已存在的草稿")

        if pipeline_mode:
            print(f"[Pipeline] Pipeline 模式：transcribe(1) -> translate({parallel_config.get('translate_workers', 4)}) -> draft({parallel_config.get('draft_workers', 2)})")
        elif parallel:
            print(f"[Parallel] 並行處理模式：同時處理 {max_workers} 個影片")
        else:
            print("[Parallel] 單執行緒模式")
        print("=" * 60)

        # 找到所有影片
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_folder.glob(ext))

        if not video_files:
            print("[Error] 找不到任何影片檔案")
            return

        print(f"[List] 找到 {len(video_files)} 個影片")

        success_count = 0
        failed_count = 0

        if pipeline_mode and len(video_files) > 1:
            # Pipeline 模式 - 使用 TranscriptionPipeline
            pipeline = TranscriptionPipeline(self, parallel_config)
            results = pipeline.process([str(f) for f in video_files], force=force)

            success_count = results["stats"]["completed"]
            failed_count = results["stats"]["failed"]

            # 結果已在 pipeline.process() 中打印

        elif parallel and len(video_files) > 1:
            # 並行處理模式 (舊版)
            print(f"\n[Start] 使用 {max_workers} 個執行緒並行處理...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任務
                future_to_video = {
                    executor.submit(self._process_video_safe, str(video_file), force): video_file
                    for video_file in video_files
                }

                # 處理完成的任務
                completed = 0
                for future in as_completed(future_to_video):
                    video_file = future_to_video[future]
                    completed += 1

                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                            print(f"\n[Progress] {completed}/{len(video_files)} - 成功: {video_file.name}")
                        else:
                            failed_count += 1
                            print(f"\n[Progress] {completed}/{len(video_files)} - 失敗: {video_file.name}")
                    except Exception as e:
                        failed_count += 1
                        print(f"\n[Error] {completed}/{len(video_files)} - 處理失敗: {video_file.name}")
                        print(f"   錯誤: {str(e)}")

            print(f"\n{'='*60}")
            print(f"[Done] 批量處理完成!")
            print(f"   成功: {success_count}/{len(video_files)}")
            if failed_count > 0:
                print(f"   失敗: {failed_count}/{len(video_files)}")
        else:
            # 單執行緒處理模式
            for i, video_file in enumerate(video_files, 1):
                print(f"\n[{i}/{len(video_files)}]")
                try:
                    result = self.process_video(str(video_file), force=force)
                    if result:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"[Error] 處理失敗: {video_file.name}")
                    print(f"   錯誤: {str(e)}")

            print(f"\n{'='*60}")
            print(f"[Done] 批量處理完成!")
            print(f"   成功: {success_count}/{len(video_files)}")
            if failed_count > 0:
                print(f"   失敗: {failed_count}/{len(video_files)}")

    def _process_video_safe(self, video_path: str, force: bool = False) -> Optional[str]:
        """安全地處理影片（用於並行處理，捕獲例外）

        Args:
            video_path: 影片路徑
            force: 強制重新處理

        Returns:
            成功返回草稿名稱，失敗返回 None
        """
        try:
            video_name = Path(video_path).stem
            print(f"\n[Start] 處理: {video_name}")
            result = self.process_video(video_path, force=force)
            if result:
                print(f"[OK] 完成: {video_name}")
            return result
        except Exception as e:
            video_name = Path(video_path).stem
            print(f"\n[Error] 處理失敗: {video_name}")
            print(f"   錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="翻譯影片工作流程")
    parser.add_argument("video", nargs="?", help="影片路徑")
    parser.add_argument("--batch", "-b", action="store_true", help="批量處理模式")
    parser.add_argument("--folder", "-f", help="指定影片資料夾")
    parser.add_argument("--skip-transcribe", action="store_true", help="跳過語音識別")
    parser.add_argument("--skip-translate", action="store_true", help="跳過翻譯")
    parser.add_argument("--force", action="store_true", help="強制重新處理（即使草稿已存在）")
    parser.add_argument("--parallel", "-p", action="store_true", help="啟用並行處理（舊版模式）")
    parser.add_argument("--pipeline", action="store_true", help="啟用 Pipeline 模式（推薦）")
    parser.add_argument("--workers", type=int, help="並行處理的執行緒數量（預設使用設定檔）")
    parser.add_argument("--translate-workers", type=int, help="翻譯並行數量（Pipeline 模式，預設 4）")
    parser.add_argument("--draft-workers", type=int, help="草稿生成並行數量（Pipeline 模式，預設 2）")

    args = parser.parse_args()

    workflow = TranslationWorkflow()

    # 如果指定了 pipeline 相關參數，更新設定
    if args.translate_workers or args.draft_workers:
        parallel_config = workflow.config.get("parallel", {})
        if args.translate_workers:
            parallel_config["translate_workers"] = args.translate_workers
        if args.draft_workers:
            parallel_config["draft_workers"] = args.draft_workers
        workflow.config["parallel"] = parallel_config

    if args.batch or args.folder:
        workflow.batch_process(
            args.folder,
            force=args.force,
            parallel=args.parallel if args.parallel else None,
            max_workers=args.workers,
            pipeline_mode=args.pipeline if args.pipeline else None
        )
    elif args.video:
        workflow.process_video(
            args.video,
            skip_transcribe=args.skip_transcribe,
            skip_translate=args.skip_translate,
            force=args.force
        )
    else:
        # 預設批量處理
        videos_folder = workflow.config.get("input", {}).get("videos_folder", "videos/translate_raw")
        parallel_config = workflow.config.get("parallel", {})
        print("翻譯影片工作流程")
        print("=" * 60)
        print("使用方式:")
        print("  python translate_video.py video.mp4                      # 處理單個影片")
        print(f"  python translate_video.py --batch                        # 批量處理 {videos_folder}")
        print("  python translate_video.py --batch --parallel             # 並行批量處理（舊版）")
        print("  python translate_video.py --batch --pipeline             # Pipeline 模式（推薦）")
        print("  python translate_video.py --batch --pipeline --translate-workers 6 --draft-workers 3")
        print("  python translate_video.py --folder <path>                # 指定資料夾")
        print()
        print("Pipeline 模式說明:")
        print("  - 語音識別：單執行緒（GPU bound）")
        print(f"  - 翻譯：{parallel_config.get('translate_workers', 4)} 個執行緒（I/O bound）")
        print(f"  - 草稿生成：{parallel_config.get('draft_workers', 2)} 個執行緒（I/O bound）")
        print("  - 各階段可並行處理不同影片")
        print()

        user_input = input(f"是否開始批量處理 {videos_folder}? (y/n/pipeline): ").strip().lower()
        if user_input == 'y':
            workflow.batch_process()
        elif user_input == 'pipeline':
            workflow.batch_process(pipeline_mode=True)


if __name__ == "__main__":
    main()
