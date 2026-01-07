# -*- coding: utf-8 -*-
"""
翻譯處理面板 - 核心 GUI 元件
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog
import threading
import queue
import os
import sys
from typing import List, Optional, Callable
from datetime import datetime

# 加入專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.utils.theme import COLORS, FONTS
from gui.utils.config_manager import ConfigManager


class VideoListItem(ctk.CTkFrame):
    """影片列表項目"""

    def __init__(self, parent, video_path: str, is_processed: bool = False, **kwargs):
        super().__init__(parent, **kwargs)

        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.file_size = self._get_file_size()

        # Checkbox 變數
        self.selected = ctk.BooleanVar(value=not is_processed)
        self.is_processed = is_processed

        self._setup_ui()

    def _get_file_size(self) -> str:
        """取得檔案大小"""
        try:
            size = os.path.getsize(self.video_path)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        except:
            return "N/A"

    def _setup_ui(self):
        """建立 UI"""
        self.grid_columnconfigure(1, weight=1)

        # Checkbox
        self.checkbox = ctk.CTkCheckBox(
            self,
            text="",
            variable=self.selected,
            width=24,
            checkbox_width=20,
            checkbox_height=20
        )
        self.checkbox.grid(row=0, column=0, padx=(5, 10), pady=5)

        # 檔名
        status_text = " (已處理)" if self.is_processed else ""
        name_color = COLORS["text_secondary"] if self.is_processed else COLORS["text"]
        self.name_label = ctk.CTkLabel(
            self,
            text=f"{self.video_name}{status_text}",
            font=ctk.CTkFont(size=12),
            text_color=name_color,
            anchor="w"
        )
        self.name_label.grid(row=0, column=1, sticky="w", pady=5)

        # 檔案大小
        self.size_label = ctk.CTkLabel(
            self,
            text=self.file_size,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"],
            width=80
        )
        self.size_label.grid(row=0, column=2, padx=10, pady=5)


class TranslatePanel(ctk.CTkFrame):
    """翻譯處理面板"""

    def __init__(self, parent, config_manager: ConfigManager):
        super().__init__(parent)

        self.config_manager = config_manager
        self.video_items: List[VideoListItem] = []
        self.is_processing = False
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self._stop_flag = threading.Event()

        # 預設資料夾路徑
        self.videos_folder = PROJECT_ROOT / self.config_manager.get("input", "videos_folder", default="videos/translate_raw")

        self._setup_ui()

    def _setup_ui(self):
        """建立 UI"""
        # 配置 grid - 左右兩欄
        self.grid_columnconfigure(0, weight=3)  # 左側較寬
        self.grid_columnconfigure(1, weight=2)  # 右側較窄
        self.grid_rowconfigure(0, weight=1)

        # === 左側面板：影片列表 ===
        self._create_left_panel()

        # === 右側面板：設定與進度 ===
        self._create_right_panel()

    def _create_left_panel(self):
        """建立左側面板 - 影片列表"""
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(2, weight=1)

        # --- 資料夾選擇 ---
        folder_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        folder_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        folder_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            folder_frame,
            text="📁 輸入資料夾",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))

        self.folder_entry = ctk.CTkEntry(
            folder_frame,
            placeholder_text="選擇影片資料夾...",
            height=35
        )
        self.folder_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 10))
        self.folder_entry.insert(0, str(self.videos_folder))

        self.browse_btn = ctk.CTkButton(
            folder_frame,
            text="瀏覽",
            width=80,
            height=35,
            command=self._browse_folder
        )
        self.browse_btn.grid(row=1, column=2)

        # --- 影片列表標題 ---
        list_header = ctk.CTkFrame(left_frame, fg_color="transparent")
        list_header.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 5))
        list_header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            list_header,
            text="📋 待處理影片",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self.video_count_label = ctk.CTkLabel(
            list_header,
            text="(0 個影片)",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        self.video_count_label.grid(row=0, column=1, sticky="w", padx=10)

        self.refresh_btn = ctk.CTkButton(
            list_header,
            text="🔄 重新掃描",
            width=100,
            height=28,
            font=ctk.CTkFont(size=11),
            command=self._scan_videos
        )
        self.refresh_btn.grid(row=0, column=2)

        # --- 影片列表 (Scrollable) ---
        self.video_list_frame = ctk.CTkScrollableFrame(
            left_frame,
            label_text="",
            fg_color=COLORS["surface"]
        )
        self.video_list_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        self.video_list_frame.grid_columnconfigure(0, weight=1)

        # --- 批量操作按鈕 ---
        action_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        action_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(
            action_frame,
            text="全選",
            width=70,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            border_width=1,
            command=self._select_all
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            action_frame,
            text="取消全選",
            width=80,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            border_width=1,
            command=self._deselect_all
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            action_frame,
            text="選擇未處理",
            width=90,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            border_width=1,
            command=self._select_unprocessed
        ).pack(side="left", padx=5)

        # 初始掃描
        self._scan_videos()

    def _create_right_panel(self):
        """建立右側面板 - 設定與進度"""
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(3, weight=1)  # 日誌區域可伸展

        # --- 處理設定 ---
        settings_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        settings_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        settings_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            settings_frame,
            text="⚙️ 處理設定",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Whisper 模型
        ctk.CTkLabel(settings_frame, text="Whisper 模型:", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.model_menu = ctk.CTkOptionMenu(
            settings_frame,
            values=["tiny", "base", "small", "medium", "large-v3"],
            width=150
        )
        self.model_menu.set(self.config_manager.get("whisper", "model", default="base"))
        self.model_menu.grid(row=1, column=1, sticky="e", pady=5)

        # 處理模式
        ctk.CTkLabel(settings_frame, text="處理模式:", font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.mode_menu = ctk.CTkOptionMenu(
            settings_frame,
            values=["pipeline", "sequential", "parallel"],
            width=150
        )
        self.mode_menu.set(self.config_manager.get("parallel", "mode", default="pipeline"))
        self.mode_menu.grid(row=2, column=1, sticky="e", pady=5)

        # 翻譯執行緒數
        ctk.CTkLabel(settings_frame, text="翻譯執行緒:", font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, sticky="w", pady=5
        )
        self.workers_menu = ctk.CTkOptionMenu(
            settings_frame,
            values=["1", "2", "4", "8"],
            width=150
        )
        self.workers_menu.set(str(self.config_manager.get("parallel", "translate_workers", default=4)))
        self.workers_menu.grid(row=3, column=1, sticky="e", pady=5)

        # 強制重新處理
        self.force_var = ctk.BooleanVar(value=False)
        self.force_checkbox = ctk.CTkCheckBox(
            settings_frame,
            text="強制重新處理 (覆蓋已存在的字幕/草稿)",
            variable=self.force_var,
            font=ctk.CTkFont(size=11)
        )
        self.force_checkbox.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 5))

        # --- 控制按鈕 ---
        control_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        control_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=10)
        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)

        self.start_btn = ctk.CTkButton(
            control_frame,
            text="▶ 開始翻譯",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["primary"],
            hover_color="#00a8cc",
            command=self._start_processing
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="⏹ 停止",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["error"],
            hover_color="#cc3333",
            state="disabled",
            command=self.stop_processing
        )
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # --- 進度顯示 ---
        progress_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        progress_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=10)
        progress_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            progress_frame,
            text="📊 處理進度",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        # 整體進度
        ctk.CTkLabel(progress_frame, text="整體進度:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, sticky="w"
        )
        self.overall_progress = ctk.CTkProgressBar(progress_frame, height=15)
        self.overall_progress.grid(row=2, column=0, sticky="ew", pady=(2, 5))
        self.overall_progress.set(0)

        self.overall_label = ctk.CTkLabel(
            progress_frame,
            text="0 / 0 (0%)",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        self.overall_label.grid(row=3, column=0, sticky="e")

        # 當前任務
        ctk.CTkLabel(progress_frame, text="當前任務:", font=ctk.CTkFont(size=11)).grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        self.current_task_label = ctk.CTkLabel(
            progress_frame,
            text="-",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["primary"]
        )
        self.current_task_label.grid(row=5, column=0, sticky="w")

        self.stage_progress = ctk.CTkProgressBar(progress_frame, height=12, mode="indeterminate")
        self.stage_progress.grid(row=6, column=0, sticky="ew", pady=(5, 0))

        # --- 日誌 ---
        log_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        log_frame.grid(row=3, column=0, sticky="nsew", padx=15, pady=(10, 15))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_frame,
            text="📝 處理日誌",
            font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.log_textbox = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=COLORS["surface"],
            wrap="word"
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew")

    # === 功能方法 ===

    def _browse_folder(self):
        """瀏覽資料夾"""
        folder = filedialog.askdirectory(
            initialdir=str(self.videos_folder),
            title="選擇影片資料夾"
        )
        if folder:
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, folder)
            self.videos_folder = Path(folder)
            self._scan_videos()

    def _scan_videos(self):
        """掃描影片資料夾"""
        # 清除現有列表
        for item in self.video_items:
            item.destroy()
        self.video_items.clear()

        folder = Path(self.folder_entry.get())
        if not folder.exists():
            self.video_count_label.configure(text="(資料夾不存在)")
            return

        # 掃描影片檔案
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder.glob(f"*{ext}"))
            video_files.extend(folder.glob(f"*{ext.upper()}"))

        # 排序
        video_files = sorted(set(video_files), key=lambda x: x.name.lower())

        # 檢查哪些已處理
        subtitle_folder = PROJECT_ROOT / self.config_manager.get("output", "subtitle_folder", default="subtitles")

        for video_path in video_files:
            video_name = video_path.stem
            # 檢查是否已有字幕 JSON
            is_processed = (subtitle_folder / f"{video_name}.json").exists()

            item = VideoListItem(
                self.video_list_frame,
                str(video_path),
                is_processed=is_processed
            )
            item.pack(fill="x", pady=2)
            self.video_items.append(item)

        # 更新計數
        total = len(video_files)
        processed = sum(1 for item in self.video_items if item.is_processed)
        self.video_count_label.configure(text=f"({total} 個影片, {processed} 已處理)")

    def _select_all(self):
        """全選"""
        for item in self.video_items:
            item.selected.set(True)

    def _deselect_all(self):
        """取消全選"""
        for item in self.video_items:
            item.selected.set(False)

    def _select_unprocessed(self):
        """只選擇未處理的"""
        for item in self.video_items:
            item.selected.set(not item.is_processed)

    def _get_selected_videos(self) -> List[str]:
        """取得選中的影片路徑"""
        return [item.video_path for item in self.video_items if item.selected.get()]

    def _log(self, message: str):
        """添加日誌訊息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def _update_ui(self):
        """更新 UI (從主執行緒呼叫)"""
        # 處理日誌佇列
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_textbox.insert("end", msg + "\n")
                self.log_textbox.see("end")
            except queue.Empty:
                break

        # 處理進度佇列
        while not self.progress_queue.empty():
            try:
                progress_data = self.progress_queue.get_nowait()
                self._apply_progress(progress_data)
            except queue.Empty:
                break

        # 如果還在處理中，繼續更新
        if self.is_processing:
            self.after(100, self._update_ui)

    def _apply_progress(self, data: dict):
        """套用進度更新"""
        if "overall" in data:
            completed, total = data["overall"]
            progress = completed / total if total > 0 else 0
            self.overall_progress.set(progress)
            self.overall_label.configure(text=f"{completed} / {total} ({int(progress * 100)}%)")

        if "current" in data:
            self.current_task_label.configure(text=data["current"])

        if "stage" in data:
            stage = data["stage"]
            if stage == "done":
                self.stage_progress.stop()
                self.stage_progress.configure(mode="determinate")
                self.stage_progress.set(1)
            else:
                self.stage_progress.configure(mode="indeterminate")
                self.stage_progress.start()

    def _start_processing(self):
        """開始處理"""
        selected_videos = self._get_selected_videos()

        if not selected_videos:
            self._log("⚠️ 請先選擇要處理的影片")
            return

        # 更新 UI 狀態
        self.is_processing = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._stop_flag.clear()

        # 清空日誌
        self.log_textbox.delete("1.0", "end")

        # 更新配置
        self.config_manager.set("whisper", "model", self.model_menu.get())
        self.config_manager.set("parallel", "mode", self.mode_menu.get())
        self.config_manager.set("parallel", "translate_workers", int(self.workers_menu.get()))
        self.config_manager.save()

        force = self.force_var.get()

        self._log(f"開始處理 {len(selected_videos)} 個影片...")
        self._log(f"模型: {self.model_menu.get()}, 模式: {self.mode_menu.get()}")

        # 初始化進度
        self.progress_queue.put({"overall": (0, len(selected_videos))})
        self.progress_queue.put({"stage": "running"})

        # 啟動背景執行緒
        thread = threading.Thread(
            target=self._run_translation,
            args=(selected_videos, force),
            daemon=True
        )
        thread.start()

        # 啟動 UI 更新
        self._update_ui()

    def _run_translation(self, video_files: List[str], force: bool):
        """在背景執行緒中執行翻譯"""
        try:
            from translate_video import TranslationWorkflow, TranscriptionPipeline

            # 建立 workflow
            workflow = TranslationWorkflow()

            # 使用 Pipeline 模式
            config = self.config_manager.config.get("parallel", {})
            pipeline = TranscriptionPipeline(workflow, config)

            # 設定停止旗標
            original_stop = pipeline._stop_event

            # 自訂進度回調
            completed_count = 0
            total = len(video_files)

            # 覆寫 task 完成處理
            original_handle = pipeline._handle_task_error
            original_draft = pipeline._draft_worker

            def on_task_complete(task):
                nonlocal completed_count
                completed_count += 1
                self.progress_queue.put({"overall": (completed_count, total)})
                self._log(f"✓ 完成: {task.video_name}")

            # 開始處理
            for i, video_path in enumerate(video_files):
                if self._stop_flag.is_set():
                    self._log("⚠️ 使用者中止處理")
                    break

                video_name = Path(video_path).stem
                self.progress_queue.put({"current": video_name})
                self._log(f"處理中: {video_name} ({i+1}/{total})")

            # 執行 pipeline
            result = pipeline.process(video_files, force=force)

            # 處理結果
            if not self._stop_flag.is_set():
                success_count = len(result.get("success", []))
                failed_count = len(result.get("failed", []))

                self.progress_queue.put({"overall": (total, total)})
                self.progress_queue.put({"stage": "done"})

                self._log(f"")
                self._log(f"{'='*40}")
                self._log(f"處理完成！")
                self._log(f"成功: {success_count} 個")
                if failed_count > 0:
                    self._log(f"失敗: {failed_count} 個")
                    for item in result.get("failed", []):
                        self._log(f"  - {item['video']}: {item['error'][:50]}...")

        except Exception as e:
            self._log(f"❌ 錯誤: {str(e)}")
            import traceback
            self._log(traceback.format_exc())

        finally:
            # 更新 UI 狀態
            self.is_processing = False
            self.after(0, self._on_processing_done)

    def _on_processing_done(self):
        """處理完成後的 UI 更新"""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_queue.put({"stage": "done"})
        self.progress_queue.put({"current": "完成"})

        # 重新掃描影片列表
        self._scan_videos()

    def stop_processing(self):
        """停止處理"""
        self._stop_flag.set()
        self._log("正在停止...")
        self.stop_btn.configure(state="disabled")
