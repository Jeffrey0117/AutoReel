# -*- coding: utf-8 -*-
"""
影片管理面板 - 預覽影片並重新命名
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog, messagebox
import os
import sys
import subprocess
import threading
from typing import List, Optional

# 加入專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.utils.theme import COLORS
from gui.utils.config_manager import ConfigManager


class VideoItem(ctk.CTkFrame):
    """影片列表項目"""

    def __init__(self, parent, video_path: str, on_select_callback=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.extension = Path(video_path).suffix
        self.on_select_callback = on_select_callback
        self.is_selected = False

        self._setup_ui()

    def _get_file_size(self) -> str:
        """取得檔案大小"""
        try:
            size = os.path.getsize(self.video_path)
            if size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        except:
            return "N/A"

    def _get_duration(self) -> str:
        """取得影片長度 (使用 ffprobe)"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', self.video_path],
                capture_output=True, text=True, timeout=5
            )
            duration = float(result.stdout.strip())
            mins = int(duration // 60)
            secs = int(duration % 60)
            return f"{mins}:{secs:02d}"
        except:
            return "--:--"

    def _setup_ui(self):
        """建立 UI"""
        self.configure(fg_color="transparent", cursor="hand2")
        self.grid_columnconfigure(0, weight=1)

        # 點擊事件
        self.bind("<Button-1>", self._on_click)

        # 檔名
        self.name_label = ctk.CTkLabel(
            self,
            text=self.video_name,
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.name_label.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 2))
        self.name_label.bind("<Button-1>", self._on_click)

        # 資訊列
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))

        self.size_label = ctk.CTkLabel(
            info_frame,
            text=self._get_file_size(),
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_secondary"]
        )
        self.size_label.pack(side="left")

        ctk.CTkLabel(
            info_frame,
            text=" | ",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_secondary"]
        ).pack(side="left")

        self.duration_label = ctk.CTkLabel(
            info_frame,
            text="--:--",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_secondary"]
        )
        self.duration_label.pack(side="left")

        # 背景載入時長
        threading.Thread(target=self._load_duration, daemon=True).start()

    def _load_duration(self):
        """背景載入影片時長"""
        duration = self._get_duration()
        self.after(0, lambda: self.duration_label.configure(text=duration))

    def _on_click(self, event=None):
        """點擊事件"""
        if self.on_select_callback:
            self.on_select_callback(self)

    def set_selected(self, selected: bool):
        """設定選中狀態"""
        self.is_selected = selected
        if selected:
            self.configure(fg_color=COLORS["primary"])
            self.name_label.configure(text_color="#000000")
        else:
            self.configure(fg_color="transparent")
            self.name_label.configure(text_color=COLORS["text"])


class VideoManagerPanel(ctk.CTkFrame):
    """影片管理面板"""

    def __init__(self, parent, config_manager: ConfigManager):
        super().__init__(parent)

        self.config_manager = config_manager
        self.video_items: List[VideoItem] = []
        self.selected_item: Optional[VideoItem] = None

        # 預設資料夾路徑
        self.videos_folder = PROJECT_ROOT / self.config_manager.get(
            "input", "videos_folder", default="videos/translate_raw"
        )

        self._setup_ui()

    def _setup_ui(self):
        """建立 UI"""
        # 配置 grid - 左右兩欄
        self.grid_columnconfigure(0, weight=2)  # 左側：影片列表
        self.grid_columnconfigure(1, weight=3)  # 右側：預覽與編輯
        self.grid_rowconfigure(0, weight=1)

        # === 右側面板：預覽與重新命名 (先建立，因為左側會用到) ===
        self._create_right_panel()

        # === 左側面板：影片列表 ===
        self._create_left_panel()

    def _create_left_panel(self):
        """建立左側面板 - 影片列表"""
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(2, weight=1)

        # --- 標題與資料夾選擇 ---
        header_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame,
            text="📁 影片列表",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        self.refresh_btn = ctk.CTkButton(
            header_frame,
            text="🔄",
            width=30,
            height=30,
            command=self._scan_videos
        )
        self.refresh_btn.grid(row=0, column=1, padx=5)

        # 資料夾路徑
        folder_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        folder_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        folder_frame.grid_columnconfigure(0, weight=1)

        self.folder_entry = ctk.CTkEntry(
            folder_frame,
            placeholder_text="影片資料夾路徑...",
            height=32
        )
        self.folder_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.folder_entry.insert(0, str(self.videos_folder))

        self.browse_btn = ctk.CTkButton(
            folder_frame,
            text="瀏覽",
            width=60,
            height=32,
            command=self._browse_folder
        )
        self.browse_btn.grid(row=0, column=1)

        # --- 影片列表 ---
        self.video_list_frame = ctk.CTkScrollableFrame(
            left_frame,
            fg_color=COLORS["surface"]
        )
        self.video_list_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.video_list_frame.grid_columnconfigure(0, weight=1)

        # --- 影片計數 ---
        self.count_label = ctk.CTkLabel(
            left_frame,
            text="0 個影片",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        self.count_label.grid(row=3, column=0, sticky="w", padx=10, pady=(0, 10))

        # 初始掃描
        self._scan_videos()

    def _create_right_panel(self):
        """建立右側面板 - 預覽與編輯"""
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)

        # --- 預覽區 ---
        preview_frame = ctk.CTkFrame(right_frame)
        preview_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        preview_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            preview_frame,
            text="🎬 影片預覽",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # 預覽框 (顯示縮圖或播放按鈕)
        self.preview_container = ctk.CTkFrame(
            preview_frame,
            fg_color=COLORS["surface"],
            height=250
        )
        self.preview_container.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        self.preview_container.grid_columnconfigure(0, weight=1)
        self.preview_container.grid_rowconfigure(0, weight=1)
        self.preview_container.grid_propagate(False)

        self.preview_label = ctk.CTkLabel(
            self.preview_container,
            text="選擇一個影片來預覽",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        self.preview_label.grid(row=0, column=0, pady=50)

        # 播放按鈕
        self.play_btn = ctk.CTkButton(
            preview_frame,
            text="▶ 用預設播放器開啟",
            height=35,
            state="disabled",
            command=self._play_video
        )
        self.play_btn.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        # --- 重新命名區 ---
        rename_frame = ctk.CTkFrame(right_frame)
        rename_frame.grid(row=1, column=0, sticky="new", padx=15, pady=(0, 15))
        rename_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            rename_frame,
            text="✏️ 重新命名",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(15, 10))

        # 目前檔名
        ctk.CTkLabel(
            rename_frame,
            text="目前檔名:",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).grid(row=1, column=0, sticky="w", padx=10)

        self.current_name_label = ctk.CTkLabel(
            rename_frame,
            text="-",
            font=ctk.CTkFont(size=12),
            wraplength=350,
            anchor="w",
            justify="left"
        )
        self.current_name_label.grid(row=2, column=0, sticky="w", padx=10, pady=(2, 10))

        # 新檔名輸入
        ctk.CTkLabel(
            rename_frame,
            text="新檔名:",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        ).grid(row=3, column=0, sticky="w", padx=10)

        self.new_name_entry = ctk.CTkEntry(
            rename_frame,
            placeholder_text="輸入新的檔案名稱 (不含副檔名)",
            height=38
        )
        self.new_name_entry.grid(row=4, column=0, sticky="ew", padx=10, pady=(2, 10))

        # 副檔名顯示
        self.extension_label = ctk.CTkLabel(
            rename_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"]
        )
        self.extension_label.grid(row=5, column=0, sticky="w", padx=10)

        # 重新命名按鈕
        self.rename_btn = ctk.CTkButton(
            rename_frame,
            text="💾 套用新名稱",
            height=40,
            fg_color=COLORS["primary"],
            hover_color="#00a8cc",
            state="disabled",
            command=self._rename_video
        )
        self.rename_btn.grid(row=6, column=0, sticky="ew", padx=10, pady=15)

        # --- 提示 ---
        tip_label = ctk.CTkLabel(
            right_frame,
            text="💡 提示: 好的檔名有助於識別影片內容，建議使用中文描述主題",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_secondary"],
            wraplength=350
        )
        tip_label.grid(row=2, column=0, sticky="w", padx=15, pady=(0, 15))

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
        self.selected_item = None

        folder = Path(self.folder_entry.get())
        if not folder.exists():
            self.count_label.configure(text="資料夾不存在")
            return

        # 掃描影片檔案
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder.glob(f"*{ext}"))
            video_files.extend(folder.glob(f"*{ext.upper()}"))

        # 排序 (按修改時間，最新在前)
        video_files = sorted(set(video_files), key=lambda x: x.stat().st_mtime, reverse=True)

        for video_path in video_files:
            item = VideoItem(
                self.video_list_frame,
                str(video_path),
                on_select_callback=self._on_video_select
            )
            item.pack(fill="x", pady=2, padx=5)
            self.video_items.append(item)

        # 更新計數
        self.count_label.configure(text=f"{len(video_files)} 個影片")

        # 重置右側面板
        self._reset_right_panel()

    def _on_video_select(self, item: VideoItem):
        """選中影片"""
        # 取消之前的選中
        if self.selected_item:
            self.selected_item.set_selected(False)

        # 設定新選中
        item.set_selected(True)
        self.selected_item = item

        # 更新右側面板
        self._update_right_panel(item)

    def _update_right_panel(self, item: VideoItem):
        """更新右側面板"""
        # 更新預覽標籤
        self.preview_label.configure(text=f"📹 {item.video_name}")

        # 啟用播放按鈕
        self.play_btn.configure(state="normal")

        # 更新重新命名區
        self.current_name_label.configure(text=item.video_name + item.extension)
        self.new_name_entry.delete(0, "end")
        self.new_name_entry.insert(0, item.video_name)
        self.extension_label.configure(text=f"副檔名: {item.extension}")
        self.rename_btn.configure(state="normal")

    def _reset_right_panel(self):
        """重置右側面板"""
        self.preview_label.configure(text="選擇一個影片來預覽")
        self.play_btn.configure(state="disabled")
        self.current_name_label.configure(text="-")
        self.new_name_entry.delete(0, "end")
        self.extension_label.configure(text="")
        self.rename_btn.configure(state="disabled")

    def _play_video(self):
        """用預設播放器開啟影片"""
        if not self.selected_item:
            return

        video_path = self.selected_item.video_path

        try:
            if sys.platform == 'win32':
                os.startfile(video_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', video_path])
            else:
                subprocess.run(['xdg-open', video_path])
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟影片:\n{e}")

    def _rename_video(self):
        """重新命名影片"""
        if not self.selected_item:
            return

        new_name = self.new_name_entry.get().strip()
        if not new_name:
            messagebox.showwarning("警告", "請輸入新的檔案名稱")
            return

        # 檢查非法字元
        invalid_chars = '<>:"/\\|?*'
        if any(c in new_name for c in invalid_chars):
            messagebox.showwarning("警告", f"檔名不能包含以下字元:\n{invalid_chars}")
            return

        old_path = Path(self.selected_item.video_path)
        new_path = old_path.parent / (new_name + self.selected_item.extension)

        # 檢查是否同名
        if old_path == new_path:
            messagebox.showinfo("提示", "檔名沒有變更")
            return

        # 檢查是否已存在
        if new_path.exists():
            messagebox.showwarning("警告", f"檔案已存在:\n{new_path.name}")
            return

        try:
            old_path.rename(new_path)
            messagebox.showinfo("成功", f"已重新命名為:\n{new_path.name}")

            # 重新掃描列表
            self._scan_videos()

        except Exception as e:
            messagebox.showerror("錯誤", f"重新命名失敗:\n{e}")
