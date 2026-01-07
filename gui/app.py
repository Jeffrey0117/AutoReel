# -*- coding: utf-8 -*-
"""
Video Translate Studio - 主應用程式
CustomTkinter GUI 主控台
"""

import customtkinter as ctk
from pathlib import Path
import sys
import os

# 加入專案根目錄到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.utils.config_manager import ConfigManager
from gui.utils.theme import COLORS, FONTS
from gui.components.translate_panel import TranslatePanel


class VideoTranslateApp(ctk.CTk):
    """Video Translate Studio 主視窗"""

    def __init__(self):
        super().__init__()

        # 載入配置
        self.config_manager = ConfigManager()

        # 視窗設定
        self.title("Video Translate Studio")
        self.geometry("1100x750")
        self.minsize(900, 650)

        # 設定主題
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # 設定視窗圖示 (如果存在)
        icon_path = PROJECT_ROOT / "gui" / "assets" / "icon.ico"
        if icon_path.exists():
            self.iconbitmap(str(icon_path))

        # 建立 UI
        self._setup_ui()

        # 綁定關閉事件
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        """建立 UI 元件"""
        # 配置 grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # 標題列
        self.grid_rowconfigure(1, weight=1)  # 主內容
        self.grid_rowconfigure(2, weight=0)  # 狀態列

        # === 標題列 ===
        self._create_header()

        # === 主內容區 (翻譯處理面板) ===
        self.translate_panel = TranslatePanel(self, self.config_manager)
        self.translate_panel.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 10))

        # === 狀態列 ===
        self._create_status_bar()

    def _create_header(self):
        """建立標題列"""
        header_frame = ctk.CTkFrame(self, height=50, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)

        # 標題
        title_label = ctk.CTkLabel(
            header_frame,
            text="🎬 Video Translate Studio",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        # 版本資訊
        version_label = ctk.CTkLabel(
            header_frame,
            text="v1.0.0",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        version_label.grid(row=0, column=1, sticky="w", padx=10)

    def _create_status_bar(self):
        """建立狀態列"""
        self.status_frame = ctk.CTkFrame(self, height=30, fg_color=COLORS["surface"])
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="就緒",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10, pady=5)

    def set_status(self, text: str):
        """更新狀態列文字"""
        self.status_label.configure(text=text)

    def _on_closing(self):
        """視窗關閉事件"""
        # 停止任何進行中的處理
        if hasattr(self, 'translate_panel') and self.translate_panel.is_processing:
            self.translate_panel.stop_processing()

        self.destroy()


def main():
    """主程式入口"""
    app = VideoTranslateApp()
    app.mainloop()


if __name__ == "__main__":
    main()
