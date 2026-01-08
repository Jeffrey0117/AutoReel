# -*- coding: utf-8 -*-
"""
Video Translate Studio - 主應用程式
CustomTkinter GUI 主控台
"""

import customtkinter as ctk
from pathlib import Path
import sys
import os
import subprocess

# 加入專案根目錄到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.utils.config_manager import ConfigManager
from gui.utils.theme import COLORS, FONTS
from gui.components.translate_panel import TranslatePanel
from gui.components.video_manager_panel import VideoManagerPanel


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

        # 目前顯示的面板
        self.current_panel = None

        # 建立 UI
        self._setup_ui()

        # 綁定關閉事件
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        """建立 UI 元件"""
        # 配置 grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # 標題列
        self.grid_rowconfigure(1, weight=0)  # 導航列
        self.grid_rowconfigure(2, weight=1)  # 主內容
        self.grid_rowconfigure(3, weight=0)  # 狀態列

        # === 標題列 ===
        self._create_header()

        # === 導航列 ===
        self._create_nav_bar()

        # === 主內容區 (容器) ===
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=(5, 10))
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # 建立各個面板 (延遲載入)
        self.panels = {}

        # 預設顯示翻譯處理面板
        self._show_panel("translate")

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
            text="v1.1.0",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"]
        )
        version_label.grid(row=0, column=1, sticky="w", padx=10)

    def _create_nav_bar(self):
        """建立導航列"""
        nav_frame = ctk.CTkFrame(self, height=45, fg_color=COLORS["surface"])
        nav_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 5))

        # 導航按鈕
        self.nav_buttons = {}

        # 影片管理
        self.nav_buttons["video_manager"] = ctk.CTkButton(
            nav_frame,
            text="📁 影片管理",
            width=120,
            height=35,
            fg_color="transparent",
            text_color=COLORS["text"],
            hover_color=COLORS["background"],
            command=lambda: self._show_panel("video_manager")
        )
        self.nav_buttons["video_manager"].pack(side="left", padx=5, pady=5)

        # 翻譯處理
        self.nav_buttons["translate"] = ctk.CTkButton(
            nav_frame,
            text="🔄 翻譯處理",
            width=120,
            height=35,
            fg_color=COLORS["primary"],
            text_color="#000000",
            hover_color="#00a8cc",
            command=lambda: self._show_panel("translate")
        )
        self.nav_buttons["translate"].pack(side="left", padx=5, pady=5)

        # Web 工具按鈕 (右側)
        web_btn = ctk.CTkButton(
            nav_frame,
            text="🌐 Web 工具",
            width=100,
            height=35,
            fg_color="transparent",
            border_width=1,
            command=self._open_web_tools
        )
        web_btn.pack(side="right", padx=5, pady=5)

    def _show_panel(self, panel_name: str):
        """顯示指定面板"""
        # 更新導航按鈕樣式
        for name, btn in self.nav_buttons.items():
            if name == panel_name:
                btn.configure(fg_color=COLORS["primary"], text_color="#000000")
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text"])

        # 隱藏當前面板
        if self.current_panel:
            self.current_panel.grid_forget()

        # 建立或顯示面板
        if panel_name not in self.panels:
            if panel_name == "translate":
                self.panels["translate"] = TranslatePanel(self.content_frame, self.config_manager)
            elif panel_name == "video_manager":
                self.panels["video_manager"] = VideoManagerPanel(self.content_frame, self.config_manager)

        self.panels[panel_name].grid(row=0, column=0, sticky="nsew")
        self.current_panel = self.panels[panel_name]

    def _create_status_bar(self):
        """建立狀態列"""
        self.status_frame = ctk.CTkFrame(self, height=30, fg_color=COLORS["surface"])
        self.status_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 10))

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

    def _open_web_tools(self):
        """啟動 Web 工具伺服器"""
        server_script = PROJECT_ROOT / "video_rename_server.py"
        if server_script.exists():
            subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            self.set_status("Web 工具伺服器啟動中，瀏覽器將自動開啟...")
        else:
            self.set_status("找不到 video_rename_server.py")

    def _on_closing(self):
        """視窗關閉事件"""
        # 停止任何進行中的處理
        if "translate" in self.panels:
            panel = self.panels["translate"]
            if hasattr(panel, 'is_processing') and panel.is_processing:
                panel.stop_processing()

        self.destroy()


def main():
    """主程式入口"""
    app = VideoTranslateApp()
    app.mainloop()


if __name__ == "__main__":
    main()
