# -*- coding: utf-8 -*-
"""
Video Translate Studio - 主程式入口
"""

import sys
from pathlib import Path

# 確保專案根目錄在 path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """檢查必要依賴"""
    missing = []

    try:
        import customtkinter
    except ImportError:
        missing.append("customtkinter")

    if missing:
        print("缺少必要套件，請執行以下指令安裝:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def main():
    """主程式入口"""
    check_dependencies()

    from gui.app import VideoTranslateApp

    app = VideoTranslateApp()
    app.mainloop()


if __name__ == "__main__":
    main()
