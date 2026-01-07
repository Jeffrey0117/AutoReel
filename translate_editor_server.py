#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻譯專案編輯器 - 後端 API 伺服器
"""

import os
import sys
import json
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# 設定專案根目錄
PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "translation_config.json"
TEMPLATE_PATH = PROJECT_ROOT / "翻譯專案" / "draft_content.json"
VIDEOS_FOLDER = PROJECT_ROOT / "videos" / "translate_raw"


class TranslateEditorHandler(SimpleHTTPRequestHandler):
    """翻譯編輯器 API Handler"""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self.serve_html()
        elif parsed.path == '/api/config':
            self.handle_get_config()
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/config':
            self.handle_save_config()
        elif parsed.path == '/api/start':
            self.handle_start_process()
        else:
            self.send_error(404)

    def serve_html(self):
        """提供 HTML 頁面"""
        html_path = PROJECT_ROOT / "translate_editor.html"
        if html_path.exists():
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "HTML file not found")

    def handle_get_config(self):
        """取得設定檔"""
        try:
            # 讀取設定檔
            config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)

            # 讀取模板文字
            template_text = ""
            if TEMPLATE_PATH.exists():
                with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                    # 從模板中找到文字素材
                    texts = template_data.get("materials", {}).get("texts", [])
                    if texts:
                        for text_item in texts:
                            if "content" in text_item:
                                try:
                                    content = json.loads(text_item["content"])
                                    if "text" in content:
                                        template_text = content["text"]
                                        break
                                except:
                                    pass

            # 列出影片
            videos = []
            if VIDEOS_FOLDER.exists():
                for f in VIDEOS_FOLDER.iterdir():
                    if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                        videos.append(f.name)

            response = {
                "config": config,
                "template_text": template_text,
                "videos": videos
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def handle_save_config(self):
        """儲存設定檔"""
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            # 讀取現有設定
            existing_config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)

            # 合併新設定
            if "config" in data:
                new_config = data["config"]
                # 更新 subtitle_style
                if "subtitle_style" in new_config:
                    if "subtitle_style" not in existing_config:
                        existing_config["subtitle_style"] = {}
                    existing_config["subtitle_style"].update(new_config["subtitle_style"])

            # 儲存設定檔
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, ensure_ascii=False, indent=2)

            # 更新模板文字
            if "template_text" in data and TEMPLATE_PATH.exists():
                self.update_template_text(data["template_text"])

            self.send_json_response({"success": True})

        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def update_template_text(self, new_text: str):
        """更新模板中的文字"""
        try:
            with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                template_data = json.load(f)

            texts = template_data.get("materials", {}).get("texts", [])
            for text_item in texts:
                if "content" in text_item:
                    try:
                        content = json.loads(text_item["content"])
                        if "text" in content:
                            content["text"] = new_text
                            # 更新 range
                            if "styles" in content:
                                for style in content["styles"]:
                                    if "range" in style:
                                        style["range"] = [0, len(new_text)]
                            text_item["content"] = json.dumps(content, ensure_ascii=False)
                            break
                    except:
                        pass

            with open(TEMPLATE_PATH, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False)

        except Exception as e:
            print(f"更新模板文字失敗: {e}")

    def handle_start_process(self):
        """開始執行翻譯流程 - 直接在程序中執行避免編碼問題"""
        import io
        from contextlib import redirect_stdout, redirect_stderr

        # 捕獲輸出
        output_buffer = io.StringIO()

        try:
            # 切換工作目錄
            old_cwd = os.getcwd()
            os.chdir(str(PROJECT_ROOT))

            # 確保 DEEPSEEK_API_KEY 存在
            if "DEEPSEEK_API_KEY" not in os.environ:
                # 從 translate.bat 讀取
                bat_path = PROJECT_ROOT / "translate.bat"
                if bat_path.exists():
                    with open(bat_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if 'DEEPSEEK_API_KEY=' in line:
                                key = line.split('=', 1)[1].strip()
                                os.environ['DEEPSEEK_API_KEY'] = key
                                break

            # 動態 import 並執行
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                # 重新載入模組以獲取最新代碼
                import importlib
                if 'translate_video' in sys.modules:
                    del sys.modules['translate_video']
                if 'subtitle_generator' in sys.modules:
                    del sys.modules['subtitle_generator']

                from translate_video import TranslationWorkflow
                workflow = TranslationWorkflow()
                workflow.batch_process()

            output = output_buffer.getvalue()
            output += "\n\n=== 處理完成 ==="

        except Exception as e:
            output = output_buffer.getvalue()
            output += f"\n\n=== 錯誤: {str(e)} ==="
            import traceback
            output += f"\n{traceback.format_exc()}"

        finally:
            os.chdir(old_cwd)

        # 發送回應
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        response_bytes = output.encode('utf-8')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

    def send_error_response(self, message):
        """發送錯誤回應"""
        self.send_response(500)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        response_bytes = message.encode('utf-8')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)

    def send_json_response(self, data, status=200):
        """傳送 JSON 回應"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        """自訂日誌格式"""
        print(f"[Server] {args[0]}")


def main():
    port = 8765
    print("=" * 50)
    print("  翻譯專案編輯器")
    print("=" * 50)
    print(f"  啟動伺服器: http://localhost:{port}")
    print(f"  專案目錄: {PROJECT_ROOT}")
    print(f"  影片目錄: {VIDEOS_FOLDER}")
    print("=" * 50)
    print("  按 Ctrl+C 停止伺服器")
    print()

    # 自動開啟瀏覽器
    import webbrowser
    webbrowser.open(f"http://localhost:{port}")

    # 啟動伺服器
    server = HTTPServer(('localhost', port), TranslateEditorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n伺服器已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
