# -*- coding: utf-8 -*-
"""
影片重新命名 - 後端 API 伺服器
提供 Web 介面真正重命名檔案的功能
"""

import http.server
import json
import os
import urllib.parse
from pathlib import Path
import socketserver
import webbrowser
import mimetypes
import socket

# 預設影片資料夾
DEFAULT_VIDEO_FOLDER = "videos/translate_raw"
PROJECT_ROOT = Path(__file__).parent

# 支援的影片格式
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


class VideoRenameHandler(http.server.SimpleHTTPRequestHandler):
    """處理影片重命名的 HTTP Handler"""

    def __init__(self, *args, **kwargs):
        # 設定根目錄為專案資料夾
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def do_GET(self):
        """處理 GET 請求"""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/videos':
            self._handle_list_videos(parsed.query)
        elif parsed.path == '/api/folders':
            self._handle_list_folders()
        elif parsed.path.startswith('/video/'):
            self._handle_serve_video(parsed.path)
        else:
            # 靜態檔案
            super().do_GET()

    def do_POST(self):
        """處理 POST 請求"""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/rename':
            self._handle_rename()
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data, status=200):
        """送出 JSON 回應"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _handle_list_folders(self):
        """列出可用資料夾"""
        folders = []

        # 掃描專案中的資料夾
        for item in PROJECT_ROOT.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 檢查是否有影片
                has_videos = any(
                    f.suffix.lower() in VIDEO_EXTENSIONS
                    for f in item.iterdir() if f.is_file()
                )
                if has_videos:
                    folders.append({
                        'path': str(item.relative_to(PROJECT_ROOT)),
                        'name': item.name
                    })

        # 加入 videos 子資料夾
        videos_dir = PROJECT_ROOT / 'videos'
        if videos_dir.exists():
            for item in videos_dir.iterdir():
                if item.is_dir():
                    has_videos = any(
                        f.suffix.lower() in VIDEO_EXTENSIONS
                        for f in item.iterdir() if f.is_file()
                    )
                    if has_videos:
                        folders.append({
                            'path': str(item.relative_to(PROJECT_ROOT)),
                            'name': f'videos/{item.name}'
                        })

        self._send_json({'folders': folders})

    def _handle_list_videos(self, query_string):
        """列出指定資料夾的影片"""
        params = urllib.parse.parse_qs(query_string)
        folder = params.get('folder', [DEFAULT_VIDEO_FOLDER])[0]

        folder_path = PROJECT_ROOT / folder

        if not folder_path.exists():
            self._send_json({'error': '資料夾不存在', 'videos': []}, 400)
            return

        videos = []
        for f in sorted(folder_path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                stat = f.stat()
                videos.append({
                    'name': f.name,
                    'path': str(f.relative_to(PROJECT_ROOT)),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                })

        self._send_json({
            'folder': folder,
            'videos': videos,
            'count': len(videos)
        })

    def _handle_serve_video(self, path):
        """提供影片檔案"""
        # /video/videos/translate_raw/xxx.mp4 -> videos/translate_raw/xxx.mp4
        video_path = urllib.parse.unquote(path[7:])  # 移除 /video/
        full_path = PROJECT_ROOT / video_path

        if not full_path.exists() or not full_path.is_file():
            self.send_error(404, "Video not found")
            return

        # 取得檔案大小
        file_size = full_path.stat().st_size

        # 處理 Range 請求 (影片 seek)
        range_header = self.headers.get('Range')

        if range_header:
            # 解析 Range: bytes=start-end
            range_match = range_header.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1

            self.send_response(206)  # Partial Content
            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            content_length = end - start + 1
        else:
            self.send_response(200)
            start = 0
            end = file_size - 1
            content_length = file_size

        # 設定 headers
        mime_type, _ = mimetypes.guess_type(str(full_path))
        self.send_header('Content-Type', mime_type or 'video/mp4')
        self.send_header('Content-Length', content_length)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # 送出檔案內容
        with open(full_path, 'rb') as f:
            f.seek(start)
            remaining = content_length
            chunk_size = 64 * 1024  # 64KB chunks

            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _handle_rename(self):
        """處理重新命名請求"""
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            old_path = data.get('oldPath', '')
            new_name = data.get('newName', '').strip()

            if not old_path or not new_name:
                self._send_json({'error': '缺少必要參數'}, 400)
                return

            # 檢查非法字元
            invalid_chars = '<>:"/\\|?*'
            if any(c in new_name for c in invalid_chars):
                self._send_json({'error': f'檔名不能包含 {invalid_chars}'}, 400)
                return

            # 完整路徑
            old_full = PROJECT_ROOT / old_path

            if not old_full.exists():
                self._send_json({'error': '檔案不存在'}, 404)
                return

            # 組合新路徑
            extension = old_full.suffix
            new_full = old_full.parent / (new_name + extension)

            if old_full == new_full:
                self._send_json({'error': '檔名沒有變更'}, 400)
                return

            if new_full.exists():
                self._send_json({'error': '已有同名檔案'}, 400)
                return

            # 執行重命名
            old_full.rename(new_full)

            self._send_json({
                'success': True,
                'oldName': old_full.name,
                'newName': new_full.name,
                'newPath': str(new_full.relative_to(PROJECT_ROOT))
            })

            print(f"[Renamed] {old_full.name} → {new_full.name}")

        except json.JSONDecodeError:
            self._send_json({'error': 'JSON 解析錯誤'}, 400)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def log_message(self, format, *args):
        """自訂 log 格式"""
        if '/api/' in args[0] or args[0].startswith('"POST'):
            print(f"[API] {args[0]}")


def find_available_port(start_port=8765, max_attempts=10):
    """尋找可用的 port"""
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


def run_server(port=8765):
    """啟動伺服器"""
    # 自動尋找可用 port
    available_port = find_available_port(port)

    if available_port is None:
        print(f"\n[錯誤] 無法找到可用的 port (嘗試了 {port}-{port+9})")
        print("請關閉佔用 port 的程式，或手動指定其他 port")
        return

    if available_port != port:
        print(f"\n[注意] Port {port} 已被佔用，改用 {available_port}")

    # 設定 socket 可重用
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", available_port), VideoRenameHandler) as httpd:
        url = f"http://localhost:{available_port}/video_rename.html"
        print(f"\n{'='*50}")
        print(f"  Video Rename Server")
        print(f"{'='*50}")
        print(f"  URL: {url}")
        print(f"  按 Ctrl+C 停止伺服器")
        print(f"{'='*50}\n")

        # 自動開啟瀏覽器
        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n伺服器已停止")


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    run_server(port)
