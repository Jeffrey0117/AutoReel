#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字幕位置調整 - 簡易伺服器"""

import os
import json
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "translation_config.json"


def get_jianying_draft_root():
    """取得剪映草稿根路徑"""
    config_path = PROJECT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return Path(config.get("jianying_draft_folder", ""))
        except:
            pass
    username = os.environ.get("USERNAME") or os.getlogin()
    return Path(rf"C:\Users\{username}\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft")


class PositionEditorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_html()
        elif self.path == '/editor' or self.path == '/subtitle-editor':
            self.serve_subtitle_editor()
        elif self.path == '/ig-caption':
            self.serve_ig_caption_editor()
        elif self.path == '/api/config':
            self.handle_get_config()
        elif self.path == '/api/drafts':
            self.handle_list_drafts()
        elif self.path.startswith('/api/subtitles?'):
            self.handle_get_subtitles()
        elif self.path == '/api/ig-examples':
            self.handle_get_ig_examples()
        elif self.path == '/api/ig-captions':
            self.handle_get_ig_captions()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/position':
            self.handle_save_position()
        elif self.path == '/api/update-draft':
            self.handle_update_draft()
        elif self.path == '/api/subtitles/replace':
            self.handle_replace_subtitles()
        elif self.path == '/api/ig-examples':
            self.handle_post_ig_examples()
        elif self.path == '/api/ig-generate':
            self.handle_ig_generate()
        else:
            self.send_error(404)

    def serve_html(self):
        html_path = PROJECT_ROOT / "subtitle_position_editor.html"
        if html_path.exists():
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def serve_subtitle_editor(self):
        html_path = PROJECT_ROOT / "subtitle_editor.html"
        if html_path.exists():
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def serve_ig_caption_editor(self):
        html_path = PROJECT_ROOT / "ig_caption_editor.html"
        if html_path.exists():
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def handle_get_config(self):
        try:
            config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            self.send_json({"config": config})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_save_position(self):
        try:
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            position_y = data.get('position_y', -0.45)
            font_size = data.get('font_size', 8.0)
            line_max_width = data.get('line_max_width', 0.82)
            background_alpha = data.get('background_alpha', 0.64)
            text_color = data.get('text_color', '#FFFFFF')
            text_color_random = data.get('text_color_random', False)

            # 讀取設定
            config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)

            # 更新設定
            if 'subtitle_style' not in config:
                config['subtitle_style'] = {}
            config['subtitle_style']['position_y'] = position_y
            config['subtitle_style']['font_size'] = font_size
            config['subtitle_style']['line_max_width'] = line_max_width
            config['subtitle_style']['background_alpha'] = background_alpha
            config['subtitle_style']['background_style'] = 1 if background_alpha > 0 else 0
            config['subtitle_style']['text_color'] = text_color
            config['subtitle_style']['text_color_random'] = text_color_random

            # 儲存
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            print(f"[OK] pos={position_y}, size={font_size}, color={text_color}, random={text_color_random}")
            self.send_json({"success": True})

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_list_drafts(self):
        """列出翻譯專案草稿"""
        try:
            draft_root = get_jianying_draft_root()
            drafts = []

            # 讀取設定取得前綴
            config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            prefix = config.get("output", {}).get("output_prefix", "翻譯專案_")

            # 列出符合前綴的草稿
            if draft_root.exists():
                for folder in draft_root.iterdir():
                    if folder.is_dir() and folder.name.startswith(prefix):
                        draft_file = folder / "draft_content.json"
                        if draft_file.exists():
                            drafts.append(folder.name)

            self.send_json({"drafts": sorted(drafts, reverse=True)})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_update_draft(self):
        """更新現有草稿的字幕位置、字號、寬度、背景、顏色"""
        try:
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            draft_name = data.get('draft_name')
            position_y = data.get('position_y', -0.45)
            font_size = data.get('font_size', 8.0)
            line_max_width = data.get('line_max_width', 0.82)
            background_alpha = data.get('background_alpha', 0.64)
            background_style = 1 if background_alpha > 0 else 0
            text_color = data.get('text_color', '#FFFFFF')
            text_color_random = data.get('text_color_random', False)
            color_options = ['#FFFFFF', '#ffe759', '#00ff00', '#00d4ff', '#ff6699']

            if not draft_name:
                self.send_json({"error": "未指定草稿"}, 400)
                return

            draft_root = get_jianying_draft_root()
            draft_path = draft_root / draft_name / "draft_content.json"

            if not draft_path.exists():
                self.send_json({"error": f"找不到草稿: {draft_name}"}, 404)
                return

            # 讀取草稿
            with open(draft_path, 'r', encoding='utf-8') as f:
                draft_data = json.load(f)

            # 收集字幕軌道的素材 ID（根據內容識別，跳過標題和 @html_cat）
            texts = draft_data.get("materials", {}).get("texts", [])
            subtitle_material_ids = set()
            skipped_count = 0

            for i, text in enumerate(texts):
                # 解析文字內容來識別類型
                content_str = text.get("content", "")
                try:
                    content_data = json.loads(content_str)
                    text_content = content_data.get("text", "")
                except:
                    text_content = ""

                # 跳過 @html_cat（根據內容識別）
                if "@html_cat" in text_content:
                    skipped_count += 1
                    print(f"   [Skip] 跳過 @html_cat (index={i})")
                    continue

                # 跳過標題（假設是前 2 個中的非 @html_cat）
                if i < 2 and "@html_cat" not in text_content:
                    skipped_count += 1
                    print(f"   [Skip] 跳過標題 (index={i}): {text_content[:20]}...")
                    continue

                # 這是翻譯字幕，進行更新
                subtitle_material_ids.add(text.get("id"))

                # 決定這個字幕的顏色
                if text_color_random:
                    current_color = random.choice(color_options)
                else:
                    current_color = text_color

                # 更新素材的字號、寬度、背景、顏色
                text["font_size"] = font_size
                text["line_max_width"] = line_max_width
                text["background_alpha"] = background_alpha
                text["background_style"] = background_style
                text["background_color"] = "#000000"
                text["text_color"] = current_color

                # 更新 content 中的字號和顏色
                try:
                    content = json.loads(text.get("content", "{}"))
                    if "styles" in content:
                        # 將 hex 顏色轉換為 RGB (0-1)
                        hex_color = current_color.lstrip('#')
                        r = int(hex_color[0:2], 16) / 255
                        g = int(hex_color[2:4], 16) / 255
                        b = int(hex_color[4:6], 16) / 255

                        for style in content["styles"]:
                            style["size"] = font_size
                            # 更新文字顏色
                            if "fill" not in style:
                                style["fill"] = {"content": {"solid": {"color": [r, g, b]}}}
                            else:
                                style["fill"]["content"]["solid"]["color"] = [r, g, b]
                        text["content"] = json.dumps(content, ensure_ascii=False)
                except:
                    pass

            # 更新字幕軌道片段的位置
            updated_count = 0
            for track in draft_data.get("tracks", []):
                if track.get("type") != "text":
                    continue
                for segment in track.get("segments", []):
                    if segment.get("material_id") in subtitle_material_ids:
                        if "clip" in segment and "transform" in segment["clip"]:
                            segment["clip"]["transform"]["y"] = position_y
                            updated_count += 1

            # 儲存草稿
            with open(draft_path, 'w', encoding='utf-8') as f:
                json.dump(draft_data, f, ensure_ascii=False)

            color_info = "隨機" if text_color_random else text_color
            print(f"[OK] 更新 {draft_name}: {updated_count} 字幕, 跳過 {skipped_count} 個 (pos={position_y}, color={color_info})")
            self.send_json({"success": True, "updated": updated_count, "skipped": skipped_count})

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_get_subtitles(self):
        """取得指定草稿的所有字幕文字"""
        try:
            # 解析 query string
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            draft_name = params.get('draft', [None])[0]

            if not draft_name:
                self.send_json({"error": "未指定草稿"}, 400)
                return

            draft_root = get_jianying_draft_root()
            draft_path = draft_root / draft_name / "draft_content.json"

            if not draft_path.exists():
                self.send_json({"error": f"找不到草稿: {draft_name}"}, 404)
                return

            # 讀取草稿
            with open(draft_path, 'r', encoding='utf-8') as f:
                draft_data = json.load(f)

            # 提取字幕文字（跳過標題和 @html_cat）
            texts = draft_data.get("materials", {}).get("texts", [])
            subtitles = []

            for i, text in enumerate(texts):
                # 解析文字內容
                content_str = text.get("content", "")
                try:
                    content_data = json.loads(content_str)
                    text_content = content_data.get("text", "")
                except:
                    text_content = ""

                # 跳過 @html_cat
                if "@html_cat" in text_content:
                    continue

                # 跳過標題（假設是前 2 個中的非 @html_cat）
                if i < 2 and "@html_cat" not in text_content:
                    continue

                # 這是翻譯字幕
                subtitles.append(text_content)

            print(f"[OK] 載入 {draft_name}: {len(subtitles)} 條字幕")
            self.send_json({"subtitles": subtitles})

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_replace_subtitles(self):
        """執行字幕文字批量取代並儲存"""
        try:
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            draft_name = data.get('draft_name')
            find_text = data.get('find')
            replace_text = data.get('replace', '')

            if not draft_name:
                self.send_json({"error": "未指定草稿"}, 400)
                return

            if not find_text:
                self.send_json({"error": "未指定尋找文字"}, 400)
                return

            draft_root = get_jianying_draft_root()
            draft_path = draft_root / draft_name / "draft_content.json"

            if not draft_path.exists():
                self.send_json({"error": f"找不到草稿: {draft_name}"}, 404)
                return

            # 讀取草稿
            with open(draft_path, 'r', encoding='utf-8') as f:
                draft_data = json.load(f)

            # 執行取代
            texts = draft_data.get("materials", {}).get("texts", [])
            replaced_count = 0

            for i, text in enumerate(texts):
                # 解析文字內容
                content_str = text.get("content", "")
                try:
                    content_data = json.loads(content_str)
                    text_content = content_data.get("text", "")
                except:
                    continue

                # 跳過 @html_cat
                if "@html_cat" in text_content:
                    continue

                # 跳過標題（假設是前 2 個中的非 @html_cat）
                if i < 2 and "@html_cat" not in text_content:
                    continue

                # 檢查是否包含要尋找的文字
                if find_text in text_content:
                    # 執行取代
                    new_text = text_content.replace(find_text, replace_text)
                    content_data["text"] = new_text

                    # 更新 styles 中的 range 以匹配新文字長度
                    new_len = len(new_text)
                    if "styles" in content_data:
                        for style in content_data["styles"]:
                            if "range" in style:
                                style["range"] = [0, new_len]

                    text["content"] = json.dumps(content_data, ensure_ascii=False)
                    replaced_count += 1

            # 儲存草稿
            with open(draft_path, 'w', encoding='utf-8') as f:
                json.dump(draft_data, f, ensure_ascii=False)

            print(f"[OK] 取代 {draft_name}: {replaced_count} 條字幕 ('{find_text}' -> '{replace_text}')")
            self.send_json({"success": True, "replaced": replaced_count})

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_get_ig_examples(self):
        """取得 IG 文案範例"""
        try:
            config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            examples = config.get("ig_caption", {}).get("examples", [])
            self.send_json({"examples": examples})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_post_ig_examples(self):
        """新增/刪除 IG 文案範例"""
        try:
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            action = data.get('action')

            # 讀取設定
            config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)

            if 'ig_caption' not in config:
                config['ig_caption'] = {}
            if 'examples' not in config['ig_caption']:
                config['ig_caption']['examples'] = []

            if action == 'add':
                text = data.get('text', '').strip()
                if text:
                    config['ig_caption']['examples'].append(text)
                    print(f"[IG] 新增範例 ({len(text)} 字)")
            elif action == 'delete':
                index = data.get('index', -1)
                if 0 <= index < len(config['ig_caption']['examples']):
                    config['ig_caption']['examples'].pop(index)
                    print(f"[IG] 刪除範例 #{index}")

            # 儲存
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            self.send_json({"success": True})

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_get_ig_captions(self):
        """取得已生成的 IG 文案列表"""
        try:
            captions_folder = PROJECT_ROOT / "subtitles"
            captions = []

            if captions_folder.exists():
                for file in captions_folder.glob("*_ig_caption.txt"):
                    video_name = file.stem.replace("_ig_caption", "")
                    with open(file, 'r', encoding='utf-8') as f:
                        caption = f.read()
                    captions.append({
                        "video_name": video_name,
                        "caption": caption,
                        "file_path": str(file).replace("\\", "/")
                    })

            # 按修改時間排序（最新的在前）
            captions.sort(key=lambda x: x['video_name'], reverse=True)
            self.send_json({"captions": captions})

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_ig_generate(self):
        """根據草稿重新生成 IG 文案"""
        try:
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length).decode('utf-8'))
            draft_name = data.get('draft_name')

            if not draft_name:
                self.send_json({"error": "未指定草稿"}, 400)
                return

            # 讀取設定取得範例
            config = {}
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)

            examples = config.get("ig_caption", {}).get("examples", [])
            if not examples:
                self.send_json({"error": "請先新增文案範例"}, 400)
                return

            # 讀取草稿字幕
            draft_root = get_jianying_draft_root()
            draft_path = draft_root / draft_name / "draft_content.json"

            if not draft_path.exists():
                self.send_json({"error": f"找不到草稿: {draft_name}"}, 404)
                return

            with open(draft_path, 'r', encoding='utf-8') as f:
                draft_data = json.load(f)

            # 提取字幕文字
            texts = draft_data.get("materials", {}).get("texts", [])
            subtitles = []
            for i, text in enumerate(texts):
                content_str = text.get("content", "")
                try:
                    content_data = json.loads(content_str)
                    text_content = content_data.get("text", "")
                except:
                    continue
                if "@html_cat" in text_content:
                    continue
                if i < 2:
                    continue
                subtitles.append(text_content)

            if not subtitles:
                self.send_json({"error": "草稿中沒有字幕"}, 400)
                return

            # 呼叫 AI 生成文案
            video_content = "\n".join(subtitles)
            caption = self._generate_ig_caption(video_content, examples, config)

            if not caption:
                self.send_json({"error": "生成失敗"}, 500)
                return

            # 儲存文案
            video_name = draft_name.replace("翻譯專案_", "")
            captions_folder = PROJECT_ROOT / "subtitles"
            captions_folder.mkdir(exist_ok=True)
            caption_file = captions_folder / f"{video_name}_ig_caption.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption)

            print(f"[IG] 生成文案: {video_name}")
            self.send_json({"success": True, "caption": caption, "file_path": str(caption_file).replace("\\", "/")})

        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def _generate_ig_caption(self, video_content, examples, config):
        """呼叫 DeepSeek API 生成 IG 文案"""
        try:
            import requests

            translation_config = config.get("translation", {})
            # 優先使用 config 中的 api_key，否則從環境變數讀取
            api_key = translation_config.get("api_key") or os.environ.get(translation_config.get("api_key_env", "DEEPSEEK_API_KEY"))
            base_url = translation_config.get("base_url", "https://api.deepseek.com")
            model = translation_config.get("model", "deepseek-chat")

            if not api_key:
                print("[IG] 錯誤: 找不到 API Key，請在 translation_config.json 的 translation 區塊加入 api_key")
                return None

            # 準備範例
            examples_text = "\n\n---\n\n".join(examples[:3])  # 最多用 3 個範例

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
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"[IG] API 錯誤: {response.status_code}")
                return None

        except Exception as e:
            print(f"[IG] 生成錯誤: {e}")
            return None

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        print(f"[Server] {args[0]}")


def main():
    port = 8766
    print("=" * 50)
    print("  字幕編輯器")
    print("=" * 50)
    print(f"  字幕位置調整: http://localhost:{port}")
    print(f"  字幕批量取代: http://localhost:{port}/editor")
    print(f"  IG 文案產生: http://localhost:{port}/ig-caption")
    print("  按 Ctrl+C 停止")
    print("=" * 50)

    import webbrowser
    webbrowser.open(f"http://localhost:{port}")

    server = HTTPServer(('localhost', port), PositionEditorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
