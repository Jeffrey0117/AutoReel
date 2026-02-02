"""
Caption Service - IG caption generation using DeepSeek API.
Reads Jianying drafts, extracts subtitles, generates IG captions.
"""

import os
import sys
import json
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_jianying_draft_root() -> Path:
    """Get Jianying draft root path from config.json or default."""
    config_path = PROJECT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                folder = config.get("jianying_draft_folder", "")
                if folder:
                    return Path(folder)
        except (json.JSONDecodeError, OSError):
            pass
    username = os.environ.get("USERNAME") or os.getlogin()
    return Path(
        rf"C:\Users\{username}\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft"
    )


class CaptionService:
    """Service for IG caption generation, bridging sync API calls to async WebSocket."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.config_path = self.project_root / "translation_config.json"
        self.subtitles_folder = self.project_root / "subtitles"

        self.is_generating = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_manager = None
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
                self._loop,
            )

    def _read_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _write_config(self, config: dict):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    # --- Examples ---

    def get_examples(self) -> List[str]:
        config = self._read_config()
        return config.get("ig_caption", {}).get("examples", [])

    def add_example(self, text: str) -> List[str]:
        config = self._read_config()
        if "ig_caption" not in config:
            config["ig_caption"] = {}
        if "examples" not in config["ig_caption"]:
            config["ig_caption"]["examples"] = []
        config["ig_caption"]["examples"].append(text)
        self._write_config(config)
        return config["ig_caption"]["examples"]

    def delete_example(self, index: int) -> List[str]:
        config = self._read_config()
        examples = config.get("ig_caption", {}).get("examples", [])
        if 0 <= index < len(examples):
            examples.pop(index)
            config["ig_caption"]["examples"] = examples
            self._write_config(config)
        return config.get("ig_caption", {}).get("examples", [])

    # --- Drafts ---

    def list_drafts(self) -> List[str]:
        config = self._read_config()
        prefix = config.get("output", {}).get("output_prefix", "翻譯專案_")
        draft_root = get_jianying_draft_root()
        drafts = []

        if draft_root.exists():
            for folder in draft_root.iterdir():
                if folder.is_dir() and folder.name.startswith(prefix):
                    draft_file = folder / "draft_content.json"
                    if draft_file.exists():
                        drafts.append(folder.name)

        return sorted(drafts, reverse=True)

    # --- Captions ---

    def _get_published_set(self) -> set:
        config = self._read_config()
        return set(config.get("ig_caption", {}).get("published", []))

    def toggle_published(self, video_name: str) -> bool:
        """Toggle published status for a caption. Returns new published state."""
        config = self._read_config()
        if "ig_caption" not in config:
            config["ig_caption"] = {}
        published = set(config["ig_caption"].get("published", []))
        if video_name in published:
            published.discard(video_name)
            is_published = False
        else:
            published.add(video_name)
            is_published = True
        config["ig_caption"]["published"] = sorted(published)
        self._write_config(config)
        return is_published

    def list_captions(self) -> List[Dict[str, Any]]:
        self.subtitles_folder.mkdir(parents=True, exist_ok=True)
        published = self._get_published_set()
        captions = []
        for f in sorted(self.subtitles_folder.glob("*_ig_caption.txt")):
            video_name = f.stem.replace("_ig_caption", "")
            try:
                content = f.read_text(encoding="utf-8")
            except OSError:
                content = ""
            captions.append({
                "video_name": video_name,
                "filename": f.name,
                "path": str(f),
                "content": content,
                "published": video_name in published,
            })
        return captions

    def update_caption(self, video_name: str, text: str) -> bool:
        caption_file = self.subtitles_folder / f"{video_name}_ig_caption.txt"
        if not caption_file.exists():
            return False
        caption_file.write_text(text, encoding="utf-8")
        return True

    def delete_caption(self, video_name: str) -> bool:
        caption_file = self.subtitles_folder / f"{video_name}_ig_caption.txt"
        if caption_file.exists():
            caption_file.unlink()
            return True
        return False

    # --- Subtitle extraction ---

    def extract_subtitles_from_draft(self, draft_name: str) -> Optional[str]:
        draft_root = get_jianying_draft_root()
        draft_file = draft_root / draft_name / "draft_content.json"
        if not draft_file.exists():
            return None

        try:
            with open(draft_file, "r", encoding="utf-8") as f:
                draft_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        texts = draft_data.get("materials", {}).get("texts", [])
        subtitles = []
        for i, text in enumerate(texts):
            content_str = text.get("content", "")
            try:
                content_data = json.loads(content_str)
                text_content = content_data.get("text", "")
            except (json.JSONDecodeError, ValueError):
                continue
            if "@html_cat" in text_content:
                continue
            if i < 2:
                continue
            subtitles.append(text_content)

        if not subtitles:
            return None
        return "\n".join(subtitles)

    # --- Generation ---

    def _call_deepseek_api(
        self, content: str, examples: List[str], config: dict
    ) -> Optional[str]:
        """Call DeepSeek API to generate IG caption."""
        try:
            import requests

            translation_config = config.get("translation", {})
            api_key = translation_config.get("api_key") or os.environ.get(
                translation_config.get("api_key_env", "DEEPSEEK_API_KEY")
            )
            base_url = translation_config.get("base_url", "https://api.deepseek.com")
            model = translation_config.get("model", "deepseek-chat")

            if not api_key:
                return None

            examples_text = "\n\n---\n\n".join(examples[:3])

            prompt = f"""你是一位專業的社群媒體文案寫手。請根據以下影片內容，用我的風格寫一篇 Instagram 文案。

## 我的文案風格範例：
{examples_text}

## 影片內容（字幕）：
{content}

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
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                },
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            return None

        except Exception:
            return None

    def generate_caption(self, draft_name: str) -> Dict[str, Any]:
        """Generate caption for a single draft. Returns result dict."""
        config = self._read_config()
        examples = config.get("ig_caption", {}).get("examples", [])
        prefix = config.get("output", {}).get("output_prefix", "翻譯專案_")

        subtitles = self.extract_subtitles_from_draft(draft_name)
        if not subtitles:
            return {"success": False, "error": "無法從草稿提取字幕"}

        if not examples:
            return {"success": False, "error": "請先新增至少一個風格範例"}

        caption = self._call_deepseek_api(subtitles, examples, config)
        if not caption:
            return {"success": False, "error": "API 呼叫失敗，請檢查 API Key"}

        video_name = draft_name.replace(prefix, "", 1) if draft_name.startswith(prefix) else draft_name
        self.subtitles_folder.mkdir(parents=True, exist_ok=True)
        output_path = self.subtitles_folder / f"{video_name}_ig_caption.txt"
        output_path.write_text(caption, encoding="utf-8")

        return {
            "success": True,
            "video_name": video_name,
            "path": str(output_path),
            "content": caption,
        }

    def start_batch_generation(self, draft_names: List[str]) -> bool:
        """Start batch generation in a background thread."""
        with self._lock:
            if self.is_generating:
                return False
            self.is_generating = True

        thread = threading.Thread(
            target=self._run_batch_generation,
            args=(draft_names,),
            daemon=True,
            name="caption-batch",
        )
        thread.start()
        return True

    def _run_batch_generation(self, draft_names: List[str]):
        """Run batch generation (called from background thread)."""
        total = len(draft_names)
        success_count = 0
        failed_count = 0

        try:
            config = self._read_config()
            examples = config.get("ig_caption", {}).get("examples", [])
            prefix = config.get("output", {}).get("output_prefix", "翻譯專案_")

            self._broadcast({
                "type": "caption_batch_start",
                "data": {"total": total},
            })

            for i, draft_name in enumerate(draft_names):
                self._broadcast({
                    "type": "caption_generate_start",
                    "data": {
                        "draft_name": draft_name,
                        "index": i + 1,
                        "total": total,
                    },
                })

                try:
                    subtitles = self.extract_subtitles_from_draft(draft_name)
                    if not subtitles:
                        raise ValueError("無法從草稿提取字幕")

                    if not examples:
                        raise ValueError("沒有風格範例")

                    caption = self._call_deepseek_api(subtitles, examples, config)
                    if not caption:
                        raise ValueError("API 呼叫失敗")

                    video_name = (
                        draft_name.replace(prefix, "", 1)
                        if draft_name.startswith(prefix)
                        else draft_name
                    )
                    self.subtitles_folder.mkdir(parents=True, exist_ok=True)
                    output_path = self.subtitles_folder / f"{video_name}_ig_caption.txt"
                    output_path.write_text(caption, encoding="utf-8")

                    success_count += 1
                    self._broadcast({
                        "type": "caption_generate_done",
                        "data": {
                            "draft_name": draft_name,
                            "video_name": video_name,
                            "index": i + 1,
                            "total": total,
                        },
                    })

                except Exception as e:
                    failed_count += 1
                    self._broadcast({
                        "type": "caption_generate_error",
                        "data": {
                            "draft_name": draft_name,
                            "index": i + 1,
                            "total": total,
                            "error": str(e),
                        },
                    })

            self._broadcast({
                "type": "caption_batch_done",
                "data": {
                    "total": total,
                    "success_count": success_count,
                    "failed_count": failed_count,
                },
            })

        except Exception as e:
            self._broadcast({
                "type": "caption_batch_done",
                "data": {
                    "total": total,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "error": str(e),
                },
            })
        finally:
            with self._lock:
                self.is_generating = False


caption_service = CaptionService()
