"""
Draft API Routes — JianYing draft manipulation + Chinese text conversion.

Endpoints:
  GET  /api/draft/list                     — 列出所有剪映草稿
  GET  /api/draft/{draft_name}/texts       — 預覽草稿文字
  GET  /api/draft/{draft_name}/info        — 草稿詳細資訊
  GET  /api/draft/{draft_name}/export-srt  — 匯出字幕為 SRT
  POST /api/draft/convert-s2t              — 草稿文字簡→繁
  POST /api/draft/convert-srt             — SRT 檔案簡→繁
  POST /api/draft/{draft_name}/replace-text — 批量替換文字
"""

import json
import re
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from opencc import OpenCC

from services.caption_service import get_jianying_draft_root

router = APIRouter(prefix="/api/draft", tags=["draft"])


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ConvertRequest(BaseModel):
    draft_name: str
    variant: Optional[str] = "s2tw"


class ConvertSrtRequest(BaseModel):
    srt_path: str
    variant: Optional[str] = "s2tw"


class TextEntry(BaseModel):
    index: int
    text: str


class ConvertChange(BaseModel):
    index: int
    before: str
    after: str


class TextReplacement(BaseModel):
    index: int
    text: str


class ReplaceTextRequest(BaseModel):
    replacements: List[TextReplacement]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_VARIANTS = {"s2t", "s2tw", "s2twp"}


def _validate_variant(variant: Optional[str]) -> str:
    v = variant or "s2tw"
    if v not in _ALLOWED_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的轉換模式 '{v}'，可用: {', '.join(sorted(_ALLOWED_VARIANTS))}",
        )
    return v


def _load_draft_json(draft_name: str) -> tuple[Path, dict]:
    """Load draft_content.json for the given draft name. Returns (path, data)."""
    draft_root = get_jianying_draft_root()
    draft_dir = draft_root / draft_name
    if not draft_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"草稿 '{draft_name}' 不存在")
    draft_file = draft_dir / "draft_content.json"
    if not draft_file.exists():
        raise HTTPException(status_code=404, detail="找不到 draft_content.json")
    # Detect encrypted drafts (newer JianYing versions encrypt content)
    with open(draft_file, "rb") as fb:
        head = fb.read(1)
    if head != b"{":
        raise HTTPException(
            status_code=422,
            detail=f"草稿 '{draft_name}' 的 draft_content.json 已加密，無法讀取",
        )
    try:
        with open(draft_file, "r", encoding="utf-8") as f:
            return draft_file, json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"draft_content.json 格式錯誤: {e}")


def _extract_texts(data: dict) -> List[TextEntry]:
    """Extract text entries from draft materials."""
    entries: list[TextEntry] = []
    texts = data.get("materials", {}).get("texts", [])
    for idx, item in enumerate(texts):
        raw_content = item.get("content", "")
        if not raw_content:
            continue
        try:
            content = json.loads(raw_content)
            text = content.get("text", "")
        except (json.JSONDecodeError, TypeError, ValueError):
            text = str(raw_content) if raw_content else ""
        if text:
            entries.append(TextEntry(index=idx, text=text))
    return entries


def _us_to_srt_time(us: int) -> str:
    """Convert microseconds to SRT timestamp (HH:MM:SS,mmm)."""
    ms = us // 1000
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/list")
async def list_drafts():
    """列出所有剪映草稿名稱。"""
    draft_root = get_jianying_draft_root()
    if not draft_root.exists():
        raise HTTPException(status_code=404, detail=f"草稿根目錄不存在: {draft_root}")

    drafts = []
    for d in sorted(draft_root.iterdir(), key=lambda x: x.name.lower()):
        if not d.is_dir():
            continue
        draft_file = d / "draft_content.json"
        encrypted = False
        if draft_file.exists():
            with open(draft_file, "rb") as fb:
                encrypted = fb.read(1) != b"{"
        drafts.append({
            "name": d.name,
            "encrypted": encrypted,
            "has_draft": draft_file.exists(),
        })

    return {"draft_root": str(draft_root), "count": len(drafts), "drafts": drafts}


@router.get("/{draft_name}/texts")
async def preview_draft_texts(draft_name: str):
    """預覽草稿中的文字內容（不修改）。"""
    _, data = _load_draft_json(draft_name)
    entries = _extract_texts(data)
    return {"draft_name": draft_name, "count": len(entries), "texts": entries}


@router.get("/{draft_name}/info")
async def draft_info(draft_name: str):
    """取得剪映草稿詳細資訊（解析度、軌道、素材統計）。"""
    draft_file, data = _load_draft_json(draft_name)

    canvas = data.get("canvas_config", {})
    materials = data.get("materials", {})
    tracks = data.get("tracks", [])

    # Duration in microseconds → seconds
    duration_us = data.get("duration", 0)

    # Track summary
    track_summary: dict[str, int] = {}
    for t in tracks:
        ttype = t.get("type", "unknown")
        track_summary[ttype] = track_summary.get(ttype, 0) + 1

    return {
        "draft_name": draft_name,
        "width": canvas.get("width", 0),
        "height": canvas.get("height", 0),
        "fps": data.get("fps", 30),
        "duration_seconds": round(duration_us / 1_000_000, 2),
        "tracks": track_summary,
        "track_count": len(tracks),
        "materials": {
            "videos": len(materials.get("videos", [])),
            "audios": len(materials.get("audios", [])),
            "texts": len(materials.get("texts", [])),
            "stickers": len(materials.get("stickers", [])),
            "effects": len(materials.get("video_effects", [])),
        },
    }


@router.get("/{draft_name}/export-srt")
async def export_srt(
    draft_name: str,
    save: bool = Query(False, description="是否存檔到草稿目錄"),
):
    """把剪映草稿字幕匯出為 SRT 格式。"""
    draft_file, data = _load_draft_json(draft_name)

    materials = data.get("materials", {})
    texts = materials.get("texts", [])
    tracks = data.get("tracks", [])

    # Build material id → text mapping
    mat_map: dict[str, str] = {}
    for item in texts:
        mid = item.get("id", "")
        raw = item.get("content", "")
        if not raw:
            continue
        try:
            content = json.loads(raw)
            mat_map[mid] = content.get("text", "")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Collect subtitle entries from text tracks with timing
    srt_entries: list[tuple[int, int, str]] = []  # (start_us, duration_us, text)
    for track in tracks:
        if track.get("type") != "text":
            continue
        for seg in track.get("segments", []):
            mid = seg.get("material_id", "")
            text = mat_map.get(mid, "")
            if not text:
                continue
            tr = seg.get("target_timerange", {})
            start = tr.get("start", 0)
            duration = tr.get("duration", 0)
            srt_entries.append((start, duration, text))

    # Sort by start time
    srt_entries.sort(key=lambda x: x[0])

    # Build SRT content
    lines: list[str] = []
    for i, (start, duration, text) in enumerate(srt_entries, 1):
        end = start + duration
        lines.append(str(i))
        lines.append(f"{_us_to_srt_time(start)} --> {_us_to_srt_time(end)}")
        lines.append(text)
        lines.append("")

    srt_content = "\n".join(lines)

    if save:
        out_path = draft_file.parent / f"{draft_name}.srt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        return {
            "draft_name": draft_name,
            "entries": len(srt_entries),
            "saved_to": str(out_path),
            "srt": srt_content,
        }

    return PlainTextResponse(content=srt_content, media_type="text/plain; charset=utf-8")


@router.post("/convert-s2t")
async def convert_draft_s2t(req: ConvertRequest):
    """把剪映草稿中所有文字從簡體中文轉換為繁體中文。"""
    variant = _validate_variant(req.variant)
    draft_file, data = _load_draft_json(req.draft_name)
    converter = OpenCC(variant)

    texts = data.get("materials", {}).get("texts", [])
    changes: list[ConvertChange] = []

    for idx, item in enumerate(texts):
        raw_content = item.get("content", "")
        try:
            content = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            continue

        original = content.get("text", "")
        if not original:
            continue

        converted = converter.convert(original)
        if converted == original:
            continue

        content["text"] = converted

        # Update style ranges to match new text length
        if "styles" in content:
            for style in content["styles"]:
                if "range" in style:
                    style["range"] = [0, len(converted)]

        item["content"] = json.dumps(content, ensure_ascii=False)
        changes.append(ConvertChange(index=idx, before=original, after=converted))

    if not changes:
        return {
            "success": True,
            "draft_name": req.draft_name,
            "variant": variant,
            "converted_count": 0,
            "message": "沒有需要轉換的文字（可能已經是繁體）",
            "changes": [],
        }

    # Save back
    with open(draft_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return {
        "success": True,
        "draft_name": req.draft_name,
        "variant": variant,
        "converted_count": len(changes),
        "changes": changes,
    }


@router.post("/convert-srt")
async def convert_srt(req: ConvertSrtRequest):
    """把 SRT 字幕檔從簡體中文轉換為繁體中文，輸出加 _繁體 後綴。"""
    variant = _validate_variant(req.variant)

    srt_path = Path(req.srt_path)

    # Security: only allow .srt files
    if srt_path.suffix.lower() != ".srt":
        raise HTTPException(status_code=400, detail="只支援 .srt 檔案")
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail=f"檔案不存在: {srt_path}")
    if not srt_path.is_file():
        raise HTTPException(status_code=400, detail="路徑不是檔案")

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    converter = OpenCC(variant)
    converted = converter.convert(content)

    # Count changed subtitle lines (skip sequence numbers and timestamps)
    orig_lines = content.split("\n")
    conv_lines = converted.split("\n")
    total_text = 0
    changed_text = 0
    for orig, conv in zip(orig_lines, conv_lines):
        stripped = orig.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        total_text += 1
        if orig != conv:
            changed_text += 1

    # Output path: original_繁體.srt
    out_path = srt_path.parent / f"{srt_path.stem}_繁體.srt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(converted)

    return {
        "success": True,
        "input_path": str(srt_path),
        "output_path": str(out_path),
        "variant": variant,
        "total_lines": total_text,
        "lines_changed": changed_text,
    }


@router.post("/{draft_name}/replace-text")
async def replace_text(draft_name: str, req: ReplaceTextRequest):
    """批量替換剪映草稿中指定 index 的文字內容。"""
    if not req.replacements:
        raise HTTPException(status_code=400, detail="replacements 不能為空")

    draft_file, data = _load_draft_json(draft_name)
    texts = data.get("materials", {}).get("texts", [])

    changes: list[ConvertChange] = []
    for rep in req.replacements:
        if rep.index < 0 or rep.index >= len(texts):
            raise HTTPException(
                status_code=400,
                detail=f"index {rep.index} 超出範圍 (0-{len(texts) - 1})",
            )
        item = texts[rep.index]
        raw_content = item.get("content", "")
        try:
            content = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            content = {"text": ""}

        original = content.get("text", "")
        content["text"] = rep.text

        # Update style ranges
        if "styles" in content:
            for style in content["styles"]:
                if "range" in style:
                    style["range"] = [0, len(rep.text)]

        item["content"] = json.dumps(content, ensure_ascii=False)
        changes.append(ConvertChange(index=rep.index, before=original, after=rep.text))

    # Save back
    with open(draft_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return {
        "success": True,
        "draft_name": draft_name,
        "replaced_count": len(changes),
        "changes": changes,
    }
