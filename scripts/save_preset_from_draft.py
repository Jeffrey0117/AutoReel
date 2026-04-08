"""
從剪映草稿讀回字幕 / 標題樣式，覆蓋存回 style_presets/{preset_id}.json

用法:
    py -3 scripts/save_preset_from_draft.py <preset_id>
    py -3 scripts/save_preset_from_draft.py --all     # 全部 preset 一次讀回

草稿名稱必須是 樣式預覽_{preset_id}
"""
import sys
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from services.preset_service import preset_service, PresetNotFoundError

PREVIEW_PREFIX = "樣式預覽_"
PRESETS_DIR = PROJECT_ROOT / "style_presets"


def get_draft_root() -> Path:
    """取得剪映草稿根資料夾"""
    config_path = PROJECT_ROOT / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            folder = cfg.get("jianying_draft_folder")
            if folder:
                return Path(folder)
        except Exception:
            pass

    username = os.environ.get("USERNAME") or os.getlogin()
    return Path(rf"C:\Users\{username}\AppData\Local\JianyingPro\User Data\Projects\com.lveditor.draft")


def rgb_float_to_hex(rgb: list) -> str:
    """[0.5, 0.1, 0.2] -> #801933"""
    if not rgb or len(rgb) < 3:
        return "#000000"
    r, g, b = (max(0, min(255, int(round(v * 255)))) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_content_styles(material: dict) -> dict:
    """從 material['content'] (JSON string) 解析 styles[0]"""
    try:
        content = json.loads(material.get("content", "{}"))
        styles = content.get("styles", [])
        if styles:
            return styles[0]
    except Exception:
        pass
    return {}


def find_track(draft: dict, name: str) -> dict:
    """依 name 找軌道"""
    for t in draft.get("tracks", []):
        if t.get("type") == "text" and t.get("name") == name:
            return t
    return {}


def get_material_by_id(draft: dict, mid: str) -> dict:
    for m in draft.get("materials", {}).get("texts", []):
        if m.get("id") == mid:
            return m
    return {}


def extract_subtitle_style(draft: dict) -> dict:
    """從「字幕軌道」第一條取出字幕樣式"""
    track = find_track(draft, "字幕軌道")
    segments = track.get("segments", [])
    if not segments:
        raise RuntimeError("找不到字幕軌道或軌道沒有任何字幕")

    first_seg = segments[0]
    material_id = first_seg.get("material_id")
    mat = get_material_by_id(draft, material_id)
    if not mat:
        raise RuntimeError(f"找不到字幕 material: {material_id}")

    # 從 material 直接讀大部分欄位
    style = {
        "font_resource_id": mat.get("font_resource_id", ""),
        "font_path": mat.get("font_path", ""),
        "font_size": mat.get("font_size"),
        "background_color": mat.get("background_color", "#000000"),
        "background_alpha": mat.get("background_alpha", 0.65),
        "background_style": mat.get("background_style", 1),
        "border_width": mat.get("border_width"),
        "border_color": mat.get("border_color", "#000000"),
        "bold_width": mat.get("bold_width"),
        "shadow_alpha": mat.get("shadow_alpha"),
        "shadow_distance": mat.get("shadow_distance"),
        "shadow_smoothing": mat.get("shadow_smoothing"),
        "line_max_width": mat.get("line_max_width"),
        "fonts": mat.get("fonts", []),
    }

    # text_color / stroke 從 content.styles[0] 解析
    style_entry = parse_content_styles(mat)

    # text_color
    fill = style_entry.get("fill", {}).get("content", {}).get("solid", {})
    if "color" in fill:
        style["text_color"] = rgb_float_to_hex(fill["color"])
    else:
        style["text_color"] = mat.get("text_color", "#FFFFFF")

    # stroke_width / stroke_color
    strokes = style_entry.get("strokes", [])
    if strokes:
        s0 = strokes[0]
        style["stroke_width"] = s0.get("width", 0.173)
        stroke_color_rgb = s0.get("content", {}).get("solid", {}).get("color")
        if stroke_color_rgb:
            style["stroke_color"] = rgb_float_to_hex(stroke_color_rgb)
        else:
            style["stroke_color"] = "#000000"
    else:
        style["stroke_width"] = 0.173
        style["stroke_color"] = "#000000"

    # position_y from segment.clip.transform.y
    clip = first_seg.get("clip", {})
    transform = clip.get("transform", {})
    if "y" in transform:
        style["position_y"] = transform["y"]

    # max_chars_per_line 保留原 preset 的值（草稿裡看不出來）
    return style


def extract_title_style(draft: dict) -> dict:
    """從「標題軌道」取出標題樣式（不含 random_bg_colors，那個要保留原 preset 的）"""
    track = find_track(draft, "標題軌道")
    segments = track.get("segments", [])
    if not segments:
        return {}

    first_seg = segments[0]
    material_id = first_seg.get("material_id")
    mat = get_material_by_id(draft, material_id)
    if not mat:
        return {}

    style = {
        "font_resource_id": mat.get("font_resource_id", ""),
        "font_size": mat.get("font_size"),
        "border_width": mat.get("border_width"),
        "border_color": mat.get("border_color", "#666666"),
        "shadow_alpha": mat.get("shadow_alpha"),
        "background_alpha": mat.get("background_alpha"),
    }

    clip = first_seg.get("clip", {})
    transform = clip.get("transform", {})
    if "y" in transform:
        style["position_y"] = transform["y"]

    return style


def save_one(preset_id: str) -> None:
    """從剪映草稿讀回樣式，覆蓋 style_presets/{preset_id}.json"""
    draft_root = get_draft_root()
    draft_name = f"{PREVIEW_PREFIX}{preset_id}"
    draft_file = draft_root / draft_name / "draft_content.json"

    if not draft_file.exists():
        raise FileNotFoundError(f"找不到草稿: {draft_file}")

    draft = json.loads(draft_file.read_text(encoding="utf-8"))

    # 讀回舊 preset（保留 name, description, random_bg_colors, max_chars_per_line 等草稿讀不到的欄位）
    try:
        old_preset = preset_service.get_preset(preset_id)
    except PresetNotFoundError:
        raise FileNotFoundError(f"Preset 不存在: {preset_id}")

    new_subtitle_style = extract_subtitle_style(draft)
    new_title_style = extract_title_style(draft)

    old_subtitle = old_preset.get("subtitle_style", {})

    # 若原 preset 是隨機色，讀回草稿會只抓到某一個隨機選中的顏色 — 保留原本設定
    if old_subtitle.get("text_color_random"):
        new_subtitle_style.pop("text_color", None)

    # 合併：新樣式覆蓋舊值，但保留舊 preset 的 max_chars_per_line / text_color_random / text_color_options
    merged_subtitle = {
        **old_subtitle,
        **new_subtitle_style,
    }
    merged_title = {
        **old_preset.get("title_style", {}),
        **new_title_style,
    }

    new_preset = {
        "id": old_preset.get("id", preset_id),
        "name": old_preset.get("name", preset_id),
        "description": old_preset.get("description", ""),
        "subtitle_style": merged_subtitle,
        "title_style": merged_title,
    }

    # 原子寫入
    target = PRESETS_DIR / f"{preset_id}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(new_preset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(target)

    print(f"[OK] {preset_id}")
    print(f"     字幕: font_size={merged_subtitle.get('font_size')}, "
          f"text_color={merged_subtitle.get('text_color')}, "
          f"bg={merged_subtitle.get('background_color')}, "
          f"position_y={merged_subtitle.get('position_y')}")
    if merged_title:
        print(f"     標題: font_size={merged_title.get('font_size')}, "
              f"position_y={merged_title.get('position_y')}")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("用法: py -3 scripts/save_preset_from_draft.py <preset_id>")
        print("      py -3 scripts/save_preset_from_draft.py --all")
        return 1

    if args[0] == "--all":
        presets = preset_service.list_presets()
        targets = [p["id"] for p in presets]
    else:
        targets = args

    errors: list[str] = []
    for pid in targets:
        try:
            save_one(pid)
        except Exception as e:
            errors.append(f"{pid}: {e}")
            print(f"[FAIL] {pid}: {e}")

    print(f"\n完成 {len(targets) - len(errors)}/{len(targets)}")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
