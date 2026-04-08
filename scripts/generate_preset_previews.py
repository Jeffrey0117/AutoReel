"""
生成預設集預覽草稿

為每個 style_presets/*.json 產生 1 個剪映草稿（含範例字幕 + 標題）。
你在剪映裡打開草稿手動調整字幕 → 然後用 save_preset_from_draft.py 存回。

用法:
    py -3 scripts/generate_preset_previews.py

草稿名稱: 樣式預覽_{preset_id}
位置: 剪映草稿資料夾 (translation_config.json / config.json 指定)
"""
import sys
import copy
import json
import shutil
from pathlib import Path

# 加入 project root 到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from translate_video import TranslationWorkflow
from subtitle_generator import SubtitleEntry
from services.preset_service import preset_service

# 範例字幕（中文 + 英文混排，讓你在剪映能看到各種情況）
SAMPLE_ENTRIES = [
    SubtitleEntry(
        index=1, start_time=0.0, end_time=3.0,
        text_original="This is a preview subtitle",
        text_translated="這是一行預覽字幕",
    ),
    SubtitleEntry(
        index=2, start_time=3.0, end_time=6.0,
        text_original="Adjust the style you want in JianYing",
        text_translated="在剪映裡調整成你要的樣式",
    ),
    SubtitleEntry(
        index=3, start_time=6.0, end_time=9.0,
        text_original="Font, color, stroke, shadow, position",
        text_translated="字體、顏色、描邊、陰影、位置",
    ),
    SubtitleEntry(
        index=4, start_time=9.0, end_time=12.0,
        text_original="Then tell me the preset id",
        text_translated="調好後告訴我預設集 id",
    ),
    SubtitleEntry(
        index=5, start_time=12.0, end_time=15.0,
        text_original="I will read it back into the preset JSON",
        text_translated="我會把樣式讀回預設集 JSON",
    ),
]

PREVIEW_PREFIX = "樣式預覽_"
CONFIG_PATH = PROJECT_ROOT / "translation_config.json"


def backup_config() -> dict:
    """備份當前 translation_config.json"""
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def restore_config(data: dict) -> None:
    """還原 translation_config.json"""
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_one(preset_id: str) -> str:
    """為單一 preset 生成預覽草稿，回傳草稿資料夾名稱"""
    # 1. 套用 preset → 寫入 translation_config.json
    preset_service.apply_preset(preset_id)

    # 2. 重新 new workflow（會讀到新的 style 設定）
    workflow = TranslationWorkflow(str(CONFIG_PATH))

    # 3. 載入 template
    template_data = workflow._load_template()
    if template_data is None:
        raise RuntimeError(f"無法載入 template: {workflow.template_name}")

    # 4. Deep copy → 清乾淨 → 加字幕 → 加標題
    draft_data = copy.deepcopy(template_data)
    draft_data = workflow._clean_template_texts(draft_data, keep_count=2)
    draft_data = workflow._add_subtitles_to_draft(draft_data, SAMPLE_ENTRIES, template_data)
    draft_data = workflow._add_title_to_draft(
        draft_data,
        video_name=f"預覽-{preset_id}",
        duration_us=draft_data.get("duration", 15_000_000),
    )

    # 5. 寫入剪映草稿資料夾
    draft_name = f"{PREVIEW_PREFIX}{preset_id}"
    output_folder = workflow.jianying_draft_root / draft_name

    if output_folder.exists():
        shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    (output_folder / "draft_content.json").write_text(
        json.dumps(draft_data, ensure_ascii=False),
        encoding="utf-8",
    )

    # 複製 template 的附屬檔案
    template_folder = workflow.jianying_draft_root / workflow.template_name
    for fname in ("draft_meta_info.json", "draft_settings"):
        src = template_folder / fname
        if src.exists():
            shutil.copy(src, output_folder / fname)

    return draft_name


def main() -> int:
    # 取得所有 preset
    presets = preset_service.list_presets()
    if not presets:
        print("[Error] style_presets/ 裡沒有任何 preset")
        return 1

    print(f"[Info] 找到 {len(presets)} 個 preset: {[p['id'] for p in presets]}")

    # 備份原本的 config
    original_config = backup_config()
    print(f"[Info] 已備份 translation_config.json (active_preset_id={original_config.get('active_preset_id')})")

    results: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []

    try:
        for p in presets:
            preset_id = p["id"]
            try:
                draft_name = generate_one(preset_id)
                results.append((preset_id, draft_name))
                print(f"  [OK] {preset_id} -> {draft_name}")
            except Exception as e:
                failures.append((preset_id, str(e)))
                print(f"  [FAIL] {preset_id}: {e}")
    finally:
        # 不管成敗都還原原本的 config
        restore_config(original_config)
        print(f"[Info] 已還原 translation_config.json")

    print("\n=== 結果 ===")
    print(f"成功: {len(results)}  失敗: {len(failures)}")
    for preset_id, draft_name in results:
        print(f"  ✓ {preset_id}: 剪映打開「{draft_name}」")
    for preset_id, err in failures:
        print(f"  ✗ {preset_id}: {err}")

    if results:
        print("\n下一步:")
        print("  1. 打開剪映 → 找到「樣式預覽_xxx」草稿 → 調整字幕樣式")
        print("  2. 存檔後執行: py -3 scripts/save_preset_from_draft.py <preset_id>")

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
