"""Tests for PresetService."""
import json
import pytest
from pathlib import Path

from services.preset_service import PresetService


@pytest.fixture
def preset_dir(tmp_path):
    """Provide a temp style_presets directory."""
    d = tmp_path / "style_presets"
    d.mkdir()
    return d


@pytest.fixture
def config_path(tmp_path):
    """Provide a temp translation_config.json path."""
    return tmp_path / "translation_config.json"


@pytest.fixture
def service(preset_dir, config_path):
    return PresetService(presets_dir=preset_dir, config_path=config_path)


def write_preset(preset_dir: Path, id: str, name: str, description: str = ""):
    """Helper to write a minimal preset file."""
    data = {
        "id": id,
        "name": name,
        "description": description,
        "subtitle_style": {"font_size": 14, "text_color": "#ffffff"},
        "title_style": {"font_size": 5.0, "random_bg_colors": ["#ff0000"]},
    }
    (preset_dir / f"{id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data


# --- list_presets ---

def test_list_presets_empty_dir(service):
    assert service.list_presets() == []


def test_list_presets_returns_summary_only(service, preset_dir):
    write_preset(preset_dir, "default", "預設", "default look")
    write_preset(preset_dir, "yellow", "黃色粗體", "")

    result = service.list_presets()
    assert len(result) == 2
    ids = {p["id"] for p in result}
    assert ids == {"default", "yellow"}
    # Each entry must have exactly id/name/description, no style data
    for p in result:
        assert set(p.keys()) == {"id", "name", "description"}


def test_list_presets_skips_hidden_files(service, preset_dir):
    write_preset(preset_dir, "visible", "Visible")
    # Hidden file (starts with .)
    (preset_dir / ".bag_state.json").write_text("{}", encoding="utf-8")

    result = service.list_presets()
    assert [p["id"] for p in result] == ["visible"]


def test_list_presets_sorted_by_id(service, preset_dir):
    write_preset(preset_dir, "zebra", "Z")
    write_preset(preset_dir, "apple", "A")
    write_preset(preset_dir, "mango", "M")

    result = service.list_presets()
    assert [p["id"] for p in result] == ["apple", "mango", "zebra"]


# --- get_preset ---

def test_get_preset_returns_full_content(service, preset_dir):
    expected = write_preset(preset_dir, "default", "預設", "the look")
    result = service.get_preset("default")
    assert result == expected


def test_get_preset_missing_raises(service):
    from services.preset_service import PresetNotFoundError
    with pytest.raises(PresetNotFoundError):
        service.get_preset("does_not_exist")


def test_get_preset_ignores_hidden_id(service, preset_dir):
    # Create a real preset and a hidden state file
    write_preset(preset_dir, "real", "Real")
    (preset_dir / ".bag_state.json").write_text('{"foo":"bar"}', encoding="utf-8")

    from services.preset_service import PresetNotFoundError
    with pytest.raises(PresetNotFoundError):
        service.get_preset(".bag_state")


# --- apply_preset ---

def test_apply_preset_overwrites_style_sections(service, preset_dir, config_path):
    # Existing config with old style values + unrelated sections
    config_path.write_text(json.dumps({
        "whisper": {"model": "base"},
        "subtitle_style": {"font_size": 99, "text_color": "#000000"},
        "title_style": {"font_size": 99, "random_bg_colors": ["#000000"]},
        "active_preset_id": "old",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    write_preset(preset_dir, "yellow", "黃色")

    result = service.apply_preset("yellow")

    # Returned config has new style sections
    assert result["subtitle_style"]["text_color"] == "#ffffff"
    assert result["subtitle_style"]["font_size"] == 14
    assert result["title_style"]["random_bg_colors"] == ["#ff0000"]
    # Active preset id updated
    assert result["active_preset_id"] == "yellow"
    # Unrelated sections untouched
    assert result["whisper"] == {"model": "base"}

    # File on disk reflects same content
    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert on_disk == result


def test_apply_preset_creates_config_when_missing(service, preset_dir, config_path):
    write_preset(preset_dir, "yellow", "黃色")
    assert not config_path.exists()

    result = service.apply_preset("yellow")
    assert config_path.exists()
    assert result["active_preset_id"] == "yellow"


def test_apply_preset_missing_raises(service):
    from services.preset_service import PresetNotFoundError
    with pytest.raises(PresetNotFoundError):
        service.apply_preset("nope")


# --- save_preset ---

def test_save_preset_merges_subtitle_style(service, preset_dir):
    # Existing preset has extra keys we must preserve
    full = {
        "id": "mine",
        "name": "My Preset",
        "description": "desc",
        "subtitle_style": {
            "font_size": 14,
            "text_color": "#ffffff",
            "fonts": [{"id": "f1"}],
            "text_color_options": ["#aaa", "#bbb"],
        },
        "title_style": {"font_size": 5.0, "random_bg_colors": ["#111"]},
    }
    (preset_dir / "mine.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = service.save_preset(
        "mine",
        subtitle_style={"font_size": 20, "text_color": "#123456"},
    )

    # Modified fields updated
    assert result["subtitle_style"]["font_size"] == 20
    assert result["subtitle_style"]["text_color"] == "#123456"
    # Unmodified sub-keys preserved
    assert result["subtitle_style"]["fonts"] == [{"id": "f1"}]
    assert result["subtitle_style"]["text_color_options"] == ["#aaa", "#bbb"]
    # id/name/description preserved
    assert result["id"] == "mine"
    assert result["name"] == "My Preset"
    assert result["description"] == "desc"
    # title_style untouched (no payload)
    assert result["title_style"] == {"font_size": 5.0, "random_bg_colors": ["#111"]}

    # Disk file matches
    on_disk = json.loads((preset_dir / "mine.json").read_text(encoding="utf-8"))
    assert on_disk == result


def test_save_preset_merges_title_style(service, preset_dir):
    write_preset(preset_dir, "mine", "My Preset")

    result = service.save_preset(
        "mine",
        title_style={"font_size": 7.5, "position_y": 0.9},
    )

    assert result["title_style"]["font_size"] == 7.5
    assert result["title_style"]["position_y"] == 0.9
    # random_bg_colors (not in payload) preserved
    assert result["title_style"]["random_bg_colors"] == ["#ff0000"]
    # subtitle_style untouched
    assert result["subtitle_style"] == {"font_size": 14, "text_color": "#ffffff"}


def test_save_preset_merges_both_sections(service, preset_dir):
    write_preset(preset_dir, "mine", "My Preset")

    result = service.save_preset(
        "mine",
        subtitle_style={"font_size": 18},
        title_style={"font_size": 6.0},
    )

    assert result["subtitle_style"]["font_size"] == 18
    assert result["subtitle_style"]["text_color"] == "#ffffff"  # preserved
    assert result["title_style"]["font_size"] == 6.0
    assert result["title_style"]["random_bg_colors"] == ["#ff0000"]  # preserved


def test_save_preset_no_payload_is_noop(service, preset_dir):
    original = write_preset(preset_dir, "mine", "My Preset")
    result = service.save_preset("mine")
    assert result == original


def test_save_preset_missing_raises(service):
    from services.preset_service import PresetNotFoundError
    with pytest.raises(PresetNotFoundError):
        service.save_preset("nope", subtitle_style={"font_size": 10})


def test_save_preset_rejects_path_traversal(service, preset_dir):
    write_preset(preset_dir, "mine", "My Preset")
    from services.preset_service import PresetNotFoundError
    with pytest.raises(PresetNotFoundError):
        service.save_preset("../../../etc/hosts", subtitle_style={})
    with pytest.raises(PresetNotFoundError):
        service.save_preset(".hidden", subtitle_style={})


# --- draw_title_color (shuffle bag) ---

def write_preset_with_colors(preset_dir: Path, id: str, colors: list[str]):
    data = {
        "id": id,
        "name": id,
        "description": "",
        "subtitle_style": {},
        "title_style": {"random_bg_colors": colors},
    }
    (preset_dir / f"{id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_draw_title_color_single_color(service, preset_dir):
    write_preset_with_colors(preset_dir, "mono", ["#abcdef"])
    for _ in range(5):
        assert service.draw_title_color("mono") == "#abcdef"


def test_draw_title_color_no_repeats_within_cycle(service, preset_dir):
    colors = ["#aa0000", "#00aa00", "#0000aa", "#aaaa00", "#aa00aa"]
    write_preset_with_colors(preset_dir, "rainbow", colors)

    drawn = [service.draw_title_color("rainbow") for _ in range(len(colors))]
    assert sorted(drawn) == sorted(colors), "every color should appear exactly once per cycle"


def test_draw_title_color_no_repeat_across_cycle_boundary(service, preset_dir):
    colors = ["#aa0000", "#00aa00", "#0000aa", "#aaaa00", "#aa00aa"]
    write_preset_with_colors(preset_dir, "rainbow", colors)

    # Draw 100 cycles' worth and check no two consecutive draws are equal
    sequence = [service.draw_title_color("rainbow") for _ in range(len(colors) * 100)]
    for i in range(1, len(sequence)):
        assert sequence[i] != sequence[i - 1], (
            f"consecutive repeat at index {i}: {sequence[i-1]} -> {sequence[i]}"
        )


def test_draw_title_color_resets_when_color_list_changes(service, preset_dir):
    write_preset_with_colors(preset_dir, "rainbow", ["#aa0000", "#00aa00"])
    service.draw_title_color("rainbow")
    service.draw_title_color("rainbow")  # bag exhausted

    # Edit preset: replace colors entirely
    write_preset_with_colors(preset_dir, "rainbow", ["#ffffff", "#000000", "#888888"])

    # Should detect change and reshuffle from new source
    drawn = [service.draw_title_color("rainbow") for _ in range(3)]
    assert sorted(drawn) == ["#000000", "#888888", "#ffffff"]


def test_draw_title_color_state_persists_to_disk(service, preset_dir):
    colors = ["#a", "#b", "#c"]
    write_preset_with_colors(preset_dir, "p", colors)

    service.draw_title_color("p")
    service.draw_title_color("p")

    # Bag state file should exist and record cursor=2
    state = json.loads((preset_dir / ".bag_state.json").read_text(encoding="utf-8"))
    assert state["p"]["cursor"] == 2
    assert sorted(state["p"]["source_colors"]) == sorted(colors)


def test_draw_title_color_isolated_per_preset(service, preset_dir):
    write_preset_with_colors(preset_dir, "a", ["#a1", "#a2"])
    write_preset_with_colors(preset_dir, "b", ["#b1", "#b2"])

    # Drawing from "a" should not affect "b"'s state
    service.draw_title_color("a")
    service.draw_title_color("a")  # exhaust a
    drawn_b = [service.draw_title_color("b") for _ in range(2)]
    assert sorted(drawn_b) == ["#b1", "#b2"]


def test_draw_title_color_handles_empty_color_list(service, preset_dir):
    write_preset_with_colors(preset_dir, "empty", [])
    # Should not raise; falls back to a sentinel default
    color = service.draw_title_color("empty")
    assert isinstance(color, str)
    assert color.startswith("#")


# --- ensure_default_preset (seed) ---

def test_ensure_default_preset_seeds_from_config(service, preset_dir, config_path):
    config_path.write_text(json.dumps({
        "whisper": {"model": "base"},
        "subtitle_style": {"font_size": 14, "text_color": "#ffe759"},
        "title_style": {"font_size": 5.0, "random_bg_colors": ["#ff0000"]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    service.ensure_default_preset()

    seeded_path = preset_dir / "default.json"
    assert seeded_path.exists()

    seeded = json.loads(seeded_path.read_text(encoding="utf-8"))
    assert seeded["id"] == "default"
    assert seeded["subtitle_style"]["text_color"] == "#ffe759"
    assert seeded["title_style"]["random_bg_colors"] == ["#ff0000"]


def test_ensure_default_preset_does_not_overwrite_existing(service, preset_dir, config_path):
    config_path.write_text(json.dumps({
        "subtitle_style": {"text_color": "#NEW"},
        "title_style": {},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Pre-existing default.json with custom content
    existing = {
        "id": "default",
        "name": "我自訂的",
        "description": "",
        "subtitle_style": {"text_color": "#OLD"},
        "title_style": {},
    }
    (preset_dir / "default.json").write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    service.ensure_default_preset()

    on_disk = json.loads((preset_dir / "default.json").read_text(encoding="utf-8"))
    assert on_disk["subtitle_style"]["text_color"] == "#OLD"
    assert on_disk["name"] == "我自訂的"


def test_ensure_default_preset_no_config_no_op(service, preset_dir, config_path):
    # config doesn't exist
    assert not config_path.exists()
    service.ensure_default_preset()
    assert not (preset_dir / "default.json").exists()
