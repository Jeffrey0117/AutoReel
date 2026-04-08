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
