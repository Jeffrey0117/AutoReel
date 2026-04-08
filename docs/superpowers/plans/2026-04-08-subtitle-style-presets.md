# Subtitle Style Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a preset-based subtitle/title style system to AutoReel so users can save/switch between visual styles via a dropdown, plus a shuffle-bag algorithm to fix title color repetition.

**Architecture:** Presets are JSON files in `style_presets/`. A `PresetService` class manages CRUD + the shuffle bag. New FastAPI routes expose preset list/get/apply. The frontend's existing subtitle editor is migrated from the legacy `subtitle_position_server.py` (port 8000) to the FastAPI backend (port 8001) to consolidate config-handling onto one server. The translate pipeline reads `active_preset_id` from config and calls `preset_service.draw_title_color()` instead of `random.choice()`.

**Tech Stack:** Python 3.13, FastAPI, pytest, Vue 3 (in `app.html`), JSON file persistence.

**Spec:** See `docs/superpowers/specs/2026-04-08-subtitle-style-presets-design.md` for design rationale.

---

## File Structure

### New files
- `style_presets/` — directory containing preset JSON files
- `style_presets/default.json` — auto-seeded from current `translation_config.json`
- `style_presets/.bag_state.json` — shuffle bag state (gitignored)
- `backend/services/preset_service.py` — `PresetService` class (list/get/apply + shuffle bag)
- `tests/__init__.py` — empty
- `tests/services/__init__.py` — empty
- `tests/services/test_preset_service.py` — pytest tests for `PresetService`

### Modified files
- `backend/api/translate_routes.py` — add 3 new routes (`GET /presets`, `GET /presets/{id}`, `POST /presets/{id}/apply`)
- `translate_video.py:1134` and `translate_video.py:1360` — replace `random.choice(bg_colors)` with `preset_service.draw_title_color(active_preset_id)`
- `app.html` — add a new `FASTAPI_BASE` constant (port 8001); migrate subtitle editor's fetch URLs from `${API_BASE}/api/config` (legacy) to `${FASTAPI_BASE}/api/translate/config`; add preset dropdown UI; add `loadPresets()` / `applyPreset()` methods
- `.gitignore` — add `style_presets/.bag_state.json`
- `requirements.txt` — uncomment `pytest>=7.0.0`

### Untouched
- `subtitle_position_server.py` (legacy) — left alone; the frontend stops calling it for config, but other standalone HTML pages may still use it. Cleanup is out of scope.
- `translation_config.json` schema is unchanged except for one new top-level field `active_preset_id` (added by `apply_preset()`).
- The translate pipeline (`translate_video.py`) is otherwise unchanged.

---

## Task 1: Set up test infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/services/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Uncomment pytest in requirements.txt**

Open `requirements.txt`, find the line `# pytest>=7.0.0` and remove the `# `:

```
pytest>=7.0.0
```

- [ ] **Step 2: Install pytest**

Run: `pip install pytest>=7.0.0`
Expected: Successfully installs pytest.

- [ ] **Step 3: Create empty test packages**

```bash
mkdir -p tests/services
```

Create `tests/__init__.py` (empty file).
Create `tests/services/__init__.py` (empty file).

- [ ] **Step 4: Create conftest.py with project root path setup**

Create `tests/conftest.py` with this content:

```python
"""Pytest config — adds project root and backend to sys.path so tests can import."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"

for path in (PROJECT_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
```

- [ ] **Step 5: Verify pytest can discover tests**

Run: `pytest tests/ -v --collect-only`
Expected: `no tests ran` (no tests yet, but pytest works).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "test: 加入 pytest 測試骨架"
```

---

## Task 2: PresetService — list_presets()

**Files:**
- Create: `backend/services/preset_service.py`
- Create: `tests/services/test_preset_service.py`

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_preset_service.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_preset_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'services.preset_service'`.

- [ ] **Step 3: Create PresetService skeleton + list_presets()**

Create `backend/services/preset_service.py`:

```python
"""
Preset Service - manages subtitle/title style presets and shuffle-bag color rotation.

A "preset" is a JSON file in style_presets/ containing a complete snapshot of
subtitle_style + title_style. Applying a preset overwrites those sections in
translation_config.json. Hidden files (starting with `.`) are not exposed as presets.
"""

import json
from pathlib import Path
from typing import Optional


class PresetNotFoundError(Exception):
    """Raised when a preset id doesn't exist."""


class PresetService:
    def __init__(
        self,
        presets_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        # Allow injection for tests; default to project layout
        if presets_dir is None:
            presets_dir = Path(__file__).parent.parent.parent / "style_presets"
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "translation_config.json"

        self.presets_dir = Path(presets_dir)
        self.config_path = Path(config_path)
        self.bag_state_path = self.presets_dir / ".bag_state.json"

    def list_presets(self) -> list[dict]:
        """Return all presets as [{id, name, description}], sorted by id."""
        if not self.presets_dir.exists():
            return []

        summaries = []
        for f in sorted(self.presets_dir.glob("*.json")):
            if f.name.startswith("."):
                continue
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                summaries.append({
                    "id": data.get("id", f.stem),
                    "name": data.get("name", f.stem),
                    "description": data.get("description", ""),
                })
            except (json.JSONDecodeError, OSError):
                # Skip malformed files silently — should not break the listing
                continue
        return summaries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/services/test_preset_service.py -v`
Expected: All 4 `list_presets` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/preset_service.py tests/services/test_preset_service.py
git commit -m "feat: PresetService.list_presets() with tests"
```

---

## Task 3: PresetService — get_preset()

**Files:**
- Modify: `backend/services/preset_service.py`
- Modify: `tests/services/test_preset_service.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/services/test_preset_service.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/services/test_preset_service.py::test_get_preset_returns_full_content -v`
Expected: FAIL with `AttributeError: 'PresetService' object has no attribute 'get_preset'`.

- [ ] **Step 3: Implement get_preset**

Add this method to `PresetService` in `backend/services/preset_service.py`:

```python
    def get_preset(self, id: str) -> dict:
        """Return the full JSON content of one preset. Raises PresetNotFoundError if missing."""
        if not id or id.startswith(".") or "/" in id or "\\" in id:
            raise PresetNotFoundError(f"Invalid preset id: {id!r}")

        path = self.presets_dir / f"{id}.json"
        if not path.exists():
            raise PresetNotFoundError(f"Preset not found: {id!r}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/services/test_preset_service.py -v`
Expected: All `get_preset` tests PASS along with prior `list_presets` tests.

- [ ] **Step 5: Commit**

```bash
git add backend/services/preset_service.py tests/services/test_preset_service.py
git commit -m "feat: PresetService.get_preset() with tests"
```

---

## Task 4: PresetService — apply_preset()

**Files:**
- Modify: `backend/services/preset_service.py`
- Modify: `tests/services/test_preset_service.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/services/test_preset_service.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_preset_service.py -v -k apply_preset`
Expected: FAIL with `AttributeError: 'PresetService' object has no attribute 'apply_preset'`.

- [ ] **Step 3: Implement apply_preset**

Add this method to `PresetService`:

```python
    def apply_preset(self, id: str) -> dict:
        """
        Read preset and overwrite subtitle_style + title_style + active_preset_id
        in translation_config.json. Returns the new full config.
        Raises PresetNotFoundError if preset doesn't exist.
        """
        preset = self.get_preset(id)  # raises PresetNotFoundError if missing

        # Load existing config (or start fresh)
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        # Overwrite style sections wholesale
        config["subtitle_style"] = preset.get("subtitle_style", {})
        config["title_style"] = preset.get("title_style", {})
        config["active_preset_id"] = id

        # Atomic-ish write: write to temp then rename
        tmp_path = self.config_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.config_path)

        return config
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/services/test_preset_service.py -v`
Expected: All tests so far PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/services/preset_service.py tests/services/test_preset_service.py
git commit -m "feat: PresetService.apply_preset() overwrites config style sections"
```

---

## Task 5: PresetService — draw_title_color() shuffle bag

**Files:**
- Modify: `backend/services/preset_service.py`
- Modify: `tests/services/test_preset_service.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/services/test_preset_service.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_preset_service.py -v -k draw_title`
Expected: FAIL with `AttributeError: 'PresetService' object has no attribute 'draw_title_color'`.

- [ ] **Step 3: Implement draw_title_color**

Add these imports at the top of `backend/services/preset_service.py`:

```python
import random
import threading
```

Add a class-level lock and the method to `PresetService`:

```python
    # Class-level lock for bag state file (parallel pipeline safety)
    _bag_lock = threading.Lock()

    DEFAULT_FALLBACK_COLOR = "#000000"

    def draw_title_color(self, preset_id: str) -> str:
        """
        Draw the next title background color for the given preset using a shuffle-bag.
        Guarantees no repeat within a cycle and no immediate repeat across cycle boundaries.
        Falls back to DEFAULT_FALLBACK_COLOR if the preset has an empty color list.
        """
        with self._bag_lock:
            # Read current source colors from preset
            try:
                preset = self.get_preset(preset_id)
            except PresetNotFoundError:
                return self.DEFAULT_FALLBACK_COLOR

            source_colors = list(preset.get("title_style", {}).get("random_bg_colors", []))

            if not source_colors:
                return self.DEFAULT_FALLBACK_COLOR

            if len(source_colors) == 1:
                return source_colors[0]

            # Load bag state
            state = self._load_bag_state()
            preset_state = state.get(preset_id)

            needs_reshuffle = (
                preset_state is None
                or preset_state.get("source_colors") != source_colors
                or preset_state.get("cursor", 0) >= len(preset_state.get("shuffled", []))
            )

            if needs_reshuffle:
                previous_last = None
                if (
                    preset_state is not None
                    and preset_state.get("source_colors") == source_colors
                    and preset_state.get("shuffled")
                ):
                    previous_last = preset_state["shuffled"][-1]

                shuffled = source_colors.copy()
                random.shuffle(shuffled)

                # Avoid placing previous_last as the first of the new cycle
                if previous_last is not None and shuffled[0] == previous_last:
                    swap_idx = random.randint(1, len(shuffled) - 1)
                    shuffled[0], shuffled[swap_idx] = shuffled[swap_idx], shuffled[0]

                preset_state = {
                    "source_colors": source_colors,
                    "shuffled": shuffled,
                    "cursor": 0,
                }

            # Draw and advance cursor
            color = preset_state["shuffled"][preset_state["cursor"]]
            preset_state["cursor"] += 1

            state[preset_id] = preset_state
            self._save_bag_state(state)

            return color

    def _load_bag_state(self) -> dict:
        if not self.bag_state_path.exists():
            return {}
        try:
            with open(self.bag_state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_bag_state(self, state: dict) -> None:
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.bag_state_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        tmp.replace(self.bag_state_path)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/services/test_preset_service.py -v`
Expected: All tests PASS (including all shuffle bag tests).

- [ ] **Step 5: Commit**

```bash
git add backend/services/preset_service.py tests/services/test_preset_service.py
git commit -m "feat: shuffle-bag title color rotation in PresetService"
```

---

## Task 6: Auto-seed default.json

**Files:**
- Modify: `backend/services/preset_service.py`
- Modify: `tests/services/test_preset_service.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/services/test_preset_service.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/services/test_preset_service.py -v -k ensure_default`
Expected: FAIL with `AttributeError: 'PresetService' object has no attribute 'ensure_default_preset'`.

- [ ] **Step 3: Implement ensure_default_preset**

Add this method to `PresetService`:

```python
    def ensure_default_preset(self) -> None:
        """
        Create style_presets/default.json from current translation_config.json
        if it doesn't already exist. Never overwrites an existing default.json.
        Silently no-ops if the source config doesn't exist.
        """
        default_path = self.presets_dir / "default.json"
        if default_path.exists():
            return
        if not self.config_path.exists():
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        seeded = {
            "id": "default",
            "name": "預設",
            "description": "從目前的 translation_config.json 自動產生",
            "subtitle_style": config.get("subtitle_style", {}),
            "title_style": config.get("title_style", {}),
        }

        self.presets_dir.mkdir(parents=True, exist_ok=True)
        with open(default_path, "w", encoding="utf-8") as f:
            json.dump(seeded, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/services/test_preset_service.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Add module-level singleton**

Append to the bottom of `backend/services/preset_service.py`:

```python
# Module singleton (default project paths)
preset_service = PresetService()
```

- [ ] **Step 6: Commit**

```bash
git add backend/services/preset_service.py tests/services/test_preset_service.py
git commit -m "feat: PresetService.ensure_default_preset() seeds from current config"
```

---

## Task 7: FastAPI preset routes

**Files:**
- Modify: `backend/api/translate_routes.py`
- Modify: `backend/main.py` (auto-seed on lifespan startup)

- [ ] **Step 1: Add new routes to translate_routes.py**

Open `backend/api/translate_routes.py`. After the existing imports at line 11, add:

```python
from services.preset_service import preset_service, PresetNotFoundError
```

After the existing `update_config` route (line 179), append:

```python


# --- Style Presets ---

@router.get("/presets")
async def list_presets():
    """List all subtitle/title style presets."""
    return {"presets": preset_service.list_presets()}


@router.get("/presets/{preset_id}")
async def get_preset(preset_id: str):
    """Get full content of one preset."""
    try:
        return preset_service.get_preset(preset_id)
    except PresetNotFoundError:
        raise HTTPException(404, f"Preset not found: {preset_id}")


@router.post("/presets/{preset_id}/apply")
async def apply_preset(preset_id: str):
    """Apply preset → overwrite subtitle_style + title_style in translation_config.json."""
    try:
        new_config = preset_service.apply_preset(preset_id)
        return {"success": True, "config": new_config}
    except PresetNotFoundError:
        raise HTTPException(404, f"Preset not found: {preset_id}")
```

- [ ] **Step 2: Add auto-seed call to backend/main.py lifespan**

Open `backend/main.py`. In the `lifespan` function (starts at line 41), find the line `init_db()` (line 43). Add right after it:

```python
    # Seed default style preset if missing
    from services.preset_service import preset_service
    try:
        preset_service.ensure_default_preset()
    except Exception as e:
        print(f"[main] preset seed failed (non-fatal): {e}")
```

- [ ] **Step 3: Manually test the routes**

Start the backend (use whatever you normally use, e.g. `python backend/main.py` or `pm2 restart autoreel`).

Then in another terminal:

```bash
curl http://localhost:8001/api/translate/presets
```
Expected: `{"presets":[{"id":"default","name":"預設","description":"..."}]}` (default.json was auto-seeded on startup).

```bash
curl http://localhost:8001/api/translate/presets/default
```
Expected: Full JSON with subtitle_style + title_style.

```bash
curl -X POST http://localhost:8001/api/translate/presets/default/apply
```
Expected: `{"success":true,"config":{...}}` and `translation_config.json` now has `active_preset_id: "default"`.

- [ ] **Step 4: Commit**

```bash
git add backend/api/translate_routes.py backend/main.py
git commit -m "feat: FastAPI routes for style preset list/get/apply"
```

---

## Task 8: Wire shuffle bag into translate_video.py

**Files:**
- Modify: `translate_video.py`

- [ ] **Step 1: Read the current code at the two replacement sites**

Open `translate_video.py` and locate lines 1125-1140 and 1350-1365. The two relevant sites are:

Around line 1125-1134:
```python
title_style = self.config.get("title_style", {})
...
bg_colors = title_style.get("random_bg_colors", ["#ff0000", "#00ff00", "#0066ff"])
...
bg_color = random.choice(bg_colors)
```

Around line 1353-1360 (same pattern, second usage).

- [ ] **Step 2: Add preset_service import at the top of translate_video.py**

Find the existing imports near the top of `translate_video.py`. Add this block immediately after the existing `import random` line (or near other imports):

```python
# Preset service for shuffle-bag title color rotation
import sys as _sys
from pathlib import Path as _Path
_backend_path = _Path(__file__).parent / "backend"
if str(_backend_path) not in _sys.path:
    _sys.path.insert(0, str(_backend_path))
try:
    from services.preset_service import preset_service as _preset_service
except ImportError:
    _preset_service = None
```

- [ ] **Step 3: Replace line 1134**

Replace this line in `translate_video.py` (around line 1134):

```python
        bg_color = random.choice(bg_colors)
```

with:

```python
        active_preset_id = self.config.get("active_preset_id", "default")
        if _preset_service is not None:
            bg_color = _preset_service.draw_title_color(active_preset_id)
        else:
            bg_color = random.choice(bg_colors)
```

- [ ] **Step 4: Replace line 1360**

Replace the same pattern at line 1360:

```python
                bg_color = random.choice(bg_colors)
```

with the same block:

```python
                active_preset_id = self.config.get("active_preset_id", "default")
                if _preset_service is not None:
                    bg_color = _preset_service.draw_title_color(active_preset_id)
                else:
                    bg_color = random.choice(bg_colors)
```

- [ ] **Step 5: Smoke test by importing**

Run: `python -c "import translate_video; print('OK')"`
Expected: `OK` printed (no import errors).

- [ ] **Step 6: Commit**

```bash
git add translate_video.py
git commit -m "feat: translate_video uses shuffle-bag for title bg color"
```

---

## Task 9: Frontend — migrate config endpoints to FastAPI

**Files:**
- Modify: `app.html`

> **Background:** `app.html` currently defines `const API_BASE = location.hostname === 'localhost' ? 'http://localhost:8000' : '';` which points to the legacy `subtitle_position_server.py` (port 8000). The FastAPI backend is reachable through the Node proxy on port 8001. We add a separate `FASTAPI_BASE` constant rather than re-route `API_BASE` so any other in-page features that still hit the legacy server keep working until they're individually migrated.

- [ ] **Step 1: Add FASTAPI_BASE constant**

In `app.html`, find the existing `API_BASE` declaration around line 3279:

```javascript
const API_BASE = location.hostname === 'localhost' ? 'http://localhost:8000' : '';
```

Immediately after that line, add:

```javascript
const FASTAPI_BASE = location.hostname === 'localhost' ? 'http://localhost:8001' : '';
```

- [ ] **Step 2: Update loadSubtitleConfig**

In `app.html`, find `loadSubtitleConfig` around line 3892. The current first line of the function body is:

```javascript
const res = await fetch(`${API_BASE}/api/config`);
```

Replace with:

```javascript
const res = await fetch(`${FASTAPI_BASE}/api/translate/config`);
```

The next block reads `data.config`. Since FastAPI's GET `/api/translate/config` returns the raw config (not wrapped), update the assignment block:

```javascript
const data = await res.json();
if (data.config) {
    subtitleConfig.value = {
        ...subtitleConfig.value,
        ...data.config,
        subtitle_style: {
            ...subtitleConfig.value.subtitle_style,
            ...data.config.subtitle_style
        }
    };
}
```

Replace with:

```javascript
const rawConfig = await res.json();
// FastAPI returns raw config dict (no wrapper)
const cfg = rawConfig.config || rawConfig;
if (cfg && (cfg.subtitle_style || cfg.title_style)) {
    subtitleConfig.value = {
        ...subtitleConfig.value,
        ...cfg,
        subtitle_style: {
            ...subtitleConfig.value.subtitle_style,
            ...(cfg.subtitle_style || {})
        }
    };
}
```

Note: this preserves backward-compat with the old wrapped shape too (in case anything else still wraps it).

The original code also checks `if (data.template_text !== undefined)` and `if (data.videos)`. Since FastAPI's translate config doesn't return those fields, those branches will simply be skipped — no behavior change needed.

- [ ] **Step 3: Update saveSubtitleConfig**

Find `saveSubtitleConfig` around line 3920. The current fetch is:

```javascript
const res = await fetch(`${API_BASE}/api/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        config: subtitleConfig.value,
        template_text: subtitleTemplateText.value
    })
});
```

Replace with:

```javascript
const res = await fetch(`${FASTAPI_BASE}/api/translate/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        config: subtitleConfig.value
    })
});
```

(FastAPI uses PUT with `{config: {...}}` body — see translate_routes.py line 175.)

- [ ] **Step 4: Manually test in browser**

Open AutoReel in browser (whatever URL it normally runs on). Navigate to the subtitle settings panel. Click "重新載入" (reload). Verify subtitle style fields populate correctly. Then change a value and click "儲存設定" (save). Verify `translation_config.json` on disk reflects the change.

- [ ] **Step 5: Commit**

```bash
git add app.html
git commit -m "refactor: 字幕設定面板改用 FastAPI /api/translate/config"
```

---

## Task 10: Frontend — preset dropdown UI + apply handler

**Files:**
- Modify: `app.html`

- [ ] **Step 1: Add the dropdown markup**

In `app.html`, find the subtitle style settings section header at line 1797:

```html
<h3 class="subtitle-section-title">{{ t('subtitle.styleSettings') }}</h3>
```

Immediately after that line, insert this preset selector block:

```html
                                <!-- Preset Selector -->
                                <div class="subtitle-form-group">
                                    <label class="subtitle-form-label">樣式預設集</label>
                                    <div style="display: flex; gap: 8px; align-items: center;">
                                        <select v-model="selectedPresetId" style="flex: 1; padding: 6px 10px; border-radius: 4px;">
                                            <option v-if="presetList.length === 0" value="">尚未建立預設集</option>
                                            <option v-for="p in presetList" :key="p.id" :value="p.id">
                                                {{ p.name }}
                                            </option>
                                        </select>
                                        <button class="btn btn-secondary" @click="applyPreset" :disabled="!selectedPresetId">套用</button>
                                    </div>
                                </div>
```

- [ ] **Step 2: Add reactive state and methods**

Find where Vue reactive refs are defined for the subtitle panel (search for `subtitleConfig` definition — likely a `ref()` call near the top of the Vue setup function).

Add new refs alongside:

```javascript
                const presetList = ref([]);
                const selectedPresetId = ref(localStorage.getItem('autoreel.lastPresetId') || '');
```

Add two new methods near `loadSubtitleConfig`:

```javascript
                const loadPresets = async () => {
                    try {
                        const res = await fetch(`${FASTAPI_BASE}/api/translate/presets`);
                        const data = await res.json();
                        presetList.value = data.presets || [];
                        // If selected id is no longer available, clear it
                        if (selectedPresetId.value && !presetList.value.some(p => p.id === selectedPresetId.value)) {
                            selectedPresetId.value = '';
                        }
                    } catch (e) {
                        console.error('loadPresets failed:', e);
                        presetList.value = [];
                    }
                };

                const applyPreset = async () => {
                    if (!selectedPresetId.value) return;
                    try {
                        const res = await fetch(
                            `${FASTAPI_BASE}/api/translate/presets/${encodeURIComponent(selectedPresetId.value)}/apply`,
                            { method: 'POST' }
                        );
                        const data = await res.json();
                        if (data.success && data.config) {
                            // Refresh local subtitleConfig from response
                            subtitleConfig.value = {
                                ...subtitleConfig.value,
                                ...data.config,
                                subtitle_style: {
                                    ...subtitleConfig.value.subtitle_style,
                                    ...(data.config.subtitle_style || {})
                                }
                            };
                            localStorage.setItem('autoreel.lastPresetId', selectedPresetId.value);
                            showToast(`已套用樣式：${selectedPresetId.value}`, 'success');
                        } else {
                            showToast('套用失敗', 'error');
                        }
                    } catch (e) {
                        console.error('applyPreset failed:', e);
                        showToast(`套用失敗: ${e.message}`, 'error');
                    }
                };
```

- [ ] **Step 3: Wire loadPresets into mount**

Find where `loadSubtitleConfig` is called on view mount. There are two such call sites (around lines 3983 and 4716 based on earlier grep). At each site, add a call to `loadPresets()` immediately after `loadSubtitleConfig()`:

```javascript
                        loadSubtitleConfig();
                        loadPresets();
```

- [ ] **Step 4: Export the new refs and methods from setup()**

Find the return object of the Vue setup function (around line 4830). The current return includes `loadSubtitleConfig, saveSubtitleConfig`. Add the new exports alongside them:

```javascript
                    loadSubtitleConfig,
                    saveSubtitleConfig,
                    presetList,
                    selectedPresetId,
                    loadPresets,
                    applyPreset,
```

- [ ] **Step 5: Manual smoke test**

1. Restart backend (so the auto-seed runs and `default.json` exists).
2. Open AutoReel → navigate to subtitle settings panel.
3. Verify the dropdown shows "預設" (the seeded default).
4. Click "套用" → toast appears, the live editor below should re-populate from the applied preset.
5. Reload the page → dropdown still shows "預設" (localStorage persistence).
6. Manually create a 2nd preset by copying `style_presets/default.json` to `style_presets/test2.json` and changing `id` to `"test2"` and `name` to `"測試 2"`. Reload page → dropdown now shows both.
7. Switch dropdown to test2, click 套用, verify config updates.

- [ ] **Step 6: Commit**

```bash
git add app.html
git commit -m "feat: 字幕樣式預設集 dropdown 與套用按鈕"
```

---

## Task 11: gitignore + final smoke test

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add bag state to gitignore**

Open `.gitignore` and append:

```
# Style preset shuffle bag state (per-machine, not source)
style_presets/.bag_state.json
```

- [ ] **Step 2: End-to-end test (real translation run)**

1. Make sure `style_presets/default.json` exists (auto-seeded).
2. Open AutoReel, drop a short test video into translate_raw.
3. Verify dropdown shows the preset.
4. Click 套用, then click 翻譯 on the test video.
5. Watch the pipeline log for any errors related to preset or shuffle bag.
6. After it completes, open `style_presets/.bag_state.json` and verify it has an entry for `"default"` with cursor > 0.
7. Run the translation again on a 2nd test video → verify the title bg color is **different** from the first run (shuffle bag rotation).
8. Repeat enough times to exhaust the bag and confirm cross-cycle no-repeat (manually inspect the colors used).

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Final commit**

```bash
git add .gitignore
git commit -m "chore: gitignore shuffle bag state file"
```

---

## Done

After all tasks, you should have:
- A `style_presets/default.json` auto-seeded from your existing config.
- A working dropdown in AutoReel's subtitle settings panel.
- Title background colors that rotate without short-term repeats.
- A unified config endpoint (FastAPI), no longer dependent on the legacy `subtitle_position_server.py` for the subtitle settings panel.
- Test coverage for the entire `PresetService`.

To add a new style preset later:
1. In AutoReel, dial in the look you want via the live editor and save.
2. Open `translation_config.json`, copy `subtitle_style` and `title_style` blocks.
3. Create `style_presets/<your_id>.json` with `id`, `name`, `description`, `subtitle_style`, `title_style`.
4. Reload the AutoReel page → new preset appears in dropdown.

(Phase 2 work — out of scope for this plan: a "save current style as preset" button.)
