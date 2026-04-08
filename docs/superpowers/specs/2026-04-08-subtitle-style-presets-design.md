# Subtitle Style Presets — Design Spec

**Date:** 2026-04-08
**Status:** Draft (awaiting user approval)
**Author:** Brainstorm session

## 1. Problem Statement

AutoReel currently has a single hardcoded subtitle style baked into `translation_config.json`. Two pain points:

1. **No way to switch between visual styles.** Every video gets the same look. Trying a different style requires hand-editing config values, which is slow and error-prone, and there's no way to "go back" to a previous style without manually re-editing.
2. **Title background colors repeat too quickly.** The current `random.choice(random_bg_colors)` is independent random sampling, so 5 consecutive videos can produce sequences like `橘 藍 橘 紅 藍` — effectively non-random feel.

## 2. Goals

- Let the user save N pre-made subtitle/title style "presets" and pick one per video via a dropdown in the AutoReel UI (`app.html`).
- Replace the title color random selection with a shuffle-bag algorithm that guarantees no repeats within one full cycle of the color list.
- Minimal disruption to existing translate pipeline (`translate_video.py`) and config file structure.

## 3. Non-Goals

- **No in-app preset editor.** Users create new presets by editing JSON files manually (or in Phase 2, by clicking a "save current as preset" button — explicitly out of scope for MVP).
- **No per-track / per-segment style overrides.** A preset applies to the whole video.
- **No split between subtitle and title preset selection.** A preset bundles both `subtitle_style` and `title_style`. (User explicitly chose this for simplicity.)
- **No batch-mode preset rotation.** Each translate run uses whatever preset is currently active.

## 4. Architecture Overview

### File Structure (new files)

```
autoreel/
└── style_presets/                    # NEW directory
    ├── default.json                  # Initial seed copied from translation_config.json
    ├── (user-added presets).json
    └── .bag_state.json               # Shuffle bag state for title colors
```

### Data Flow

```
[User picks preset in dropdown → clicks "Apply"]
       ↓
[Frontend: POST /api/translate/presets/{id}/apply]
       ↓
[preset_service.apply_preset(id):
   1. Read style_presets/{id}.json
   2. Overwrite subtitle_style + title_style in translation_config.json
   3. Return new full config]
       ↓
[Frontend: existing reactive editor updates from response]
       ↓
[User clicks 翻譯 → translate_video.py reads config as before
 (subtitle_style + title_style + new active_preset_id field)]
       ↓
[When picking title bg color, translate_video.py calls
 preset_service.draw_title_color(active_preset_id) instead of
 random.choice(random_bg_colors). This is the only line that
 changes in translate_video.py.]
```

### Why this approach (vs alternatives)

- **Direction 1 (chosen):** Presets are files. "Apply" rewrites `translation_config.json`. Pipeline unchanged.
- **Direction 2 (rejected):** Merge presets at translate time. Adds runtime complexity and a "what's actually running?" mental load.
- **Direction 3 (rejected):** Refactor entire config into preset array. Too invasive for the value delivered.

## 5. Preset File Format

Each `style_presets/{id}.json` is a complete snapshot:

```json
{
  "id": "yellow_bold",
  "name": "黃色粗體",
  "description": "高對比黃字，適合教學類影片",
  "subtitle_style": {
    "font_resource_id": "...",
    "font_path": "...",
    "font_size": 14,
    "text_color": "#ffe759",
    "background_alpha": 0.65,
    "position_y": -0.45,
    "...": "(all subtitle_style fields, full snapshot)"
  },
  "title_style": {
    "font_resource_id": "...",
    "font_size": 5.0,
    "position_y": 0.8,
    "random_bg_colors": ["#ff0000", "#00cccc", "..."],
    "...": "(all title_style fields, full snapshot)"
  }
}
```

### Field Notes

| Field | Notes |
|---|---|
| `id` | Filename (without `.json`). Unique identifier. Used as shuffle bag state key. |
| `name` | Display name shown in dropdown. Free text. |
| `description` | Optional. Shown as tooltip. |
| `subtitle_style` | Full snapshot, same shape as `translation_config.json`'s `subtitle_style`. |
| `title_style` | Full snapshot, same shape as `translation_config.json`'s `title_style`. Each preset can carry its own `random_bg_colors` palette. |

### Design rationale

- **Full snapshot, not partial:** Apply = direct overwrite. No merge logic, no risk of stale fields.
- **`id` decoupled from `name`:** Renaming a preset's display name doesn't break shuffle bag state.
- **Per-preset color palette:** A "cool tones" preset can carry blue/green colors; a "warm" preset can carry red/orange. No global palette needed.

## 6. Backend API

Add to `backend/api/translate_routes.py`:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/translate/presets` | List all presets. Returns `[{id, name, description}]`. |
| `GET` | `/api/translate/presets/{id}` | Return one preset's full content. |
| `POST` | `/api/translate/presets/{id}/apply` | Apply preset → overwrite `subtitle_style` + `title_style` in `translation_config.json`. Returns the new full config. |
| `POST` | `/api/translate/presets` | **Phase 2 (out of MVP scope):** Save current `translation_config.json` style as a new preset. |

New service file `backend/services/preset_service.py`:

```python
class PresetService:
    PRESETS_DIR = Path("style_presets")
    BAG_STATE_FILE = PRESETS_DIR / ".bag_state.json"

    def list_presets() -> list[dict]:
        """Scan style_presets/*.json (excluding hidden files), return [{id, name, description}]."""

    def get_preset(id: str) -> dict:
        """Read and return one preset's full JSON."""

    def apply_preset(id: str) -> dict:
        """Read preset, overwrite subtitle_style + title_style in translation_config.json,
        return the new full config."""

    def draw_title_color(preset_id: str) -> str:
        """Shuffle-bag draw from preset's random_bg_colors. See Section 7."""
```

### Pipeline integration

In `translate_video.py`, the existing line that picks a random title bg color is replaced with a call to `preset_service.draw_title_color(active_preset_id)`. The "active preset id" comes from a new top-level field added to `translation_config.json`:

```json
{
  "whisper": { "..." },
  "translation": { "..." },
  "subtitle_style": { "..." },
  "title_style": { "..." },
  "active_preset_id": "default"
}
```

Top-level (not nested under `output`) because it's a global runtime setting, not an output destination. This field is set by `apply_preset()` whenever the user applies a preset. If absent, fall back to `"default"`.

## 7. Title Color Shuffle Bag

### Algorithm

1. Each preset has its own bag state, keyed by preset id.
2. State file `style_presets/.bag_state.json`:
   ```json
   {
     "yellow_bold": {
       "source_colors": ["#ff0000", "#00ff00", "#0066ff"],
       "shuffled": ["#0066ff", "#ff0000", "#00ff00"],
       "cursor": 1
     }
   }
   ```
3. **Draw flow:**
   - If state for this preset doesn't exist OR `source_colors` differs from preset's current `random_bg_colors` → reshuffle (no previous-last constraint).
   - Else if `cursor >= len(shuffled)` → record `previous_last = shuffled[-1]`, reshuffle, then if `new_shuffled[0] == previous_last` swap it with a random later index. Reset cursor to 0.
   - Return `shuffled[cursor]`, increment cursor, save state.

This guarantees N-color no-repeat **within** a cycle and also no immediate repeat **across** cycle boundaries (when N >= 2).

### Edge Cases

| Case | Behavior |
|---|---|
| Color list has 1 entry | Return that color, no shuffle, no state mutation. |
| Color list edited (added/removed colors) | `source_colors` mismatch → reset bag and reshuffle. |
| Concurrent writes (parallel pipeline) | Read-modify-write inside a single function call; for batch mode use a `threading.Lock` or simple file lock. |
| Bag state file corrupt / missing | Treat as no state, reshuffle. |
| `random_bg_colors` empty | Fall back to a hardcoded default color (e.g. `#000000`) — should never happen in practice. |

### Why store state on disk

The user runs translation in batches that may span multiple invocations (close app, reopen, run more videos). In-memory state would re-shuffle every restart, defeating the purpose.

## 8. Frontend UI Changes

In `app.html`, add a row to the top of the existing 字幕樣式 (subtitle style) section:

```
┌─────────────────────────────────────────────────┐
│  字幕樣式：[預設集 ▼ 黃色粗體      ] [套用]      │
├─────────────────────────────────────────────────┤
│  字體大小  ━━━●━━━━━  14                         │
│  文字顏色  [#ffe759]                             │
│  ...（既有的 live 編輯器）                       │
└─────────────────────────────────────────────────┘
```

### Behavior

- On panel mount, call `GET /api/translate/presets` to populate the dropdown.
- Dropdown selection alone does **not** apply the preset (avoids accidental clobbering of in-progress edits).
- "套用" button calls `POST /apply` and updates the live editor with the response.
- After apply, the user can keep tweaking via the existing live editor (writes go through existing `PUT /api/translate/config`).
- Last-applied preset id is persisted to `localStorage` (`autoreel.lastPresetId`) so it's remembered across page reloads.
- If `style_presets/` is empty, the dropdown shows a placeholder ("尚未建立預設集") and the apply button is disabled.

## 9. Initial Seed Content

When the system is first deployed, ship with:

- `style_presets/default.json` — generated from the user's current `translation_config.json` (`subtitle_style` + `title_style` extracted, `id: "default"`, `name: "預設"`).

**No other presets are pre-shipped.** The user creates additional presets by:
1. Editing `translation_config.json` (or live editor) until satisfied with a new look.
2. Manually copying `subtitle_style` and `title_style` blocks into a new `style_presets/{new_id}.json` file with `id`, `name`, `description` fields.

## 10. Testing Strategy

| Layer | Tests |
|---|---|
| `PresetService` unit tests | list (empty / multiple), get (existing / missing → 404), apply (config gets overwritten correctly), draw_title_color (shuffle exhaustion, list change → reset, cross-cycle no-repeat, single-color edge case). |
| API integration tests | Each route returns expected status + shape; apply round-trip via real config file in tmp dir. |
| Frontend manual smoke | Load page → dropdown populated → apply → live editor reflects new values → translate runs → output uses new style. |

E2E tests for the dropdown are nice-to-have but not blocking MVP.

## 11. Migration & Rollout

1. Create `style_presets/` directory.
2. Auto-seed `default.json` from current `translation_config.json` on `PresetService` init if either (a) `style_presets/` directory is empty, or (b) `default.json` doesn't exist. **Never overwrite an existing `default.json`** — if the user has customized it, leave it alone.
3. Add top-level `active_preset_id` field to `translation_config.json` (default `"default"`).
4. Deploy backend + frontend together. No data migration needed beyond seed.

## 12. Open Questions

None at design time. All decisions confirmed with user during brainstorming session:

- Q1: Style-only changes (not template structure). ✅
- Q2: Per-video manual selection via dropdown. ✅
- Q3: User dials in styles in JianyingPro then copies values. ✅
- Q4: MVP = manual JSON; "save as preset" button is Phase 2. ✅
- Q5: One preset bundles both subtitle_style and title_style. ✅
- Q6: Direction 1 (file-based, apply = overwrite config). ✅

## 13. Future Work (Phase 2+)

- "Save current style as preset" button + `POST /api/translate/presets` endpoint.
- Preset preview thumbnails.
- Per-video preset override at translate time (currently global via active_preset_id).
- Optional: shuffle bag for `subtitle_style.text_color_options` (currently `text_color_random: false`).
