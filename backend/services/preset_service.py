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

    def get_preset(self, id: str) -> dict:
        """Return the full JSON content of one preset. Raises PresetNotFoundError if missing."""
        if not id or id.startswith(".") or "/" in id or "\\" in id:
            raise PresetNotFoundError(f"Invalid preset id: {id!r}")

        path = self.presets_dir / f"{id}.json"
        if not path.exists():
            raise PresetNotFoundError(f"Preset not found: {id!r}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

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
