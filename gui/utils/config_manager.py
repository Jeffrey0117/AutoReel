# -*- coding: utf-8 -*-
"""
配置管理器 - 讀寫 translation_config.json
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """管理 translation_config.json"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # 預設路徑為專案根目錄
            self.config_path = Path(__file__).parent.parent.parent / "translation_config.json"
        else:
            self.config_path = Path(config_path)

        self._config: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        """載入配置"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            self._config = self.get_defaults()
        return self._config

    def save(self):
        """儲存配置"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)

    def get(self, *keys, default=None) -> Any:
        """取得配置值 (支援巢狀 key)

        Example:
            config.get("whisper", "model")  # 取得 config["whisper"]["model"]
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys_and_value):
        """設定配置值 (支援巢狀 key)

        Example:
            config.set("whisper", "model", "base")  # 設定 config["whisper"]["model"] = "base"
        """
        if len(keys_and_value) < 2:
            raise ValueError("At least one key and one value required")

        *keys, value = keys_and_value

        # 確保路徑存在
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    @property
    def config(self) -> Dict[str, Any]:
        """取得完整配置"""
        return self._config

    @staticmethod
    def get_defaults() -> Dict[str, Any]:
        """取得預設配置"""
        return {
            "whisper": {
                "model": "base",
                "language": "en",
                "device": "auto",
                "max_words_per_segment": 8,
                "engine": "faster-whisper",
                "compute_type": "int8",
                "vad_filter": True
            },
            "translation": {
                "provider": "deepseek",
                "source_lang": "en",
                "target_lang": "zh-TW"
            },
            "input": {
                "videos_folder": "videos/translate_raw"
            },
            "output": {
                "template_name": "翻譯專案",
                "output_prefix": "翻譯專案_",
                "subtitle_folder": "subtitles"
            },
            "parallel": {
                "enabled": True,
                "mode": "pipeline",
                "translate_workers": 4,
                "draft_workers": 2
            }
        }
