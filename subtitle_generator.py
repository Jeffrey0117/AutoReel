#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字幕生成模組 - Whisper 語音識別 + 翻譯
支援 openai-whisper 和 faster-whisper 雙引擎
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Generator
import re


@dataclass
class SubtitleEntry:
    """字幕條目"""
    index: int
    start_time: float  # 秒
    end_time: float    # 秒
    text_original: str
    text_translated: str = ""

    @property
    def start_time_us(self) -> int:
        """開始時間（微秒）"""
        return int(self.start_time * 1_000_000)

    @property
    def end_time_us(self) -> int:
        """結束時間（微秒）"""
        return int(self.end_time * 1_000_000)

    @property
    def duration_us(self) -> int:
        """持續時間（微秒）"""
        return self.end_time_us - self.start_time_us


class SubtitleGenerator:
    """
    字幕生成器 - Whisper 語音識別

    支援引擎:
    - openai-whisper: OpenAI 官方 Whisper 實現
    - faster-whisper: CTranslate2 優化版本，速度更快、記憶體更省

    Config 設定範例:
    {
        "whisper": {
            "engine": "faster-whisper",  # 或 "openai-whisper"
            "model": "base",
            "language": "en",
            "device": "auto",
            "compute_type": "float16",   # faster-whisper 專用: float16, int8, int8_float16
            "vad_filter": true,          # faster-whisper 專用: 語音活動偵測過濾
            "vad_parameters": {          # VAD 進階參數
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100
            },
            "max_words_per_segment": 8
        }
    }
    """

    # 支援的引擎列表
    SUPPORTED_ENGINES = ["openai-whisper", "faster-whisper"]

    # 預設 compute_type 對應
    DEFAULT_COMPUTE_TYPES = {
        "cuda": "float16",
        "cpu": "int8"
    }

    def __init__(self, config_path: str = "translation_config.json"):
        self.config = self._load_config(config_path)
        self.whisper_model = None
        self._engine = None  # 實際使用的引擎

    def _load_config(self, config_path: str) -> dict:
        """載入設定檔"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "whisper": {
                "engine": "openai-whisper",  # 預設使用 openai-whisper 保持向後相容
                "model": "base",
                "language": "en",
                "device": "auto",
                "compute_type": "auto",
                "vad_filter": False
            },
            "translation": {"provider": "openai", "target_lang": "zh-TW"}
        }

    @property
    def engine(self) -> str:
        """取得目前使用的引擎"""
        if self._engine:
            return self._engine
        return self.config.get("whisper", {}).get("engine", "openai-whisper")

    def _get_device(self) -> str:
        """取得運算裝置"""
        device = self.config.get("whisper", {}).get("device", "auto")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return device

    def _get_compute_type(self, device: str) -> str:
        """取得 compute_type (faster-whisper 專用)"""
        compute_type = self.config.get("whisper", {}).get("compute_type", "auto")
        if compute_type == "auto":
            return self.DEFAULT_COMPUTE_TYPES.get(device, "int8")
        return compute_type

    def _load_whisper_model(self):
        """延遲載入 Whisper 模型（支援雙引擎）"""
        if self.whisper_model is not None:
            return self.whisper_model

        whisper_config = self.config.get("whisper", {})
        engine = whisper_config.get("engine", "openai-whisper")
        model_name = whisper_config.get("model", "base")
        device = self._get_device()

        # 驗證引擎設定
        if engine not in self.SUPPORTED_ENGINES:
            print(f"[Warning] 不支援的引擎 '{engine}'，改用 openai-whisper")
            engine = "openai-whisper"

        print(f"[...] 載入 Whisper 模型: {model_name} (引擎: {engine})")

        if engine == "faster-whisper":
            self._load_faster_whisper_model(model_name, device, whisper_config)
        else:
            self._load_openai_whisper_model(model_name, device)

        self._engine = engine
        return self.whisper_model

    def _load_openai_whisper_model(self, model_name: str, device: str):
        """載入 OpenAI Whisper 模型"""
        try:
            import whisper

            self.whisper_model = whisper.load_model(model_name, device=device)
            print(f"[OK] OpenAI Whisper 模型載入完成 (device: {device})")

        except ImportError:
            print("[Error] 請先安裝 openai-whisper: pip install openai-whisper")
            raise

    def _load_faster_whisper_model(self, model_name: str, device: str, whisper_config: dict):
        """載入 Faster Whisper 模型"""
        try:
            from faster_whisper import WhisperModel

            compute_type = self._get_compute_type(device)

            # 取得額外參數
            cpu_threads = whisper_config.get("cpu_threads", 0)  # 0 = 自動
            num_workers = whisper_config.get("num_workers", 1)

            print(f"   compute_type: {compute_type}")

            self.whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers
            )
            print(f"[OK] Faster Whisper 模型載入完成 (device: {device}, compute_type: {compute_type})")

        except ImportError:
            print("[Error] 請先安裝 faster-whisper: pip install faster-whisper")
            raise

    def transcribe(self, video_path: str, language: str = None) -> List[SubtitleEntry]:
        """
        使用 Whisper 轉錄影片

        Args:
            video_path: 影片路徑
            language: 語言代碼 (預設從設定檔讀取)

        Returns:
            字幕條目列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到影片: {video_path}")

        print(f"[Audio] 開始語音識別: {os.path.basename(video_path)}")

        # 載入模型（會設定 self._engine）
        self._load_whisper_model()
        lang = language or self.config.get("whisper", {}).get("language", "en")

        # 根據引擎選擇轉錄方法
        if self._engine == "faster-whisper":
            entries = self._transcribe_faster(video_path, lang)
        else:
            entries = self._transcribe_openai(video_path, lang)

        print(f"[OK] 識別完成，共 {len(entries)} 條字幕")
        return entries

    def _transcribe_openai(self, video_path: str, language: str) -> List[SubtitleEntry]:
        """
        使用 OpenAI Whisper 轉錄

        Args:
            video_path: 影片路徑
            language: 語言代碼

        Returns:
            字幕條目列表
        """
        max_words_per_segment = self.config.get("whisper", {}).get("max_words_per_segment", 8)

        print(f"   語言: {language}")
        print(f"   每段最大字數: {max_words_per_segment}")

        # OpenAI Whisper 轉錄
        result = self.whisper_model.transcribe(
            video_path,
            language=language,
            task="transcribe",
            verbose=False,
            word_timestamps=True
        )

        # 轉換為 SubtitleEntry 列表
        return self._process_openai_segments(
            result.get("segments", []),
            max_words_per_segment
        )

    def _process_openai_segments(self, segments: list, max_words_per_segment: int) -> List[SubtitleEntry]:
        """
        處理 OpenAI Whisper 的 segments

        Args:
            segments: Whisper 返回的 segments 列表
            max_words_per_segment: 每段最大字數

        Returns:
            字幕條目列表
        """
        entries = []
        entry_index = 1

        for segment in segments:
            words = segment.get("words", [])

            if not words:
                # 如果沒有 word-level timestamps，使用整個 segment
                entry = SubtitleEntry(
                    index=entry_index,
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text_original=segment["text"].strip()
                )
                entries.append(entry)
                entry_index += 1
            else:
                # 分割成較短的句子
                sub_entries = self._split_words_into_entries(
                    words,
                    entry_index,
                    max_words_per_segment,
                    word_text_key="word",
                    fallback_start=segment["start"],
                    fallback_end=segment["end"]
                )
                entries.extend(sub_entries)
                entry_index += len(sub_entries)

        return entries

    def _transcribe_faster(self, video_path: str, language: str) -> List[SubtitleEntry]:
        """
        使用 Faster Whisper 轉錄

        Args:
            video_path: 影片路徑
            language: 語言代碼

        Returns:
            字幕條目列表
        """
        whisper_config = self.config.get("whisper", {})
        max_words_per_segment = whisper_config.get("max_words_per_segment", 8)

        # VAD 設定
        vad_filter = whisper_config.get("vad_filter", False)
        vad_parameters = whisper_config.get("vad_parameters", None)

        print(f"   語言: {language}")
        print(f"   每段最大字數: {max_words_per_segment}")
        print(f"   VAD 過濾: {'啟用' if vad_filter else '停用'}")

        # 準備轉錄參數
        transcribe_options = {
            "language": language,
            "task": "transcribe",
            "word_timestamps": True,
            "vad_filter": vad_filter,
        }

        # 加入 VAD 參數（如果有設定）
        if vad_filter and vad_parameters:
            transcribe_options["vad_parameters"] = vad_parameters

        # Faster Whisper 轉錄（返回 generator）
        segments_generator, info = self.whisper_model.transcribe(
            video_path,
            **transcribe_options
        )

        print(f"   偵測語言: {info.language} (機率: {info.language_probability:.2%})")

        # 處理 segments generator
        return self._process_faster_segments(
            segments_generator,
            max_words_per_segment
        )

    def _process_faster_segments(self, segments_generator, max_words_per_segment: int) -> List[SubtitleEntry]:
        """
        處理 Faster Whisper 的 segments generator

        Args:
            segments_generator: Faster Whisper 返回的 segments generator
            max_words_per_segment: 每段最大字數

        Returns:
            字幕條目列表
        """
        entries = []
        entry_index = 1

        for segment in segments_generator:
            # faster-whisper segment 有 .words 屬性（可能為 None）
            words = segment.words

            if not words:
                # 如果沒有 word-level timestamps，使用整個 segment
                entry = SubtitleEntry(
                    index=entry_index,
                    start_time=segment.start,
                    end_time=segment.end,
                    text_original=segment.text.strip()
                )
                entries.append(entry)
                entry_index += 1
            else:
                # 轉換 faster-whisper Word 物件為 dict 格式
                words_list = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": getattr(w, "probability", 1.0)
                    }
                    for w in words
                ]

                # 分割成較短的句子
                sub_entries = self._split_words_into_entries(
                    words_list,
                    entry_index,
                    max_words_per_segment,
                    word_text_key="word",
                    fallback_start=segment.start,
                    fallback_end=segment.end
                )
                entries.extend(sub_entries)
                entry_index += len(sub_entries)

        return entries

    def _split_words_into_entries(
        self,
        words: list,
        start_index: int,
        max_words_per_segment: int,
        word_text_key: str = "word",
        fallback_start: float = 0.0,
        fallback_end: float = 0.0
    ) -> List[SubtitleEntry]:
        """
        將 word 列表分割成較短的字幕條目

        Args:
            words: word 列表 (dict 格式)
            start_index: 起始索引
            max_words_per_segment: 每段最大字數
            word_text_key: word dict 中文字的 key
            fallback_start: 備用開始時間
            fallback_end: 備用結束時間

        Returns:
            字幕條目列表
        """
        entries = []
        entry_index = start_index
        current_words = []
        current_start = None

        for word in words:
            if current_start is None:
                current_start = word.get("start", fallback_start)

            word_text = word.get(word_text_key, "").strip()
            current_words.append(word_text)

            # 檢查是否需要分割
            should_split = (
                len(current_words) >= max_words_per_segment or
                word_text.rstrip().endswith(('.', '?', '!', ','))
            )

            if should_split and current_words:
                text = " ".join(current_words).strip()
                if text:
                    entry = SubtitleEntry(
                        index=entry_index,
                        start_time=current_start,
                        end_time=word.get("end", fallback_end),
                        text_original=text
                    )
                    entries.append(entry)
                    entry_index += 1
                current_words = []
                current_start = None

        # 處理剩餘的字
        if current_words:
            text = " ".join(current_words).strip()
            if text:
                entry = SubtitleEntry(
                    index=entry_index,
                    start_time=current_start,
                    end_time=words[-1].get("end", fallback_end) if words else fallback_end,
                    text_original=text
                )
                entries.append(entry)

        return entries

    def translate_entries(self, entries: List[SubtitleEntry],
                          target_lang: str = None) -> List[SubtitleEntry]:
        """
        翻譯字幕條目

        Args:
            entries: 字幕條目列表
            target_lang: 目標語言 (預設從設定檔讀取)

        Returns:
            翻譯後的字幕條目列表
        """
        if not entries:
            return entries

        target = target_lang or self.config.get("translation", {}).get("target_lang", "zh-TW")
        provider = self.config.get("translation", {}).get("provider", "openai")

        print(f"[Web] 開始翻譯 ({provider})...")
        print(f"   目標語言: {target}")
        print(f"   字幕數量: {len(entries)}")

        if provider == "openai":
            entries = self._translate_with_openai(entries, target)
        elif provider == "deepseek":
            entries = self._translate_with_deepseek(entries, target)
        elif provider == "google":
            entries = self._translate_with_google(entries, target)
        else:
            print(f"[Warning]  不支援的翻譯提供者: {provider}，跳過翻譯")

        return entries

    def _translate_with_openai(self, entries: List[SubtitleEntry],
                                target_lang: str) -> List[SubtitleEntry]:
        """使用 OpenAI API 翻譯"""
        try:
            from openai import OpenAI

            api_key_env = self.config.get("translation", {}).get("api_key_env", "OPENAI_API_KEY")
            api_key = os.environ.get(api_key_env)

            if not api_key:
                print(f"[Warning]  未設定 {api_key_env} 環境變數，跳過翻譯")
                return entries

            client = OpenAI(api_key=api_key)

            # 批量翻譯 (每批最多 20 條)
            batch_size = 20
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                texts = [e.text_original for e in batch]

                prompt = f"""將以下英文字幕翻譯成繁體中文（台灣用語）。
保持原有的編號格式，每行一條翻譯。
只輸出翻譯結果，不要加任何解釋。

{chr(10).join([f"{j+1}. {t}" for j, t in enumerate(texts)])}"""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )

                # 解析翻譯結果
                translated_text = response.choices[0].message.content
                lines = translated_text.strip().split('\n')

                for j, line in enumerate(lines):
                    if j < len(batch):
                        # 移除編號前綴
                        clean_text = re.sub(r'^\d+\.\s*', '', line.strip())
                        batch[j].text_translated = clean_text

                print(f"   翻譯進度: {min(i + batch_size, len(entries))}/{len(entries)}")

            print("[OK] 翻譯完成")

        except ImportError:
            print("[Warning]  請先安裝 openai: pip install openai")
        except Exception as e:
            print(f"[Error] 翻譯失敗: {e}")

        return entries

    def _translate_with_deepseek(self, entries: List[SubtitleEntry],
                                  target_lang: str) -> List[SubtitleEntry]:
        """使用 DeepSeek API 翻譯 (兼容 OpenAI SDK) - 並行優化版"""
        try:
            from openai import OpenAI
            from concurrent.futures import ThreadPoolExecutor, as_completed

            translation_config = self.config.get("translation", {})
            # 優先使用 config 中的 api_key，否則從環境變數讀取
            api_key = translation_config.get("api_key") or os.environ.get(translation_config.get("api_key_env", "DEEPSEEK_API_KEY"))
            base_url = translation_config.get("base_url", "https://api.deepseek.com")
            model = translation_config.get("model", "deepseek-chat")

            if not api_key:
                print(f"[Warning]  未設定 API Key，跳過翻譯")
                return entries

            client = OpenAI(api_key=api_key, base_url=base_url)

            # 優化參數
            batch_size = translation_config.get("batch_size", 50)  # 增加到 50
            max_workers = translation_config.get("max_workers", 3)  # 並行 3 個請求

            # 準備所有批次
            batches = []
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                batches.append((i, batch))

            print(f"   批次大小: {batch_size}, 並行數: {max_workers}, 總批次: {len(batches)}")

            def translate_batch(batch_info):
                """翻譯單一批次"""
                batch_idx, batch = batch_info
                texts = [e.text_original for e in batch]

                prompt = f"""翻譯成繁體中文（台灣用語），每行一條，只輸出翻譯：

{chr(10).join([f"{j+1}. {t}" for j, t in enumerate(texts)])}"""

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )

                translated_text = response.choices[0].message.content
                lines = translated_text.strip().split('\n')

                results = []
                for j, line in enumerate(lines):
                    if j < len(batch):
                        clean_text = re.sub(r'^\d+\.\s*', '', line.strip())
                        results.append((batch_idx + j, clean_text))

                return results

            # 並行翻譯
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(translate_batch, b): b for b in batches}

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        for idx, text in results:
                            if idx < len(entries):
                                entries[idx].text_translated = text
                        completed += len(futures[future][1])
                        print(f"   翻譯進度: {completed}/{len(entries)}")
                    except Exception as e:
                        print(f"   [Warning] 批次翻譯失敗: {e}")

            print("[OK] 翻譯完成")

        except ImportError:
            print("[Warning]  請先安裝 openai: pip install openai")
        except Exception as e:
            print(f"[Error] 翻譯失敗: {e}")

        return entries

    def _translate_with_google(self, entries: List[SubtitleEntry],
                                target_lang: str) -> List[SubtitleEntry]:
        """使用 Google Translate 翻譯 (免費但不穩定)"""
        try:
            from googletrans import Translator

            translator = Translator()

            for i, entry in enumerate(entries):
                try:
                    result = translator.translate(entry.text_original, dest='zh-tw')
                    entry.text_translated = result.text

                    if (i + 1) % 10 == 0:
                        print(f"   翻譯進度: {i + 1}/{len(entries)}")

                except Exception as e:
                    print(f"   [Warning]  第 {i + 1} 條翻譯失敗: {e}")
                    entry.text_translated = entry.text_original

            print("[OK] 翻譯完成")

        except ImportError:
            print("[Warning]  請先安裝 googletrans: pip install googletrans==4.0.0rc1")

        return entries

    def export_srt(self, entries: List[SubtitleEntry], output_path: str,
                   use_translated: bool = True):
        """
        輸出 SRT 格式字幕檔

        Args:
            entries: 字幕條目列表
            output_path: 輸出路徑
            use_translated: 是否使用翻譯後的文字
        """
        def format_time(seconds: float) -> str:
            """格式化時間為 SRT 格式"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                text = entry.text_translated if (use_translated and entry.text_translated) else entry.text_original

                f.write(f"{entry.index}\n")
                f.write(f"{format_time(entry.start_time)} --> {format_time(entry.end_time)}\n")
                f.write(f"{text}\n")
                f.write("\n")

        print(f"[File] 字幕已輸出: {output_path}")

    def export_json(self, entries: List[SubtitleEntry], output_path: str):
        """輸出 JSON 格式字幕檔"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = [asdict(e) for e in entries]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[File] JSON 已輸出: {output_path}")

    def load_from_json(self, json_path: str) -> List[SubtitleEntry]:
        """從 JSON 載入字幕"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [SubtitleEntry(**item) for item in data]


def main():
    """測試用主函數"""
    import sys

    if len(sys.argv) < 2:
        print("使用方式: python subtitle_generator.py <video_path>")
        return

    video_path = sys.argv[1]
    generator = SubtitleGenerator()

    # 轉錄
    entries = generator.transcribe(video_path)

    # 翻譯
    entries = generator.translate_entries(entries)

    # 輸出
    video_name = Path(video_path).stem
    generator.export_srt(entries, f"subtitles/{video_name}_zh.srt", use_translated=True)
    generator.export_srt(entries, f"subtitles/{video_name}_en.srt", use_translated=False)
    generator.export_json(entries, f"subtitles/{video_name}.json")


if __name__ == "__main__":
    main()
