#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字幕轉錄快速測試腳本
測試 SubtitleGenerator 的各種功能

使用方式:
    python test_transcription.py test_video.mp4
    python test_transcription.py test_video.mp4 --engine faster-whisper
    python test_transcription.py test_video.mp4 --vad --no-vad
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

# 導入字幕生成器
from subtitle_generator import SubtitleGenerator, SubtitleEntry


def format_time_srt(seconds: float) -> str:
    """格式化時間為 SRT 格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def print_sample_subtitles(entries: List[SubtitleEntry], count: int = 5):
    """印出範例字幕"""
    print(f"\n--- 範例字幕 (前 {min(count, len(entries))} 條) ---")
    for entry in entries[:count]:
        print(f"\n{entry.index}")
        print(f"{format_time_srt(entry.start_time)} --> {format_time_srt(entry.end_time)}")
        print(f"原文: {entry.text_original}")
        if entry.text_translated:
            print(f"翻譯: {entry.text_translated}")
    print("-" * 40)


def test_engine(
    video_path: str,
    engine: str,
    model: str = "base",
    compute_type: str = "float16",
    vad_filter: bool = False,
    language: str = "en"
) -> Optional[List[SubtitleEntry]]:
    """測試單一引擎"""
    print(f"\n{'='*60}")
    print(f"測試引擎: {engine}")
    print(f"模型: {model}")
    if engine == "faster-whisper":
        print(f"計算類型: {compute_type}")
        print(f"VAD 過濾: {'啟用' if vad_filter else '停用'}")
    print(f"語言: {language}")
    print(f"{'='*60}")

    # 創建臨時設定
    config = {
        "whisper": {
            "engine": engine,
            "model": model,
            "language": language,
            "device": "auto",
            "compute_type": compute_type,
            "vad_filter": vad_filter,
            "max_words_per_segment": 8
        },
        "translation": {
            "provider": "openai",
            "target_lang": "zh-TW"
        }
    }

    # 寫入臨時設定檔
    temp_config_path = "_test_config_temp.json"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    try:
        # 創建生成器並測試
        generator = SubtitleGenerator(temp_config_path)

        start_time = time.time()
        entries = generator.transcribe(video_path, language=language)
        elapsed_time = time.time() - start_time

        print(f"\n[結果]")
        print(f"   轉錄時間: {elapsed_time:.2f}s")
        print(f"   字幕數量: {len(entries)}")

        if entries:
            total_duration = entries[-1].end_time
            print(f"   音訊時長: {total_duration:.2f}s")
            print(f"   RTF: {elapsed_time/total_duration:.3f}")

            # 顯示範例字幕
            print_sample_subtitles(entries)

        return entries

    except ImportError as e:
        print(f"[Error] 缺少模組: {e}")
        return None
    except Exception as e:
        print(f"[Error] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 清理臨時設定檔
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_vad_comparison(
    video_path: str,
    model: str = "base",
    compute_type: str = "float16",
    language: str = "en"
):
    """比較 VAD 開啟/關閉的差異"""
    print(f"\n{'='*60}")
    print("VAD 過濾比較測試 (faster-whisper)")
    print(f"{'='*60}")

    results = {}

    for vad in [False, True]:
        vad_label = "VAD 開啟" if vad else "VAD 關閉"
        print(f"\n>>> {vad_label}")

        entries = test_engine(
            video_path,
            engine="faster-whisper",
            model=model,
            compute_type=compute_type,
            vad_filter=vad,
            language=language
        )

        if entries:
            results[vad_label] = {
                "segment_count": len(entries),
                "duration": entries[-1].end_time if entries else 0
            }

    # 比較結果
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("VAD 比較結果")
        print(f"{'='*60}")
        for label, data in results.items():
            print(f"{label}: {data['segment_count']} 段字幕")


def test_both_engines(
    video_path: str,
    model: str = "base",
    compute_type: str = "float16",
    language: str = "en"
):
    """測試兩種引擎"""
    print(f"\n{'#'*60}")
    print("# 雙引擎比較測試")
    print(f"{'#'*60}")

    results = {}

    # 測試 faster-whisper
    entries_faster = test_engine(
        video_path,
        engine="faster-whisper",
        model=model,
        compute_type=compute_type,
        vad_filter=False,
        language=language
    )
    if entries_faster:
        results["faster-whisper"] = len(entries_faster)

    # 測試 openai-whisper
    entries_openai = test_engine(
        video_path,
        engine="openai-whisper",
        model=model,
        language=language
    )
    if entries_openai:
        results["openai-whisper"] = len(entries_openai)

    # 比較結果
    if results:
        print(f"\n{'='*60}")
        print("雙引擎比較結果")
        print(f"{'='*60}")
        for engine, count in results.items():
            print(f"{engine}: {count} 段字幕")


def test_with_translation(
    video_path: str,
    engine: str = "faster-whisper",
    model: str = "base"
):
    """測試轉錄 + 翻譯"""
    print(f"\n{'='*60}")
    print("轉錄 + 翻譯測試")
    print(f"{'='*60}")

    # 使用主設定檔
    if os.path.exists("translation_config.json"):
        generator = SubtitleGenerator("translation_config.json")
    else:
        print("[Warning] 找不到 translation_config.json，使用預設設定")
        generator = SubtitleGenerator()

    try:
        # 轉錄
        print("\n[Step 1] 語音識別...")
        entries = generator.transcribe(video_path)
        print(f"   識別完成: {len(entries)} 條字幕")

        # 翻譯
        print("\n[Step 2] 翻譯...")
        entries = generator.translate_entries(entries)

        # 顯示結果
        print_sample_subtitles(entries, count=3)

        # 輸出到檔案
        video_name = Path(video_path).stem
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)

        generator.export_srt(entries, str(output_dir / f"{video_name}_test.srt"), use_translated=True)
        generator.export_json(entries, str(output_dir / f"{video_name}_test.json"))

        print(f"\n[輸出檔案]")
        print(f"   {output_dir / f'{video_name}_test.srt'}")
        print(f"   {output_dir / f'{video_name}_test.json'}")

    except Exception as e:
        print(f"[Error] 測試失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="字幕轉錄快速測試",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python test_transcription.py test_video.mp4
    python test_transcription.py test_video.mp4 --engine faster-whisper
    python test_transcription.py test_video.mp4 --compare-engines
    python test_transcription.py test_video.mp4 --vad --no-vad
    python test_transcription.py test_video.mp4 --with-translation
        """
    )

    parser.add_argument("video", help="測試影片路徑")
    parser.add_argument("--engine", "-e", choices=["openai-whisper", "faster-whisper"],
                        default="faster-whisper", help="測試引擎 (預設: faster-whisper)")
    parser.add_argument("--model", "-m", choices=["tiny", "base", "small", "medium", "large"],
                        default="base", help="測試模型 (預設: base)")
    parser.add_argument("--compute-type", "-c", choices=["float16", "float32", "int8", "int8_float16"],
                        default="float16", help="計算類型 (faster-whisper, 預設: float16)")
    parser.add_argument("--language", "-l", default="en",
                        help="語言代碼 (預設: en)")
    parser.add_argument("--vad", action="store_true",
                        help="啟用 VAD 過濾 (faster-whisper)")
    parser.add_argument("--compare-vad", action="store_true",
                        help="比較 VAD 開啟/關閉的差異")
    parser.add_argument("--compare-engines", action="store_true",
                        help="比較兩種引擎")
    parser.add_argument("--with-translation", "-t", action="store_true",
                        help="測試轉錄 + 翻譯")

    args = parser.parse_args()

    # 檢查影片是否存在
    if not os.path.exists(args.video):
        print(f"[Error] 找不到影片: {args.video}")
        sys.exit(1)

    print(f"測試影片: {args.video}")
    print(f"影片大小: {os.path.getsize(args.video) / 1024 / 1024:.2f} MB")

    # 執行測試
    if args.compare_vad:
        test_vad_comparison(
            args.video,
            model=args.model,
            compute_type=args.compute_type,
            language=args.language
        )
    elif args.compare_engines:
        test_both_engines(
            args.video,
            model=args.model,
            compute_type=args.compute_type,
            language=args.language
        )
    elif args.with_translation:
        test_with_translation(
            args.video,
            engine=args.engine,
            model=args.model
        )
    else:
        # 單一引擎測試
        test_engine(
            args.video,
            engine=args.engine,
            model=args.model,
            compute_type=args.compute_type,
            vad_filter=args.vad,
            language=args.language
        )

    print("\n[Done] 測試完成!")


if __name__ == "__main__":
    main()
