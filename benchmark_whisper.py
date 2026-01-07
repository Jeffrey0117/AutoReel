#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper 效能基準測試
比較 openai-whisper vs faster-whisper 的效能

使用方式:
    python benchmark_whisper.py test_video.mp4
    python benchmark_whisper.py test_video.mp4 --model small --compute-type int8
    python benchmark_whisper.py test_video.mp4 --iterations 5
"""

import os
import sys
import argparse
import time
import gc
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

# 嘗試導入 psutil 用於記憶體測量
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[Warning] psutil 未安裝，無法測量記憶體使用量")
    print("         安裝方式: pip install psutil")


@dataclass
class BenchmarkResult:
    """基準測試結果"""
    engine: str
    model: str
    compute_type: str
    device: str

    # 時間測量 (秒)
    load_time: float
    transcribe_time: float
    total_time: float

    # 記憶體測量 (MB)
    memory_before: float
    memory_after: float
    memory_peak: float
    memory_used: float

    # 轉錄結果
    segment_count: int
    audio_duration: float  # 音訊時長 (秒)

    # 計算指標
    real_time_factor: float  # 處理時間 / 音訊時長

    # VAD 設定
    vad_filter: bool = False

    # 迭代資訊
    iteration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WhisperBenchmark:
    """Whisper 效能基準測試器"""

    SUPPORTED_ENGINES = ["openai-whisper", "faster-whisper"]
    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]
    SUPPORTED_COMPUTE_TYPES = ["float16", "float32", "int8", "int8_float16"]

    def __init__(self, config_path: str = "translation_config.json"):
        self.config = self._load_config(config_path)
        self.results: List[BenchmarkResult] = []

    def _load_config(self, config_path: str) -> dict:
        """載入設定檔"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _get_device(self) -> str:
        """取得運算裝置"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _get_memory_usage(self) -> float:
        """取得當前記憶體使用量 (MB)"""
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        return 0.0

    def _get_gpu_memory_usage(self) -> float:
        """取得 GPU 記憶體使用量 (MB)"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return 0.0

    def benchmark_openai_whisper(
        self,
        video_path: str,
        model_name: str = "base",
        language: str = "en"
    ) -> BenchmarkResult:
        """測試 OpenAI Whisper"""
        device = self._get_device()
        memory_before = self._get_memory_usage()
        memory_peak = memory_before

        print(f"\n[OpenAI Whisper] 開始測試...")
        print(f"   模型: {model_name}")
        print(f"   裝置: {device}")

        # 載入模型
        import whisper

        load_start = time.time()
        model = whisper.load_model(model_name, device=device)
        load_time = time.time() - load_start

        memory_after_load = self._get_memory_usage()
        memory_peak = max(memory_peak, memory_after_load)
        print(f"   載入時間: {load_time:.2f}s")

        # 轉錄
        transcribe_start = time.time()
        result = model.transcribe(
            video_path,
            language=language,
            task="transcribe",
            verbose=False,
            word_timestamps=True
        )
        transcribe_time = time.time() - transcribe_start

        memory_after = self._get_memory_usage()
        memory_peak = max(memory_peak, memory_after)

        # 取得音訊時長
        segments = result.get("segments", [])
        audio_duration = segments[-1]["end"] if segments else 0

        print(f"   轉錄時間: {transcribe_time:.2f}s")
        print(f"   片段數量: {len(segments)}")

        # 清理
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        return BenchmarkResult(
            engine="openai-whisper",
            model=model_name,
            compute_type="float32" if device == "cpu" else "float16",
            device=device,
            load_time=load_time,
            transcribe_time=transcribe_time,
            total_time=load_time + transcribe_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=memory_peak,
            memory_used=memory_peak - memory_before,
            segment_count=len(segments),
            audio_duration=audio_duration,
            real_time_factor=transcribe_time / audio_duration if audio_duration > 0 else 0,
            vad_filter=False
        )

    def benchmark_faster_whisper(
        self,
        video_path: str,
        model_name: str = "base",
        compute_type: str = "float16",
        language: str = "en",
        vad_filter: bool = False
    ) -> BenchmarkResult:
        """測試 Faster Whisper"""
        device = self._get_device()

        # CPU 不支援 float16
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        memory_before = self._get_memory_usage()
        memory_peak = memory_before

        print(f"\n[Faster Whisper] 開始測試...")
        print(f"   模型: {model_name}")
        print(f"   裝置: {device}")
        print(f"   計算類型: {compute_type}")
        print(f"   VAD 過濾: {'啟用' if vad_filter else '停用'}")

        # 載入模型
        from faster_whisper import WhisperModel

        load_start = time.time()
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )
        load_time = time.time() - load_start

        memory_after_load = self._get_memory_usage()
        memory_peak = max(memory_peak, memory_after_load)
        print(f"   載入時間: {load_time:.2f}s")

        # 轉錄
        transcribe_start = time.time()
        segments_generator, info = model.transcribe(
            video_path,
            language=language,
            task="transcribe",
            word_timestamps=True,
            vad_filter=vad_filter
        )

        # 消耗 generator 取得所有片段
        segments = list(segments_generator)
        transcribe_time = time.time() - transcribe_start

        memory_after = self._get_memory_usage()
        memory_peak = max(memory_peak, memory_after)

        # 取得音訊時長
        audio_duration = info.duration if hasattr(info, 'duration') else (
            segments[-1].end if segments else 0
        )

        print(f"   轉錄時間: {transcribe_time:.2f}s")
        print(f"   片段數量: {len(segments)}")
        print(f"   偵測語言: {info.language} ({info.language_probability:.2%})")

        # 清理
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        return BenchmarkResult(
            engine="faster-whisper",
            model=model_name,
            compute_type=compute_type,
            device=device,
            load_time=load_time,
            transcribe_time=transcribe_time,
            total_time=load_time + transcribe_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=memory_peak,
            memory_used=memory_peak - memory_before,
            segment_count=len(segments),
            audio_duration=audio_duration,
            real_time_factor=transcribe_time / audio_duration if audio_duration > 0 else 0,
            vad_filter=vad_filter
        )

    def run_benchmark(
        self,
        video_path: str,
        engines: List[str] = None,
        models: List[str] = None,
        compute_types: List[str] = None,
        iterations: int = 3,
        vad_filter: bool = False,
        language: str = "en"
    ) -> List[BenchmarkResult]:
        """
        執行完整基準測試

        Args:
            video_path: 測試影片路徑
            engines: 要測試的引擎列表
            models: 要測試的模型列表
            compute_types: 要測試的計算類型列表 (faster-whisper)
            iterations: 每個組合的測試次數
            vad_filter: 是否啟用 VAD 過濾
            language: 語言代碼

        Returns:
            測試結果列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到測試影片: {video_path}")

        # 預設值
        if engines is None:
            engines = ["faster-whisper", "openai-whisper"]
        if models is None:
            models = ["tiny", "base", "small"]
        if compute_types is None:
            compute_types = ["float16", "int8"]

        results = []
        total_tests = 0

        # 計算總測試數量
        for engine in engines:
            if engine == "faster-whisper":
                total_tests += len(models) * len(compute_types) * iterations
            else:
                total_tests += len(models) * iterations

        current_test = 0

        print(f"\n{'='*70}")
        print(f"Whisper 效能基準測試")
        print(f"{'='*70}")
        print(f"測試影片: {video_path}")
        print(f"測試引擎: {', '.join(engines)}")
        print(f"測試模型: {', '.join(models)}")
        print(f"計算類型: {', '.join(compute_types)} (faster-whisper)")
        print(f"測試次數: {iterations}")
        print(f"VAD 過濾: {'啟用' if vad_filter else '停用'}")
        print(f"總測試數: {total_tests}")
        print(f"{'='*70}")

        for engine in engines:
            for model in models:
                if engine == "faster-whisper":
                    for compute_type in compute_types:
                        for i in range(iterations):
                            current_test += 1
                            print(f"\n[{current_test}/{total_tests}] {engine} / {model} / {compute_type} (第 {i+1} 次)")

                            try:
                                result = self.benchmark_faster_whisper(
                                    video_path,
                                    model_name=model,
                                    compute_type=compute_type,
                                    language=language,
                                    vad_filter=vad_filter
                                )
                                result.iteration = i + 1
                                results.append(result)
                            except Exception as e:
                                print(f"   [Error] 測試失敗: {e}")

                            # 等待記憶體釋放
                            gc.collect()
                            time.sleep(1)
                else:
                    for i in range(iterations):
                        current_test += 1
                        print(f"\n[{current_test}/{total_tests}] {engine} / {model} (第 {i+1} 次)")

                        try:
                            result = self.benchmark_openai_whisper(
                                video_path,
                                model_name=model,
                                language=language
                            )
                            result.iteration = i + 1
                            results.append(result)
                        except Exception as e:
                            print(f"   [Error] 測試失敗: {e}")

                        # 等待記憶體釋放
                        gc.collect()
                        time.sleep(1)

        self.results = results
        return results

    def calculate_averages(self, results: List[BenchmarkResult] = None) -> Dict[str, BenchmarkResult]:
        """計算每個組合的平均值"""
        if results is None:
            results = self.results

        # 按照 engine + model + compute_type 分組
        groups = {}
        for result in results:
            key = f"{result.engine}/{result.model}/{result.compute_type}"
            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        # 計算平均值
        averages = {}
        for key, group in groups.items():
            n = len(group)
            avg = BenchmarkResult(
                engine=group[0].engine,
                model=group[0].model,
                compute_type=group[0].compute_type,
                device=group[0].device,
                load_time=sum(r.load_time for r in group) / n,
                transcribe_time=sum(r.transcribe_time for r in group) / n,
                total_time=sum(r.total_time for r in group) / n,
                memory_before=sum(r.memory_before for r in group) / n,
                memory_after=sum(r.memory_after for r in group) / n,
                memory_peak=sum(r.memory_peak for r in group) / n,
                memory_used=sum(r.memory_used for r in group) / n,
                segment_count=int(sum(r.segment_count for r in group) / n),
                audio_duration=group[0].audio_duration,
                real_time_factor=sum(r.real_time_factor for r in group) / n,
                vad_filter=group[0].vad_filter,
                iteration=0  # 表示這是平均值
            )
            averages[key] = avg

        return averages

    def print_results_table(self, results: List[BenchmarkResult] = None):
        """輸出結果表格"""
        if results is None:
            results = self.results

        if not results:
            print("\n[Warning] 沒有測試結果")
            return

        # 計算平均值
        averages = self.calculate_averages(results)

        print(f"\n{'='*100}")
        print("基準測試結果摘要 (平均值)")
        print(f"{'='*100}")

        # 表頭
        headers = ["引擎", "模型", "計算類型", "載入時間", "轉錄時間", "總時間", "記憶體", "RTF"]
        widths = [16, 8, 12, 10, 10, 10, 10, 8]

        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        print(header_line)
        print("-" * len(header_line))

        # 按總時間排序
        sorted_keys = sorted(averages.keys(), key=lambda k: averages[k].total_time)

        for key in sorted_keys:
            avg = averages[key]
            row = [
                avg.engine,
                avg.model,
                avg.compute_type,
                f"{avg.load_time:.2f}s",
                f"{avg.transcribe_time:.2f}s",
                f"{avg.total_time:.2f}s",
                f"{avg.memory_used:.0f}MB",
                f"{avg.real_time_factor:.3f}"
            ]
            print(" | ".join(str(r).ljust(w) for r, w in zip(row, widths)))

        print(f"{'='*100}")
        print("RTF = Real-Time Factor (轉錄時間 / 音訊時長，越低越好)")
        print(f"音訊時長: {results[0].audio_duration:.2f}s")

        # 找出最快的配置
        fastest_key = sorted_keys[0]
        fastest = averages[fastest_key]
        print(f"\n最快配置: {fastest.engine} / {fastest.model} / {fastest.compute_type}")
        print(f"   總時間: {fastest.total_time:.2f}s")
        print(f"   RTF: {fastest.real_time_factor:.3f}")

    def export_results(self, output_path: str, results: List[BenchmarkResult] = None):
        """輸出結果到 JSON 檔案"""
        if results is None:
            results = self.results

        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
            "averages": {k: v.to_dict() for k, v in self.calculate_averages(results).items()}
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n[File] 結果已輸出: {output_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="Whisper 效能基準測試",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python benchmark_whisper.py test_video.mp4
    python benchmark_whisper.py test_video.mp4 --model small --compute-type int8
    python benchmark_whisper.py test_video.mp4 --engine faster-whisper --iterations 5
    python benchmark_whisper.py test_video.mp4 --all-models --all-compute-types
        """
    )

    parser.add_argument("video", help="測試影片路徑")
    parser.add_argument("--engine", "-e", choices=["openai-whisper", "faster-whisper", "both"],
                        default="both", help="測試引擎 (預設: both)")
    parser.add_argument("--model", "-m", choices=["tiny", "base", "small", "medium", "large"],
                        default="base", help="測試模型 (預設: base)")
    parser.add_argument("--all-models", action="store_true",
                        help="測試所有模型 (tiny, base, small)")
    parser.add_argument("--compute-type", "-c", choices=["float16", "float32", "int8", "int8_float16"],
                        default="float16", help="計算類型 (faster-whisper, 預設: float16)")
    parser.add_argument("--all-compute-types", action="store_true",
                        help="測試所有計算類型 (float16, int8)")
    parser.add_argument("--iterations", "-i", type=int, default=3,
                        help="每個組合的測試次數 (預設: 3)")
    parser.add_argument("--vad", action="store_true",
                        help="啟用 VAD 過濾 (faster-whisper)")
    parser.add_argument("--language", "-l", default="en",
                        help="語言代碼 (預設: en)")
    parser.add_argument("--output", "-o", help="輸出 JSON 檔案路徑")

    args = parser.parse_args()

    # 確定測試配置
    if args.engine == "both":
        engines = ["faster-whisper", "openai-whisper"]
    else:
        engines = [args.engine]

    if args.all_models:
        models = ["tiny", "base", "small"]
    else:
        models = [args.model]

    if args.all_compute_types:
        compute_types = ["float16", "int8"]
    else:
        compute_types = [args.compute_type]

    # 執行基準測試
    benchmark = WhisperBenchmark()

    try:
        results = benchmark.run_benchmark(
            video_path=args.video,
            engines=engines,
            models=models,
            compute_types=compute_types,
            iterations=args.iterations,
            vad_filter=args.vad,
            language=args.language
        )

        # 輸出結果
        benchmark.print_results_table()

        # 輸出 JSON
        if args.output:
            benchmark.export_results(args.output)
        else:
            # 預設輸出到 benchmark_results.json
            video_name = Path(args.video).stem
            output_path = f"benchmark_results_{video_name}.json"
            benchmark.export_results(output_path)

    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except ImportError as e:
        print(f"[Error] 缺少必要模組: {e}")
        print("請安裝必要的套件:")
        print("  pip install openai-whisper")
        print("  pip install faster-whisper")
        print("  pip install psutil")
        sys.exit(1)


if __name__ == "__main__":
    main()
