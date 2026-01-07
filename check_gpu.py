"""
GPU Diagnostic Utility for Video Translation Project

This module provides functions to check GPU availability and capabilities
for PyTorch, CUDA, and CTranslate2. It can auto-detect the best compute_type
based on available hardware.

Usage:
    python check_gpu.py
"""

import sys
from typing import Optional, Dict, Any, Tuple


def check_pytorch_cuda() -> Dict[str, Any]:
    """
    Check PyTorch CUDA availability and version information.

    Returns:
        Dictionary containing:
        - available: bool - Whether CUDA is available
        - pytorch_version: str - PyTorch version
        - cuda_version: str or None - CUDA version if available
        - cudnn_version: str or None - cuDNN version if available
    """
    result = {
        "available": False,
        "pytorch_version": None,
        "cuda_version": None,
        "cudnn_version": None,
    }

    try:
        import torch
        result["pytorch_version"] = torch.__version__
        result["available"] = torch.cuda.is_available()

        if result["available"]:
            result["cuda_version"] = torch.version.cuda
            if torch.backends.cudnn.is_available():
                result["cudnn_version"] = str(torch.backends.cudnn.version())
    except ImportError:
        pass

    return result


def check_gpu_info() -> Dict[str, Any]:
    """
    Get GPU name and VRAM information.

    Returns:
        Dictionary containing:
        - gpu_count: int - Number of GPUs detected
        - gpus: list - List of GPU info dictionaries with name, vram_total, vram_free
    """
    result = {
        "gpu_count": 0,
        "gpus": [],
    }

    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_count"] = torch.cuda.device_count()

            for i in range(result["gpu_count"]):
                props = torch.cuda.get_device_properties(i)

                # Get memory info
                torch.cuda.set_device(i)
                vram_total = props.total_memory / (1024 ** 3)  # Convert to GB
                vram_free = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)

                gpu_info = {
                    "index": i,
                    "name": props.name,
                    "vram_total_gb": round(vram_total, 2),
                    "vram_free_gb": round(vram_free, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
                result["gpus"].append(gpu_info)
    except ImportError:
        pass
    except Exception as e:
        result["error"] = str(e)

    return result


def check_ctranslate2_support() -> Dict[str, Any]:
    """
    Check CTranslate2 availability and CUDA support.

    Returns:
        Dictionary containing:
        - installed: bool - Whether CTranslate2 is installed
        - version: str or None - CTranslate2 version
        - cuda_support: bool - Whether CUDA is supported
        - supported_compute_types: list - Available compute types
    """
    result = {
        "installed": False,
        "version": None,
        "cuda_support": False,
        "supported_compute_types": [],
    }

    try:
        import ctranslate2
        result["installed"] = True
        result["version"] = ctranslate2.__version__

        # Check available compute types
        compute_types = ["default", "auto", "int8", "int8_float32", "int8_float16",
                        "int8_bfloat16", "int16", "float16", "bfloat16", "float32"]

        # Check if CUDA device is available
        try:
            # Try to get CUDA device
            supported = ctranslate2.get_supported_compute_types("cuda")
            result["cuda_support"] = True
            result["supported_compute_types"] = list(supported)
        except Exception:
            # Fall back to CPU compute types
            try:
                supported = ctranslate2.get_supported_compute_types("cpu")
                result["supported_compute_types"] = list(supported)
            except Exception:
                result["supported_compute_types"] = ["float32", "int8"]

    except ImportError:
        pass
    except Exception as e:
        result["error"] = str(e)

    return result


def check_faster_whisper_support() -> Dict[str, Any]:
    """
    Check faster-whisper availability.

    Returns:
        Dictionary containing:
        - installed: bool - Whether faster-whisper is installed
        - version: str or None - faster-whisper version
    """
    result = {
        "installed": False,
        "version": None,
    }

    try:
        import faster_whisper
        result["installed"] = True
        result["version"] = getattr(faster_whisper, "__version__", "unknown")
    except ImportError:
        pass
    except Exception as e:
        result["error"] = str(e)

    return result


def auto_detect_compute_type() -> Tuple[str, str]:
    """
    Auto-detect the best compute_type based on available hardware.

    Returns:
        Tuple of (compute_type, reason)
        - compute_type: Recommended compute type string
        - reason: Explanation for the recommendation
    """
    pytorch_info = check_pytorch_cuda()
    gpu_info = check_gpu_info()
    ct2_info = check_ctranslate2_support()

    # Check if CUDA is available
    if not pytorch_info["available"]:
        return "int8", "No CUDA available - using CPU-optimized int8"

    # Check GPU compute capability
    if gpu_info["gpus"]:
        gpu = gpu_info["gpus"][0]
        compute_cap = gpu.get("compute_capability", "0.0")
        major, minor = map(int, compute_cap.split("."))
        vram_gb = gpu.get("vram_total_gb", 0)

        # Check CTranslate2 supported types
        supported = ct2_info.get("supported_compute_types", [])

        # Compute capability 7.0+ (Volta, Turing, Ampere, Ada Lovelace) supports float16 well
        if major >= 7:
            if "float16" in supported:
                if vram_gb >= 8:
                    return "float16", f"GPU {gpu['name']} (CC {compute_cap}) with {vram_gb}GB VRAM supports efficient float16"
                elif vram_gb >= 4:
                    if "int8_float16" in supported:
                        return "int8_float16", f"GPU {gpu['name']} with {vram_gb}GB VRAM - using int8_float16 for memory efficiency"
                    return "float16", f"GPU {gpu['name']} with {vram_gb}GB VRAM supports float16"

        # Compute capability 6.0+ (Pascal) - float16 with some overhead
        if major >= 6:
            if vram_gb >= 6 and "float16" in supported:
                return "float16", f"GPU {gpu['name']} (CC {compute_cap}) supports float16"
            if "int8_float32" in supported:
                return "int8_float32", f"GPU {gpu['name']} - using int8_float32 for balance of speed and precision"

        # Older GPUs or limited VRAM
        if "int8" in supported:
            return "int8", f"GPU {gpu['name']} - using int8 for memory efficiency"

        return "float32", f"GPU {gpu['name']} - using float32 for compatibility"

    return "float32", "No GPU detected - using float32 on CPU"


def print_diagnostic_report() -> None:
    """
    Print a comprehensive diagnostic report of GPU capabilities.
    """
    print("=" * 60)
    print("GPU DIAGNOSTIC REPORT")
    print("=" * 60)
    print()

    # PyTorch CUDA
    print("[PyTorch CUDA]")
    print("-" * 40)
    pytorch_info = check_pytorch_cuda()
    if pytorch_info["pytorch_version"]:
        print(f"  PyTorch Version:  {pytorch_info['pytorch_version']}")
        print(f"  CUDA Available:   {pytorch_info['available']}")
        if pytorch_info["available"]:
            print(f"  CUDA Version:     {pytorch_info['cuda_version']}")
            print(f"  cuDNN Version:    {pytorch_info['cudnn_version']}")
    else:
        print("  PyTorch not installed")
    print()

    # GPU Info
    print("[GPU Information]")
    print("-" * 40)
    gpu_info = check_gpu_info()
    if gpu_info["gpu_count"] > 0:
        print(f"  GPU Count: {gpu_info['gpu_count']}")
        for gpu in gpu_info["gpus"]:
            print(f"\n  GPU {gpu['index']}: {gpu['name']}")
            print(f"    VRAM Total:         {gpu['vram_total_gb']} GB")
            print(f"    VRAM Free:          {gpu['vram_free_gb']} GB")
            print(f"    Compute Capability: {gpu['compute_capability']}")
            print(f"    Multiprocessors:    {gpu['multi_processor_count']}")
    else:
        print("  No GPU detected")
        if "error" in gpu_info:
            print(f"  Error: {gpu_info['error']}")
    print()

    # CTranslate2
    print("[CTranslate2]")
    print("-" * 40)
    ct2_info = check_ctranslate2_support()
    if ct2_info["installed"]:
        print(f"  Version:        {ct2_info['version']}")
        print(f"  CUDA Support:   {ct2_info['cuda_support']}")
        print(f"  Compute Types:  {', '.join(ct2_info['supported_compute_types'])}")
    else:
        print("  CTranslate2 not installed")
        if "error" in ct2_info:
            print(f"  Error: {ct2_info['error']}")
    print()

    # Faster-Whisper
    print("[Faster-Whisper]")
    print("-" * 40)
    fw_info = check_faster_whisper_support()
    if fw_info["installed"]:
        print(f"  Version: {fw_info['version']}")
    else:
        print("  Faster-Whisper not installed")
        if "error" in fw_info:
            print(f"  Error: {fw_info['error']}")
    print()

    # Recommendation
    print("[Recommendation]")
    print("-" * 40)
    compute_type, reason = auto_detect_compute_type()
    print(f"  Best compute_type: {compute_type}")
    print(f"  Reason: {reason}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
For translation_config.json, recommended whisper settings:

  "whisper": {{
    "engine": "faster-whisper",
    "compute_type": "{compute_type}",
    "device": "{'cuda' if pytorch_info['available'] else 'cpu'}",
    ...
  }}
""")


def get_diagnostic_dict() -> Dict[str, Any]:
    """
    Get all diagnostic information as a dictionary.

    Returns:
        Dictionary containing all diagnostic information.
    """
    compute_type, reason = auto_detect_compute_type()

    return {
        "pytorch": check_pytorch_cuda(),
        "gpu": check_gpu_info(),
        "ctranslate2": check_ctranslate2_support(),
        "faster_whisper": check_faster_whisper_support(),
        "recommendation": {
            "compute_type": compute_type,
            "reason": reason,
        }
    }


if __name__ == "__main__":
    print_diagnostic_report()
