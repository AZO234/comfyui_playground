#!/usr/bin/env python3
"""check_env.py - 学習/生成に必要な環境を一括チェックする。

torch+cuda、cuDNN、GPU、AMP 対応、onnxruntime providers、主要パッケージのバージョンと
簡単な CUDA matmul スモークテストをまとめて表示する。

問題の切り分けに使う。CUDA が動かない時はここで原因がほぼ特定できる。
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
import traceback
from pathlib import Path


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def kv(key: str, value: object) -> None:
    print(f"  {key:24s} {value}")


def main() -> None:
    section("Python / OS")
    kv("python", sys.version.split()[0])
    kv("executable", sys.executable)
    kv("platform", platform.platform())

    # PyTorch
    section("PyTorch")
    try:
        import torch
        kv("torch", torch.__version__)
        kv("cuda_built_with", torch.version.cuda)
        cuda_ok = torch.cuda.is_available()
        kv("cuda_available", cuda_ok)
        if cuda_ok:
            kv("cudnn_version", torch.backends.cudnn.version())
            kv("device_count", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                kv(f"gpu[{i}]", f"{p.name} sm_{p.major}{p.minor} {p.total_memory/1024**3:.1f}GB")
            kv("bf16_supported", torch.cuda.is_bf16_supported())
            try:
                a = torch.randn(512, 512, device="cuda")
                b = torch.randn(512, 512, device="cuda")
                _ = (a @ b).sum().item()
                kv("matmul_smoke_test", "OK")
            except Exception as e:
                kv("matmul_smoke_test", f"FAIL: {e}")
    except Exception as e:
        kv("torch", f"NOT IMPORTABLE: {e}")
        torch = None  # type: ignore

    # onnxruntime - CUDA DLL を torch から借りるパッチを適用してから取得
    section("onnxruntime")
    if torch is not None:
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(torch_lib))
                kv("dll_search_added", str(torch_lib))
            except Exception as e:
                kv("dll_search_added", f"FAIL: {e}")
    try:
        import onnxruntime as ort
        kv("onnxruntime", ort.__version__)
        kv("available_providers", ort.get_available_providers())
    except Exception as e:
        kv("onnxruntime", f"NOT IMPORTABLE: {e}")

    # 主要パッケージ
    section("Packages")
    for name in [
        "diffusers", "transformers", "peft", "accelerate",
        "safetensors", "huggingface_hub", "PIL", "numpy",
        "tomli", "tomli_w", "torchvision",
    ]:
        try:
            mod = importlib.import_module(name)
            kv(name, getattr(mod, "__version__", "?"))
        except Exception as e:
            kv(name, f"FAIL: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
