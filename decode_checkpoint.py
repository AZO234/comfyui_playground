#!/usr/bin/env python3
"""decode_checkpoint.py - highrender.py がクラッシュしたとき、5_8_highrend_gen に残った
中間 latent (.pt) を別途 decode → png に変換するレスキューツール。

使い方:
    python decode_checkpoint.py 5_8_highrend_gen/<stem>_step240.pt --base fantasticmix_v10
    # または --base 省略で 5_1_XL_base 配下を順に試す

出力: 同じ stem の .png を _8_highrend_gen に書き出す
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from common import (
    HIGHREND_GEN_DIR,
    XL_BASE_DIR,
    build_pipeline,
    pick_device_dtype,
    resolve_tensor_path,
    save_image_no_metadata,
)


def decode_latent(latent_path: Path, base_path: Path, device: str, dtype) -> Image.Image:
    """latent_path (.pt) を読み込んで VAE decode → PIL Image。"""
    latents = torch.load(latent_path, map_location="cpu")
    print(f"latent shape: {tuple(latents.shape)} dtype: {latents.dtype}")

    # SDXL pipe (vae_fp32 / ControlNet 不要、VAE decode のみ使う)
    pipe = build_pipeline(base_path, "sdxl", device, dtype, vram_gb=64.0,
                           scheduler="dpmpp-2m-karras", vae_fp32=True, progress=False)
    vae = pipe.vae
    latents = latents.to(device=device, dtype=torch.float32)
    # SDXL scaling factor: latent → image 変換のスケールは pipe.vae.config.scaling_factor
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


def main() -> None:
    ap = argparse.ArgumentParser(description="latent .pt を decode → png に復元")
    ap.add_argument("latent_path", help="5_8_highrend_gen/<stem>_step{N}.pt")
    ap.add_argument("--base", default=None,
                    help="VAE 用 SDXL base (省略時は 5_1_XL_base 配下の任意 1 つ)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cpu",
                    help="decode 実行 device (既定 cpu — VAE decode は CPU で 1-2 分で済む)")
    args = ap.parse_args()

    latent_path = Path(args.latent_path)
    if not latent_path.exists():
        raise SystemExit(f"latent ファイルが見つかりません: {latent_path}")

    if args.base:
        base_path = resolve_tensor_path(args.base)
        if base_path is None:
            raise SystemExit(f"base が見つかりません: {args.base}")
    else:
        bases = sorted(XL_BASE_DIR.glob("*.safetensors"))
        if not bases:
            raise SystemExit(f"{XL_BASE_DIR} に SDXL base がありません")
        base_path = bases[0]
        print(f"--base 省略、{base_path.name} を使用")

    if args.device == "cpu":
        device, dtype = "cpu", torch.float32
    else:
        device, dtype = pick_device_dtype(64.0)

    img = decode_latent(latent_path, base_path, device, dtype)
    out_path = HIGHREND_GEN_DIR / f"{latent_path.stem}_decoded.png"
    save_image_no_metadata(img, out_path)
    print(f"\n復元 → {out_path}  {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(0)
