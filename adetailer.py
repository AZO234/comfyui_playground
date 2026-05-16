"""adetailer.py - 顔/手を YOLO で検出して個別に img2img で振り替えるモジュール。

generate.py から `run_adetailer(i2i_pipe, image, prompt, negative, ...)` で呼ばれる。
ultralytics と Bingsu/adetailer の YOLO 重みを使う。重みは初回呼び出し時に
huggingface_hub 経由でローカルキャッシュへ DL される。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

# YOLO モデルのキャッシュ。プロセス内で 1 回だけロード
_YOLO_CACHE: dict[str, object] = {}

# 使う検出モデル一覧。Bingsu/adetailer リポジトリの公式アセット
_YOLO_FILES = ("face_yolov8n.pt", "hand_yolov8n.pt")


def _load_yolo_models():
    """YOLO モデルを必要に応じて DL & ロード。返り値は YOLO インスタンスのリスト。"""
    if "loaded" in _YOLO_CACHE:
        return _YOLO_CACHE["loaded"]
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    models = []
    for fname in _YOLO_FILES:
        path = hf_hub_download(repo_id="Bingsu/adetailer", filename=fname)
        models.append(YOLO(path))
    _YOLO_CACHE["loaded"] = models
    return models


def _feather_mask(w: int, h: int, feather: int = 24) -> Image.Image:
    """周囲が透明、内側が不透明な L-mode マスク。境界に feather px のグラデーション。"""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    f = max(1, min(feather, w // 3, h // 3))
    for i in range(f):
        alpha = int(255 * (i + 1) / f)
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=alpha)
    draw.rectangle([f, f, w - 1 - f, h - 1 - f], fill=255)
    return mask


def _detect_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """全モデルで検出して boxes (x1,y1,x2,y2) のリストを返す。"""
    boxes: list[tuple[int, int, int, int]] = []
    for model in _load_yolo_models():
        result = model(image, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for xyxy in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            boxes.append((x1, y1, x2, y2))
    return boxes


def run_adetailer(
    i2i_pipe,
    image: Image.Image,
    prompt: str,
    negative: str,
    *,
    steps: int = 30,
    cfg: float = 7.0,
    strength: float = 0.4,
    generator=None,
    pad_ratio: float = 0.3,
    target_size: int = 512,
    compel=None,
    version: str = "sd15",
) -> Image.Image:
    """画像に対して顔/手検出 → 個別 img2img → 合成 を実行する。
    検出ゼロや YOLO ロード失敗時は元画像をそのまま返す。
    """
    try:
        boxes = _detect_boxes(image)
    except Exception as e:
        print(f"  ADetailer 検出失敗: {e}")
        return image
    if not boxes:
        return image

    W, H = image.size
    out = image.copy()
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(bw, bh) * (1.0 + pad_ratio)
        x1p = max(0, int(cx - size / 2))
        y1p = max(0, int(cy - size / 2))
        x2p = min(W, int(cx + size / 2))
        y2p = min(H, int(cy + size / 2))
        cw, ch = x2p - x1p, y2p - y1p
        if cw < 32 or ch < 32:
            continue
        crop = out.crop((x1p, y1p, x2p, y2p))
        # img2img は 8 の倍数が必要
        tw = max(target_size, cw)
        th = max(target_size, ch)
        tw = (tw // 8) * 8
        th = (th // 8) * 8
        crop_resized = crop.resize((tw, th), Image.LANCZOS)
        try:
            if compel is not None:
                from common import compel_call
                fixed = compel_call(
                    i2i_pipe, compel, prompt, negative, version,
                    image=crop_resized,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                ).images[0]
            else:
                fixed = i2i_pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=crop_resized,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                ).images[0]
        except Exception as e:
            print(f"  ADetailer 修正失敗 ({cw}x{ch}): {e}")
            continue
        fixed_back = fixed.resize((cw, ch), Image.LANCZOS)
        mask = _feather_mask(cw, ch, feather=max(8, min(cw, ch) // 12))
        out.paste(fixed_back, (x1p, y1p), mask)
    return out
