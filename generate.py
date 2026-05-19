#!/usr/bin/env python3
"""generate.py - ComfyUI HTTP API 経由で SDXL 画像を連続生成する CLI。

実行モデル:
    [Python (この generate.py)] ─── HTTP POST /prompt ──▶ [常駐 ComfyUI server]
                                  ◀── GET /history/{id} ──
                                  ◀── GET /view?filename ──
    各 source ループで:
        prompt.toml → build_prompt → checkpoint 抽選 → workflow JSON 組立 →
        ComfyUI に投入 → 完成画像を fetch → A1111 メタ付き PNG で 3_1_generated に保存

前提:
    `python ComfyUI/main.py --listen 127.0.0.1 --port 8188` で ComfyUI が常駐している
    (`--listen 0.0.0.0` でも可、外部 LAN 公開する場合)

Phase 1 実装範囲:
    - `--prompt auto` のみ (prompt.toml 駆動)
    - 単純 SDXL txt2img (LoRA / ControlNet / ADetailer / upscale なし)
    - checkpoint は 2_1_checkpoint からランダム
    - Ctrl+C でループ停止
    - A1111 互換メタを 3_1_generated/{YYYYMMDDHHMMSS}.png に書き込み
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
import tomli_w

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# プロンプト組立 + LoRA 抽選は既存の common 関数を流用
from common import (
    build_prompt,
    load_prompt_config,
    build_lora_corpus,
    pick_n_loras_by_keywords,
    current_gpu_temp,
)
# A1111 メタ書き込みは pngutil の serializer を流用
from pngutil import serialize_a1111_parameters, write_text_chunks
# tensors triage は起動時に必ず実行
from tensors import check_tensors

# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #
ROOT             = Path(__file__).parent
CHECKPOINT_DIR   = ROOT / "2_1_checkpoint"
LORA_DIR         = ROOT / "2_2_LoRA"
EMBEDDING_DIR    = ROOT / "2_3_Embedding"
CONTROLNET_DIR   = ROOT / "2_4_ControlNet"
PROMPTS_DIR      = ROOT / "1_prompts"
GENERATED_DIR    = ROOT / "3_1_generated"
UPSCALED_DIR     = ROOT / "3_2_upscaled"
CHECKPOINT_TOML  = ROOT / "checkpoint.toml"
LORA_KEYWORDS_TOML = ROOT / "LoRA_keywords.toml"

# checkpoint.toml の `style` → 使う Real-ESRGAN モデル
_UPSCALE_MODEL_BY_STYLE = {
    "anime": "RealESRGAN_x4plus_anime_6B.pth",
    "real":  "RealESRGAN_x4plus.pth",
}
_UPSCALE_MODEL_DEFAULT = "RealESRGAN_x4plus_anime_6B.pth"  # mix / empty / unknown

# ControlNet ファイル名 stem → 前処理 mode → ComfyUI preprocessor node クラス名 + 追加 input
# comfyui_controlnet_aux の preprocessor は class ごとに必須 input が違うので、汎用引数を整える。
_MODE_TO_PREPROCESSOR = {
    "canny":      ("CannyEdgePreprocessor", {"low_threshold": 100, "high_threshold": 200}),
    "depth":      ("DepthAnythingV2Preprocessor", {"ckpt_name": "depth_anything_v2_vitl.pth"}),
    "softedge":   ("HEDPreprocessor", {"safe": "enable"}),
    "openpose":   ("DWPreprocessor", {"detect_hand": "enable", "detect_body": "enable",
                                       "detect_face": "enable",
                                       "bbox_detector": "yolox_l.onnx",
                                       "pose_estimator": "dw-ll_ucoco_384.onnx"}),
    "lineart":    ("AnimeLineArtPreprocessor", {}),
    "passthrough": (None, {}),
}

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_WS   = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"


# --------------------------------------------------------------------------- #
# ComfyUI HTTP client (最小実装)
# --------------------------------------------------------------------------- #
def _http_post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def submit_prompt(workflow: dict, client_id: str) -> str:
    """workflow を ComfyUI に投入して prompt_id を返す。"""
    resp = _http_post_json(f"{COMFY_BASE}/prompt",
                            {"prompt": workflow, "client_id": client_id})
    return resp["prompt_id"]


def wait_for_history(prompt_id: str, poll_interval: float = 1.0,
                     timeout: float = 1800.0) -> dict:
    """history が出るまで poll、出たら結果 dict を返す。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            hist = _http_get_json(f"{COMFY_BASE}/history/{prompt_id}")
        except Exception as e:
            print(f"  [history poll error] {e}", flush=True)
            time.sleep(poll_interval)
            continue
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(poll_interval)
    raise TimeoutError(f"ComfyUI history wait timeout: {prompt_id}")


def wait_for_completion_ws(prompt_id: str, client_id: str,
                            timeout: float = 1800.0) -> dict:
    """ComfyUI WebSocket に接続して per-step progress を tqdm 表示、完了したら history を返す。

    監視する event:
      - `progress`: 各 KSampler step (value / max) → tqdm 更新
      - `executing` with `node=null` and matching prompt_id → 完了サイン
    完了後 /history/{prompt_id} を取得して返す。
    WebSocket 失敗時は wait_for_history にフォールバック。
    """
    try:
        import websocket  # websocket-client
        from tqdm import tqdm
    except ImportError:
        return wait_for_history(prompt_id, timeout=timeout)

    deadline = time.time() + timeout
    try:
        ws = websocket.create_connection(f"{COMFY_WS}?clientId={client_id}", timeout=30)
    except Exception as e:
        print(f"  [ws error] {e}、HTTP poll にフォールバック", flush=True)
        return wait_for_history(prompt_id, timeout=timeout)

    pbar: Optional[object] = None
    current_node: Optional[str] = None
    try:
        while time.time() < deadline:
            ws.settimeout(min(30.0, deadline - time.time()))
            try:
                raw = ws.recv()
            except Exception:
                continue
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            data = msg.get("data") or {}
            if mtype == "progress":
                # data: {value, max, prompt_id, node}
                if data.get("prompt_id") != prompt_id:
                    continue
                value = int(data.get("value", 0))
                maxv  = int(data.get("max", 0))
                node  = str(data.get("node", "?"))
                if node != current_node:
                    if pbar is not None:
                        pbar.close()
                    current_node = node
                    pbar = tqdm(total=maxv, desc=f"  node{node}", ncols=80,
                                bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}]")
                if pbar is not None:
                    pbar.n = value
                    pbar.refresh()
            elif mtype == "executed":
                # ノード完了 (例: VAEDecode が終わった等)
                if data.get("prompt_id") == prompt_id and pbar is not None:
                    pbar.close()
                    pbar = None
                    current_node = None
            elif mtype == "execution_success":
                if data.get("prompt_id") == prompt_id:
                    break
            elif mtype == "execution_error":
                if data.get("prompt_id") == prompt_id:
                    print(f"  [ComfyUI error] {data}", flush=True)
                    break
    finally:
        if pbar is not None:
            pbar.close()
        try:
            ws.close()
        except Exception:
            pass

    # 完了後の history 取得 (最終結果)
    try:
        hist = _http_get_json(f"{COMFY_BASE}/history/{prompt_id}")
        if prompt_id in hist:
            return hist[prompt_id]
    except Exception:
        pass
    return wait_for_history(prompt_id, timeout=max(10.0, deadline - time.time()))


def fetch_image(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """ComfyUI 出力 dir から画像 bytes を取得。"""
    params = urllib.parse.urlencode({
        "filename": filename, "subfolder": subfolder, "type": folder_type
    })
    return _http_get_bytes(f"{COMFY_BASE}/view?{params}")


# --------------------------------------------------------------------------- #
# ComfyUI server 制御 (--arch 切替で自動 restart)
# --------------------------------------------------------------------------- #
COMFYUI_DIR = ROOT / "ComfyUI"

def get_comfyui_device() -> Optional[str]:
    """現 server の device 種別 ('cuda' / 'cpu') を返す。接続不能なら None。"""
    try:
        info = _http_get_json(f"{COMFY_BASE}/system_stats")
    except Exception:
        return None
    devices = info.get("devices") or []
    if not devices:
        return None
    return str(devices[0].get("type") or "").lower() or None


def _find_comfyui_processes() -> list:
    """ComfyUI main.py --listen を実行中の Python プロセスを返す。"""
    import psutil
    result = []
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(str(a) for a in cmdline)
            if "main.py" in joined and "--listen" in joined and "ComfyUI" in joined:
                result.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def kill_comfyui_server(timeout: float = 30.0) -> None:
    """ComfyUI server を kill し、ポート 8188 が空くまで待つ。"""
    import psutil
    procs = _find_comfyui_processes()
    if not procs:
        return
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # 終了待ち (graceful → kill -9)
    gone, alive = psutil.wait_procs(procs, timeout=10)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # ポート空き待ち
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _http_get_json_safe(f"{COMFY_BASE}/system_stats") is None:
            return
        time.sleep(1)


def _http_get_json_safe(url: str) -> Optional[dict]:
    try:
        return _http_get_json(url)
    except Exception:
        return None


def start_comfyui_server(arch: str, ready_timeout: float = 120.0) -> None:
    """ComfyUI server を `arch` で起動し、ready になるまで待つ (`arch` ∈ {'cuda', 'cpu'})。"""
    import subprocess
    flags = ["--listen", COMFY_HOST, "--port", str(COMFY_PORT)]
    if arch == "cpu":
        flags.append("--cpu")
    # ComfyUI dir で main.py を起動。.venv の python を使う。
    cmd = [sys.executable, "main.py", *flags]
    # stdout/stderr は親に向けない (subprocess.DEVNULL でログを切る)
    subprocess.Popen(
        cmd, cwd=str(COMFYUI_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )
    # ready 待ち
    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if get_comfyui_device() is not None:
            return
        time.sleep(2)
    raise SystemExit(f"ComfyUI server ({arch}) の起動 timeout ({ready_timeout}s)")


def ensure_comfyui_arch(arch: str) -> None:
    """現 server の device と `arch` を比較。mismatch なら kill + restart。
    `arch` ∈ {'cuda', 'cpu'}。
    """
    cur = get_comfyui_device()
    if cur is None:
        # サーバ停止中 → そのまま起動
        print(f"  ComfyUI 未起動 → {arch} で新規起動", flush=True)
        start_comfyui_server(arch)
        return
    if cur == arch:
        # 一致 → 何もしない
        return
    # Mismatch → kill して起動
    print(f"  ComfyUI device mismatch (現 {cur}, 要 {arch})、再起動中...", flush=True)
    kill_comfyui_server()
    start_comfyui_server(arch)
    print(f"  ComfyUI server を {arch} で再起動完了", flush=True)


# --------------------------------------------------------------------------- #
# Workflow JSON 組立
# --------------------------------------------------------------------------- #
def build_workflow_txt2img(
    *,
    checkpoint: str,
    positive: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "karras",
    filename_prefix: str = "playground",
    loras: Optional[list[tuple[str, float]]] = None,
    controlnet_name: Optional[str] = None,
    controlnet_mode: str = "passthrough",
    controlnet_image: Optional[str] = None,
    controlnet_strength: float = 0.7,
    upscale_model: Optional[str] = None,
    adetailer: bool = False,
    adetailer_face_model: str = "bbox/face_yolov8n.pt",
    adetailer_hand_model: Optional[str] = "bbox/hand_yolov8n.pt",
    adetailer_person_model: Optional[str] = "segm/person_yolov8s-seg.pt",
    adetailer_denoise: float = 0.5,
    adetailer_person_denoise: float = 0.3,
    adetailer_steps: int = 30,
) -> dict:
    """SDXL txt2img の workflow JSON を組み立てる。

    LoRA stacking 対応: loras = [(lora_name.safetensors, strength), ...] を渡すと
    CheckpointLoaderSimple と CLIPTextEncode/KSampler の間に LoraLoader をチェーン挿入する。
    """
    workflow: dict = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
    }
    # 初期 model/clip ハンドル = CheckpointLoaderSimple の MODEL (out 0) / CLIP (out 1)
    model_ref = ["1", 0]
    clip_ref  = ["1", 1]

    # LoRA stacking: 順次 LoraLoader をチェーン
    for i, (lora_name, strength) in enumerate(loras or []):
        node_id = str(100 + i)  # 100, 101, 102, ... (既存ノード 1-7 と衝突しない)
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora_name,
                "strength_model": float(strength),
                "strength_clip":  float(strength),
                "model": model_ref,
                "clip":  clip_ref,
            },
        }
        model_ref = [node_id, 0]
        clip_ref  = [node_id, 1]

    # CLIP encoders (positive / negative)、最終 clip_ref を使う
    workflow["2"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": positive, "clip": clip_ref},
    }
    workflow["3"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": clip_ref},
    }
    workflow["4"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    # ControlNet が指定されてれば、CLIPTextEncode の conditioning を ControlNetApply で wrap
    ksampler_positive_ref = ["2", 0]
    ksampler_negative_ref = ["3", 0]
    if controlnet_name and controlnet_image:
        # (8) LoadImage: アップロードしたソース PNG を取得
        workflow["8"] = {
            "class_type": "LoadImage",
            "inputs": {"image": controlnet_image},
        }
        # (9) Preprocessor (passthrough は skip)
        prep_cls, prep_extra = _MODE_TO_PREPROCESSOR.get(controlnet_mode, (None, {}))
        if prep_cls:
            prep_inputs = {"image": ["8", 0], "resolution": max(width, height)}
            prep_inputs.update(prep_extra)
            workflow["9"] = {
                "class_type": prep_cls,
                "inputs": prep_inputs,
            }
            ctrl_image_ref = ["9", 0]
        else:
            ctrl_image_ref = ["8", 0]  # passthrough: 元画像をそのまま使う
        # (10) ControlNetLoader
        workflow["10"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_name},
        }
        # (11) ControlNetApplyAdvanced: 両 conditioning を wrap
        workflow["11"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "control_net": ["10", 0],
                "image": ctrl_image_ref,
                "strength": float(controlnet_strength),
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        }
        ksampler_positive_ref = ["11", 0]
        ksampler_negative_ref = ["11", 1]

    workflow["5"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "model": model_ref,  # 最終 LoraLoader (なければ CheckpointLoader) の model
            "positive": ksampler_positive_ref,
            "negative": ksampler_negative_ref,
            "latent_image": ["4", 0],
        },
    }
    workflow["6"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    }
    # ADetailer chain (FaceDetailer for face / optional hand / optional person)
    final_image_ref = ["6", 0]
    if adetailer:
        def _facedetailer_inputs(image_ref, bbox_ref, det_seed,
                                  denoise=adetailer_denoise,
                                  guide_size=512.0, max_size=1024.0):
            """FaceDetailer node の inputs を返す。"""
            return {
                "image": image_ref,
                "model": model_ref,
                "clip": clip_ref,
                "vae": ["1", 2],
                "guide_size": float(guide_size),
                "guide_size_for": True,
                "max_size": float(max_size),
                "seed": det_seed,
                "steps": int(adetailer_steps),
                "cfg": float(cfg),
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": ksampler_positive_ref,
                "negative": ksampler_negative_ref,
                "denoise": float(denoise),
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "bbox_detector": bbox_ref,
                "wildcard": "",
                "cycle": 1,
            }

        # (20) face detector
        workflow["20"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": adetailer_face_model},
        }
        # (21) FaceDetailer 顔
        workflow["21"] = {
            "class_type": "FaceDetailer",
            "inputs": _facedetailer_inputs(final_image_ref, ["20", 0], (seed + 1) & 0xFFFFFFFF),
        }
        final_image_ref = ["21", 0]

        # (22)(23) Hand detector + FaceDetailer (使い回し可能、別 detector で実行)
        if adetailer_hand_model:
            workflow["22"] = {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": adetailer_hand_model},
            }
            workflow["23"] = {
                "class_type": "FaceDetailer",
                "inputs": _facedetailer_inputs(final_image_ref, ["22", 0], (seed + 2) & 0xFFFFFFFF),
            }
            final_image_ref = ["23", 0]

        # (24)(25) Person detector + FaceDetailer (全身 inpainting、足/脚の奇形対策)
        # 全身 region なので denoise を低め (構造維持) + guide_size を 1024 で詳細リトーチ
        if adetailer_person_model:
            workflow["24"] = {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": adetailer_person_model},
            }
            workflow["25"] = {
                "class_type": "FaceDetailer",
                "inputs": _facedetailer_inputs(
                    final_image_ref, ["24", 0], (seed + 3) & 0xFFFFFFFF,
                    denoise=adetailer_person_denoise,
                    guide_size=1024.0, max_size=2048.0,
                ),
            }
            final_image_ref = ["25", 0]

    workflow["7"] = {
        "class_type": "SaveImage",
        "inputs": {"images": final_image_ref, "filename_prefix": filename_prefix},
    }

    # アップスケール chain (upscale_model 指定時のみ)
    if upscale_model:
        # (12) UpscaleModelLoader: Real-ESRGAN モデルをロード
        workflow["12"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale_model},
        }
        # (13) ImageUpscaleWithModel: ADetailer 後の image を 4x upscale
        workflow["13"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["12", 0], "image": final_image_ref},
        }
        # (14) 2 つめの SaveImage: アップスケール後
        workflow["14"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["13", 0], "filename_prefix": f"{filename_prefix}_up"},
        }
    return workflow


# --------------------------------------------------------------------------- #
# checkpoint.toml 管理
# --------------------------------------------------------------------------- #
def load_checkpoint_toml() -> dict:
    """checkpoint.toml をロード。無ければ空 dict。"""
    if not CHECKPOINT_TOML.exists():
        return {}
    try:
        return tomllib.loads(CHECKPOINT_TOML.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[警告] checkpoint.toml パース失敗 ({e})、空として扱う", flush=True)
        return {}


def save_checkpoint_toml(data: dict) -> None:
    """checkpoint.toml に保存。"""
    try:
        CHECKPOINT_TOML.write_text(tomli_w.dumps(data), encoding="utf-8")
    except Exception as e:
        print(f"[警告] checkpoint.toml 保存失敗: {e}", flush=True)


def reload_update_save_checkpoint_toml(name: str, elapsed_s: float, data: dict) -> None:
    """ディスク上の checkpoint.toml を直前に再読込してから timing 更新 → 保存。
    外部エディタで like/inference/style 等を編集中でも、その変更を踏み潰さない。
    in-memory `data` も再読込後の内容で同期させ、後続の pick_checkpoint が最新値を見れるようにする。
    """
    fresh = load_checkpoint_toml()
    update_checkpoint_timing(name, elapsed_s, fresh)
    save_checkpoint_toml(fresh)
    data.clear()
    data.update(fresh)


# --------------------------------------------------------------------------- #
# LoRA_keywords.toml 管理
# --------------------------------------------------------------------------- #
def load_lora_keywords_toml() -> dict:
    """LoRA_keywords.toml をロード。無ければ空 dict。
    形式: {stem: {keyword: "..."}}
    """
    if not LORA_KEYWORDS_TOML.exists():
        return {}
    try:
        return tomllib.loads(LORA_KEYWORDS_TOML.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[警告] LoRA_keywords.toml パース失敗 ({e})、空として扱う", flush=True)
        return {}


def build_lora_corpus_for_playground(loras: list[Path], lora_keywords_data: dict) -> dict[str, str]:
    """common.build_lora_corpus への薄いアダプタ。
    LoRA_keywords.toml の `keyword` フィールドを common 側の `trigger` 相当として渡す。
    """
    adapter = {
        stem: {"trigger": str((entry or {}).get("keyword") or "")}
        for stem, entry in lora_keywords_data.items()
    }
    return build_lora_corpus(loras, adapter)


# --------------------------------------------------------------------------- #
# ControlNet 抽選 + 前処理 mode 推論
# --------------------------------------------------------------------------- #
def infer_controlnet_mode(stem: str) -> str:
    """ControlNet ファイル名 stem から前処理 mode を推定。
    マッチしないものは passthrough (元画像をそのまま流す = Tile/Blur/ColorGrid 系の安全側)。
    """
    s = stem.lower()
    if "canny" in s:
        return "canny"
    if "depth" in s or "midas" in s:
        return "depth"
    if "openpose" in s or s.endswith("pose") or "_pose" in s:
        return "openpose"
    if "mlsd" in s:
        return "softedge"
    if "hed" in s or "softedge" in s or "soft_edge" in s:
        return "softedge"
    if "lineart" in s:
        return "lineart"
    return "passthrough"


def pick_controlnet(style: str, fixed_name: Optional[str] = None,
                     force_openpose: bool = False) -> Optional[Path]:
    """ControlNet を抽選。

    - fixed_name 指定 → そのまま返す
    - force_openpose=True → stem に 'pose'/'openpose' を含むもの から強制抽選
    - style == "anime" → ファイル名 stem に 'anime' を含むもの からランダム
    - style == "real"  → ファイル名 stem に 'real' を含むもの からランダム
    - style == "mix" or "" → 全 ControlNet からランダム
    - 候補ゼロ → None
    """
    if not CONTROLNET_DIR.exists():
        return None
    candidates = sorted(CONTROLNET_DIR.glob("*.safetensors"))
    if not candidates:
        return None
    if fixed_name:
        for c in candidates:
            if c.stem == fixed_name or c.name == fixed_name:
                return c
        raise SystemExit(f"ControlNet が見つかりません: {fixed_name}")

    if force_openpose:
        matched = [c for c in candidates
                   if "openpose" in c.stem.lower() or "_pose" in c.stem.lower()
                   or c.stem.lower().endswith("pose")]
        if not matched:
            raise SystemExit(
                "--pose 指定だが 2_4_ControlNet/ に openpose 系 ControlNet が見つかりません "
                "(stem に 'openpose' / '_pose' / 末尾 'pose' を含むファイルを配置)"
            )
        return random.choice(matched)

    s = (style or "").lower()
    if s == "anime":
        matched = [c for c in candidates if "anime" in c.stem.lower()]
    elif s == "real":
        matched = [c for c in candidates if "real" in c.stem.lower()]
    else:
        matched = candidates
    return random.choice(matched or candidates)


# --------------------------------------------------------------------------- #
# ComfyUI 画像アップロード (ControlNet source 用)
# --------------------------------------------------------------------------- #
def upload_image_to_comfyui(image_path: Path) -> str:
    """ローカル PNG を ComfyUI の input/ にアップロードし、参照名を返す。"""
    import requests  # ComfyUI 自体が依存している
    with open(image_path, "rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        data = {"type": "input", "overwrite": "true"}
        r = requests.post(f"{COMFY_BASE}/upload/image", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()["name"]


# --------------------------------------------------------------------------- #
# PNG メタからプロンプト読出 (--prompt png モード)
# --------------------------------------------------------------------------- #
def parse_png_prompt_metadata(png_path: Path) -> tuple[str, str, list[str]]:
    """PNG の A1111 'parameters' chunk から positive/negative/lora_keywords を取り出す。
    chunk が無い場合は全て空。
    """
    from pngutil import read_text_chunks, parse_a1111_parameters
    chunks = read_text_chunks(png_path)
    if "parameters" not in chunks:
        return "", "", []
    parsed = parse_a1111_parameters(chunks["parameters"])
    positive = parsed.get("positive") or ""
    negative = parsed.get("negative") or ""
    lora_kw_str = (parsed.get("params") or {}).get("Lora keywords", "")
    lora_keywords = [k.strip() for k in lora_kw_str.split(",") if k.strip()]
    return positive, negative, lora_keywords


def parse_png_full_metadata(png_path: Path) -> dict:
    """PNG の A1111 'parameters' chunk から **全フィールド** を構造化 dict で返す。
    `--prompt original` でチェックポイント・LoRA・プロンプトを丸ごと流用するために使う。

    返す dict: {positive, negative, lora_keywords, model, loras, controlnet}
        loras は [(name.safetensors, strength), ...]
    chunk が無いキーは空文字 / 空 list。
    """
    from pngutil import read_text_chunks, parse_a1111_parameters
    out: dict = {
        "positive": "", "negative": "",
        "lora_keywords": [], "model": "", "loras": [], "controlnet": "",
    }
    chunks = read_text_chunks(png_path)
    if "parameters" not in chunks:
        return out
    parsed = parse_a1111_parameters(chunks["parameters"])
    out["positive"] = parsed.get("positive") or ""
    out["negative"] = parsed.get("negative") or ""
    params = parsed.get("params") or {}
    out["model"] = params.get("Model", "")
    out["controlnet"] = params.get("ControlNet", "")
    lora_kw_str = params.get("Lora keywords", "")
    out["lora_keywords"] = [k.strip() for k in lora_kw_str.split(",") if k.strip()]
    # Loras field: "name1: 0.40, name2: 0.27" → [(name, strength)]
    loras_str = params.get("Loras", "")
    loras: list[tuple[str, float]] = []
    for part in loras_str.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, strength_str = part.rsplit(":", 1)
        try:
            loras.append((name.strip(), float(strength_str.strip())))
        except ValueError:
            continue
    out["loras"] = loras
    return out


def resolve_png_path(name: str) -> Path:
    """`--png NAME` を絶対パス / 1_prompts/NAME / 1_prompts/NAME.png の順で解決。"""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for cand in (p, PROMPTS_DIR / name, PROMPTS_DIR / f"{name}.png"):
        if cand.exists():
            return cand
    raise SystemExit(f"PNG が見つかりません: {name} (1_prompts/ を確認)")


def update_checkpoint_timing(name: str, elapsed_s: float, data: dict) -> None:
    """gear high 完走後、checkpoint.toml を in-place 更新。
    既存エントリは fast (最小) / slow (最大) を更新、新規は追記。
    """
    elapsed = int(round(elapsed_s))
    entry = data.get(name)
    if entry is None:
        # 新規追記
        data[name] = {
            "slow": elapsed,
            "fast": elapsed,
            "like": 0,
            "inference": 0,
            "style": "",
        }
        print(f"  checkpoint.toml に {name} を初期登録 (slow=fast={elapsed}s)", flush=True)
    else:
        cur_fast = int(entry.get("fast", elapsed))
        cur_slow = int(entry.get("slow", elapsed))
        new_fast = min(cur_fast, elapsed)
        new_slow = max(cur_slow, elapsed)
        if new_fast != cur_fast or new_slow != cur_slow:
            entry["fast"] = new_fast
            entry["slow"] = new_slow
            print(f"  checkpoint.toml 更新 {name}: fast={new_fast}s slow={new_slow}s", flush=True)
        # like / inference / style はユーザ管理、触らない


# --------------------------------------------------------------------------- #
# 抽選
# --------------------------------------------------------------------------- #
def _resolve_checkpoint_name(name: str) -> Path:
    """`--checkpoint NAME` 解決。stem / .safetensors 付きどちらでも可。"""
    candidates = sorted(CHECKPOINT_DIR.glob("*.safetensors"))
    for c in candidates:
        if c.stem == name or c.name == name:
            return c
    raise SystemExit(f"checkpoint が見つかりません: {name}")


def pick_checkpoint(
    data: dict,
    state: dict,
    fixed_name: Optional[str] = None,
) -> Path:
    """checkpoint を抽選 (`checkpoint.toml` 連動)。

    ルール:
        - `fixed_name` 指定 (= `--checkpoint NAME`) → そのまま返す
        - state['count'] == 0 (1 度め) + 未計測あり → 未計測からランダム
        - 2 度め以降 → 2/3 確率で計測済み (重み付き)、1/3 確率で未計測ランダム
        - 計測済み内の重み: `max(1, (max_slow*2 - (fast + slow)) / 2 + like)`
        - 片方しか無ければそちらに寄せる
    """
    candidates = sorted(CHECKPOINT_DIR.glob("*.safetensors"))
    if not candidates:
        raise SystemExit(f"{CHECKPOINT_DIR} に SDXL checkpoint がありません")
    if fixed_name:
        return _resolve_checkpoint_name(fixed_name)

    scored   = [c for c in candidates if c.stem in data]
    unscored = [c for c in candidates if c.stem not in data]

    first_pick = state.get("count", 0) == 0
    state["count"] = state.get("count", 0) + 1

    # 1 度めは未計測優先
    if first_pick and unscored:
        return random.choice(unscored)

    # 片方しか無い場合はそちらに寄せる
    if not scored:
        return random.choice(unscored)
    if not unscored:
        return _weighted_pick_scored(scored, data)

    # 通常: 2/3 計測済み / 1/3 未計測
    if random.random() < (2.0 / 3.0):
        return _weighted_pick_scored(scored, data)
    return random.choice(unscored)


def _weighted_pick_scored(scored: list[Path], data: dict) -> Path:
    """計測済み checkpoint から重み付き抽選 (`max(1, (max_slow*2-(fast+slow))/2+like)`)。"""
    max_slow = max(int(data[c.stem].get("slow", 0)) for c in scored)
    base = max_slow * 2
    weights: list[int] = []
    for c in scored:
        e = data[c.stem]
        fast = int(e.get("fast", 0))
        slow = int(e.get("slow", 0))
        like = int(e.get("like", 0))
        w = (base - (fast + slow)) // 2 + like
        weights.append(max(1, w))
    return random.choices(scored, weights=weights, k=1)[0]


# --------------------------------------------------------------------------- #
# プロンプト後処理: LoRA キーワードを (0.8/N) 重み付けで positive に append
# --------------------------------------------------------------------------- #
def collect_negative_embeddings() -> list[str]:
    """`2_3_Embedding/` 配下から負のクオリティ embedding を集める。

    判定: stem を lowercase して `neg` / `bad` / `worst` を含むものを採用。
    例: `PonyXL_NegScore-neg.safetensors` / `SmoothNegative_Hands-neg.safetensors` 等。
    """
    if not EMBEDDING_DIR.exists():
        return []
    stems: list[str] = []
    for p in sorted(EMBEDDING_DIR.glob("*.safetensors")):
        s = p.stem.lower()
        if "neg" in s or "bad" in s or "worst" in s:
            stems.append(p.stem)
    return stems


def augment_negative_with_embeddings(negative: str, embed_stems: list[str]) -> str:
    """negative 末尾に `embedding:NAME` を append (ComfyUI/A1111 構文)。"""
    if not embed_stems:
        return negative
    appended = ", ".join(f"embedding:{stem}" for stem in embed_stems)
    return f"{negative}, {appended}" if negative else appended


def augment_positive_with_lora_keywords(positive: str, lora_keywords: list[str],
                                          total_weight: float = 0.8) -> str:
    """LoRA キーワード列を atomic (`,` 区切り) に分解し、各々に `total_weight/N` の重みで
    `(kw:weight), ...` 形式で positive 末尾に連結する。

    例: positive = "a girl, vivid color"
        lora_keywords = ["nude, naked", "jewel"]
        → atoms = ["nude", "naked", "jewel"] (N=3、weight=0.27)
        → "a girl, vivid color, (nude:0.27), (naked:0.27), (jewel:0.27)"
    """
    atoms: list[str] = []
    for entry in (lora_keywords or []):
        for atom in str(entry).split(","):
            atom = atom.strip()
            if atom:
                atoms.append(atom)
    if not atoms:
        return positive
    w = total_weight / len(atoms)
    appended = ", ".join(f"({a}:{w:.2f})" for a in atoms)
    if positive:
        return f"{positive}, {appended}"
    return appended


# --------------------------------------------------------------------------- #
# プロンプト (mode 別ハンドリング)
# --------------------------------------------------------------------------- #
def get_prompt_for_iteration(
    mode: str,
    png_path: Optional[Path] = None,
    sentence: Optional[str] = None,
    lora_keywords_arg: Optional[str] = None,
) -> tuple[str, str, list[str], dict]:
    """指定 mode で 1 source 分の (positive, negative, lora_keywords, extras) を返す。

    extras dict: {model: str, loras: list[(name,strength)]} など、original モードの上書き情報。

    - auto    : prompt.toml から build_prompt
    - sentence: --sentence の文章 + --lora-keywords をそのまま使用、negative は prompt.toml の negative_always
    - png     : PNG の A1111 'parameters' chunk から positive/negative/lora_keywords を読出
    - original: PNG メタ全部 (Model / Loras / positive / negative / lora_keywords) を流用
    """
    from common import normalize_emphasis
    extras: dict = {}

    if mode == "auto":
        cfg = load_prompt_config()
        pos, neg, kws = build_prompt(cfg)
        return pos, neg, kws, extras

    if mode == "sentence":
        if not sentence:
            raise SystemExit("--prompt sentence には --sentence \"...\" が必要")
        cfg = load_prompt_config()
        positive = normalize_emphasis(sentence)
        negative = normalize_emphasis(str(cfg.get("negative_always") or ""))
        kws: list[str] = []
        if lora_keywords_arg:
            kws = [k.strip() for k in lora_keywords_arg.split(",") if k.strip()]
        return positive, negative, kws, extras

    if mode == "png":
        if png_path is None:
            raise SystemExit("--prompt png には --png <PNG> が必要")
        positive, negative, kws = parse_png_prompt_metadata(png_path)
        if not positive:
            print(f"  [info] PNG にメタ情報なし、auto モードにフォールバック", flush=True)
            cfg = load_prompt_config()
            pos, neg, kws = build_prompt(cfg)
            return pos, neg, kws, extras
        return positive, negative, kws, extras

    if mode == "original":
        if png_path is None:
            raise SystemExit("--prompt original には --png <PNG> が必要")
        meta = parse_png_full_metadata(png_path)
        positive = meta["positive"]
        negative = meta["negative"]
        kws      = meta["lora_keywords"]
        if not positive:
            print(f"  [info] PNG にメタ情報なし、auto モードにフォールバック", flush=True)
            cfg = load_prompt_config()
            pos, neg, kws = build_prompt(cfg)
            return pos, neg, kws, extras
        # checkpoint / loras を extras に詰める (main loop で上書き適用)
        if meta["model"]:
            extras["model"] = meta["model"]
        if meta["loras"]:
            extras["loras"] = meta["loras"]
        return positive, negative, kws, extras

    raise SystemExit(f"--prompt {mode} は未対応")


# --------------------------------------------------------------------------- #
# 出力保存 (A1111 メタ付き)
# --------------------------------------------------------------------------- #
def save_with_a1111_metadata(
    image_bytes: bytes,
    out_path: Path,
    *,
    positive: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    width: int,
    height: int,
    checkpoint: str,
    lora_keywords: list[str],
    loras: Optional[list[tuple[str, float]]] = None,
    controlnet_name: Optional[str] = None,
    controlnet_mode: str = "",
    controlnet_strength: float = 0.0,
    pose_source: Optional[str] = None,
    adetailer: bool = False,
    adetailer_person: bool = False,
) -> None:
    """ComfyUI から取得した画像 bytes を A1111 互換メタ付きで PNG 保存する。"""
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_bytes(image_bytes)
    parsed = {
        "positive": positive,
        "negative": negative,
        "params": {
            "Steps":         str(steps),
            "Sampler":       sampler,
            "Schedule type": scheduler,
            "CFG scale":     f"{cfg}",
            "Seed":          str(seed),
            "Size":          f"{width}x{height}",
            "Model":         checkpoint,
        },
    }
    if loras:
        # "Loras" フィールド: A1111 流の "name1: 0.40, name2: 0.40" 列挙
        parsed["params"]["Loras"] = ", ".join(f"{n}: {s:.2f}" for n, s in loras)
    if lora_keywords:
        parsed["params"]["Lora keywords"] = ", ".join(lora_keywords)
    if controlnet_name:
        parsed["params"]["ControlNet"] = f"{controlnet_name} (mode={controlnet_mode}, strength={controlnet_strength:.2f})"
    if pose_source:
        parsed["params"]["Pose source"] = pose_source
    if adetailer:
        parsed["params"]["ADetailer"] = "on (person)" if adetailer_person else "on"
    parameters_text = serialize_a1111_parameters(parsed)
    write_text_chunks(out_path, {"parameters": parameters_text})


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="ComfyUI HTTP API 経由で SDXL 画像を連続生成する (Phase 1: --prompt auto のみ)"
    )
    ap.add_argument("--prompt", choices=["auto", "sentence", "png", "original"], default="auto",
                    help="プロンプト入力モード。"
                         "auto=prompt.toml 駆動 / "
                         "sentence=--sentence で直接 / "
                         "png=PNG メタから読出 / "
                         "original=PNG メタの checkpoint+LoRA+prompt 全部を流用")
    ap.add_argument("--sentence", type=str, default=None,
                    help="--prompt sentence のとき、文章プロンプト。`**word**` 強調記法 OK")
    ap.add_argument("--lora-keywords", type=str, default=None,
                    help="--prompt sentence のとき、LoRA キーワード列 (カンマ区切り)")
    ap.add_argument("--png", type=str, default=None,
                    help="--prompt png/original のときの PNG ファイル名 (1_prompts/ 配下 or 絶対パス)")
    ap.add_argument("--controlnet", type=str, default=None,
                    help="ControlNet を固定 (name or stem)")
    ap.add_argument("--no-controlnet", action="store_true",
                    help="ControlNet を完全 OFF (--prompt png でソース PNG があっても使わない)")
    ap.add_argument("--controlnet-strength", type=float, default=0.7,
                    help="controlnet_conditioning_scale (既定 0.7)")
    ap.add_argument("--pose", type=str, default=None,
                    help="openpose 用 ソース PNG (絶対パス or 1_prompts/NAME)。"
                         "指定すると DWPose 抽出 → openpose ControlNet を強制適用 "
                         "(2_4_ControlNet/ に stem に 'openpose'/'_pose' を含むファイルが必要)。"
                         "--prompt mode とは独立 (sentence/auto/png/original 全モードで併用可)")
    ap.add_argument("--pose-strength", type=float, default=1.0,
                    help="--pose 指定時の controlnet_conditioning_scale (既定 1.0、骨格は強めが効く)")
    ap.add_argument("--gear", choices=["low", "high"], default="high",
                    help="low=ラフ (steps 30) / high=本番 (steps 100、既定)")
    ap.add_argument("--arch", choices=["cuda", "cpu"], default="cuda",
                    help="ComfyUI 側 device 切替 (Phase 1 では参考扱い、ComfyUI 起動時に決まる)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="checkpoint を固定。NAME or NAME.safetensors")
    ap.add_argument("--cfg-scale", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--sampler", type=str, default="dpmpp_2m")
    ap.add_argument("--scheduler", type=str, default="karras")
    ap.add_argument("--lora-scale", type=float, default=0.8,
                    help="LoRA n 個重ね掛け時の合計 scale (各 LoRA strength = lora_scale/n、既定 0.8)")
    ap.add_argument("--lora-stack-min", type=int, default=3,
                    help="1 枚あたりの重ね掛け LoRA 最小数 (既定 3、1 で「下限 1」)")
    ap.add_argument("--lora-stack-max", type=int, default=5,
                    help="1 枚あたりの重ね掛け LoRA 最大数 (random.randint(min, max)、既定 5、"
                         "1 で重ね無し、0 で完全 OFF)")
    ap.add_argument("--upscale", action=argparse.BooleanOptionalAction, default=None,
                    help="Real-ESRGAN x4 アップスケール (3_2_upscaled に出力)。"
                         "既定: gear high で ON / gear low で OFF。明示すれば上書き")
    ap.add_argument("--upscale-model", type=str, default=None,
                    help="アップスケール用 Real-ESRGAN モデル名 (既定: style=anime → anime6B、"
                         "real → x4plus、mix/空 → anime6B)")
    ap.add_argument("--adetailer", action=argparse.BooleanOptionalAction, default=None,
                    help="ADetailer (顔/手 YOLO inpainting)。既定: gear high で ON / low で OFF")
    ap.add_argument("--adetailer-face-model", type=str, default="bbox/face_yolov8n.pt",
                    help="ADetailer 顔検出 model (既定 face_yolov8n)")
    ap.add_argument("--adetailer-hand-model", type=str, default="bbox/hand_yolov8n.pt",
                    help="ADetailer 手検出 model (空文字で hand OFF、既定 hand_yolov8n)")
    ap.add_argument("--adetailer-person-model", type=str, default="segm/person_yolov8s-seg.pt",
                    help="ADetailer 全身検出 model (空文字で person OFF、既定 person_yolov8s-seg)。"
                         "足/脚の奇形補正に使用、denoise を低めで構造維持")
    ap.add_argument("--adetailer-denoise", type=float, default=0.5,
                    help="ADetailer (face/hand) inpaint strength (既定 0.5)")
    ap.add_argument("--adetailer-person-denoise", type=float, default=0.3,
                    help="ADetailer person inpaint strength (既定 0.3、低めで構造維持)")
    ap.add_argument("--adetailer-steps", type=int, default=30,
                    help="ADetailer 各 detected region のステップ数 (既定 30)")
    ap.add_argument("--embeddings", action=argparse.BooleanOptionalAction, default=True,
                    help="2_3_Embedding/ から負のクオリティ embedding (`*-neg` 等) を negative に自動投入 (既定 ON)")
    ap.add_argument("--cooldown", type=float, default=None,
                    help="1 枚生成後の待機秒。既定: GPU 温度 - 50 秒 (温度取れなければ 1.0 秒、--cooldown 0 で OFF)")
    args = ap.parse_args()

    steps = {"low": 30, "high": 50}[args.gear]

    # --sentence / --png 単独指定で --prompt mode を自動推定 (UX 改善)
    # 例: `generate.py --sentence "..."` → --prompt sentence を暗黙適用
    if args.prompt == "auto":
        if args.sentence:
            args.prompt = "sentence"
        elif args.png:
            args.prompt = "png"

    # アップスケール / ADetailer 既定 (gear に紐づき、明示で上書き)
    if args.upscale is None:
        args.upscale = (args.gear == "high")
    if args.adetailer is None:
        args.adetailer = (args.gear == "high")

    print(f"=== generate.py ===")
    print(f"prompt mode: {args.prompt}  gear: {args.gear} (steps={steps})  arch: {args.arch}  "
          f"upscale: {args.upscale}  adetailer: {args.adetailer}")

    print(f"\n--- tensors triage ---")
    counts = check_tensors()
    print(f"  checkpoint={counts['checkpoint']}  LoRA={counts['lora']}  embedding={counts['embedding']}  "
          f"controlnet={counts['controlnet']}  SD15={counts['sd15']}  error={counts['error']}")
    if counts["checkpoint"] == 0:
        raise SystemExit(f"\n{CHECKPOINT_DIR} に SDXL checkpoint がありません。先に 2_tensors に投入を")

    print(f"\n--- ComfyUI 接続確認 / device 整合 ---")
    ensure_comfyui_arch(args.arch)
    cur_device = get_comfyui_device()
    if cur_device is None:
        raise SystemExit(f"ComfyUI に接続できません ({COMFY_BASE})")
    print(f"  OK: {COMFY_BASE} (device={cur_device})")

    client_id = uuid.uuid4().hex
    GENERATED_DIR.mkdir(exist_ok=True)
    if args.upscale:
        UPSCALED_DIR.mkdir(exist_ok=True)

    # checkpoint.toml 連携の state
    checkpoint_data = load_checkpoint_toml()
    pick_state: dict = {}

    # LoRA_keywords.toml + LoRA 一覧 + corpus 構築 (起動時 1 回)
    lora_keywords_data = load_lora_keywords_toml()
    all_loras = sorted(LORA_DIR.glob("*.safetensors")) if args.lora_stack_max > 0 else []
    lora_corpus = build_lora_corpus_for_playground(all_loras, lora_keywords_data) if all_loras else {}
    if all_loras:
        print(f"  LoRA 候補: {len(all_loras)} 件 / keywords.toml 登録: {len(lora_keywords_data)} 件")

    # 負のクオリティ embedding (起動時 1 回スキャン)
    neg_embeddings = collect_negative_embeddings() if args.embeddings else []
    if neg_embeddings:
        print(f"  negative embeddings: {len(neg_embeddings)} 件 ({', '.join(neg_embeddings[:5])}{'...' if len(neg_embeddings) > 5 else ''})")

    # --pose 指定時: ソース PNG を起動時 1 回 resolve + upload (ループ内で使い回す)
    pose_png: Optional[Path] = None
    pose_upload_name: Optional[str] = None
    if args.pose:
        pose_png = resolve_png_path(args.pose)
        pose_upload_name = upload_image_to_comfyui(pose_png)
        print(f"  pose source: {pose_png.name} (uploaded as {pose_upload_name})")

    stop = {"flag": False}
    def handler(_s, _f):
        stop["flag"] = True
        print("\n[Ctrl+C] 中断要求 (現在の生成完了後に終了)", flush=True)
    signal.signal(signal.SIGINT, handler)

    total = 0
    while not stop["flag"]:
        try:
            iter_start = time.time()

            # --prompt png/original のときソース PNG を一度だけ用意
            src_png: Optional[Path] = None
            if args.prompt in ("png", "original"):
                if not args.png:
                    raise SystemExit(f"--prompt {args.prompt} には --png <PNG> が必要")
                src_png = resolve_png_path(args.png)

            positive, negative, lora_keywords, extras = get_prompt_for_iteration(
                args.prompt, src_png, args.sentence, args.lora_keywords,
            )

            # original モード: PNG の Model を checkpoint として優先 (--checkpoint 引数より優先)
            fixed_checkpoint = args.checkpoint
            if "model" in extras:
                fixed_checkpoint = extras["model"]
            checkpoint_path = pick_checkpoint(checkpoint_data, pick_state, fixed_checkpoint)

            seed = random.randint(0, 2**32 - 1)

            entry = checkpoint_data.get(checkpoint_path.stem, {})
            inference_bonus = int(entry.get("inference", 0))
            use_steps = max(1, steps + inference_bonus)

            # LoRA: original モードは PNG 由来をそのまま使う (抽選 skip)。
            # それ以外は通常 keyword 抽選 (90% match / 10% random)。
            picked_loras: list[tuple[Path, float]] = []
            if "loras" in extras:
                # original モード: PNG メタの (name, strength) をそのまま使う
                #  ファイル名は LoRA_DIR で存在確認、無いものは捨てる
                for name, strength in extras["loras"]:
                    cand = LORA_DIR / name
                    if not cand.exists():
                        # stem だけ書かれてる場合も拾う
                        cand2 = LORA_DIR / f"{name}.safetensors" if not name.endswith(".safetensors") else cand
                        if cand2.exists():
                            cand = cand2
                        else:
                            print(f"  [warn] PNG メタの LoRA が見つからない: {name}、スキップ", flush=True)
                            continue
                    picked_loras.append((cand, float(strength)))
            elif args.gear == "high" and all_loras and args.lora_stack_max > 0:
                picked = pick_n_loras_by_keywords(
                    all_loras, lora_keywords, lora_corpus,
                    n_max=args.lora_stack_max, n_min=args.lora_stack_min,
                )
                if picked:
                    n = len(picked)
                    strength = args.lora_scale / n
                    picked_loras = [(p, strength) for p in picked]

            # ControlNet 抽選: gear high + --no-controlnet 未指定
            # 優先順位:
            #   (1) --pose 指定 → openpose 強制 + pose PNG (strength=args.pose_strength)
            #   (2) --prompt png/original の src_png → 既存通り style ベース抽選
            #   (3) どちらもなし → ControlNet OFF
            picked_controlnet: Optional[Path] = None
            controlnet_mode = "passthrough"
            controlnet_upload_name: Optional[str] = None
            effective_cn_strength = args.controlnet_strength
            if args.gear == "high" and not args.no_controlnet:
                if pose_upload_name is not None:
                    picked_controlnet = pick_controlnet(
                        "", args.controlnet, force_openpose=True,
                    )
                    if picked_controlnet is not None:
                        controlnet_mode = "openpose"
                        controlnet_upload_name = pose_upload_name
                        effective_cn_strength = args.pose_strength
                elif src_png is not None:
                    ckpt_style = (entry.get("style") or "").strip()
                    picked_controlnet = pick_controlnet(ckpt_style, args.controlnet)
                    if picked_controlnet is not None:
                        controlnet_mode = infer_controlnet_mode(picked_controlnet.stem)
                        try:
                            controlnet_upload_name = upload_image_to_comfyui(src_png)
                        except Exception as e:
                            print(f"  [warn] ControlNet 用画像 upload 失敗 ({e})、CN OFF", flush=True)
                            picked_controlnet = None

            print(f"\n=== source {total+1} ===")
            print(f"  checkpoint: {checkpoint_path.name}"
                  f"{' (未計測)' if checkpoint_path.stem not in checkpoint_data else ''}")
            print(f"  positive  : {positive[:120]}{'...' if len(positive) > 120 else ''}")
            print(f"  negative  : {negative[:80]}{'...' if len(negative) > 80 else ''}")
            print(f"  lora_kw   : {', '.join(lora_keywords) if lora_keywords else '(none)'}")
            if picked_loras:
                names = [f"{p.name}({s:.2f})" for p, s in picked_loras]
                print(f"  LoRA x{len(picked_loras)}: " + " + ".join(names))
            if picked_controlnet is not None:
                tag = " [--pose]" if pose_upload_name is not None else ""
                print(f"  ControlNet: {picked_controlnet.name} "
                      f"(mode={controlnet_mode}, strength={effective_cn_strength:.2f}){tag}")
            print(f"  seed/steps: {seed} / {use_steps}"
                  f"{f' (= {steps} + inference {inference_bonus:+})' if inference_bonus else ''}")

            # アップスケールモデル選択: --upscale-model 指定 → そのまま、未指定 → style ベース
            upscale_model_name: Optional[str] = None
            if args.upscale:
                if args.upscale_model:
                    upscale_model_name = args.upscale_model
                else:
                    style = (entry.get("style") or "").strip().lower()
                    upscale_model_name = _UPSCALE_MODEL_BY_STYLE.get(style, _UPSCALE_MODEL_DEFAULT)

            # LoRA キーワードを (0.8/N) 重みで positive に append (README L193 仕様)
            positive_augmented = augment_positive_with_lora_keywords(positive, lora_keywords)
            if positive_augmented != positive:
                print(f"  prompt+kw : ...{positive_augmented[len(positive):][:80]}")

            # 負のクオリティ embedding を negative に追加 (奇形/低クオリティ抑制)
            negative_augmented = augment_negative_with_embeddings(negative, neg_embeddings)

            workflow_loras = [(p.name, s) for p, s in picked_loras]
            hand_model = args.adetailer_hand_model if args.adetailer_hand_model else None
            person_model = args.adetailer_person_model if args.adetailer_person_model else None
            workflow = build_workflow_txt2img(
                checkpoint=checkpoint_path.name,
                positive=positive_augmented, negative=negative_augmented,
                seed=seed, steps=use_steps, cfg=args.cfg_scale,
                width=args.width, height=args.height,
                sampler_name=args.sampler, scheduler=args.scheduler,
                loras=workflow_loras,
                controlnet_name=picked_controlnet.name if picked_controlnet else None,
                controlnet_mode=controlnet_mode,
                controlnet_image=controlnet_upload_name,
                controlnet_strength=effective_cn_strength,
                upscale_model=upscale_model_name,
                adetailer=args.adetailer,
                adetailer_face_model=args.adetailer_face_model,
                adetailer_hand_model=hand_model,
                adetailer_person_model=person_model,
                adetailer_denoise=args.adetailer_denoise,
                adetailer_person_denoise=args.adetailer_person_denoise,
                adetailer_steps=args.adetailer_steps,
            )
            if args.adetailer:
                parts = [f"face={args.adetailer_face_model}"]
                if hand_model:
                    parts.append(f"hand={hand_model}")
                if person_model:
                    parts.append(f"person={person_model}@{args.adetailer_person_denoise}")
                print(f"  ADetailer: {', '.join(parts)}"
                      f" (denoise={args.adetailer_denoise}, steps={args.adetailer_steps})")
            if upscale_model_name:
                print(f"  upscale: {upscale_model_name}")

            prompt_id = submit_prompt(workflow, client_id)
            print(f"  ComfyUI prompt_id: {prompt_id}")

            result = wait_for_completion_ws(prompt_id, client_id)

            outputs = result.get("outputs", {})
            # node 7 = 通常解像度 (3_1_generated)
            save_node = outputs.get("7", {})
            images = save_node.get("images", [])
            if not images:
                print(f"  [warn] 出力画像が見つからない、スキップ")
                continue
            img_info = images[0]
            img_bytes = fetch_image(img_info["filename"],
                                     img_info.get("subfolder", ""),
                                     img_info.get("type", "output"))

            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            out_path = GENERATED_DIR / f"{ts}.png"
            save_with_a1111_metadata(
                img_bytes, out_path,
                positive=positive_augmented, negative=negative_augmented, seed=seed,
                steps=use_steps, cfg=args.cfg_scale,
                sampler=args.sampler, scheduler=args.scheduler,
                width=args.width, height=args.height,
                checkpoint=checkpoint_path.name,
                lora_keywords=lora_keywords,
                loras=[(p.name, s) for p, s in picked_loras],
                controlnet_name=picked_controlnet.name if picked_controlnet else None,
                controlnet_mode=controlnet_mode,
                controlnet_strength=effective_cn_strength,
                pose_source=pose_png.name if pose_png else None,
                adetailer=args.adetailer,
                adetailer_person=bool(person_model) and args.adetailer,
            )
            elapsed = time.time() - iter_start
            total += 1
            print(f"  → {out_path.name}  {args.width}x{args.height}  ({elapsed:.1f}s)")

            # node 14 = アップスケール後 (3_2_upscaled)
            if upscale_model_name:
                up_node = outputs.get("14", {})
                up_images = up_node.get("images", [])
                if up_images:
                    up_info = up_images[0]
                    up_bytes = fetch_image(up_info["filename"],
                                            up_info.get("subfolder", ""),
                                            up_info.get("type", "output"))
                    up_path = UPSCALED_DIR / f"{ts}.png"
                    save_with_a1111_metadata(
                        up_bytes, up_path,
                        positive=positive_augmented, negative=negative_augmented, seed=seed,
                        steps=use_steps, cfg=args.cfg_scale,
                        sampler=args.sampler, scheduler=args.scheduler,
                        width=args.width * 4, height=args.height * 4,
                        checkpoint=checkpoint_path.name,
                        lora_keywords=lora_keywords,
                        loras=[(p.name, s) for p, s in picked_loras],
                        controlnet_name=picked_controlnet.name if picked_controlnet else None,
                        controlnet_mode=controlnet_mode,
                        controlnet_strength=effective_cn_strength,
                        pose_source=pose_png.name if pose_png else None,
                        adetailer=args.adetailer,
                        adetailer_person=bool(person_model) and args.adetailer,
                    )
                    print(f"      up → 3_2_upscaled/{up_path.name}  {args.width*4}x{args.height*4} ({upscale_model_name})")
                else:
                    print(f"  [warn] アップスケール出力が見つからない")

            # gear high のみ checkpoint.toml の fast/slow を更新 / 新規追記
            # 直前にディスクから再読込してマージ → ユーザが外部エディタで編集中の他フィールドを潰さない
            if args.gear == "high":
                reload_update_save_checkpoint_toml(checkpoint_path.stem, elapsed, checkpoint_data)

            # cooldown: --cooldown 明示なら固定、未指定なら (GPU 温度 - 50) 秒、取れなければ 1.0 秒
            if not stop["flag"]:
                if args.cooldown is not None:
                    wait_s = max(0.0, args.cooldown)
                else:
                    temp = current_gpu_temp()
                    wait_s = max(0.0, float((temp or 51) - 50))
                if wait_s > 0:
                    if args.cooldown is None and temp is not None:
                        print(f"  cooldown: GPU {temp}°C → {wait_s:.0f}s 待機")
                    time.sleep(wait_s)

        except Exception as e:
            print(f"\n[エラー] {e}", flush=True)
            if not stop["flag"]:
                time.sleep(5.0)

    print(f"\n総計: {total} 枚")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(0)
