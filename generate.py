#!/usr/bin/env python3
"""generate.py - ComfyUI HTTP API 経由で SDXL 画像を連続生成する CLI。

実行モデル:
    [Python (この generate.py)] ─── HTTP POST /prompt ──▶ [常駐 ComfyUI server]
                                  ◀── GET /history/{id} ──
                                  ◀── GET /view?filename ──
    各 source ループで:
        prompt.toml → build_prompt → checkpoint 抽選 → workflow JSON 組立 →
        ComfyUI に投入 → 完成画像を fetch → A1111 メタ付き PNG で 5_1_generated に保存

前提:
    `python ComfyUI/main.py --listen 127.0.0.1 --port 8188` で ComfyUI が常駐している
    (`--listen 0.0.0.0` でも可、外部 LAN 公開する場合)

Phase 1 実装範囲:
    - `--prompt auto` のみ (prompt.toml 駆動)
    - 単純 SDXL txt2img (LoRA / ControlNet / ADetailer / upscale なし)
    - checkpoint は 4_1_SDXL_checkpoint (sd15 時 3_1_SD15_checkpoint) からランダム
    - Ctrl+C でループ停止
    - A1111 互換メタを 5_1_generated/{YYYYMMDDHHMMSS}.png に書き込み
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
import urllib.error
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
    L,
)
# A1111 メタ書き込みは pngutil の serializer を流用
from pngutil import serialize_a1111_parameters, write_text_chunks
# tensors triage は起動時に必ず実行
from dist_tensors import check_tensors

# --------------------------------------------------------------------------- #
# 定数
# --------------------------------------------------------------------------- #
ROOT             = Path(__file__).parent
# --- 固定 dir 定義 (lane 非依存。extra_model_paths.yaml 生成にも使う) ---
SD15_CHECKPOINT_DIR = ROOT / "3_1_SD15_checkpoint"
SD15_LORA_DIR       = ROOT / "3_2_SD15_LoRA"
SD15_EMBED_DIR      = ROOT / "3_3_SD15_Embedding"
SD15_ROUGH_DIR      = ROOT / "3_9_SD15_rough"   # 2段チェーンの SD15 下書き保存先
SDXL_CHECKPOINT_DIR = ROOT / "4_1_SDXL_checkpoint"
SDXL_LORA_DIR       = ROOT / "4_2_SDXL_LoRA"
SDXL_CONTROLNET_DIR = ROOT / "4_3_SDXL_ControlNet"
SDXL_EMBED_DIR      = ROOT / "4_4_SDXL_Embedding"
# --- アクティブ dir (既定 sdxl。--version sd15 で main() が SD15 に差し替え) ---
CHECKPOINT_DIR   = SDXL_CHECKPOINT_DIR
LORA_DIR         = SDXL_LORA_DIR
EMBEDDING_DIR    = SDXL_EMBED_DIR
CONTROLNET_DIR   = SDXL_CONTROLNET_DIR
PROMPTS_DIR      = ROOT / "1_0_prompts"
GENERATED_DIR    = ROOT / "5_1_generated"
UPSCALED_DIR     = ROOT / "5_2_upscaled"
WORKFLOW_DUMP_DIR = ROOT / "workflow_dump"   # --dump-workflow: 組んだ API workflow JSON の出力先
CHECKPOINT_TOML  = ROOT / "checkpoint.toml"
LORA_KEYWORDS_TOML = ROOT / "LoRA_keywords.toml"
SDXL_LORA_HINT_TOML = ROOT / "SDXL_LoRA_hint.toml"   # SDXL LoRA の subject (pose のみ機能的)

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
def _format_comfy_error(body: bytes) -> str:
    """ComfyUI の /prompt 400 レスポンス本文を、原因が分かる形に整形する。
    本文には error.message と node_errors (どのノードの何の入力が不正か) が入る。"""
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        text = body.decode("utf-8", "replace").strip()
        return text[:1500] if text else L("(レスポンス本文なし)", "(empty response body)")
    lines: list[str] = []
    err = data.get("error") or {}
    if err:
        msg = err.get("message") or err.get("type") or ""
        det = err.get("details") or ""
        lines.append(f"{msg}{(' - ' + det) if det else ''}".strip())
    for node_id, ne in (data.get("node_errors") or {}).items():
        cls = ne.get("class_type", "?")
        for e in (ne.get("errors") or []):
            em = e.get("message") or e.get("type") or ""
            ed = e.get("details") or ""
            lines.append(f"  node {node_id} ({cls}): {em}{(' — ' + ed) if ed else ''}")
    return "\n".join([ln for ln in lines if ln]) or (json.dumps(data, ensure_ascii=False)[:1500])


def _http_post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # ComfyUI はバリデーション失敗 (例: モデル名が候補に無い) を 400 + JSON 本文で返す。
        # 本文を読まないと「Bad Request」しか出ず原因が分からないので、ここで整形して再 raise。
        try:
            detail = _format_comfy_error(e.read())
        except Exception:
            detail = L("(本文の読取に失敗)", "(failed to read response body)")
        raise RuntimeError(f"ComfyUI {e.code} {e.reason} @ {url}\n{detail}") from None


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
        print(L(f"  [ws error] {e}、HTTP poll にフォールバック", f"  [ws error] {e}, falling back to HTTP poll"), flush=True)
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
            elif mtype == "executing":
                # node=null は全ノード実行完了の合図。execution_success を取り損ねても
                # (例: 完全キャッシュヒットで step が走らない等) ここで確実に抜ける。
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
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
EXTRA_MODEL_PATHS_YAML = COMFYUI_DIR / "extra_model_paths.yaml"


def write_extra_model_paths() -> bool:
    """playground の model dir を ComfyUI に認識させる extra_model_paths.yaml を
    dir 定数から自動生成する。SD15(3_x) / SDXL(4_x) 両レーンを登録 (ComfyUI は
    ファイル名で全 path を横断検索するので、--version に関係なく解決できる)。

    内容が変わったら True を返す (= ComfyUI 再起動が必要)。
    dir リネームで yaml がズレて 400 になる事故を構造的に防ぐための自動生成。"""
    content = (
        "# 自動生成 (generate.py write_extra_model_paths)。手で編集しない。\n"
        "# playground の model dir を ComfyUI に登録 (SD15=3_x / SDXL=4_x 両レーン)。\n"
        "comfyui_playground:\n"
        f"    base_path: {ROOT.as_posix()}/\n"
        "    is_default: true\n"
        "    checkpoints: |\n"
        f"        {SD15_CHECKPOINT_DIR.name}/\n"
        f"        {SDXL_CHECKPOINT_DIR.name}/\n"
        "    loras: |\n"
        f"        {SD15_LORA_DIR.name}/\n"
        f"        {SDXL_LORA_DIR.name}/\n"
        "    embeddings: |\n"
        f"        {SD15_EMBED_DIR.name}/\n"
        f"        {SDXL_EMBED_DIR.name}/\n"
        f"    controlnet: {SDXL_CONTROLNET_DIR.name}/\n"
    )
    old = EXTRA_MODEL_PATHS_YAML.read_text(encoding="utf-8") if EXTRA_MODEL_PATHS_YAML.exists() else ""
    if old == content:
        return False
    EXTRA_MODEL_PATHS_YAML.write_text(content, encoding="utf-8")
    print(L(f"  [extra_model_paths] {EXTRA_MODEL_PATHS_YAML.name} を更新 (model dir 登録)", f"  [extra_model_paths] {EXTRA_MODEL_PATHS_YAML.name} updated (model dir registered)"), flush=True)
    return True


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
    raise SystemExit(L(f"ComfyUI server ({arch}) の起動 timeout ({ready_timeout}s)", f"ComfyUI server ({arch}) startup timeout ({ready_timeout}s)"))


def ensure_comfyui_arch(arch: str, force_restart: bool = False) -> None:
    """現 server の device と `arch` を比較。mismatch なら kill + restart。
    `force_restart=True` なら device 一致でも再起動 (extra_model_paths.yaml 更新時など、
    起動時設定を読み直させたいケース)。`arch` ∈ {'cuda', 'cpu'}。
    """
    cur = get_comfyui_device()
    if cur is None:
        # サーバ停止中 → そのまま起動
        print(L(f"  ComfyUI 未起動 → {arch} で新規起動", f"  ComfyUI not running → starting fresh with {arch}"), flush=True)
        start_comfyui_server(arch)
        return
    if cur == arch and not force_restart:
        # 一致 + 再起動不要 → 何もしない
        return
    reason = L("model path 更新で設定再読込", "reloading config after model path update") if (cur == arch and force_restart) else L(f"device mismatch (現 {cur}, 要 {arch})", f"device mismatch (current {cur}, required {arch})")
    print(L(f"  ComfyUI 再起動中... ({reason})", f"  ComfyUI restarting... ({reason})"), flush=True)
    kill_comfyui_server()
    start_comfyui_server(arch)
    print(L(f"  ComfyUI server を {arch} で再起動完了", f"  ComfyUI server restarted with {arch}"), flush=True)


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
    init_image: Optional[str] = None,
    denoise: float = 1.0,
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
    adetailer_person_model: Optional[str] = "segm/person_yolov8n-seg.pt",
    adetailer_denoise: float = 0.5,
    adetailer_person_denoise: float = 0.3,
    adetailer_steps: int = 30,
    hires_fix: bool = False,
    hires_scale: float = 1.5,
    hires_denoise: float = 0.35,
    hires_steps: int = 20,
) -> dict:
    """txt2img / img2img の workflow JSON を組み立てる (SD15 / SDXL 共通)。

    LoRA stacking 対応: loras = [(lora_name.safetensors, strength), ...] を渡すと
    CheckpointLoaderSimple と CLIPTextEncode/KSampler の間に LoraLoader をチェーン挿入する。

    init_image (ComfyUI 上のアップロード名) を渡すと img2img になる:
    その画像を width×height にスケール → VAEEncode → denoise (既定 1.0、img2img では <1) で再描画。
    未指定なら EmptyLatentImage の txt2img (denoise は内部で 1.0 固定)。

    Hires Fix (hires_fix=True): width/height を **base 解像度** (= 1段目 sampling 解像度)
    として扱い、LatentUpscaleBy(scale) → 2 段目 KSampler(hires_denoise) で refine →
    最終 latent は width×scale × height×scale で VAEDecode。memory 「Hires Fix 既定 ON
    (512→768 二段)」を ComfyUI 版に移植した実装で、引数の width/height は **必ず base 側**
    (SD15 なら 512〜768、SDXL なら 1024 等のネイティブ解像度) を渡す。Hires Fix off のときは
    width/height がそのまま最終解像度。txt2img / img2img どちらでも動く。
    """
    # Hires Fix は width/height を base として扱うので、追加の縮小は不要
    base_w, base_h = width, height
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
    # latent ソース: txt2img は EmptyLatentImage、img2img (init_image 指定) は
    # LoadImage → ImageScale(width×height) → VAEEncode で init latent を作り、denoise<1 で再描画。
    # SD15 下書き → SDXL 清書チェーンの「清書」段がこの img2img 経路を使う。
    if init_image:
        # node ID は 40-42 を使う (ADetailer 部位ループが 26-31 を使うため衝突回避)
        # init_image は base 解像度にスケール (Hires Fix 段で 1.5× にアップ)
        workflow["40"] = {
            "class_type": "LoadImage",
            "inputs": {"image": init_image},
        }
        workflow["41"] = {
            "class_type": "ImageScale",
            "inputs": {"image": ["40", 0], "width": base_w, "height": base_h,
                       "upscale_method": "lanczos", "crop": "disabled"},
        }
        workflow["42"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["41", 0], "vae": ["1", 2]},
        }
        latent_ref = ["42", 0]
        ksampler_denoise = float(denoise)
    else:
        workflow["4"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": base_w, "height": base_h, "batch_size": 1},
        }
        latent_ref = ["4", 0]
        ksampler_denoise = 1.0
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
            "denoise": ksampler_denoise,
            "model": model_ref,  # 最終 LoraLoader (なければ CheckpointLoader) の model
            "positive": ksampler_positive_ref,
            "negative": ksampler_negative_ref,
            "latent_image": latent_ref,
        },
    }
    workflow["6"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    }
    # Hires Fix: ImageScale 方式 (旧 sd_playground の diffusers Img2ImgPipeline と同等)。
    #   1段目 latent (node 5) → VAEDecode (29) → ImageScale lanczos (30)
    #   → VAEEncode (33) → KSampler 2段目 refine (31) → VAEDecode 最終 (32)
    # LatentUpscaleBy 方式 (旧実装) は latent 空間の補間誤差を 2段目で増幅し、
    # タイル/chromatic aberration/色彩破綻を多発した (2026-05-27 実機確認)。
    # RGB 空間で lanczos 拡大すれば品質劣化が大幅に減る (ただし VAE 2 回追加で時間増)。
    if hires_fix:
        # target は 8 倍数に丸め (VAEEncode 要件)
        target_w = max(64, (int(base_w * hires_scale) // 8) * 8)
        target_h = max(64, (int(base_h * hires_scale) // 8) * 8)
        workflow["29"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        }
        workflow["30"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["29", 0],
                "width": target_w,
                "height": target_h,
                "upscale_method": "lanczos",
                "crop": "disabled",
            },
        }
        workflow["33"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["30", 0], "vae": ["1", 2]},
        }
        workflow["31"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": (seed + 7) & 0xFFFFFFFF,   # 2 段目は別ノイズで refine
                "steps": int(hires_steps),
                "cfg": float(cfg),
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": float(hires_denoise),
                "model": model_ref,
                "positive": ksampler_positive_ref,
                "negative": ksampler_negative_ref,
                "latent_image": ["33", 0],
            },
        }
        workflow["32"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["31", 0], "vae": ["1", 2]},
        }
        final_image_ref = ["32", 0]
    else:
        final_image_ref = ["6", 0]
    # ADetailer chain (FaceDetailer for face / optional hand / optional person)
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

        # 実行順は person → face → hand。全身の構造を先に直し、顔・手のディテールを
        # 最後に乗せる (詳細パスが person の再描画で上書きされないように)。
        # node ID は固定 (face=20/21 / hand=22/23 / person=24/25)、final_image_ref で連結。

        # (24)(25) Person detector + FaceDetailer を先に (全身 inpainting、足/脚の奇形・体の構造)
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

        # (20)(21) face detector + FaceDetailer (構造の後にディテールを乗せる)
        workflow["20"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": adetailer_face_model},
        }
        workflow["21"] = {
            "class_type": "FaceDetailer",
            "inputs": _facedetailer_inputs(final_image_ref, ["20", 0], (seed + 1) & 0xFFFFFFFF),
        }
        final_image_ref = ["21", 0]

        # (22)(23) Hand detector + FaceDetailer (最後 = 上書きされない)
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
        print(L(f"[警告] checkpoint.toml パース失敗 ({e})、空として扱う", f"[warn] checkpoint.toml parse failed ({e}), treating as empty"), flush=True)
        return {}


def save_checkpoint_toml(data: dict) -> None:
    """checkpoint.toml に保存。"""
    try:
        CHECKPOINT_TOML.write_text(tomli_w.dumps(data), encoding="utf-8")
    except Exception as e:
        print(L(f"[警告] checkpoint.toml 保存失敗: {e}", f"[warn] checkpoint.toml save failed: {e}"), flush=True)


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
        print(L(f"[警告] LoRA_keywords.toml パース失敗 ({e})、空として扱う", f"[warn] LoRA_keywords.toml parse failed ({e}), treating as empty"), flush=True)
        return {}


def load_sdxl_lora_subjects() -> dict[str, str]:
    """SDXL_LoRA_hint.toml から {stem: subject(lower)} を返す。subject="pose" のみ機能的
    (OpenPose 段で除外)。無ければ空 dict。"""
    if not SDXL_LORA_HINT_TOML.exists():
        return {}
    try:
        data = tomllib.loads(SDXL_LORA_HINT_TOML.read_text(encoding="utf-8"))
    except Exception as e:
        print(L(f"[警告] SDXL_LoRA_hint.toml パース失敗 ({e})、空として扱う", f"[warn] SDXL_LoRA_hint.toml parse failed ({e}), treating as empty"), flush=True)
        return {}
    return {stem: str((v or {}).get("subject") or "").strip().lower() for stem, v in data.items()}


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
        raise SystemExit(L(f"ControlNet が見つかりません: {fixed_name}", f"ControlNet not found: {fixed_name}"))

    if force_openpose:
        matched = [c for c in candidates
                   if "openpose" in c.stem.lower() or "_pose" in c.stem.lower()
                   or c.stem.lower().endswith("pose")]
        if not matched:
            raise SystemExit(
                L("--pose 指定だが 4_3_SDXL_ControlNet/ に openpose 系 ControlNet が見つかりません "
                  "(stem に 'openpose' / '_pose' / 末尾 'pose' を含むファイルを配置)",
                  "--pose specified but no openpose ControlNet found in 4_3_SDXL_ControlNet/ "
                  "(place a file with 'openpose' / '_pose' / ending in 'pose' in its stem)")
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


def upload_bytes_to_comfyui(data: bytes, filename: str) -> str:
    """画像 bytes を ComfyUI の input/ にアップロードし参照名を返す (中間下書きの受け渡し用)。"""
    import io
    import requests
    files = {"image": (filename, io.BytesIO(data), "image/png")}
    r = requests.post(f"{COMFY_BASE}/upload/image", files=files,
                      data={"type": "input", "overwrite": "true"}, timeout=60)
    r.raise_for_status()
    return r.json()["name"]


def _submit_and_fetch(workflow: dict, client_id: str, save_node: str = "7"):
    """workflow を投入 → 完了待ち → 指定 SaveImage ノードの画像 bytes を取得。

    返り値: (image_bytes|None, image_info|None, outputs)。画像が無ければ bytes=None。
    """
    prompt_id = submit_prompt(workflow, client_id)
    print(f"  ComfyUI prompt_id: {prompt_id}", flush=True)
    result = wait_for_completion_ws(prompt_id, client_id)
    outputs = result.get("outputs", {})
    imgs = outputs.get(save_node, {}).get("images", [])
    if not imgs:
        return None, None, outputs
    info = imgs[0]
    data = fetch_image(info["filename"], info.get("subfolder", ""), info.get("type", "output"))
    return data, info, outputs


def _dump_workflow(workflow: dict, kind: str) -> Path:
    """組んだ API 形式 workflow を JSON ファイルに保存し、パスを返す。

    ComfyUI v0.21 のフロントは API 形式 JSON をキャンバスに **ドラッグ＆ドロップ**
    すると自動レイアウトでノードグラフに展開してくれる。つまりこの JSON を
    WebUI (http://127.0.0.1:8188) に放り込めば「generate.py が実際に組んだグラフ」が
    そのまま絵で見える。出力先は workflow_dump/<時刻>_<kind>.json。
    """
    WORKFLOW_DUMP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out = WORKFLOW_DUMP_DIR / f"{ts}_{kind}.json"
    out.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    print(L(f"  [dump] workflow ({kind}, {len(workflow)} nodes) → "
            f"workflow_dump/{out.name}  ※WebUI canvas にドロップで可視化",
            f"  [dump] workflow ({kind}, {len(workflow)} nodes) → "
            f"workflow_dump/{out.name}  drop onto WebUI canvas to visualize"), flush=True)
    return out


# --------------------------------------------------------------------------- #
# Upscale モデル (Real-ESRGAN) の自動 DL
# --------------------------------------------------------------------------- #
UPSCALE_MODELS_DIR = COMFYUI_DIR / "models" / "upscale_models"
# 既知の Real-ESRGAN モデル → 公式 GitHub release URL
_UPSCALE_MODEL_URLS = {
    "RealESRGAN_x4plus_anime_6B.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x4plus.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def ensure_upscale_model(name: Optional[str]) -> Optional[str]:
    """upscale_models/<name> が無ければ公式 GitHub release から DL する。
    成功 / 既存 → name を返す。URL 未登録 / DL 失敗 → None を返し caller がスキップ。"""
    if not name:
        return None
    full = UPSCALE_MODELS_DIR / name
    if full.is_file():
        return name
    url = _UPSCALE_MODEL_URLS.get(name)
    if not url:
        print(L(f"  [upscale][warn] {name} の DL URL 未登録 → スキップ",
                f"  [upscale][warn] no download URL registered for {name} → skipping"), flush=True)
        return None
    print(L(f"  [upscale] {name} が無い → {url} から DL 試行...",
            f"  [upscale] {name} not found → attempting download from {url}..."), flush=True)
    try:
        import urllib.request
        full.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, full)
        print(L(f"  [upscale] DL 完了 → {full} ({full.stat().st_size // (1024*1024)} MB)",
                f"  [upscale] download complete → {full} ({full.stat().st_size // (1024*1024)} MB)"), flush=True)
        return name
    except Exception as e:
        print(L(f"  [upscale][warn] {name} の DL 失敗 ({type(e).__name__}: {e}) → スキップ",
                f"  [upscale][warn] {name} download failed ({type(e).__name__}: {e}) → skipping"), flush=True)
        return None


# --------------------------------------------------------------------------- #
# ADetailer (Ultralytics) モデルの自動 DL
# --------------------------------------------------------------------------- #
ULTRALYTICS_DIR = COMFYUI_DIR / "models" / "ultralytics"
_ADETAILER_HF_REPO = "Bingsu/adetailer"  # face/hand/person の公式配布元


def ensure_adetailer_model(rel_name: Optional[str]) -> Optional[str]:
    """ADetailer モデル (例 'segm/person_yolov8n-seg.pt') が無ければ HF から DL する。

    - 既に ComfyUI/models/ultralytics/<rel_name> があればそのまま返す。
    - 無ければ Bingsu/adetailer から basename を DL してコピー → rel_name を返す。
    - HF に無い (= NSFW 部位系など Civitai 産) で DL 失敗したら、警告して None を返す
      (= その detector を無効化し、生成 workflow が落ちないようにする)。
    """
    if not rel_name:
        return None
    full = ULTRALYTICS_DIR / rel_name
    if full.is_file():
        return rel_name
    basename = full.name
    print(L(f"  [adetailer] {rel_name} が無い → {_ADETAILER_HF_REPO} から DL 試行...", f"  [adetailer] {rel_name} not found → attempting download from {_ADETAILER_HF_REPO}..."), flush=True)
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        src = hf_hub_download(_ADETAILER_HF_REPO, basename)
        full.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, full)
        print(L(f"  [adetailer] DL 完了 → {full} ({full.stat().st_size // 1024} KB)", f"  [adetailer] download complete → {full} ({full.stat().st_size // 1024} KB)"), flush=True)
        return rel_name
    except Exception as e:
        print(L(f"  [adetailer][warn] {basename} は自動 DL できない ({type(e).__name__})。"
                f"この detector を無効化 (手動で {full} に配置すれば有効化)",
                f"  [adetailer][warn] {basename} cannot be auto-downloaded ({type(e).__name__}). "
                f"Disabling this detector (place it manually at {full} to enable)"), flush=True)
        return None


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
    """`--png NAME` を絶対パス / 1_0_prompts/NAME / 1_0_prompts/NAME.png の順で解決。"""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for cand in (p, PROMPTS_DIR / name, PROMPTS_DIR / f"{name}.png"):
        if cand.exists():
            return cand
    raise SystemExit(L(f"PNG が見つかりません: {name} (1_0_prompts/ を確認)", f"PNG not found: {name} (check 1_0_prompts/)"))


def update_checkpoint_timing(name: str, elapsed_s: float, data: dict) -> None:
    """gear high 完走後、checkpoint.toml を in-place 更新。
    既存エントリは fast (最小) / slow (最大) を更新、新規は追記。
    """
    elapsed = int(round(elapsed_s))
    entry = data.get(name)
    if entry is None:
        # 新規追記。family はファイル名から推定 (pony/illustrious/real)、外れは手で直す。
        fam = _family_from_name(name)
        data[name] = {
            "slow": elapsed,
            "fast": elapsed,
            "like": 0,
            "inference": 0,
            "style": "",
            "family": fam,
        }
        print(L(f"  checkpoint.toml に {name} を初期登録 (slow=fast={elapsed}s, family={fam or '?'})", f"  checkpoint.toml: registered {name} (slow=fast={elapsed}s, family={fam or '?'})"), flush=True)
    else:
        cur_fast = int(entry.get("fast", elapsed))
        cur_slow = int(entry.get("slow", elapsed))
        new_fast = min(cur_fast, elapsed)
        new_slow = max(cur_slow, elapsed)
        if new_fast != cur_fast or new_slow != cur_slow:
            entry["fast"] = new_fast
            entry["slow"] = new_slow
            print(L(f"  checkpoint.toml 更新 {name}: fast={new_fast}s slow={new_slow}s", f"  checkpoint.toml updated {name}: fast={new_fast}s slow={new_slow}s"), flush=True)
        # like / inference / style はユーザ管理、触らない


# --------------------------------------------------------------------------- #
# 抽選
# --------------------------------------------------------------------------- #
_CKPT_MIN_BYTES = 256 * 1024 * 1024  # 256MB 未満は checkpoint ではない


def _gather_checkpoints(dirs: list[Path]) -> list[Path]:
    """複数 dir の checkpoint を 1 プールに集約 (name でソート)。

    256MB 未満のファイルは除外する: 実体が embedding / LoRA なのに base と誤分類されて
    checkpoint dir に紛れたファイル (例: ng_deepnegative [75,768] TI) を抽選プールから外し、
    CheckpointLoaderSimple が "Could not detect model type" で落ちるのを防ぐ。
    """
    out: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.safetensors"):
            try:
                if p.stat().st_size >= _CKPT_MIN_BYTES:
                    out.append(p)
            except OSError:
                continue
    return sorted(out, key=lambda p: p.name)


def checkpoint_version(path: Path) -> str:
    """checkpoint の版を置き場 dir で判定 (tensors.py が分類済なので確実)。"""
    return "sd15" if path.parent == SD15_CHECKPOINT_DIR else "sdxl"


def _is_pony_name(stem: str) -> bool:
    """ファイル名から Pony 系かを推定 (pony / pdxl / pny / pxl を含む)。
    ユーザは収集時に pony/pny/pxl を意図的に付与 (無記載=オリジナル名)。"""
    s = stem.lower()
    return ("pony" in s) or ("pdxl" in s) or ("pny" in s) or ("pxl" in s)


def checkpoint_is_pony(checkpoint_path: Path, checkpoint_data: dict) -> bool:
    """checkpoint が Pony 系か。checkpoint.toml の `family` を優先、無ければファイル名から推定。
    (family タグは後段 (b) で付与。未設定の間はファイル名推定で動く)。"""
    fam = str((checkpoint_data.get(checkpoint_path.stem) or {}).get("family") or "").strip().lower()
    if fam:
        return fam == "pony"
    return _is_pony_name(checkpoint_path.stem)


def gate_neg_embeddings(neg_stems: list[str], is_pony: bool, pony_cap: int = 3) -> list[str]:
    """neg embedding を系統に応じて取捨:
    - 非 Pony checkpoint → Pony 専用 embed (名前に Pny/Pony/PDXL) を**除外** (色崩壊/主体消失防止)。
    - Pony checkpoint → 汎用 embed は全通し、Pony 専用 embed は **pony_cap 個まで** (過剰ネガ抑制)。
    汎用 embed は系統問わず常に通す。"""
    general = [e for e in neg_stems if not _is_pony_name(e)]
    if not is_pony:
        return general
    pony = [e for e in neg_stems if _is_pony_name(e)]
    return general + pony[:pony_cap]


# Pony 系 checkpoint に自動前置する quality タグ (score 条件付け)
PONY_SCORE_PREFIX = "score_9, score_8_up, score_7_up, source_anime, rating_explicit"


def _family_from_name(stem: str) -> str:
    """ファイル名から checkpoint 系統を推定 (checkpoint.toml `family` の初期値)。
    判定不能は "" (ユーザが手で補正する想定)。メタには系統が無いのでファイル名が頼り。"""
    s = stem.lower()
    if ("pony" in s) or ("pdxl" in s) or ("pny" in s) or ("pxl" in s):
        return "pony"                       # 2.5D-3D 主流 (Pony lineage)
    if ("ill" in s) or ("noob" in s) or ("nai" in s):
        return "2d"                         # 2D 純粋 (Illustrious / NoobAI / NAI 系)
    if ("real" in s) or ("photo" in s):
        return "real"                       # 実写寄り
    return ""


def _resolve_checkpoint_name(name: str, dirs: Optional[list[Path]] = None) -> Path:
    """`--checkpoint NAME` 解決。stem / .safetensors 付きどちらでも可。"""
    candidates = _gather_checkpoints(dirs or [CHECKPOINT_DIR])
    for c in candidates:
        if c.stem == name or c.name == name:
            return c
    raise SystemExit(L(f"checkpoint が見つかりません: {name}", f"checkpoint not found: {name}"))


def pick_checkpoint(
    data: dict,
    state: dict,
    fixed_name: Optional[str] = None,
    pool_dirs: Optional[list[Path]] = None,
) -> Path:
    """checkpoint を抽選 (`checkpoint.toml` 連動)。

    `pool_dirs` を渡すと複数 dir (SD15 3_1 + SDXL 4_1 等) を 1 つの統合プールとして
    重み付き抽選する。版を区別せず checkpoint.toml の重みのみで引く (偏りは like で手動調整)。
    未指定なら従来どおり CHECKPOINT_DIR 単独。

    ルール:
        - `fixed_name` 指定 (= `--checkpoint NAME`) → そのまま返す
        - state['count'] == 0 (1 度め) + 未計測あり → 未計測からランダム
        - 2 度め以降 → 2/3 確率で計測済み (重み付き)、1/3 確率で未計測ランダム
        - 計測済み内の重み: `max(1, (max_slow*2 - (fast + slow)) / 2 + like)`
        - 片方しか無ければそちらに寄せる
    """
    dirs = pool_dirs or [CHECKPOINT_DIR]
    candidates = _gather_checkpoints(dirs)
    if not candidates:
        raise SystemExit(L(f"{', '.join(d.name for d in dirs)} に checkpoint がありません", f"no checkpoints found in {', '.join(d.name for d in dirs)}"))
    if fixed_name:
        return _resolve_checkpoint_name(fixed_name, dirs)

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
def collect_negative_embeddings(embed_dir: Optional[Path] = None) -> list[str]:
    """`embed_dir` (既定 EMBEDDING_DIR、sdxl=4_4 / sd15=3_2) 配下から負のクオリティ embedding を集める。

    判定: stem を lowercase して `neg` / `bad` / `worst` を含むものを採用。
    例: `PonyXL_NegScore-neg.safetensors` / `SmoothNegative_Hands-neg.safetensors` 等。
    """
    d = embed_dir or EMBEDDING_DIR
    if not d.exists():
        return []
    stems: list[str] = []
    for p in sorted(d.glob("*.safetensors")):
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
        pos, neg, kws, many = build_prompt(cfg)
        extras["many"] = many
        return pos, neg, kws, extras

    if mode == "sentence":
        if not sentence:
            raise SystemExit(L("--prompt sentence には --sentence \"...\" が必要", "--prompt sentence requires --sentence \"...\""))
        cfg = load_prompt_config()
        positive = normalize_emphasis(sentence)
        negative = normalize_emphasis(str(cfg.get("negative_always") or ""))
        kws: list[str] = []
        if lora_keywords_arg:
            kws = [k.strip() for k in lora_keywords_arg.split(",") if k.strip()]
        return positive, negative, kws, extras

    if mode == "refine":
        # 画質アップ: PNG の埋込プロンプトを採用 (無ければ --sentence)。negative 無しは negative_always。
        # 画像そのものは loop 側で init_image に使う (img2img)。
        positive = negative = ""
        kws = []
        if png_path is not None:
            positive, negative, kws = parse_png_prompt_metadata(png_path)
        if not positive and sentence:
            positive = normalize_emphasis(sentence)
        if not negative:
            cfg = load_prompt_config()
            negative = normalize_emphasis(str(cfg.get("negative_always") or ""))
        if lora_keywords_arg:
            kws = [k.strip() for k in lora_keywords_arg.split(",") if k.strip()]
        extras["refine"] = True
        return positive, negative, kws, extras

    if mode == "png":
        if png_path is None:
            raise SystemExit(L("--prompt png には --png <PNG> が必要", "--prompt png requires --png <PNG>"))
        positive, negative, kws = parse_png_prompt_metadata(png_path)
        if not positive:
            print(L(f"  [info] PNG にメタ情報なし、auto モードにフォールバック", f"  [info] no metadata in PNG, falling back to auto mode"), flush=True)
            cfg = load_prompt_config()
            pos, neg, kws, many = build_prompt(cfg)
            extras["many"] = many
            return pos, neg, kws, extras
        return positive, negative, kws, extras

    if mode == "original":
        if png_path is None:
            raise SystemExit(L("--prompt original には --png <PNG> が必要", "--prompt original requires --png <PNG>"))
        meta = parse_png_full_metadata(png_path)
        positive = meta["positive"]
        negative = meta["negative"]
        kws      = meta["lora_keywords"]
        if not positive:
            print(L(f"  [info] PNG にメタ情報なし、auto モードにフォールバック", f"  [info] no metadata in PNG, falling back to auto mode"), flush=True)
            cfg = load_prompt_config()
            pos, neg, kws, many = build_prompt(cfg)
            extras["many"] = many
            return pos, neg, kws, extras
        # checkpoint / loras を extras に詰める (main loop で上書き適用)
        if meta["model"]:
            extras["model"] = meta["model"]
        if meta["loras"]:
            extras["loras"] = meta["loras"]
        return positive, negative, kws, extras

    raise SystemExit(L(f"--prompt {mode} は未対応", f"--prompt {mode} is not supported"))


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
    adetailer_parts: Optional[list[str]] = None,
    pipeline: Optional[str] = None,
    draft_checkpoint: Optional[str] = None,
    draft_loras: Optional[list[tuple[str, float]]] = None,
) -> None:
    """ComfyUI から取得した画像 bytes を A1111 互換メタ付きで PNG 保存する。

    2段チェーン時は draft_checkpoint / draft_loras に SD15 下書き段の情報を渡すと、
    `Draft model:` / `Draft loras:` として記録する (清書段は Model: / Loras:)。
    """
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
        tags = (["person"] if adetailer_person else []) + list(adetailer_parts or [])
        parsed["params"]["ADetailer"] = f"on ({', '.join(tags)})" if tags else "on"
    if pipeline:
        parsed["params"]["Pipeline"] = pipeline
    if draft_checkpoint:
        parsed["params"]["Draft model"] = draft_checkpoint
    if draft_loras:
        parsed["params"]["Draft loras"] = ", ".join(f"{n}: {s:.2f}" for n, s in draft_loras)
    parameters_text = serialize_a1111_parameters(parsed)
    write_text_chunks(out_path, {"parameters": parameters_text})


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=L("ComfyUI HTTP API 経由で SDXL 画像を連続生成する (Phase 1: --prompt auto のみ)",
                      "Continuously generate SDXL images via ComfyUI HTTP API (Phase 1: --prompt auto only)")
    )
    ap.add_argument("--prompt", choices=["auto", "sentence", "png", "original"], default="auto",
                    help=L("プロンプト入力モード。"
                           "auto=prompt.toml 駆動 / "
                           "sentence=--sentence で直接 / "
                           "png=PNG メタから読出 / "
                           "original=PNG メタの checkpoint+LoRA+prompt 全部を流用",
                           "prompt input mode. "
                           "auto=prompt.toml driven / "
                           "sentence=direct via --sentence / "
                           "png=read from PNG metadata / "
                           "original=reuse all PNG metadata (checkpoint+LoRA+prompt)"))
    ap.add_argument("--sentence", type=str, default=None,
                    help=L("--prompt sentence のとき、文章プロンプト。`**word**` 強調記法 OK",
                           "sentence prompt for --prompt sentence. `**word**` emphasis syntax supported"))
    ap.add_argument("--lora-keywords", type=str, default=None,
                    help=L("--prompt sentence のとき、LoRA キーワード列 (カンマ区切り)",
                           "LoRA keyword list for --prompt sentence (comma-separated)"))
    ap.add_argument("--png", type=str, default=None,
                    help=L("画質アップ refine 用 PNG (1_0_prompts/ 配下 or 絶対パス)。"
                           "その画像を SDXL img2img で描き直して SD15→SDXL 画質に上げる (1 枚で終了)",
                           "PNG for quality-up refine (under 1_0_prompts/ or absolute path). "
                           "Redraws the image via SDXL img2img to upgrade quality to SDXL level (single image, then exits)"))
    ap.add_argument("--png-sentence", type=str, default=None,
                    help=L("PNG の埋込プロンプト『文章』で生成する PNG (画像は使わない)。"
                           "統合抽選 (SD15/SDXL→1/2段) で連続量産。--prompt original と併用で全メタ流用",
                           "PNG whose embedded prompt sentence is used for generation (image itself is not used). "
                           "Continuous high-volume generation via unified draw (SD15/SDXL→1/2-stage). "
                           "Combine with --prompt original to reuse all metadata"))
    ap.add_argument("--refine-denoise", type=float, default=0.5,
                    help=L("--png refine の img2img denoise (既定 0.5。低=元忠実 / 高=SDXL が大きく描き直す)",
                           "--png refine img2img denoise (default 0.5. low=faithful to original / high=SDXL redraws more)"))
    ap.add_argument("--controlnet", type=str, default=None,
                    help=L("ControlNet を固定 (name or stem)", "fix ControlNet (name or stem)"))
    ap.add_argument("--no-controlnet", action="store_true",
                    help=L("ControlNet を完全 OFF (--prompt png でソース PNG があっても使わない)",
                           "disable ControlNet entirely (even if a source PNG is present with --prompt png)"))
    ap.add_argument("--controlnet-strength", type=float, default=0.7,
                    help=L("controlnet_conditioning_scale (既定 0.7)", "controlnet_conditioning_scale (default 0.7)"))
    ap.add_argument("--pose", type=str, default=None,
                    help=L("openpose 用 ソース PNG (絶対パス or 1_0_prompts/NAME)。"
                           "指定すると DWPose 抽出 → openpose ControlNet を強制適用 "
                           "(4_3_SDXL_ControlNet/ に stem に 'openpose'/'_pose' を含むファイルが必要)。"
                           "--prompt mode とは独立 (sentence/auto/png/original 全モードで併用可)",
                           "source PNG for OpenPose (absolute path or 1_0_prompts/NAME). "
                           "When specified, extracts DWPose and forces openpose ControlNet "
                           "(requires a file with 'openpose'/'_pose' in its stem in 4_3_SDXL_ControlNet/). "
                           "Independent of --prompt mode (works with sentence/auto/png/original)"))
    ap.add_argument("--pose-strength", type=float, default=1.0,
                    help=L("--pose 指定時の controlnet_conditioning_scale (既定 1.0、骨格は強めが効く)",
                           "controlnet_conditioning_scale when --pose is specified (default 1.0, stronger works better for skeleton)"))
    ap.add_argument("--gear", choices=["low", "high"], default="high",
                    help=L("low=ラフ (steps 50) / high=本番 (steps 100、既定)",
                           "low=rough (steps 50) / high=production (steps 100, default)"))
    ap.add_argument("--arch", choices=["cuda", "cpu"], default="cuda",
                    help=L("ComfyUI 側 device 切替 (Phase 1 では参考扱い、ComfyUI 起動時に決まる)",
                           "ComfyUI device selection (informational in Phase 1; determined at ComfyUI startup)"))
    ap.add_argument("--version", choices=["auto", "sdxl", "sd15"], default="auto",
                    help=L("checkpoint 抽選プールの絞り込み。auto=SD15+SDXL 統合 (既定) / "
                           "sdxl=4_1 のみ / sd15=3_1 のみ。当選 checkpoint の版で 1 段/2 段が決まる "
                           "(gear high + SD15 当選 → SD15 下書き→SDXL 清書の 2 段)",
                           "narrow the checkpoint draw pool. auto=unified SD15+SDXL (default) / "
                           "sdxl=4_1 only / sd15=3_1 only. The drawn checkpoint's version determines 1-stage or 2-stage "
                           "(gear high + SD15 drawn → SD15 draft→SDXL clean two-stage chain)"))
    ap.add_argument("--chain-denoise", type=float, default=0.45,
                    help=L("gear high で SD15 が当選したときの SDXL 清書 img2img の denoise "
                           "(既定 0.45。低=下書きの構図/色を保持、高=SDXL が描き直す)",
                           "SDXL clean img2img denoise when SD15 is drawn in gear high "
                           "(default 0.45. low=preserves draft composition/color, high=SDXL redraws more)"))
    ap.add_argument("--save-draft", action=argparse.BooleanOptionalAction, default=True,
                    help=L("2 段チェーン時、中間の SD15 下書きを 3_9_SD15_rough に保存 (既定 ON、--no-save-draft で OFF)",
                           "save intermediate SD15 draft to 3_9_SD15_rough during two-stage chain (default ON, --no-save-draft to disable)"))
    ap.add_argument("--checkpoint", type=str, default=None,
                    help=L("checkpoint を固定。NAME or NAME.safetensors",
                           "fix checkpoint. NAME or NAME.safetensors"))
    ap.add_argument("--cfg-scale", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=None,
                    help=L("生成幅 (未指定: sdxl=1024 / sd15=512)", "generation width (default: sdxl=1024 / sd15=512)"))
    ap.add_argument("--height", type=int, default=None,
                    help=L("生成高さ (未指定: sdxl=1024 / sd15=512)", "generation height (default: sdxl=1024 / sd15=512)"))
    ap.add_argument("--many-width", type=int, default=None,
                    help=L("many=true のとき使う幅 (未指定: sdxl=1216 / sd15=768、横長で複数人の融合抑制)",
                           "width when many=true (default: sdxl=1216 / sd15=768, landscape to suppress multi-person merging)"))
    ap.add_argument("--many-height", type=int, default=None,
                    help=L("many=true のとき使う高さ (未指定: sdxl=832 / sd15=512)",
                           "height when many=true (default: sdxl=832 / sd15=512)"))
    ap.add_argument("--many", action="store_true",
                    help=L("複数人モードを強制 ON (横長キャンバスで生成)。--sentence/--png 等 "
                           "prompt.toml 由来でない入力で複数人を描くとき指定。auto モードでは "
                           "who エントリの many 判定と OR で効く",
                           "force multi-person mode ON (generates on landscape canvas). "
                           "Use when drawing multiple people with --sentence/--png or other non-prompt.toml inputs. "
                           "In auto mode, ORed with the who-entry many flag"))
    ap.add_argument("--sampler", type=str, default="dpmpp_2m")
    ap.add_argument("--scheduler", type=str, default="karras")
    ap.add_argument("--lora-scale", type=float, default=0.8,
                    help=L("LoRA n 個重ね掛け時の合計 scale (各 LoRA strength = lora_scale/n、既定 0.8)",
                           "total scale when stacking n LoRAs (each LoRA strength = lora_scale/n, default 0.8)"))
    ap.add_argument("--lora-stack-min", type=int, default=3,
                    help=L("1 枚あたりの重ね掛け LoRA 最小数 (既定 3、1 で「下限 1」)",
                           "minimum number of stacked LoRAs per image (default 3, set 1 for min of 1)"))
    ap.add_argument("--lora-stack-max", type=int, default=5,
                    help=L("1 枚あたりの重ね掛け LoRA 最大数 (random.randint(min, max)、既定 5、"
                           "1 で重ね無し、0 で完全 OFF)",
                           "maximum number of stacked LoRAs per image (random.randint(min, max), default 5, "
                           "1 for no stacking, 0 to disable entirely)"))
    ap.add_argument("--upscale", action=argparse.BooleanOptionalAction, default=None,
                    help=L("Real-ESRGAN x4 アップスケール (5_2_upscaled に出力)。"
                           "既定: gear high で ON / gear low で OFF。明示すれば上書き",
                           "Real-ESRGAN x4 upscale (output to 5_2_upscaled). "
                           "Default: ON for gear high / OFF for gear low. Explicit flag overrides"))
    ap.add_argument("--upscale-model", type=str, default=None,
                    help=L("アップスケール用 Real-ESRGAN モデル名 (既定: style=anime → anime6B、"
                           "real → x4plus、mix/空 → anime6B)",
                           "Real-ESRGAN model name for upscaling (default: style=anime → anime6B, "
                           "real → x4plus, mix/empty → anime6B)"))
    ap.add_argument("--adetailer", action=argparse.BooleanOptionalAction, default=None,
                    help=L("ADetailer (顔/手 YOLO inpainting)。既定: gear high で ON / low で OFF",
                           "ADetailer (face/hand YOLO inpainting). Default: ON for gear high / OFF for low"))
    ap.add_argument("--adetailer-face-model", type=str, default="bbox/face_yolov8n.pt",
                    help=L("ADetailer 顔検出 model (既定 face_yolov8n)", "ADetailer face detection model (default face_yolov8n)"))
    ap.add_argument("--adetailer-hand-model", type=str, default="bbox/hand_yolov8n.pt",
                    help=L("ADetailer 手検出 model (空文字で hand OFF、既定 hand_yolov8n)",
                           "ADetailer hand detection model (empty string to disable hand, default hand_yolov8n)"))
    ap.add_argument("--adetailer-person-model", type=str, default="segm/person_yolov8n-seg.pt",
                    help=L("ADetailer 全身検出 model (空文字で person OFF、既定 person_yolov8n-seg)。"
                           "足/脚の奇形補正に使用、denoise を低めで構造維持",
                           "ADetailer full-body detection model (empty string to disable, default person_yolov8n-seg). "
                           "Used to correct leg/foot artifacts; lower denoise to preserve structure"))
    ap.add_argument("--adetailer-denoise", type=float, default=0.5,
                    help=L("ADetailer (face/hand) inpaint strength (既定 0.5)",
                           "ADetailer (face/hand) inpaint strength (default 0.5)"))
    ap.add_argument("--adetailer-person-denoise", type=float, default=0.3,
                    help=L("ADetailer person inpaint strength (既定 0.3、低めで構造維持)",
                           "ADetailer person inpaint strength (default 0.3, lower to preserve structure)"))
    ap.add_argument("--adetailer-steps", type=int, default=30,
                    help=L("ADetailer 各 detected region のステップ数 (既定 30)",
                           "ADetailer inference steps per detected region (default 30)"))
    ap.add_argument("--hires-fix", action=argparse.BooleanOptionalAction, default=None,
                    help=L("Hires Fix (低解像度→1.5×二段)。draft / 清書段の両方に効く。"
                           "既定: gear high で ON / low で OFF",
                           "Hires Fix (low-res then 1.5× refine pass). Applies to both draft and clean stages. "
                           "Default: ON for gear high / OFF for low"))
    ap.add_argument("--hires-scale", type=float, default=1.5,
                    help=L("Hires Fix のスケール係数 (既定 1.5、512→768 の比率)",
                           "Hires Fix scale factor (default 1.5, matches 512→768)"))
    ap.add_argument("--hires-denoise", type=float, default=0.35,
                    help=L("Hires Fix 2 段目 denoise (既定 0.35、高いと tile/seamless 化)",
                           "Hires Fix 2nd-pass denoise (default 0.35; higher values risk tile/seamless artifacts)"))
    ap.add_argument("--hires-steps", type=int, default=20,
                    help=L("Hires Fix 2 段目 step 数 (既定 20)",
                           "Hires Fix 2nd-pass steps (default 20)"))
    ap.add_argument("--embeddings", action=argparse.BooleanOptionalAction, default=True,
                    help=L("Embedding dir (sdxl=4_4 / sd15=3_2) から負のクオリティ embedding (`*-neg` 等) を negative に自動投入 (既定 ON)",
                           "auto-inject negative quality embeddings (`*-neg` etc.) from embedding dir (sdxl=4_4 / sd15=3_2) into negative (default ON)"))
    ap.add_argument("--quality-prefix", type=str, default="",
                    help=L("positive の先頭に常時前置する quality タグ列 (例 'score_9, score_8_up, score_7_up')。"
                           "checkpoint の系統に合わせて指定。--pony 指定時は Pony 標準値が自動で入る",
                           "quality tag string always prepended to positive (e.g. 'score_9, score_8_up, score_7_up'). "
                           "Match to checkpoint lineage. When --pony is set, Pony defaults are inserted automatically"))
    ap.add_argument("--pony", action="store_true",
                    help=L("Pony 系 checkpoint 用。--quality-prefix 未指定なら "
                           "'score_9, score_8_up, score_7_up, source_anime, rating_explicit' を前置",
                           "for Pony lineage checkpoints. Prepends "
                           "'score_9, score_8_up, score_7_up, source_anime, rating_explicit' if --quality-prefix is not set"))
    ap.add_argument("--cooldown", type=float, default=None,
                    help=L("1 枚生成後の待機秒。既定: GPU 温度 - 50 秒 (温度取れなければ 1.0 秒、--cooldown 0 で OFF)",
                           "cooldown interval in seconds after each image. Default: GPU temp - 50s (1.0s if temp unavailable, --cooldown 0 to disable)"))
    ap.add_argument("--dump-workflow", action="store_true",
                    help=L("投入する API workflow JSON を workflow_dump/ にも保存 (生成は通常通り実行)。"
                           "出力 JSON を ComfyUI WebUI の canvas にドラッグすればグラフを可視化できる",
                           "also save the submitted API workflow JSON to workflow_dump/ (generation runs normally). "
                           "Drag the output JSON onto the ComfyUI WebUI canvas to visualize the graph"))
    ap.add_argument("--dump-only", action="store_true",
                    help=L("workflow JSON を吐くだけで ComfyUI への投入はしない (GPU を使わずグラフ確認)。"
                           "1 枚分の単一パス workflow を吐いて即終了 (chain/refine は無効化)",
                           "dump workflow JSON only without submitting to ComfyUI (graph inspection without GPU). "
                           "Dumps a single-pass workflow for one image then exits immediately (chain/refine disabled)"))
    args = ap.parse_args()

    # --version は checkpoint 抽選プールの絞り込みのみ (auto=両レーン統合)。
    # 当選 checkpoint の版 (置き場 dir) で 1 段 / 2 段が決まるので、ここで lane は固定しない。
    if args.version == "sd15":
        pool_dirs = [SD15_CHECKPOINT_DIR]
    elif args.version == "sdxl":
        pool_dirs = [SDXL_CHECKPOINT_DIR]
    else:  # auto
        pool_dirs = [SD15_CHECKPOINT_DIR, SDXL_CHECKPOINT_DIR]

    # 解像度は当選版で決まる (loop の lane_resolution で per-pick 解決)。
    # 明示があればそれ優先、無ければ sd15=512 / sdxl=1024、many は sd15=768x512 / sdxl=1216x832。
    def lane_resolution(lane: str, many: bool) -> tuple[int, int]:
        if lane == "sd15":
            w, h, mw, mh = args.width or 512, args.height or 512, args.many_width or 768, args.many_height or 512
        else:
            w, h, mw, mh = args.width or 1024, args.height or 1024, args.many_width or 1216, args.many_height or 832
        return (mw, mh) if many else (w, h)

    # quality 前置の確定: --quality-prefix 明示 > --pony 標準値 > 無し
    # quality 前置の方針 (実際の付与は清書/単一段で family を見て per-checkpoint に行う):
    #   明示 --quality-prefix > --pony (全 checkpoint に強制) > family=="pony" の checkpoint に自動
    # SD15 下書き段には前置しない (Pony は SDXL のみ)。
    explicit_prefix = args.quality_prefix.strip()
    if explicit_prefix:
        print(L(f"[quality prefix] 明示 (全 checkpoint): {explicit_prefix}",
                f"[quality prefix] explicit (all checkpoints): {explicit_prefix}"))
    elif args.pony:
        print(L(f"[quality prefix] --pony: 全 checkpoint に Pony score を前置",
                f"[quality prefix] --pony: prepending Pony score to all checkpoints"))
    else:
        print(L(f"[quality prefix] family=pony の checkpoint にのみ Pony score を自動前置",
                f"[quality prefix] auto-prepending Pony score only for family=pony checkpoints"))

    steps = {"low": 50, "high": 100}[args.gear]

    # 入力ソースから mode を確定 (UX): --png=画質アップ refine / --png-sentence=PNG文章生成 / --sentence=文章
    #   --png は最優先 (refine)。--png-sentence は png(文章) だが --prompt original 明示は尊重。
    if args.png:
        args.prompt = "refine"
    elif args.png_sentence:
        if args.prompt not in ("png", "original"):
            args.prompt = "png"
    elif args.sentence and args.prompt == "auto":
        args.prompt = "sentence"

    # アップスケール / ADetailer / Hires Fix 既定 (gear に紐づき、明示で上書き)
    if args.upscale is None:
        args.upscale = (args.gear == "high")
    if args.adetailer is None:
        args.adetailer = (args.gear == "high")
    if args.hires_fix is None:
        args.hires_fix = (args.gear == "high")

    pool_label = {"auto": "SD15+SDXL", "sd15": "SD15", "sdxl": "SDXL"}[args.version]
    print(f"=== generate.py ===")
    print(f"checkpoint pool: {pool_label}  prompt mode: {args.prompt}  "
          f"gear: {args.gear} (steps={steps})  arch: {args.arch}  "
          f"upscale: {args.upscale}  adetailer: {args.adetailer}  "
          f"hires_fix: {args.hires_fix}")
    if args.gear == "high":
        print(L(f"  gear high: SDXL 当選→1パス清書 / SD15 当選→SD15下書き→SDXL清書 "
                f"(img2img denoise {args.chain_denoise})",
                f"  gear high: SDXL drawn→single-pass clean / SD15 drawn→SD15 draft→SDXL clean "
                f"(img2img denoise {args.chain_denoise})"))

    print(f"\n--- tensors triage ---")
    counts = check_tensors()
    print(f"  SDXL: ckpt={counts['checkpoint']} LoRA={counts['lora']} "
          f"embed={counts['embedding']} CN={counts['controlnet']}   "
          f"SD15: ckpt={counts['sd15_checkpoint']} LoRA={counts['sd15_lora']} "
          f"embed={counts['sd15_embedding']}   error={counts['error']}")
    if not _gather_checkpoints(pool_dirs):
        raise SystemExit(L(f"\n抽選プール ({', '.join(d.name for d in pool_dirs)}) に "
                           f"checkpoint がありません。先に 2_0_tensors に投入を",
                           f"\nno checkpoints in draw pool ({', '.join(d.name for d in pool_dirs)}). "
                           f"Place tensors in 2_0_tensors first"))

    print(L(f"\n--- ComfyUI 接続確認 / device 整合 ---", f"\n--- ComfyUI connection check / device match ---"))
    yaml_changed = write_extra_model_paths()  # model dir を ComfyUI に登録 (dir 定数から自動生成)
    ensure_comfyui_arch(args.arch, force_restart=yaml_changed)
    cur_device = get_comfyui_device()
    if cur_device is None:
        raise SystemExit(L(f"ComfyUI に接続できません ({COMFY_BASE})", f"cannot connect to ComfyUI ({COMFY_BASE})"))
    print(f"  OK: {COMFY_BASE} (device={cur_device})")

    client_id = uuid.uuid4().hex
    GENERATED_DIR.mkdir(exist_ok=True)
    if args.upscale:
        UPSCALED_DIR.mkdir(exist_ok=True)

    # checkpoint.toml 連携の state
    checkpoint_data = load_checkpoint_toml()
    pick_state: dict = {}

    # LoRA / neg embedding を両レーン分まとめて準備 (起動時 1 回)。
    # 統合抽選で SD15/SDXL どちらが当選しても、その版の LoRA・embed を使えるようにする。
    lora_keywords_data = load_lora_keywords_toml()
    sdxl_lora_subjects = load_sdxl_lora_subjects()  # {stem: subject}。pose は OpenPose 段で除外

    def _prep_lane(lora_dir: Path, embed_dir: Path) -> dict:
        loras = sorted(lora_dir.glob("*.safetensors")) if args.lora_stack_max > 0 else []
        corpus = build_lora_corpus_for_playground(loras, lora_keywords_data) if loras else {}
        negs = collect_negative_embeddings(embed_dir) if args.embeddings else []
        return {"loras": loras, "corpus": corpus, "neg": negs, "lora_dir": lora_dir}

    lane_assets = {
        "sdxl": _prep_lane(SDXL_LORA_DIR, SDXL_EMBED_DIR),
        "sd15": _prep_lane(SD15_LORA_DIR, SD15_EMBED_DIR),
    }
    for ln in ("sdxl", "sd15"):
        a = lane_assets[ln]
        print(L(f"  [{ln}] LoRA 候補: {len(a['loras'])} 件 / neg embed: {len(a['neg'])} 件",
                f"  [{ln}] LoRA candidates: {len(a['loras'])} / neg embed: {len(a['neg'])}"))

    # ADetailer モデルを起動時 1 回 resolve (無ければ HF から DL、不可なら無効化)
    face_model = person_model = hand_model = None
    if args.adetailer:
        face_model    = ensure_adetailer_model(args.adetailer_face_model)
        hand_model    = ensure_adetailer_model(args.adetailer_hand_model or None)
        person_model  = ensure_adetailer_model(args.adetailer_person_model or None)
        if not face_model:
            print(L("  [adetailer][warn] face model が無く DL も不可 → ADetailer 全体を OFF",
                    "  [adetailer][warn] face model missing and download failed → disabling ADetailer entirely"), flush=True)
            args.adetailer = False

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
        print(L("\n[Ctrl+C] 中断要求 (現在の生成完了後に終了)",
                "\n[Ctrl+C] stop requested (will exit after current generation finishes)"), flush=True)
    signal.signal(signal.SIGINT, handler)

    total = 0
    while not stop["flag"]:
        try:
            iter_start = time.time()

            # ソース PNG 解決: refine=--png (画像を使う) / png,original=--png-sentence (文章を使う)
            src_png: Optional[Path] = None
            if args.prompt == "refine":
                if not args.png:
                    raise SystemExit(L("--png <PNG> が必要", "--png <PNG> is required"))
                src_png = resolve_png_path(args.png)
            elif args.prompt in ("png", "original"):
                if not args.png_sentence:
                    raise SystemExit(L(f"--prompt {args.prompt} には --png-sentence <PNG> が必要",
                                       f"--prompt {args.prompt} requires --png-sentence <PNG>"))
                src_png = resolve_png_path(args.png_sentence)

            positive, negative, lora_keywords, extras = get_prompt_for_iteration(
                args.prompt, src_png, args.sentence, args.lora_keywords,
            )
            is_refine = bool(extras.get("refine"))  # 画質アップ (PNG を init に SDXL img2img、1枚)
            # quality 前置は清書/単一段 (SDXL) で family を見て付与する (下書き段には付けない)。

            # checkpoint 抽選: refine は SDXL 専用プール (画質アップ目的)、他は --version プール
            fixed_checkpoint = args.checkpoint
            if "model" in extras:
                fixed_checkpoint = extras["model"]
            iter_pool = [SDXL_CHECKPOINT_DIR] if is_refine else pool_dirs
            checkpoint_path = pick_checkpoint(checkpoint_data, pick_state, fixed_checkpoint, pool_dirs=iter_pool)
            ckpt_lane = checkpoint_version(checkpoint_path)  # 当選版 (3_1=sd15 / 4_1=sdxl)
            seed = random.randint(0, 2**32 - 1)
            many = bool(extras.get("many") or args.many)

            print(f"\n=== source {total+1} ===")

            # ---- 2 段チェーン: gear high + SD15 当選 → SD15 下書き → SDXL 清書 ----
            # --dump-only は下書き生成 (GPU) を避けたいので chain を切り単一パス build にする
            chain = (args.gear == "high" and ckpt_lane == "sd15"
                     and not is_refine and not args.dump_only)
            init_image_name: Optional[str] = None
            draft_checkpoint_meta: Optional[str] = None      # 2段時の SD15 下書き checkpoint (メタ記録用)
            draft_loras_meta: list[tuple[str, float]] = []   # 2段時の SD15 下書き LoRA (メタ記録用)
            if is_refine:
                # 画質アップ: PNG をそのまま init に SDXL img2img (チェーンの清書段だけを単体実行)
                init_image_name = upload_image_to_comfyui(src_png)
                pipeline_label = f"refine (src: {src_png.stem}, denoise {args.refine_denoise})"
                print(L(f"  [refine] {src_png.name} を SDXL img2img で画質アップ "
                        f"(denoise {args.refine_denoise})",
                        f"  [refine] quality-up {src_png.name} via SDXL img2img "
                        f"(denoise {args.refine_denoise})"), flush=True)
            elif chain:
                draft_ckpt = checkpoint_path
                sd15a = lane_assets["sd15"]
                d_w, d_h = lane_resolution("sd15", many)
                d_loras: list[tuple[Path, float]] = []
                if sd15a["loras"] and args.lora_stack_max > 0:
                    dp = pick_n_loras_by_keywords(sd15a["loras"], lora_keywords, sd15a["corpus"],
                                                  n_max=args.lora_stack_max, n_min=args.lora_stack_min)
                    if dp:
                        ds = args.lora_scale / len(dp)
                        d_loras = [(p, ds) for p in dp]
                draft_checkpoint_meta = draft_ckpt.name
                draft_loras_meta = [(p.name, s) for p, s in d_loras]
                d_steps = max(1, steps + int(checkpoint_data.get(draft_ckpt.stem, {}).get("inference", 0)))
                d_pos = augment_positive_with_lora_keywords(positive, lora_keywords)
                d_neg = augment_negative_with_embeddings(negative, gate_neg_embeddings(sd15a["neg"], False))
                lk = f"  LoRA x{len(d_loras)}" if d_loras else ""
                print(L(f"  [chain] SD15 下書き生成中: {draft_ckpt.name} ({d_w}x{d_h}){lk}",
                        f"  [chain] generating SD15 draft: {draft_ckpt.name} ({d_w}x{d_h}){lk}"), flush=True)
                draft_wf = build_workflow_txt2img(
                    checkpoint=draft_ckpt.name, positive=d_pos, negative=d_neg,
                    seed=seed, steps=d_steps, cfg=args.cfg_scale, width=d_w, height=d_h,
                    sampler_name=args.sampler, scheduler=args.scheduler,
                    loras=[(p.name, s) for p, s in d_loras],
                    filename_prefix="playground_draft",
                    hires_fix=args.hires_fix, hires_scale=args.hires_scale,
                    hires_denoise=args.hires_denoise, hires_steps=args.hires_steps,
                )
                if args.dump_workflow:
                    _dump_workflow(draft_wf, "draft_sd15")
                d_bytes, _d_info, _ = _submit_and_fetch(draft_wf, client_id)
                if d_bytes is None:
                    print(L("  [warn] 下書き生成に失敗、この枚をスキップ",
                            "  [warn] draft generation failed, skipping this image"), flush=True)
                    continue
                if args.save_draft:
                    SD15_ROUGH_DIR.mkdir(exist_ok=True)
                    dts = datetime.now().strftime("%Y%m%d%H%M%S")
                    draft_path = SD15_ROUGH_DIR / f"{dts}_draft.png"
                    # ラフ画にも A1111 互換メタを残す (失敗解析・gallery で内容確認しやすく)。
                    # 解像度は Hires Fix on のとき base × scale が実画像サイズ
                    d_final_w = int(d_w * args.hires_scale) if args.hires_fix else d_w
                    d_final_h = int(d_h * args.hires_scale) if args.hires_fix else d_h
                    save_with_a1111_metadata(
                        d_bytes, draft_path,
                        positive=d_pos, negative=d_neg, seed=seed,
                        steps=d_steps, cfg=args.cfg_scale,
                        sampler=args.sampler, scheduler=args.scheduler,
                        width=d_final_w, height=d_final_h,
                        checkpoint=draft_ckpt.name,
                        lora_keywords=lora_keywords,
                        loras=[(p.name, s) for p, s in d_loras],
                        pipeline="SD15 draft (chain 1st pass)",
                    )
                    print(L(f"  下書き保存: 3_9_SD15_rough/{dts}_draft.png",
                            f"  draft saved: 3_9_SD15_rough/{dts}_draft.png"))
                init_image_name = upload_bytes_to_comfyui(d_bytes, f"draft_{seed}.png")
                # 清書は SDXL を別途抽選 → 以降は SDXL stage として通常 body を実行
                checkpoint_path = pick_checkpoint(checkpoint_data, pick_state, args.checkpoint,
                                                  pool_dirs=[SDXL_CHECKPOINT_DIR])
                ckpt_lane = "sdxl"
                # pipeline_label は PNG メタにも書かれる。メタは英語に統一するため英語固定 (L で切替えない)。
                pipeline_label = (f"SD15→SDXL 2-stage (draft: {draft_ckpt.stem} / "
                                  f"clean: {checkpoint_path.stem}, denoise {args.chain_denoise})")
            else:
                pipeline_label = f"{ckpt_lane.upper()} single-pass"

            # ---- 清書 / 単一パス stage (active lane = ckpt_lane の資産を使う) ----
            active = lane_assets[ckpt_lane]
            gen_width, gen_height = lane_resolution(ckpt_lane, many)
            if is_refine:
                # 元 PNG のアスペクトを保ったまま ~1MP (SDXL ネイティブ) にスケール
                from PIL import Image as _PILImage
                with _PILImage.open(src_png) as _im:
                    _ow, _oh = _im.size
                _sc = (1024 * 1024 / max(1, _ow * _oh)) ** 0.5
                gen_width  = max(512, round(_ow * _sc / 8) * 8)
                gen_height = max(512, round(_oh * _sc / 8) * 8)
            entry = checkpoint_data.get(checkpoint_path.stem, {})
            inference_bonus = int(entry.get("inference", 0))
            use_steps = max(1, steps + inference_bonus)

            # LoRA: original モードは PNG 由来 (chain 時は不可)、それ以外は active lane で keyword 抽選
            picked_loras: list[tuple[Path, float]] = []
            if "loras" in extras and not chain:
                for name, strength in extras["loras"]:
                    cand = active["lora_dir"] / name
                    if not cand.exists():
                        cand2 = active["lora_dir"] / f"{name}.safetensors" if not name.endswith(".safetensors") else cand
                        if cand2.exists():
                            cand = cand2
                        else:
                            print(L(f"  [warn] PNG メタの LoRA が見つからない: {name}、スキップ",
                            f"  [warn] LoRA from PNG metadata not found: {name}, skipping"), flush=True)
                            continue
                    picked_loras.append((cand, float(strength)))
            elif args.gear == "high" and active["loras"] and args.lora_stack_max > 0:
                picked = pick_n_loras_by_keywords(
                    active["loras"], lora_keywords, active["corpus"],
                    n_max=args.lora_stack_max, n_min=args.lora_stack_min,
                )
                if picked:
                    n = len(picked)
                    strength = args.lora_scale / n
                    picked_loras = [(p, strength) for p in picked]

            # ControlNet 抽選 (SDXL stage のみ。SD15 stage に SDXL CN は載らない)
            picked_controlnet: Optional[Path] = None
            controlnet_mode = "passthrough"
            controlnet_upload_name: Optional[str] = None
            effective_cn_strength = args.controlnet_strength
            if args.gear == "high" and not args.no_controlnet and ckpt_lane == "sdxl" and not is_refine:
                if pose_upload_name is not None:
                    picked_controlnet = pick_controlnet("", args.controlnet, force_openpose=True)
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
                            print(L(f"  [warn] ControlNet 用画像 upload 失敗 ({e})、CN OFF",
                            f"  [warn] ControlNet source image upload failed ({e}), CN OFF"), flush=True)
                            picked_controlnet = None

            # OpenPose 有効時は pose 系 LoRA を除外 (姿勢の取り合い回避、SDXL_LoRA_hint.toml subject=pose)
            if controlnet_mode == "openpose" and picked_loras:
                dropped_pose = [p.name for p, _ in picked_loras
                                if sdxl_lora_subjects.get(p.stem) == "pose"]
                if dropped_pose:
                    picked_loras = [(p, s) for p, s in picked_loras
                                    if sdxl_lora_subjects.get(p.stem) != "pose"]
                    print(L(f"  [pose-gate] OpenPose 有効 → pose LoRA 除外: {', '.join(dropped_pose)}",
                            f"  [pose-gate] OpenPose active → dropping pose LoRAs: {', '.join(dropped_pose)}"))

            # family ゲート: 非 Pony base → Pony LoRA を除外。
            # Pony LoRA は Pony Diffusion XL 専用 (score_9 系前提) で、Illustrious/realistic
            # SDXL に乗せると subject が崩壊して人物が消える。embedding と同じ非対称ガード。
            if picked_loras and ckpt_lane == "sdxl" \
                    and not checkpoint_is_pony(checkpoint_path, checkpoint_data):
                dropped_pony = [p.name for p, _ in picked_loras if _is_pony_name(p.stem)]
                if dropped_pony:
                    picked_loras = [(p, s) for p, s in picked_loras
                                    if not _is_pony_name(p.stem)]
                    print(L(f"  [family-gate] 非 Pony base → Pony LoRA 除外: {', '.join(dropped_pony)}",
                            f"  [family-gate] non-Pony base → dropping Pony LoRAs: {', '.join(dropped_pony)}"))

            print(f"  path      : {pipeline_label}")
            print(f"  checkpoint: {checkpoint_path.name} ({ckpt_lane.upper()})"
                  f"{L(' (未計測)', ' (unscored)') if checkpoint_path.stem not in checkpoint_data else ''}")
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
            if many:
                print(L(f"  size      : {gen_width}x{gen_height} (many 横長)",
                        f"  size      : {gen_width}x{gen_height} (many landscape)"))
            print(f"  seed/steps: {seed} / {use_steps}"
                  f"{f' (= {steps} + inference {inference_bonus:+})' if inference_bonus else ''}"
                  f"{f' / img2img denoise {args.chain_denoise}' if chain else ''}")

            # アップスケールモデル選択: --upscale-model 指定 → そのまま、未指定 → style ベース
            upscale_model_name: Optional[str] = None
            if args.upscale:
                if args.upscale_model:
                    upscale_model_name = args.upscale_model
                else:
                    style = (entry.get("style") or "").strip().lower()
                    upscale_model_name = _UPSCALE_MODEL_BY_STYLE.get(style, _UPSCALE_MODEL_DEFAULT)
                # 必要なら HF release から自動 DL。失敗時は None で upscale 段を無効化 (workflow が落ちないように)
                upscale_model_name = ensure_upscale_model(upscale_model_name)

            # この段 (清書/単一パス = checkpoint_path) の系統を確定 (family タグ優先・無ければ名前)
            ckpt_is_pony = checkpoint_is_pony(checkpoint_path, checkpoint_data)

            # quality 前置: 明示 > --pony > family==pony 自動。SD15 下書き段には付けない (ここは清書/単一段)
            eff_prefix = explicit_prefix
            if not eff_prefix and (args.pony or ckpt_is_pony):
                eff_prefix = PONY_SCORE_PREFIX
            pos_for_stage = positive
            if eff_prefix and not positive.lower().startswith(eff_prefix.split(",")[0].strip().lower()):
                pos_for_stage = f"{eff_prefix}, {positive}" if positive else eff_prefix
                src = L("明示", "explicit") if explicit_prefix else ("--pony" if args.pony else "family=pony")
                print(f"  [score] {src} → {eff_prefix.split(',')[0].strip()}…")

            # LoRA キーワードを (0.8/N) 重みで positive に append (README L193 仕様)
            positive_augmented = augment_positive_with_lora_keywords(pos_for_stage, lora_keywords)
            if positive_augmented != pos_for_stage:
                print(f"  prompt+kw : ...{positive_augmented[len(pos_for_stage):][:80]}")

            # 負のクオリティ embedding を negative に追加 (active lane の embed)。
            # Pony 専用 embed は Pony checkpoint のときだけ通す (非 Pony への誤爆 = 色崩壊/主体消失を防ぐ)。
            gated_neg = gate_neg_embeddings(active["neg"], ckpt_is_pony)
            if len(gated_neg) != len(active["neg"]):
                dropped = [e for e in active["neg"] if e not in gated_neg]
                why = L("Pony embed 上限3超過", "Pony embed cap exceeded (3)") if ckpt_is_pony else L("非Pony → Pony embed", "non-Pony → Pony embed")
                print(L(f"  [neg-gate] {why} 除外: {', '.join(dropped)}",
                        f"  [neg-gate] {why} excluded: {', '.join(dropped)}"))
            negative_augmented = augment_negative_with_embeddings(negative, gated_neg)

            workflow_loras = [(p.name, s) for p, s in picked_loras]
            workflow = build_workflow_txt2img(
                checkpoint=checkpoint_path.name,
                positive=positive_augmented, negative=negative_augmented,
                seed=seed, steps=use_steps, cfg=args.cfg_scale,
                width=gen_width, height=gen_height,
                sampler_name=args.sampler, scheduler=args.scheduler,
                init_image=init_image_name,                          # chain/refine 時: init 画像 (img2img)
                denoise=(args.refine_denoise if is_refine else (args.chain_denoise if chain else 1.0)),
                loras=workflow_loras,
                controlnet_name=picked_controlnet.name if picked_controlnet else None,
                controlnet_mode=controlnet_mode,
                controlnet_image=controlnet_upload_name,
                controlnet_strength=effective_cn_strength,
                upscale_model=upscale_model_name,
                adetailer=args.adetailer,
                adetailer_face_model=face_model,
                adetailer_hand_model=hand_model,
                adetailer_person_model=person_model,
                adetailer_denoise=args.adetailer_denoise,
                adetailer_person_denoise=args.adetailer_person_denoise,
                adetailer_steps=args.adetailer_steps,
                hires_fix=args.hires_fix, hires_scale=args.hires_scale,
                hires_denoise=args.hires_denoise, hires_steps=args.hires_steps,
            )
            if args.adetailer:
                parts = [f"face={face_model}"]
                if hand_model:
                    parts.append(f"hand={hand_model}")
                if person_model:
                    parts.append(f"person={person_model}@{args.adetailer_person_denoise}")
                print(f"  ADetailer: {', '.join(parts)}"
                      f" (denoise={args.adetailer_denoise}, steps={args.adetailer_steps})")
            if upscale_model_name:
                print(f"  upscale: {upscale_model_name}")

            if args.dump_workflow or args.dump_only:
                kind = "refine" if is_refine else ("chain_clean" if chain else f"{ckpt_lane}_single")
                _dump_workflow(workflow, kind)
            if args.dump_only:
                print(L("  [dump-only] ComfyUI への投入はスキップ。"
                        "上記 JSON を WebUI canvas にドロップしてグラフ確認",
                        "  [dump-only] skipping ComfyUI submission. "
                        "Drop the JSON above onto the WebUI canvas to inspect the graph"), flush=True)
                break

            prompt_id = submit_prompt(workflow, client_id)
            print(f"  ComfyUI prompt_id: {prompt_id}")

            result = wait_for_completion_ws(prompt_id, client_id)

            outputs = result.get("outputs", {})
            # node 7 = 通常解像度 (5_1_generated)
            save_node = outputs.get("7", {})
            images = save_node.get("images", [])
            if not images:
                print(L(f"  [warn] 出力画像が見つからない、スキップ",
                        f"  [warn] output image not found, skipping"))
                continue
            img_info = images[0]
            img_bytes = fetch_image(img_info["filename"],
                                     img_info.get("subfolder", ""),
                                     img_info.get("type", "output"))

            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            out_path = GENERATED_DIR / f"{ts}.png"
            # 画像の最終解像度 (Hires Fix on のとき base × scale)。メタ Size はここを書く
            final_w = int(gen_width * args.hires_scale) if args.hires_fix else gen_width
            final_h = int(gen_height * args.hires_scale) if args.hires_fix else gen_height
            save_with_a1111_metadata(
                img_bytes, out_path,
                positive=positive_augmented, negative=negative_augmented, seed=seed,
                steps=use_steps, cfg=args.cfg_scale,
                sampler=args.sampler, scheduler=args.scheduler,
                width=final_w, height=final_h,
                checkpoint=checkpoint_path.name,
                lora_keywords=lora_keywords,
                loras=[(p.name, s) for p, s in picked_loras],
                controlnet_name=picked_controlnet.name if picked_controlnet else None,
                controlnet_mode=controlnet_mode,
                controlnet_strength=effective_cn_strength,
                pose_source=pose_png.name if pose_png else None,
                adetailer=args.adetailer,
                adetailer_person=bool(person_model) and args.adetailer,
                pipeline=pipeline_label,
                draft_checkpoint=draft_checkpoint_meta,
                draft_loras=draft_loras_meta,
            )
            elapsed = time.time() - iter_start
            total += 1
            print(f"  → {out_path.name}  {final_w}x{final_h}  ({elapsed:.1f}s)")
            if is_refine:
                stop["flag"] = True  # --png refine は 1 枚で終了 (この後の upscale 保存まではやる)

            # node 14 = アップスケール後 (5_2_upscaled)
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
                        width=final_w * 4, height=final_h * 4,
                        checkpoint=checkpoint_path.name,
                        lora_keywords=lora_keywords,
                        loras=[(p.name, s) for p, s in picked_loras],
                        controlnet_name=picked_controlnet.name if picked_controlnet else None,
                        controlnet_mode=controlnet_mode,
                        controlnet_strength=effective_cn_strength,
                        pose_source=pose_png.name if pose_png else None,
                        adetailer=args.adetailer,
                        adetailer_person=bool(person_model) and args.adetailer,
                                pipeline=pipeline_label,
                        draft_checkpoint=draft_checkpoint_meta,
                        draft_loras=draft_loras_meta,
                    )
                    print(f"      up → 5_2_upscaled/{up_path.name}  {gen_width*4}x{gen_height*4} ({upscale_model_name})")
                else:
                    print(L(f"  [warn] アップスケール出力が見つからない",
                            f"  [warn] upscaled output not found"))

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
                        print(L(f"  cooldown: GPU {temp}°C → {wait_s:.0f}s 待機",
                                f"  cooldown: GPU {temp}°C → waiting {wait_s:.0f}s"))
                    time.sleep(wait_s)

        except Exception as e:
            print(L(f"\n[エラー] {e}", f"\n[error] {e}"), flush=True)
            if not stop["flag"]:
                time.sleep(5.0)

    print(L(f"\n総計: {total} 枚", f"\ntotal: {total} image(s)"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(L("\n中断", "\ninterrupted"))
        sys.exit(0)
