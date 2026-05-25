#!/usr/bin/env python3
"""tensors.py - `2_0_tensors` 受入トレーのテンソルを分類して各 dir へ振り分ける CLI。

処理内容 (idempotent、何度実行しても安全):
    - 2_0_tensors の zip 展開 / ckpt → safetensors 変換
    - hash 取得 + 重複検出 (mtime 古い方を 2_1_errortensors へ)
    - classify_tensor で base / LoRA / Embedding / ControlNet / inpainting / broken を判別
    - SDXL → 4_1_SDXL_checkpoint / 4_2_SDXL_LoRA / 4_4_SDXL_Embedding (本番レーン)
    - SD15  → 3_1_SD15_checkpoint / 3_3_SD15_LoRA / 3_2_SD15_Embedding (rough/量産レーン)
    - ControlNet → 4_3_SDXL_ControlNet
    - broken / inpainting / 判別不能 → 2_1_errortensors
    - キャッシュは tensors_cache.toml (path: {size, mtime, hash, kind, version})

policy:
    - 4_3_SDXL_ControlNet と 2_1_errortensors は **scan しない** (手動配置を尊重、再分類しない)
    - SD15 / SDXL の base・LoRA・Embedding dir は scan する (直接投入分も dedup + 再分類)

使い方:
    python tensors.py
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore
import tomli_w

from common import (
    L,
    classify_tensor,
    convert_to_safetensors,
    detect_controlnet_version,
    detect_embedding_version,
    detect_sd_version,
    file_sha256,
    lora_target_version,
)

# Windows console 絵文字落ち防止
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# --------------------------------------------------------------------------- #
# dir 構成
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent
TENSORS_DIR    = ROOT / "2_0_tensors"        # 受入トレー (ユーザがここに投入)
ERROR_DIR      = ROOT / "2_1_errortensors"   # 不正 / 破損 / inpainting
# --- SD15 レーン (rough/量産) ---
SD15_CKPT_DIR  = ROOT / "3_1_SD15_checkpoint" # SD15 base
SD15_LORA_DIR  = ROOT / "3_2_SD15_LoRA"       # SD15 LoRA
SD15_EMBED_DIR = ROOT / "3_3_SD15_Embedding"  # SD15 Embedding
# --- SDXL レーン (本番) ---
CHECKPOINT_DIR = ROOT / "4_1_SDXL_checkpoint" # SDXL base
LORA_DIR       = ROOT / "4_2_SDXL_LoRA"       # SDXL LoRA
CONTROLNET_DIR = ROOT / "4_3_SDXL_ControlNet" # ControlNet (手動配置、scan しない)
EMBED_DIR      = ROOT / "4_4_SDXL_Embedding"  # SDXL Embedding

CACHE_FILE = ROOT / "tensors_cache.toml"

TENSOR_EXTS = (".safetensors", ".ckpt", ".pt", ".bin")


# --------------------------------------------------------------------------- #
# cache I/O (TOML)
# --------------------------------------------------------------------------- #
def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return tomllib.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(tomli_w.dumps(cache), encoding="utf-8")
    except Exception as e:
        print(L(f"キャッシュ保存失敗: {e}", f"Cache save failed: {e}"), flush=True)


# --------------------------------------------------------------------------- #
# 分類先解決
# --------------------------------------------------------------------------- #
def _classified_destination(kind: str, entry: dict) -> Path:
    """kind + version 情報から振り分け先 dir を返す。"""
    if kind in ("broken", "inpainting"):
        return ERROR_DIR
    if kind == "controlnet":
        return CONTROLNET_DIR
    if kind == "base":
        ver = entry.get("base_version")
        if ver == "sdxl":
            return CHECKPOINT_DIR
        if ver == "sd15":
            return SD15_CKPT_DIR
        return ERROR_DIR
    if kind == "lora":
        ver = entry.get("lora_version")
        if ver == "sdxl":
            return LORA_DIR
        if ver == "sd15":
            return SD15_LORA_DIR
        return ERROR_DIR
    if kind == "embedding":
        ver = entry.get("embedding_version")
        if ver == "sdxl":
            return EMBED_DIR
        if ver == "sd15":
            return SD15_EMBED_DIR
        return ERROR_DIR
    return ERROR_DIR


# --------------------------------------------------------------------------- #
# 補助
# --------------------------------------------------------------------------- #
def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}PB"


def _is_torch_zip(path: Path) -> bool:
    """torch.save のアーカイブ (data.pkl 入り) は内部 zip。配下に data.pkl があれば真。"""
    try:
        with zipfile.ZipFile(path) as zf:
            return any(n.endswith("data.pkl") or n.endswith("/data.pkl") for n in zf.namelist())
    except Exception:
        return False


def _safe_extract_tensors(zip_path: Path, dest: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name or not name.lower().endswith(TENSOR_EXTS):
                continue
            out = dest / name
            stem, suffix = out.stem, out.suffix
            i = 2
            while out.exists():
                out = dest / f"{stem}_{i}{suffix}"
                i += 1
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


# --------------------------------------------------------------------------- #
# メイン振り分け
# --------------------------------------------------------------------------- #
def check_tensors() -> dict:
    """テンソル振り分けを実行し、各 dir の件数を返す。"""
    for d in (TENSORS_DIR, ERROR_DIR,
              SD15_CKPT_DIR, SD15_EMBED_DIR, SD15_LORA_DIR,
              CHECKPOINT_DIR, LORA_DIR, CONTROLNET_DIR, EMBED_DIR):
        d.mkdir(exist_ok=True)
    cache = load_cache()

    # ---- Phase 1: zip 展開 / ckpt 変換 (受入トレー内のみ) ----
    candidates = sorted([
        *TENSORS_DIR.glob("*.zip"),
        *(p for p in TENSORS_DIR.iterdir()
          if p.is_file() and p.suffix.lower() not in TENSOR_EXTS + (".zip", ".txt", ".toml", ".md")),
    ])
    for path in candidates:
        if not zipfile.is_zipfile(path) or _is_torch_zip(path):
            continue
        print(f"[zip] {path.name} ({_human_size(path.stat().st_size)})", flush=True)
        try:
            n = _safe_extract_tensors(path, TENSORS_DIR)
            print(L(f"  展開 {n} 件 → 2_0_tensors / 元 zip → 2_1_errortensors",
                    f"  extracted {n} file(s) → 2_0_tensors / original zip → 2_1_errortensors"), flush=True)
            shutil.move(str(path), str(ERROR_DIR / path.name))
        except Exception as e:
            print(L(f"  展開失敗 ({e}) → 2_1_errortensors",
                    f"  extraction failed ({e}) → 2_1_errortensors"), flush=True)
            shutil.move(str(path), str(ERROR_DIR / path.name))

    for path in sorted([*TENSORS_DIR.glob("*.ckpt"), *TENSORS_DIR.glob("*.pt"), *TENSORS_DIR.glob("*.bin")]):
        print(L(f"[変換] {path.name} ({_human_size(path.stat().st_size)})",
                f"[convert] {path.name} ({_human_size(path.stat().st_size)})"), flush=True)
        try:
            convert_to_safetensors(path)
            path.unlink()
        except Exception as e:
            print(L(f"  変換失敗 ({e}) → 2_1_errortensors",
                    f"  conversion failed ({e}) → 2_1_errortensors"), flush=True)
            shutil.move(str(path), str(ERROR_DIR / path.name))

    # ---- Phase 2a: scan + hash ----
    # ControlNet (4_3) / error (2_1) は scan しない (手動配置 / 退避先)。
    # SD15 / SDXL の base・LoRA・Embedding dir は scan し、直接投入分も dedup + 再分類する。
    scan_dirs = [TENSORS_DIR,
                 SD15_CKPT_DIR, SD15_LORA_DIR, SD15_EMBED_DIR,
                 CHECKPOINT_DIR, LORA_DIR, EMBED_DIR]
    all_files: list[Path] = []
    for d in scan_dirs:
        all_files.extend(d.glob("*.safetensors"))
    all_files.sort()

    hash_groups: dict[str, list[tuple[Path, str, dict, str]]] = {}
    for st in all_files:
        key = str(st).replace("\\", "/")
        stat = st.stat()
        entry = dict(cache.get(key) or {})
        cached_hit = (
            entry.get("size") == stat.st_size
            and abs(float(entry.get("mtime", 0)) - stat.st_mtime) < 1e-3
            and "hash" in entry and "kind" in entry
        )
        if cached_hit:
            digest = entry["hash"]
            kind = entry["kind"]
        else:
            print(f"[hash] {st.name} ({_human_size(stat.st_size)})", flush=True)
            try:
                digest = file_sha256(st)
            except Exception as e:
                print(L(f"  ハッシュ失敗 ({e}) → 2_9_error",
                        f"  hash failed ({e}) → 2_9_error"), flush=True)
                shutil.move(str(st), str(ERROR_DIR / st.name))
                cache.pop(key, None)
                continue
            kind = classify_tensor(st)
            entry = {"size": stat.st_size, "mtime": stat.st_mtime, "hash": digest, "kind": kind}
            cache[key] = entry
        hash_groups.setdefault(digest, []).append((st, kind, entry, key))

    # ---- Phase 2b: 重複解決 (mtime 最新を残し、古いものを 2_9_error へ) ----
    survivors: list[tuple[Path, str, dict, str]] = []
    for digest, group in hash_groups.items():
        if len(group) == 1:
            survivors.append(group[0])
            continue
        group_sorted = sorted(
            group,
            key=lambda item: item[0].stat().st_mtime if item[0].exists() else 0,
            reverse=True,
        )
        keeper = group_sorted[0]
        for old_item in group_sorted[1:]:
            old_path, _, _, old_key = old_item
            print(L(f"  重複: {old_path.name} (古い、mtime={old_path.stat().st_mtime:.0f}) "
                    f"← keeper={keeper[0].name} → 2_9_error",
                    f"  duplicate: {old_path.name} (older, mtime={old_path.stat().st_mtime:.0f}) "
                    f"← keeper={keeper[0].name} → 2_9_error"), flush=True)
            try:
                shutil.move(str(old_path), str(ERROR_DIR / old_path.name))
            except Exception as e:
                print(L(f"    移動失敗: {e}", f"    move failed: {e}"), flush=True)
            cache.pop(old_key, None)
        survivors.append(keeper)

    # ---- Phase 2c: 生き残りを分類 + 移動 ----
    for st, kind, entry, key in survivors:
        # version 判定をキャッシュに焼く
        if kind == "lora" and "lora_version" not in entry:
            entry["lora_version"] = lora_target_version(st)
            cache[key] = entry
        if kind == "base" and "base_version" not in entry:
            try:
                entry["base_version"] = detect_sd_version(st)
            except Exception:
                entry["base_version"] = "unknown"
            cache[key] = entry
        if kind == "embedding" and "embedding_version" not in entry:
            try:
                entry["embedding_version"] = detect_embedding_version(st)
            except Exception:
                entry["embedding_version"] = None
            cache[key] = entry
        if kind == "controlnet" and "controlnet_version" not in entry:
            try:
                entry["controlnet_version"] = detect_controlnet_version(st)
            except Exception:
                entry["controlnet_version"] = None
            cache[key] = entry

        target_dir = _classified_destination(kind, entry)
        if st.parent != target_dir:
            new_path = target_dir / st.name
            try:
                shutil.move(str(st), str(new_path))
                new_key = str(new_path).replace("\\", "/")
                cache[new_key] = cache.pop(key, entry)
                key = new_key
                print(f"  {st.name}: {kind} → {target_dir.name}", flush=True)
            except Exception as e:
                print(L(f"  移動失敗 ({st.name} → {target_dir.name}): {e}",
                        f"  move failed ({st.name} → {target_dir.name}): {e}"), flush=True)
                continue

        # ERROR_DIR に行ったものは cache から外す (再分類対象にしない、ユーザ手動運用)
        if target_dir == ERROR_DIR:
            cache.pop(key, None)

    # 存在しないパスのエントリは GC
    for k in list(cache.keys()):
        if not Path(k).exists():
            del cache[k]
    # TOML 化のために None を含むエントリを掃除 (TOML は None を保存できない)
    for k, v in list(cache.items()):
        if isinstance(v, dict):
            cache[k] = {kk: vv for kk, vv in v.items() if vv is not None}
    save_cache(cache)

    # SDXL LoRA の種別表 (subject) を更新 (ユーザ記入分は保持、ヒントは再生成)
    update_sdxl_lora_toml()

    return {
        "checkpoint":      _count(CHECKPOINT_DIR),
        "lora":            _count(LORA_DIR),
        "embedding":       _count(EMBED_DIR),
        "controlnet":      _count(CONTROLNET_DIR),
        "sd15_checkpoint": _count(SD15_CKPT_DIR),
        "sd15_lora":       _count(SD15_LORA_DIR),
        "sd15_embedding":  _count(SD15_EMBED_DIR),
        "error":           _count(ERROR_DIR),
    }


def _count(d: Path) -> int:
    return len(list(d.glob("*.safetensors"))) if d.exists() else 0


# --------------------------------------------------------------------------- #
# SDXL_LoRA.toml (SDXL LoRA の種別 subject。pose のみ機能的=OpenPose ゲート用)
# --------------------------------------------------------------------------- #
SDXL_LORA_TOML = ROOT / "SDXL_LoRA.toml"
_LORA_HINT_NOISE = {"v1", "v2", "v3", "v4", "v5", "v6", "v10", "v20", "v30", "v50",
                    "sdxl", "xl", "lora", "il", "ill", "pony", "fp16", "bf16", ""}


def _lora_hint(stem: str) -> str:
    """ファイル名から整理ヒント語を抽出 (subject 記入の手がかり: 物? アクセサリ? 等)。"""
    import re
    seen: list[str] = []
    for w in re.split(r"[ _\-.,()@\[\]]+", stem):
        if not w or w.isdigit() or w.lower() in _LORA_HINT_NOISE:
            continue
        if w not in seen:
            seen.append(w)
    return ", ".join(seen)[:120]


def update_sdxl_lora_toml() -> int:
    """4_2_SDXL_LoRA の各 LoRA に `subject` を持つ SDXL_LoRA.toml を生成/更新。

    - `subject` はユーザ記入 (object/accessory/ware/facial/pose 等)。**既存値は保持**。
    - 行末 `# hint:` はファイル名由来の自動ヒント (毎回再生成・編集不要)。
    - 機能的に意味を持つのは `subject="pose"` のみ (generate.py が OpenPose 段で除外)。
    """
    loras = sorted(LORA_DIR.glob("*.safetensors"))
    existing: dict = {}
    if SDXL_LORA_TOML.exists():
        try:
            existing = tomllib.loads(SDXL_LORA_TOML.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    lines = [
        "# SDXL LoRA の種別。subject に object / accessory / ware / facial / pose 等を記入。",
        '# 機能的に意味を持つのは subject="pose" のみ (OpenPose と競合 → 清書段で自動除外)。',
        "# 行末 # hint: はファイル名由来の自動ヒント (毎回再生成・編集不要)。",
        "",
    ]
    for p in loras:
        stem = p.stem
        subj = str((existing.get(stem) or {}).get("subject") or "")
        hint = _lora_hint(stem)
        lines.append(f'["{stem}"]')
        lines.append(f'subject = "{subj}"' + (f"  # hint: {hint}" if hint else ""))
        lines.append("")
    SDXL_LORA_TOML.write_text("\n".join(lines), encoding="utf-8")
    return len(loras)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    counts = check_tensors()
    print()
    print(L("=== 振り分け結果 ===", "=== Triage Results ==="))
    print(f"  3_1_SD15_checkpoint : {counts['sd15_checkpoint']:4d}")
    print(f"  3_2_SD15_Embedding  : {counts['sd15_embedding']:4d}")
    print(f"  3_3_SD15_LoRA       : {counts['sd15_lora']:4d}")
    print(f"  4_1_SDXL_checkpoint : {counts['checkpoint']:4d}")
    print(f"  4_2_SDXL_LoRA       : {counts['lora']:4d}")
    print(f"  4_3_SDXL_ControlNet : {counts['controlnet']:4d}")
    print(f"  4_4_SDXL_Embedding  : {counts['embedding']:4d}")
    print(f"  2_1_errortensors    : {counts['error']:4d}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(L("\n中断", "\nInterrupted"))
        sys.exit(0)
