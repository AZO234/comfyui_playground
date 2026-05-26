#!/usr/bin/env python3
"""dist_tensors.py - `2_0_tensors` 受入トレーのテンソルを分類して各 dir へ振り分ける CLI。

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
    python dist_tensors.py
"""
from __future__ import annotations

import argparse
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

CACHE_FILE      = ROOT / "tensors_cache.toml"
CHECKPOINT_TOML = ROOT / "checkpoint.toml"

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

    # SDXL/SD15 LoRA の種別 hint を更新 (ユーザ記入分は保持、hint は再生成)
    update_sdxl_lora_hint_toml()
    update_sd15_lora_hint_toml()
    # LoRA_preview.toml (make_previews のプレビュー用カテゴリ) の過不足を点検・追従
    audit_lora_preview_toml()
    # checkpoint.toml に未登録 checkpoint を score 0 / family 推定 / style=anime で登録 + 空欄補完
    added, filled = update_checkpoint_toml()
    if added or filled:
        print(L(f"  checkpoint.toml: 新規 {added} 件 / 空欄補完 {filled} 件",
                f"  checkpoint.toml: added {added} / filled {filled}"), flush=True)

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
# {SDXL,SD15}_LoRA_hint.toml (LoRA の種別 subject + ヒント。subject="pose" のみ機能的)
# --------------------------------------------------------------------------- #
SDXL_LORA_HINT_TOML = ROOT / "SDXL_LoRA_hint.toml"
SD15_LORA_HINT_TOML = ROOT / "SD15_LoRA_hint.toml"
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


def _write_lora_hint_toml(toml_path: Path, lora_dir: Path, header_label: str) -> int:
    """指定 dir の LoRA を hint toml に書き出す共通実装。既存 subject は保持、hint は再生成。"""
    loras = sorted(lora_dir.glob("*.safetensors"))
    existing: dict = {}
    if toml_path.exists():
        try:
            existing = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    lines = [
        f"# {header_label} LoRA の種別。subject に object / accessory / ware / facial / pose 等を記入。",
        '# 機能的に意味を持つのは subject="pose" のみ (OpenPose と競合 → 清書段で自動除外、SDXL のみ)。',
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
    toml_path.write_text("\n".join(lines), encoding="utf-8")
    return len(loras)


def update_sdxl_lora_hint_toml() -> int:
    """4_2_SDXL_LoRA → SDXL_LoRA_hint.toml を更新。"""
    return _write_lora_hint_toml(SDXL_LORA_HINT_TOML, LORA_DIR, "SDXL")


def update_sd15_lora_hint_toml() -> int:
    """3_2_SD15_LoRA → SD15_LoRA_hint.toml を更新 (SDXL と同形式)。"""
    return _write_lora_hint_toml(SD15_LORA_HINT_TOML, SD15_LORA_DIR, "SD15")


# --------------------------------------------------------------------------- #
# checkpoint.toml (family / style / score の台帳) の追従
# --------------------------------------------------------------------------- #
def _family_from_name(stem: str) -> str:
    """ファイル名から checkpoint 系統を推定 (generate.py の同名関数と同一規則)。
    判定不能は "" (ユーザが手で補正する想定)。メタには系統が無いのでファイル名が頼り。"""
    s = stem.lower()
    if ("pony" in s) or ("pdxl" in s) or ("pny" in s) or ("pxl" in s):
        return "pony"
    if ("ill" in s) or ("noob" in s) or ("nai" in s):
        return "2d"
    if ("real" in s) or ("photo" in s):
        return "real"
    return ""


def update_checkpoint_toml() -> tuple[int, int]:
    """3_1_SD15_checkpoint + 4_1_SDXL_checkpoint を scan して checkpoint.toml を追従。

    - 新規 stem → score 0 (slow=fast=like=inference=0), family=ファイル名推定, style="anime" で登録
    - 既存 stem の family が空欄 → 推定値で補完 (空欄のみ; 上書きはしない)
    - 既存 stem の style が空欄 → "anime" で補完 (空欄のみ)
    - 既存値 (slow/fast/like/inference や非空の family/style) は維持
    - 戻り値: (新規追加数, 空欄補完数)
    """
    data: dict = {}
    if CHECKPOINT_TOML.exists():
        try:
            data = tomllib.loads(CHECKPOINT_TOML.read_text(encoding="utf-8"))
        except Exception as e:
            print(L(f"  [warn] checkpoint.toml 読込失敗 ({e}) — 既存無視で再生成",
                    f"  [warn] checkpoint.toml load failed ({e}) — regenerating from scratch"), flush=True)
            data = {}

    # 物理ファイルとして実在する stem を集める (新規追加判定用)。
    # 補完対象はゴースト entry も含めて data 全体に走らせる。
    on_disk: set = set()
    for d in (SD15_CKPT_DIR, CHECKPOINT_DIR):
        if d.exists():
            on_disk.update(p.stem for p in d.glob("*.safetensors"))

    added = 0
    filled = 0

    # (1) 物理存在するが未登録の stem を新規追加 (score 0 / family 推定 / style="anime")
    for stem in sorted(on_disk, key=str.lower):
        if stem in data:
            continue
        fam_guess = _family_from_name(stem)
        data[stem] = {
            "slow": 0, "fast": 0, "like": 0, "inference": 0,
            "style": "anime", "family": fam_guess,
        }
        added += 1
        print(L(f"  checkpoint.toml + {stem} (family={fam_guess or '?'}, style=anime)",
                f"  checkpoint.toml + {stem} (family={fam_guess or '?'}, style=anime)"), flush=True)

    # (2) 既存 entry すべての空欄補完 (ゴースト含む)。
    # 新規追加分は family/style/score が既に埋まっているので touched=False で素通り。
    for stem, entry in data.items():
        if not isinstance(entry, dict):
            continue
        fam_guess = _family_from_name(stem)
        cur_fam = str(entry.get("family") or "").strip()
        cur_style = str(entry.get("style") or "").strip()
        touched = False
        if not cur_fam and fam_guess:
            entry["family"] = fam_guess
            touched = True
        if not cur_style:
            entry["style"] = "anime"
            touched = True
        for k in ("slow", "fast", "like", "inference"):
            if k not in entry:
                entry[k] = 0
                touched = True
        if touched:
            filled += 1

    data_sorted = dict(sorted(data.items(), key=lambda kv: kv[0].lower()))
    with open(CHECKPOINT_TOML, "wb") as f:
        tomli_w.dump(data_sorted, f)
    return added, filled


# --------------------------------------------------------------------------- #
# LoRA_preview.toml (make_previews のプレビュー用カテゴリ表) の過不足チェック
# --------------------------------------------------------------------------- #
LORA_PREVIEW_TOML = ROOT / "LoRA_preview.toml"


def audit_lora_preview_toml() -> tuple[int, int]:
    """LoRA_preview.toml ({SD15,SDXL}_categories) と LoRA 実体 (SD15 3_2 + SDXL 4_2) の過不足を点検。

    - 不足 (実体はあるが該当 toml セクションに無い LoRA) → category="ware" で自動追加。
    - 余剰 (toml にあるが実体が無い) → 警告のみ (手編集の category を失わないため削除しない。
      完全に整理するなら `make_previews.py --init-categories`)。
    既存 category 値は保持。同期済みなら無出力。戻り値: (追加合計, 余剰合計)。
    """
    sd15_stems = sorted({p.stem for p in SD15_LORA_DIR.glob("*.safetensors")}) if SD15_LORA_DIR.exists() else []
    sdxl_stems = sorted({p.stem for p in LORA_DIR.glob("*.safetensors")}) if LORA_DIR.exists() else []
    if not sd15_stems and not sdxl_stems:
        return (0, 0)
    data: dict = {}
    if LORA_PREVIEW_TOML.exists():
        try:
            data = tomllib.loads(LORA_PREVIEW_TOML.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    total_missing = total_stale = 0
    dirty = False
    for prefix, stems in (("SD15", sd15_stems), ("SDXL", sdxl_stems)):
        cat_key = f"{prefix}_categories"
        existing = dict(data.get(cat_key) or {})
        stem_set = set(stems)
        missing = [s for s in stems if s not in existing]
        stale = [s for s in existing if s not in stem_set]
        if missing:
            for s in missing:
                existing[s] = "ware"
            data[cat_key] = dict(sorted(existing.items(), key=lambda kv: kv[0].lower()))
            dirty = True
            ex = ", ".join(missing[:3]) + (" …" if len(missing) > 3 else "")
            print(L(f"  [LoRA_preview/{prefix}] 不足 {len(missing)} 件を ware で追加 ({ex})",
                    f"  [LoRA_preview/{prefix}] added {len(missing)} missing as ware ({ex})"))
        if stale:
            ex = ", ".join(stale[:3]) + (" …" if len(stale) > 3 else "")
            print(L(f"  [LoRA_preview/{prefix}] 余剰 {len(stale)} 件 (実体なし。--init-categories で整理可): {ex}",
                    f"  [LoRA_preview/{prefix}] {len(stale)} stale entries (no file; tidy via --init-categories): {ex}"))
        total_missing += len(missing)
        total_stale += len(stale)
    if dirty:
        with open(LORA_PREVIEW_TOML, "wb") as f:
            tomli_w.dump(data, f)
    return (total_missing, total_stale)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=L("テンソル振り分け (既定) / 軽量メンテモード",
                      "Triage tensors (default) / lightweight maintenance modes"))
    ap.add_argument("--refresh-checkpoint-toml", action="store_true",
                    help=L("振り分け処理を行わず、3_1/4_1 を scan して checkpoint.toml の "
                           "family / style / score を読み直し・空欄補完のみ実行する",
                           "Skip triage; just rescan 3_1/4_1 and refresh family / style / score "
                           "fields in checkpoint.toml (fill-only, no overwrite)"))
    args = ap.parse_args()

    if args.refresh_checkpoint_toml:
        print(L("=== checkpoint.toml 読み直しモード (ファイル移動なし) ===",
                "=== refresh-checkpoint-toml mode (no file moves) ==="))
        added, filled = update_checkpoint_toml()
        print(L(f"checkpoint.toml: 新規 {added} 件 / 空欄補完 {filled} 件",
                f"checkpoint.toml: added {added} / filled {filled}"))
        return

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
