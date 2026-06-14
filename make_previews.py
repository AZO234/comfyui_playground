#!/usr/bin/env python3
"""make_previews.py - 各テンソル (checkpoint / LoRA) のプレビュー画像をサイドカーで焼く。real ブランチ = SDXL only。

「モデルなら最小プロンプト / LoRA なら最小ベース＋トリガー語」で 1 枚ずつ生成し、
safetensors の隣に `<name>.preview.png` として保存する (SD エコシステム標準のサイドカー)。
sd_tensors_view 等のビューアはこのサイドカーを拾って表示できる。

方針:
  - **Checkpoint**: そのモデルに固定・最小プロンプト + 固定 seed で 1 枚。
    全モデル同条件なので画風比較になる。Pony 系には score 前置を自動付与。
  - **LoRA**: 系統一致のベースを自動選択し、LoRA を適用 + トリガー語 (ss_tag_frequency) で 1 枚。
    SDXL LoRA→pony→sdxl→2d→real の順で代表ベースを使う。

生成は ComfyUI HTTP API 経由 (generate.py の build_workflow_txt2img / _submit_and_fetch を流用)。
ComfyUI 未起動なら main() 冒頭で自動起動 (ensure_comfyui_arch、--dry-run は触らない)。

使い方:
    python make_previews.py                         # 全 checkpoint + LoRA (既存サイドカーはスキップ)
    python make_previews.py --only lora --limit 2   # LoRA を 2 個だけ (動作確認)
    python make_previews.py --dry-run               # 生成せず計画 (ベース選択/プロンプト) を表示
    python make_previews.py --force                 # 既存サイドカーも焼き直す
"""
from __future__ import annotations

import argparse
import json
import random
import re
import struct
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from i18n import L
# generate.py の既存機構を流用 (import 時に torch/ComfyUI 定数が読まれる)
from generate import (
    PONY_SCORE_PREFIX,
    ROOT,
    SDXL_CHECKPOINT_DIR,
    SDXL_LORA_DIR,
    _family_from_name,
    _submit_and_fetch,
    build_workflow_txt2img,
    ensure_comfyui_arch,
    load_checkpoint_toml,
    write_extra_model_paths,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

PROMPT_TOML = ROOT / "prompt.toml"
# 最小・中立な被写体プロンプト (単一被写体・縦構図でクローンを避ける)。
# full body = 全身。トップス/ボトム/脚衣/靴など、どの部位を変える LoRA でも効果が写る
# (upper body だとボトム系 LoRA の効果が見えないため)。
DEFAULT_POSITIVE = "1lady, solo, full body, standing, looking at viewer, simple background"

CATEGORIES_FILE = ROOT / "LoRA_preview.toml"             # [SDXL_] {categories,prompts}
PREVIEW_SETTINGS_TOML = ROOT / "preview_settings.toml"   # [tensors_dirs] / [LoRA_preview_template] / [checkpoint_preview_template]

# プレビュー用カテゴリ → positive スキャフォールド。後ろに {hint}, {trigger} が足される。
#   ware=着衣 / doing{1,2,3,mob}=行為(人数別) / object=物体 / part=モデル部位 / view=視点 /
#   place=場所 / artstyle=作風 / unknown=その他
DEFAULT_TEMPLATES = {
    "ware":     DEFAULT_POSITIVE,
    "doing1":   "1lady, solo, full body, simple background",              # 1人 (自慰等)
    "doing2":   "2ladies, full body, interaction, simple background",     # 2人
    "doing3":   "3ladies, full body, interaction, simple background",     # 3人
    "doingmob": "6+ladies, full body, interaction, simple background",    # 多数 (乱交等)
    "object":   "no humans, simple background",
    "part":     "1lady, solo, upper body, close-up, simple background",   # 部位を近接で見せる
    "view":     "1lady, solo, full body, simple background",              # 視点は hint/trigger 由来
    "place":    "1lady, solo, full body, scenery",                        # 場所/環境を見せる (simple bg は外す)
    "artstyle": "1lady, solo, upper body, looking at viewer, detailed, simple background",  # 画風を細部で見せる
    "unknown":  "1lady, solo, upper body, simple background",
}
# 行為系の自動推定トークン (--init-categories --guess 用)。
# トークン単位で一致を見る (部分一致だと "Sexy"→sex, "0769"→69 のように誤爆するため)。
_DOING_TOKENS = {
    "sex", "blowjob", "blowjobs", "fellatio", "irrumatio", "cunnilingus", "paizuri",
    "titfuck", "handjob", "footjob", "fisting", "creampie", "missionary", "doggystyle",
    "cowgirl", "gangbang", "bukkake", "threesome", "foursome", "orgy", "piledriver",
    "mating", "penetration", "deepthroat", "facesitting", "tribadism", "scissoring",
    "fucking", "fuck", "anal", "vaginal",
}


# --------------------------------------------------------------------------- #
# safetensors メタからトリガー語を取得 (torch 不要のヘッダ生読み)
# --------------------------------------------------------------------------- #
def _read_metadata(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            obj = json.loads(f.read(n))
        meta = obj.get("__metadata__", {})
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def clean_name_hint(stem: str) -> str:
    """ファイル名から衣類/被写体ヒントを抽出する。

    LoRA のトリガーが抽象トークン (ruanyi0641 等) の場合、衣類カテゴリがプロンプトに
    渡らずベースが勝手にアウターを着せてしまう。ファイル名には作者が衣類名を書いている
    ことが多い ("Sexy lingerie", "Twill pantyhose" 等) ので、ノイズを除いて流用する。
    """
    s = stem
    s = re.sub(r"^\d+\s*[_\-]?\s*", "", s)          # 先頭の数値 ID ("0641 ", "0093_")
    s = re.sub(r"[_\-]+", " ", s)                    # 区切り → 空白
    s = re.sub(r"\b(v\d+|pony|pdxl|sdxl|xl|fp16|bakedvae|\d{4,})\b", "", s, flags=re.I)
    s = re.sub(r"\b\d+\b", "", s)                    # 単独の数字
    return re.sub(r"\s+", " ", s).strip()


def top_triggers(path: Path, n: int = 2) -> list[str]:
    """ss_tag_frequency から頻度上位のタグ (トリガー語候補) を返す。"""
    raw = _read_metadata(path).get("ss_tag_frequency")
    if not raw:
        return []
    try:
        tf = json.loads(raw)
    except (ValueError, TypeError):
        return []
    counts: dict[str, int] = {}
    if isinstance(tf, dict):
        for tags in tf.values():
            if isinstance(tags, dict):
                for t, c in tags.items():
                    counts[t] = counts.get(t, 0) + (c if isinstance(c, int) else 0)
    return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


# --------------------------------------------------------------------------- #
# 系統 (family) 判定 & ベース選択
# --------------------------------------------------------------------------- #
def tensor_family(path: Path) -> str:
    """pony(2.5-3D) / 2d / real / sdxl(汎用) を返す。ベース選択のキー。"""
    fam = _family_from_name(path.stem)   # pony / 2d / real / ""
    if fam in ("pony", "2d", "real"):
        return fam
    return "sdxl"


def gather(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        if d.is_dir():
            out += [p for p in d.glob("*.safetensors") if p.is_file()]
    return sorted(out, key=lambda p: p.name.lower())


def build_family_bases(overrides: dict[str, Optional[str]]) -> dict[str, Path]:
    """系統 → 代表ベース checkpoint を決める。CLI 上書き > 系統内の先頭(名前順)。"""
    checkpoints = gather([SDXL_CHECKPOINT_DIR])
    by_family: dict[str, list[Path]] = {}
    for c in checkpoints:
        by_family.setdefault(tensor_family(c), []).append(c)

    bases: dict[str, Path] = {}
    for fam, members in by_family.items():
        bases[fam] = members[0]   # 名前順の先頭を代表に
    # CLI 上書き
    for fam, name in overrides.items():
        if not name:
            continue
        hit = next((c for c in checkpoints if c.stem == name or c.name == name), None)
        if hit:
            bases[fam] = hit
        else:
            print(L(f"  [warn] --base-{fam} '{name}' が見つかりません (無視)",
                    f"  [warn] --base-{fam} '{name}' not found (ignored)"), flush=True)
    return bases


def base_for_lora(lora: Path, bases: dict[str, Path]) -> Optional[Path]:
    """LoRA に使うベースを返す。

    系統一致を優先: pony(2.5-3D) → pony / 2d(ill/noob/nai) → 2d / real → real / 汎用 → sdxl。
    Pony と 2D を混ぜると崩れる (ミュータント/ノイズ) ので環境を分ける。
    一致系統が無ければ pony → sdxl → 2d → real の順でフォールバック。
    ※系統判定はファイル名 (_family_from_name) ベースで不完全な点に注意。
    """
    fam = tensor_family(lora)
    for key in (fam, "pony", "sdxl", "2d", "real"):
        if key in bases:
            return bases[key]
    return None


# --------------------------------------------------------------------------- #
# プロンプト構築
# --------------------------------------------------------------------------- #
def load_negative() -> str:
    try:
        import tomllib
        with open(PROMPT_TOML, "rb") as f:
            return str(tomllib.load(f).get("negative_always") or "")
    except (OSError, ValueError, ModuleNotFoundError):
        return ""


def build_positive(base_positive: str, family: str, trigger: str = "") -> str:
    parts = []
    if family == "pony":               # Pony 系ベースには score 前置
        parts.append(PONY_SCORE_PREFIX)
    parts.append(base_positive)
    if trigger:
        parts.append(trigger)
    return ", ".join(parts)


SDXL_PREVIEW_RES = (832, 1216)   # SDXL 縦構図 (単一被写体)


# --------------------------------------------------------------------------- #
# プレビュー用カテゴリ (ware/doing/object/unknown) — TOML で stem ごとに上書き可
# --------------------------------------------------------------------------- #
def _name_tokens(stem: str) -> set:
    """ファイル名を camelCase + 非英数字で分割し小文字トークン集合にする。"""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)   # camelCase 境界
    return set(t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t)


def guess_category(stem: str) -> str:
    """ファイル名トークンから行為系を推定。一致なしは ware (--guess 時のみ使用)。
    行為は人数不明なので既定 doing2 (2人) に寄せる。手編集で doing1/3/mob に直す想定。"""
    return "doing2" if (_name_tokens(stem) & _DOING_TOKENS) else "ware"


def load_preview_config(template_path: Path = PREVIEW_SETTINGS_TOML,
                        lora_preview_path: Path = CATEGORIES_FILE
                        ) -> tuple[dict, dict, dict, dict]:
    """新スキーマ: preview_template.toml + LoRA_preview.toml を読む。

    戻り値:
      lora_templates: {category: scaffold}  ← preview_template.toml [LoRA_preview_template] (DEFAULT_TEMPLATES と merge)
      ckpt_templates: {family|'default': scaffold} ← [checkpoint_preview_template] (default は DEFAULT_POSITIVE)
      cats:           {stem→cat}    ← SDXL_categories
      prompts:        {stem→prompt} ← SDXL_prompts
    """
    import tomllib
    lora_templates = dict(DEFAULT_TEMPLATES)
    ckpt_templates = {"default": DEFAULT_POSITIVE}
    if template_path.exists():
        try:
            td = tomllib.loads(template_path.read_text(encoding="utf-8"))
            for k, v in (td.get("LoRA_preview_template") or {}).items():
                lora_templates[str(k)] = str(v)
            for k, v in (td.get("checkpoint_preview_template") or {}).items():
                ckpt_templates[str(k)] = str(v)
        except Exception:
            pass
    cats: dict = {}
    prompts: dict = {}
    if lora_preview_path.exists():
        try:
            lp = tomllib.loads(lora_preview_path.read_text(encoding="utf-8"))
            cats = {str(k): str(v) for k, v in (lp.get("SDXL_categories") or {}).items()}
            prompts = {str(k): str(v) for k, v in (lp.get("SDXL_prompts") or {}).items() if str(v).strip()}
        except Exception:
            pass
    return lora_templates, ckpt_templates, cats, prompts


def write_preview_categories(guess: bool = False,
                             lora_preview_path: Path = CATEGORIES_FILE) -> None:
    """全 LoRA を SDXL_categories に書き出す (既存 cats/prompts は保持)。
    --init-categories から呼ぶ。templates は preview_template.toml 側を編集する。"""
    import tomllib
    import tomli_w
    data: dict = {}
    if lora_preview_path.exists():
        try:
            data = tomllib.loads(lora_preview_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    existing = dict(data.get("SDXL_categories") or {})
    new_cats = {p.stem: existing.get(p.stem)
                   or (guess_category(p.stem) if guess else "ware")
                for p in sorted(SDXL_LORA_DIR.glob("*.safetensors"))}
    data["SDXL_categories"] = dict(sorted(new_cats.items(), key=lambda kv: kv[0].lower()))
    if "SDXL_prompts" not in data:
        data["SDXL_prompts"] = {}
    ordered = {k: data[k] for k in ("SDXL_categories", "SDXL_prompts") if k in data}
    with open(lora_preview_path, "wb") as f:
        tomli_w.dump(ordered, f)
    n_sdxl = len(ordered.get("SDXL_categories", {}))
    print(L(f"{lora_preview_path.name} を書き出し: SDXL {n_sdxl} 件",
            f"wrote {lora_preview_path.name}: SDXL {n_sdxl} entries"))


def save_preview_entry(stem: str, category: str, custom_prompt: str,
                       lora_preview_path: Path = CATEGORIES_FILE) -> None:
    """1 件分の category + custom prompt を SDXL_{categories,prompts} に書き込む。"""
    import tomllib
    import tomli_w
    data: dict = {}
    if lora_preview_path.exists():
        try:
            data = tomllib.loads(lora_preview_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    cats = dict(data.get("SDXL_categories") or {})
    prompts = dict(data.get("SDXL_prompts") or {})
    cats[stem] = category
    cp = (custom_prompt or "").strip()
    if cp:
        prompts[stem] = cp
    else:
        prompts.pop(stem, None)
    data["SDXL_categories"] = dict(sorted(cats.items(), key=lambda kv: kv[0].lower()))
    data["SDXL_prompts"] = dict(sorted(prompts.items(), key=lambda kv: kv[0].lower()))
    with open(lora_preview_path, "wb") as f:
        tomli_w.dump(data, f)


def set_preview_categories(stems: list[str], category: str,
                           lora_preview_path: Path = CATEGORIES_FILE) -> None:
    """複数 stem の category を一括設定 (prompts は保持、1 回書き込み)。"""
    import tomllib
    import tomli_w
    data: dict = {}
    if lora_preview_path.exists():
        try:
            data = tomllib.loads(lora_preview_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    cats = dict(data.get("SDXL_categories") or {})
    for s in stems:
        cats[s] = category
    data["SDXL_categories"] = dict(sorted(cats.items(), key=lambda kv: kv[0].lower()))
    with open(lora_preview_path, "wb") as f:
        tomli_w.dump(data, f)


# --------------------------------------------------------------------------- #
# 1 枚生成
# --------------------------------------------------------------------------- #
def render(*, checkpoint_name: str, loras, positive: str, negative: str,
           seed: int, steps: int, cfg: float, sampler: str, scheduler: str,
           client_id: str) -> Optional[bytes]:
    w, h = SDXL_PREVIEW_RES
    wf = build_workflow_txt2img(
        checkpoint=checkpoint_name, positive=positive, negative=negative,
        seed=seed, steps=steps, cfg=cfg, width=w, height=h,
        sampler_name=sampler, scheduler=scheduler,
        loras=loras, filename_prefix="preview",
    )
    data, _info, _outputs = _submit_and_fetch(wf, client_id)
    return data


def _grid_2x2(shots: list[bytes]) -> bytes:
    """最大4枚の画像 bytes を半分に縮小して 2x2 に並べ、1枚の PNG bytes にする。"""
    import io
    from PIL import Image
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in shots[:4]]
    w, h = imgs[0].size
    cw, ch = w // 2, h // 2
    grid = Image.new("RGB", (cw * 2, ch * 2), (0, 0, 0))
    for i, im in enumerate(imgs):
        grid.paste(im.resize((cw, ch), Image.LANCZOS), ((i % 2) * cw, (i // 2) * ch))
    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return buf.getvalue()


def render_multi(*, checkpoint_name: str, loras, positive: str, negative: str,
                 seeds: list[int], steps: int, cfg: float, sampler: str, scheduler: str,
                 client_id: str) -> Optional[bytes]:
    """seeds の数だけ生成し、複数なら 2x2 グリッドに合成して返す (checkpoint の描き味比較用)。"""
    shots = []
    for s in seeds:
        data = render(checkpoint_name=checkpoint_name, loras=loras, positive=positive,
                      negative=negative, seed=s, steps=steps, cfg=cfg,
                      sampler=sampler, scheduler=scheduler, client_id=client_id)
        if data:
            shots.append(data)
    if not shots:
        return None
    return shots[0] if len(shots) == 1 else _grid_2x2(shots)


def build_job(kind: str, path: Path, *, lora_templates: dict, ckpt_templates: dict,
              cats_map: dict, prompts_map: dict,
              bases: dict, prompt: str, extra: str = "", lora_strength: float = 0.8):
    """1 ターゲットの生成内容を組む。

    戻り値: (positive, checkpoint_name, loras, plan)。
    LoRA で適合ベースが無ければ None。main ループと regenerate() が共有する。
    checkpoint は ckpt_templates[family] (なければ default) を使う (--prompt は最後の保険)。
    """
    if kind == "checkpoint":
        family = tensor_family(path)
        scaffold = ckpt_templates.get(family) or ckpt_templates.get("default") or prompt
        positive = build_positive(scaffold, family)
        ckpt_name, loras = path.name, None
        plan = f"ckpt={path.stem} [{family}]"
    else:
        base = base_for_lora(path, bases)
        if base is None:
            return None
        triggers = top_triggers(path)
        family = tensor_family(base)
        custom = prompts_map.get(path.stem, "")
        if custom:
            # 個別カスタムプロンプト (unknown 等)。トリガー未記載なら活性化のため足す
            trig = ", ".join(triggers)
            tok = trig if (trig and trig not in custom) else ""
            positive = build_positive(custom, family, tok)
            plan = f"lora={path.stem} cat=custom base={base.stem} prompt='{custom[:48]}'"
        else:
            cat = cats_map.get(path.stem, "ware")           # 未記載は ware
            scaffold = lora_templates.get(cat, lora_templates["ware"])
            hint = clean_name_hint(path.stem)
            lora_tokens = ", ".join([x for x in ([hint] + triggers) if x])
            positive = build_positive(scaffold, family, lora_tokens)
            plan = f"lora={path.stem} cat={cat} base={base.stem} hint='{hint or '-'}' trigger='{', '.join(triggers) or '-'}'"
        ckpt_name, loras = base.name, [(path.name, lora_strength)]
    if extra:
        positive = f"{positive}, {extra}"
    return positive, ckpt_name, loras, plan


def regenerate(path: Path, *, seed: Optional[int] = None, steps: int = 24, cfg: float = 5.0,
               lora_strength: float = 0.8, sampler: str = "dpmpp_2m",
               scheduler: str = "karras", extra: str = "",
               categories_file: Path = CATEGORIES_FILE,
               client_id: Optional[str] = None) -> Path:
    """checkpoint / LoRA 1 つのプレビューを現在の LoRA_preview.toml 設定で焼き直し、
    サイドカー <name>.preview.png に保存してそのパスを返す。
    呼び出し側 (main()) で ComfyUI 起動済み前提。直接 import して使う場合は事前に
    `ensure_comfyui_arch(arch)` を呼んでおくこと。

    tensors_view の『再生成』ボタンから呼ぶ想定。失敗時は例外。
    """
    if not path.exists():
        raise FileNotFoundError(path)
    parent = path.resolve().parent
    kind = "checkpoint" if parent == SDXL_CHECKPOINT_DIR.resolve() else "lora"
    write_extra_model_paths()
    lora_templates, ckpt_templates, cats_map, prompts_map = load_preview_config(
        lora_preview_path=categories_file)
    bases = build_family_bases({})
    job = build_job(kind, path, lora_templates=lora_templates, ckpt_templates=ckpt_templates,
                    cats_map=cats_map, prompts_map=prompts_map,
                    bases=bases, prompt=DEFAULT_POSITIVE, extra=extra, lora_strength=lora_strength)
    if job is None:
        raise RuntimeError(f"no matching base for {path.stem}")
    positive, ckpt_name, loras, _plan = job
    # checkpoint は 4ショット(seed揺らし)→2x2 グリッド。LoRA は1枚。
    if kind == "checkpoint":
        seeds = [random.randint(0, 2**32 - 1) for _ in range(4)]
    else:
        seeds = [seed if seed is not None else random.randint(0, 2**32 - 1)]
    data = render_multi(checkpoint_name=ckpt_name, loras=loras, positive=positive,
                        negative=load_negative(), seeds=seeds, steps=steps,
                        cfg=cfg, sampler=sampler, scheduler=scheduler,
                        client_id=client_id or uuid.uuid4().hex)
    if not data:
        raise RuntimeError("ComfyUI returned no image")
    sidecar = path.with_suffix(".preview.png")
    sidecar.write_bytes(data)
    return sidecar


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=L("各 checkpoint / LoRA のプレビュー画像をサイドカー (<name>.preview.png) で焼く",
                      "Render preview sidecars (<name>.preview.png) for each checkpoint / LoRA"))
    ap.add_argument("--only", choices=["checkpoint", "lora", "both"], default="both",
                    help=L("対象種別 (既定 both)", "target kind (default both)"))
    ap.add_argument("--match", type=str, default="",
                    help=L("ファイル名にこの文字列を含むものだけ (部分一致・確認用)",
                           "only files whose name contains this substring (for testing)"))
    ap.add_argument("--categories", type=str, default=str(CATEGORIES_FILE),
                    help=L("カテゴリ定義 TOML (templates と stem→category)。未記載は ware",
                           "category TOML (templates and stem→category); unlisted = ware"))
    ap.add_argument("--init-categories", action="store_true",
                    help=L("全 LoRA のカテゴリ一覧を TOML に書き出して終了 (既存の手編集は保持)",
                           "write a category TOML for all LoRAs and exit (preserves manual edits)"))
    ap.add_argument("--guess", action="store_true",
                    help=L("--init-categories でファイル名から doing を自動推定 (既定は全 ware)",
                           "with --init-categories, auto-guess doing from filename (default all ware)"))
    ap.add_argument("--limit", type=int, default=0,
                    help=L("処理数上限 先頭から (0=無制限、動作確認用)", "max targets from the top (0=unlimited; for testing)"))
    ap.add_argument("--force", action="store_true",
                    help=L("既存サイドカーも焼き直す", "regenerate even if a sidecar exists"))
    ap.add_argument("--dry-run", action="store_true",
                    help=L("生成せず計画 (ベース/プロンプト) を表示", "show plan (base/prompt) without generating"))
    ap.add_argument("--seed", type=int, default=-1,
                    help=L("seed (既定 -1=画像ごとに乱数。固定したい時のみ値を指定)",
                           "seed (default -1 = random per image; set a value to pin)"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--lora-strength", type=float, default=0.8)
    ap.add_argument("--sampler", type=str, default="dpmpp_2m")
    ap.add_argument("--scheduler", type=str, default="karras")
    ap.add_argument("--prompt", type=str, default=DEFAULT_POSITIVE,
                    help=L("最小プロンプト (positive)", "minimal positive prompt"))
    ap.add_argument("--extra", type=str, default="",
                    help=L("positive 末尾に足す追加トークン (例 'colorful, vivid colors')。"
                           "既定は忠実 (色強制なし)",
                           "extra tokens appended to positive (e.g. 'colorful, vivid colors'). "
                           "default is faithful (no forced color)"))
    ap.add_argument("--files", nargs="*", default=None,
                    help=L("指定したファイル(パス)だけ再生成して終了 (tensors_view が別プロセスで呼ぶ)",
                           "regenerate only the given file paths and exit (called by tensors_view as a subprocess)"))
    ap.add_argument("--arch", choices=["cuda", "cpu"], default="cuda",
                    help=L("ComfyUI 側 device (既定 cuda)。未起動なら自動で立ち上げる",
                           "ComfyUI device (default cuda). Auto-launches ComfyUI if not running"))
    for fam in ("pony", "2d", "real", "sdxl"):
        ap.add_argument(f"--base-{fam}", type=str, default=None,
                        help=L(f"{fam} 系 LoRA のベース checkpoint を明示指定",
                               f"explicit base checkpoint for {fam} LoRAs"))
    args = ap.parse_args()

    cat_file = Path(args.categories)
    if args.init_categories:
        write_preview_categories(guess=args.guess, lora_preview_path=cat_file)
        return

    # ComfyUI 未起動なら自動起動 (起動済みなら no-op、--dry-run は触らない)
    if not args.dry_run:
        yaml_changed = write_extra_model_paths()
        ensure_comfyui_arch(args.arch, force_restart=yaml_changed)

    if args.files:
        # 指定ファイルだけ再生成 (tensors_view が subprocess で呼ぶ。GUI を torch から隔離)
        n_ok = n_fail = 0
        for f in args.files:
            p = Path(f)
            try:
                out = regenerate(p, extra=args.extra, steps=args.steps, categories_file=cat_file)
                print(L(f"  → {out.name} 保存", f"  → saved {out.name}"), flush=True)
                n_ok += 1
            except Exception as ex:
                print(L(f"  [error] {p.name}: {ex}", f"  [error] {p.name}: {ex}"), flush=True)
                n_fail += 1
        print(L(f"=== files 完了: {n_ok} ok / {n_fail} fail ===",
                f"=== files done: {n_ok} ok / {n_fail} fail ==="))
        return
    lora_templates, ckpt_templates, cats_map, prompts_map = load_preview_config(
        lora_preview_path=cat_file)
    # preview_template.toml が ware を持っていなければ --prompt を反映
    if not PREVIEW_SETTINGS_TOML.exists():
        lora_templates["ware"] = args.prompt

    negative = load_negative()
    bases = build_family_bases({
        "pony": args.base_pony, "2d": args.base_2d,
        "real": args.base_real, "sdxl": args.base_sdxl,
    })
    print(L("=== プレビュー焼き ===", "=== preview rendering ==="))
    print(L(f"系統別ベース: " + ", ".join(f"{k}={v.stem}" for k, v in bases.items()),
            f"family bases: " + ", ".join(f"{k}={v.stem}" for k, v in bases.items())))

    # write_extra_model_paths + ensure_comfyui_arch は main 冒頭で 1 回 済み

    # 処理対象を集める (kind → match → limit の順)
    targets: list[tuple[str, Path]] = []
    if args.only in ("checkpoint", "both"):
        targets += [("checkpoint", p) for p in gather([SDXL_CHECKPOINT_DIR])]
    if args.only in ("lora", "both"):
        targets += [("lora", p) for p in gather([SDXL_LORA_DIR])]
    if args.match:
        targets = [(k, p) for (k, p) in targets if args.match.lower() in p.stem.lower()]
    if args.limit:
        targets = targets[:args.limit]

    client_id = uuid.uuid4().hex
    done = skipped = failed = 0
    t_start = time.time()

    for idx, (kind, path) in enumerate(targets, 1):
        sidecar = path.with_suffix(".preview.png")
        if sidecar.exists() and not args.force:
            skipped += 1
            continue

        job = build_job(kind, path, lora_templates=lora_templates, ckpt_templates=ckpt_templates,
                        cats_map=cats_map, prompts_map=prompts_map,
                        bases=bases, prompt=args.prompt,
                        extra=args.extra, lora_strength=args.lora_strength)
        if job is None:
            print(L(f"  [warn] {path.stem}: 適合ベースなし、スキップ",
                    f"  [warn] {path.stem}: no matching base, skipped"), flush=True)
            failed += 1
            continue
        positive, ckpt_name, loras, plan = job

        print(f"[{_ts()}] ({idx}/{len(targets)}) {kind}: {plan}", flush=True)
        if args.dry_run:
            print(f"            positive: {positive[:110]}", flush=True)
            continue

        try:
            if kind == "checkpoint":
                seeds = [random.randint(0, 2**32 - 1) for _ in range(4)]   # 4ショット (seed揺らし→2x2)
            else:
                seeds = [args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)]
            data = render_multi(checkpoint_name=ckpt_name, loras=loras, positive=positive,
                                negative=negative, seeds=seeds, steps=args.steps,
                                cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler,
                                client_id=client_id)
        except Exception as ex:
            print(L(f"            [error] 生成失敗: {ex}", f"            [error] generation failed: {ex}"), flush=True)
            failed += 1
            continue
        if not data:
            print(L("            [error] 画像が返らなかった", "            [error] no image returned"), flush=True)
            failed += 1
            continue
        sidecar.write_bytes(data)
        print(L(f"            → {sidecar.name} 保存", f"            → saved {sidecar.name}"), flush=True)
        done += 1

    elapsed = time.time() - t_start
    print(L(f"=== 完了: 生成 {done} / スキップ {skipped} / 失敗 {failed} "
            f"(対象 {len(targets)}, {elapsed:.0f}s) ===",
            f"=== done: rendered {done} / skipped {skipped} / failed {failed} "
            f"(targets {len(targets)}, {elapsed:.0f}s) ==="))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(L("\n中断", "\nInterrupted"))
        sys.exit(0)
