#!/usr/bin/env python3
"""common.py - generate.py / change.py / upscale.py / hightensors.py で共有する純粋ユーティリティ。

このモジュールはスクリプトエントリーポイントを持たず、argparse / main() ループも含まない。
generate.py のような実行スクリプトはここの関数を呼んで処理を組み立てる。

提供するもの:
    定数:   ROOT 配下の各ディレクトリ / TOML パス、PIPE_SIZE_GB、DEFAULT_SIZE 等
    TOML:   load_prompt_config / load_lora_params / load_base_settings / save_base_settings /
            migrate_base_score_toml / update_lora_param_toml
    抽選:   pick_base / base_weight / lora_trigger / lora_scale_for / is_lora_skipped
    解析:   classify_tensor / detect_sd_version / lora_target_version / check_tensors
    変換:   convert_to_safetensors / file_sha256
    パイプ: pick_device_dtype / build_pipeline / build_img2img_pipe / get_upscale_model
    画像:   save_image_no_metadata / detect_degenerate / unique_output_path
    その他: current_gpu_temp / normalize_emphasis / build_prompt
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

# コンソール出力の言語切替 (英/日)。再エクスポートして from common import L を可能にする。
from i18n import L, LANG, set_lang  # noqa: F401

# --------------------------------------------------------------------------- #
# 定数 / ディレクトリ構成
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent
TENSORS_DIR      = ROOT / "2_0_tensors"      # 受入トレー: ユーザが新規ファイルを放り込む場所
BASE_DIR         = ROOT / "2_1_base"          # 分類済み SD15 base
LORA_DIR         = ROOT / "2_2_LoRA"          # 分類済み LoRA (SD15)
EMBED_DIR        = ROOT / "2_3_Embedding"     # 分類済み Textual Inversion (SD15)
LOW_DIR          = ROOT / "2_9_1_lowtensors"     # 自動退避: 破損 / 重複 / inpainting
HIGH_DIR         = ROOT / "2_9_2_hightensors"    # 旧 SDXL 隔離先 (廃止予定。migrate_high_to_xl で 5_x に移行)
ANOMALY_DIR      = ROOT / "2_9_3_anomalytensors" # 手動隔離: 修正/利用しにくいテンソル (check_tensors は触らない)
OUT_DIR          = ROOT / "3_0_generateds"
# highrender.py (SDXL + CPU + 高 step + ControlNet + tiled diffusion) 用ディレクトリ群
SELECTED5_DIR    = ROOT / "5_0_selected"         # highrender 入力 (.txt + 任意 .png)
XL_BASE_DIR      = ROOT / "5_1_XL_base"          # 分類済み SDXL base
XL_LORA_DIR      = ROOT / "5_2_XL_LoRA"          # 分類済み SDXL LoRA
XL_EMBED_DIR     = ROOT / "5_3_XL_Embedding"     # 分類済み SDXL Embedding
XL_CONTROLNET_DIR = ROOT / "5_4_XL_ControlNet"   # SDXL ControlNet テンソル (ファイル名抽選)
HIGHREND_GEN_DIR = ROOT / "5_8_highrend_gen"     # highrender 出力 (1024² + メタ)
HIGHREND_UP_DIR  = ROOT / "5_9_highrend_up"      # highrender 出力 (tiled refine 2048/4096² + メタ)
PROMPT_TOML = ROOT / "prompt.toml"
LORA_PARAM_TOML = ROOT / "LoRA_param.toml"
BASE_SETTING_TOML = ROOT / "base_setting.toml"
BASE_SCORE_TOML = ROOT / "base_score.toml"  # 旧名 (base_score.toml → base_setting.toml 移行のため互換参照)
CACHE_FILE = ROOT / ".tensor_cache.json"

# `*word*` / `**word**` / `***word***` を捉える。先頭側の `*` 数で重みが決まる。
EMPHASIS_RE = re.compile(r"(\*+)([^*]+)\*+")
# asterisk 数 → compel 重み (markdown の italics/bold/bold-italics に倣う段階表現)
_EMPHASIS_WEIGHTS = {1: 1.1, 2: 1.3}  # それ以上 (3+) は 1.5 にフォールバック

TENSOR_EXTS = (".safetensors", ".ckpt", ".pt", ".bin")

# fp16 で読んだ場合のおおよその常駐サイズ (GB)。アクティベーション余裕 1.5GB を別途見込む
PIPE_SIZE_GB = {"sd15": 2.5, "sdxl": 6.5}
# version 別のディテール優先デフォルト解像度
DEFAULT_SIZE = {"sd15": 768, "sdxl": 1024}


# --------------------------------------------------------------------------- #
# プロンプト
# --------------------------------------------------------------------------- #
def normalize_emphasis(text: str) -> str:
    """`*word*` / `**word**` / `***word***` を compel の重み記法 `(word:1.x)` に変換する。
    asterisks の数で強度: 1=1.1, 2=1.3, 3+=1.5。
    compel が prompt_embeds 構築時にこの記法をパースして per-token weight に変換する。
    """
    def _repl(m: re.Match) -> str:
        n = len(m.group(1))
        weight = _EMPHASIS_WEIGHTS.get(n, 1.5)
        return f"({m.group(2)}:{weight})"
    return EMPHASIS_RE.sub(_repl, text)


def load_prompt_config() -> dict:
    """prompt.toml をロード。存在しなければ最小デフォルトを返す。"""
    if not PROMPT_TOML.exists():
        return {
            "who": [["a girl", False]],
            "wearing": [["nothing", 1]],
            "with_items": [["nothing", 1]],
            "motion": [["standing", 1]],
            "at": ["room"],
            "lighting": ["bright lighting"],
            "positive_always": "",
            "negative_always": "",
        }
    with open(PROMPT_TOML, "rb") as f:
        return tomllib.load(f)


def _wpick_entry(pairs: list) -> list | None:
    """[[value, weight, kw?], ...] から 1 つ重み抽選してエントリ全体を返す。
    pairs 空なら None。各エントリは可変長 ([value, weight] or [value, weight, kw])。
    """
    if not pairs:
        return None
    weights = [max(0, int(p[1])) for p in pairs]
    if sum(weights) == 0:
        return random.choice(pairs)
    return random.choices(pairs, weights=weights, k=1)[0]


def _entry_value(entry: list | None) -> str:
    return str(entry[0]) if entry else ""


def _entry_keyword(entry: list | None) -> str:
    """エントリの 3 番目要素 (LoRA キーワード) を返す。無ければ空文字。"""
    if entry and len(entry) >= 3 and entry[2]:
        return str(entry[2])
    return ""


def build_prompt(cfg: dict) -> tuple[str, str, list[str], bool]:
    """prompt.toml の設定から (positive, negative, lora_keywords, many) を組み立てる。

    many は採用された who エントリが複数人 (例 "2 women") を表すフラグ。
    生成側はこれを見て横長キャンバスに切り替える等の判断に使う。

    各セクションの動作:
      who         : [character, has_wearing, has_motion?, has_where?, many?, kw?] から均等ランダム 1 つ。
                    has_wearing=true なら wearing セクション自体をスキップ。
                    has_motion=true なら motion をスキップ。
                    has_where=true なら at セクションをスキップ (= キャラ文字列に場所が内包、例
                    "a girl ware swimsuit at pool" / "a girl ware swimsuit in sea")。
                    many=true は複数人 (キャラ文字列が "2 women" 等)。生成側で横長化に使う。
                    後方互換: 3〜5 要素目の **型** で版を判別:
                      - [char, has_motion, kw_str]            旧 v01 形式 (3 要素、kw のみ。bool 1 個は has_motion 扱い)
                      - [char, has_wearing, has_motion, kw_str]    v02 形式 (4 要素、has_where=false default)
                      - [char, has_wearing, has_motion, has_where, kw_str]  v03 形式 (5 要素、index4=str)
                      - [char, has_wearing, has_motion, has_where, many, kw_str]  v04 形式 (6 要素、index4=bool)
      wearing     : [clothing, weight, kw?] から重み抽選 1 つ。"nothing" → "naked"、他は "wearing X"。
                    who.has_wearing=true ならこのセクションは走らない。
      with_items  : [item, weight, kw?] から 3 回独立重み抽選 → dedupe / "nothing" 除外 → 各 "with X"。
      motion      : [motion, weight, kw?] から重み抽選 1 つ (has_motion=false のときのみ)。
      at          : 均等抽選 1 つ → "at {place}" (has_where=false のときのみ)。
      lighting    : 均等抽選 1 つ → "with {lighting}"。
      *_always    : 末尾 / negative 本体。

    最後の kw 要素は当該エントリが採用されたときに LoRA 名マッチング用キーワードとして
    収集される (`pick_lora_by_keywords` で使用)。最終的に normalize_emphasis で
    `**word**` を compel 重み記法に変換して返す。
    """
    parts: list[str] = []
    lora_keywords: list[str] = []

    # 誰が: [char, has_wearing, has_motion?, has_where?, kw?]
    # 後方互換性のため、3〜5 要素を型判定で振り分ける。
    # v01 (3 要素 [char, has_motion, kw]) は単独 bool を has_motion として扱う旧仕様。
    # v02/v03 (4-5 要素) では [char, has_wearing, has_motion, ...] の順に変更済み。
    who_entries = cfg.get("who") or []
    has_motion = False
    has_wearing = False
    has_where = False
    many = False
    if who_entries:
        chosen = random.choice(who_entries)
        char = str(chosen[0]) if chosen else ""
        kw_value: str | None = None
        # 3 要素目 (index 2): bool (v02+ では has_motion) or str (旧 v01 kw)
        if len(chosen) >= 3:
            third = chosen[2]
            if isinstance(third, bool):
                # v02+ ルート: bool 1 個目 = has_wearing
                has_wearing = bool(chosen[1]) if len(chosen) >= 2 else False
                has_motion = third
                # 4 要素目 (index 3): bool (has_where、v03) or str (v02 kw)
                if len(chosen) >= 4:
                    fourth = chosen[3]
                    if isinstance(fourth, bool):
                        has_where = fourth
                        # 5 要素目 (index 4): bool (many、v04) or str (kw、v03)
                        if len(chosen) >= 5:
                            fifth = chosen[4]
                            if isinstance(fifth, bool):
                                many = fifth
                                # 6 要素目 (index 5): kw (v04)
                                if len(chosen) >= 6 and chosen[5]:
                                    kw_value = str(chosen[5])
                            elif fifth:
                                kw_value = str(fifth)
                    else:
                        # fourth is str = v02 kw, has_where defaults False
                        if fourth:
                            kw_value = str(fourth)
            else:
                # third is str = 旧 v01 形式 [char, has_motion, kw]、has_wearing/has_where とも False
                has_motion = bool(chosen[1]) if len(chosen) >= 2 else False
                if third:
                    kw_value = str(third)
        elif len(chosen) >= 2:
            # 2 要素のみ (kw 無し v01 風): bool 1 個目 = has_motion 扱い
            has_motion = bool(chosen[1])
        if char:
            parts.append(char)
        if kw_value:
            lora_keywords.append(kw_value)

    # 何を着て: has_wearing=true ならセクション全体をスキップ (= キャラ文字列に服装情報が
    # 内包されている前提、wearing 抽選自体を回さない)。
    # has_wearing=false は通常抽選: "nothing" → "naked"、その他は "wearing {clothing}"。
    if not has_wearing:
        w_entry = _wpick_entry(cfg.get("wearing") or [])
        w = _entry_value(w_entry)
        if w == "nothing":
            parts.append("naked")
        elif w:
            parts.append(f"wearing {w}")
        kw = _entry_keyword(w_entry)
        if kw:
            lora_keywords.append(kw)

    # with アクセサリ/状況 (最大 3、dedupe、"nothing" 除外)
    with_pool = cfg.get("with_items") or []
    if with_pool:
        weights = [max(0, int(p[1])) for p in with_pool]
        if sum(weights) > 0:
            entries = random.choices(with_pool, weights=weights, k=3)
        else:
            entries = [random.choice(with_pool) for _ in range(3)]
        seen: set[str] = set()
        for ent in entries:
            it = str(ent[0])
            if not it or it == "nothing" or it in seen:
                continue
            seen.add(it)
            parts.append(f"with {it}")
            if len(ent) >= 3 and ent[2]:
                lora_keywords.append(str(ent[2]))

    # 動作 (キャラ側に動作が無いときのみ)
    if not has_motion:
        m_entry = _wpick_entry(cfg.get("motion") or [])
        m = _entry_value(m_entry)
        if m:
            parts.append(m)
        kw = _entry_keyword(m_entry)
        if kw:
            lora_keywords.append(kw)

    # at 場所 (has_where=true なら キャラ文字列に既に場所が含まれているのでスキップ)
    if not has_where:
        at_list = cfg.get("at") or []
        if at_list:
            parts.append(f"at {random.choice(at_list)}")

    # with 明るさ
    lighting_list = cfg.get("lighting") or []
    if lighting_list:
        parts.append(f"with {random.choice(lighting_list)}")

    # 必ず付加 (positive)
    pos_always = str(cfg.get("positive_always") or "").strip()
    if pos_always:
        parts.append(pos_always)

    neg_always = str(cfg.get("negative_always") or "").strip()

    # LoRA キーワード重複排除: 各 entry をカンマ区切りで atomic に分解、
    # 大小文字無視で初出順を保つ。複数 entry に同じ kw (例: "nude") が混じっても 1 回のみ。
    seen_kws: set[str] = set()
    deduped_kws: list[str] = []
    for kw_entry in lora_keywords:
        for atom in str(kw_entry).split(","):
            atom = atom.strip()
            if not atom:
                continue
            key = atom.lower()
            if key in seen_kws:
                continue
            seen_kws.add(key)
            deduped_kws.append(atom)

    return (
        normalize_emphasis(", ".join(parts)),
        normalize_emphasis(neg_always),
        deduped_kws,
        many,
    )


# --------------------------------------------------------------------------- #
# LoRA_param.toml
# --------------------------------------------------------------------------- #
def load_lora_params() -> dict[str, dict]:
    """LoRA_param.toml をロードして {LoRA stem: {"trigger": "...", ...}} を返す。"""
    if not LORA_PARAM_TOML.exists():
        return {}
    with open(LORA_PARAM_TOML, "rb") as f:
        return tomllib.load(f) or {}


def lora_trigger(params: dict, lora: Optional[Path]) -> str:
    """LoRA に紐付く trigger 文字列を返す。未設定なら空文字。

    LoRA の trigger は「その LoRA を発動させる呪文」なので基本的に強調しないと効きが薄い。
    記号無しの素のトリガーはそれだけで 1.3 倍にラップして返し、ユーザが `*` / `**` / `***`
    で明示的に重みを指定していればそれを `normalize_emphasis` の規則 (1.1 / 1.3 / 1.5) に従わせる。
    """
    if lora is None:
        return ""
    entry = params.get(lora.stem) or {}
    raw = str(entry.get("trigger") or "").strip()
    if not raw:
        return ""
    if "*" in raw:
        # 明示制御あり → ユーザの asterisks をそのまま重みに変換 (混在も尊重)
        return normalize_emphasis(raw)
    # 記号なし → デフォルト 1.3 倍でラップ (LoRA trigger は強調が前提)
    return f"({raw}:1.3)"


def build_lora_corpus(loras: list[Path], lora_params: dict) -> dict[str, str]:
    """各 LoRA の検索用 lowercase 文字列を `{stem: corpus}` で返す。`pick_lora_by_keywords` が参照。

    マッチ対象は 3 つの情報源を結合したテキスト:
      1. **LoRA のファイル名** (stem)
      2. **プリセットキーワード**: `extract_lora_trigger_hints` で safetensors メタから抽出した
         activation_text / trigger_words / ss_tag_frequency 上位タグ
      3. **LoRA_param.toml の trigger**: ユーザが設定した手動 trigger 文字列

    起動時に 1 回作って LoRA 数ぶん回しても 200 件で 1〜2 秒程度 (メタ読みのみ)。
    """
    corpus: dict[str, str] = {}
    for lora in loras:
        stem = lora.stem
        try:
            hints = extract_lora_trigger_hints(lora)
        except Exception:
            hints = []
        trigger = str((lora_params.get(stem) or {}).get("trigger") or "")
        parts = [stem, *hints, trigger]
        corpus[stem] = " ".join(p for p in parts if p).lower()
    return corpus


def _parse_keyword_clauses(keywords: list[str]) -> list[list[str]]:
    """keyword 文字列群を AND/OR の入れ子に展開する:
      - `,` で分割 → OR の clause リスト
      - 各 clause を空白で分割 → AND の token リスト
      - 全 lowercase 化、空文字は除外

    例: ['naked, nude', 'swimsuit beach', 'wet']
        → [['naked'], ['nude'], ['swimsuit', 'beach'], ['wet']]
           (naked OR nude OR (swimsuit AND beach) OR wet の意)
    """
    clauses: list[list[str]] = []
    for kw in keywords:
        for clause_str in str(kw).split(","):
            tokens = [t.strip().lower() for t in clause_str.split() if t.strip()]
            if tokens:
                clauses.append(tokens)
    return clauses


def _text_matches_clauses(text: str, clauses: list[list[str]]) -> bool:
    """text を OR of ANDs で判定 (各 clause 内は AND、clause 間は OR)。
    text は事前に lowercase 化済み前提 (本関数では追加変換しない)。
    """
    for clause in clauses:
        if all(token in text for token in clause):
            return True
    return False


def pick_lora_by_keywords(compat_loras: list[Path], keywords: list[str],
                          corpus: dict[str, str] | None = None) -> Optional[Path]:
    """compat_loras から keywords にマッチする LoRA を抽選する。

    **keyword 構文** (大小文字を問わない):
      - `,` 区切り = **OR** (どれかの clause がマッチすれば該当)
      - 空白区切り = **AND** (clause 内の全 token が含まれる必要あり)
      - 例: `"naked oiled, bondage"` → (naked AND oiled) OR bondage

    - 各 LoRA について `corpus[stem]` (build_lora_corpus 製、stem + preset hints + trigger) を
      lowercase で部分一致検査 (corpus 未指定なら stem 単独で検査、後方互換)。
    - match があれば **90% で match 候補から random.choice**、10% で全 compat_loras から random.choice
      (= 従来抽選にフォールバック、用語に縛られない多様性確保)。
    - match が無ければ 100% 全 compat_loras から random.choice (= 従来抽選と同義)。
    - 呼び出し側で `args.lora_prob` ゲートを通す前提。compat_loras が空なら None。
    """
    if not compat_loras:
        return None
    clauses = _parse_keyword_clauses(keywords)
    if clauses:
        matches = []
        for lora in compat_loras:
            text = (corpus.get(lora.stem) if corpus is not None else None) or lora.stem.lower()
            if _text_matches_clauses(text, clauses):
                matches.append(lora)
        if matches and random.random() < 0.90:
            return random.choice(matches)
    return random.choice(compat_loras)


def pick_n_loras_by_keywords(
    compat_loras: list[Path],
    keywords: list[str],
    corpus: dict[str, str] | None = None,
    n_max: int = 3,
    n_min: int = 1,
) -> list[Path]:
    """n_min〜n_max 個のユニークな LoRA を重ね掛け用に抽選する (n は random.randint で決まる)。

    **1 pick = 1 キーワード** の原則。キーワード列をシャッフルして先頭から消費し、各 pick で
    `pick_lora_by_keywords` を **その 1 キーワードだけ** で呼ぶ。これによって 1 回の生成内で
    同じキーワードが 2 回以上 LoRA 抽選に使われない (大小文字無視で dedup 済み)。

    - keywords が空 → 全 LoRA から random.choice で n_target 個
    - keywords が n_target 未満 → 全 kw を使い切り、実 n はその数に縮む
    - 各 pick で `pick_lora_by_keywords` の 90% match / 10% random フォールバックは生きる
    - 既に選ばれた LoRA は次の pick の候補から除外、同じ LoRA は重複しない

    呼び出し側は: ① n=1 なら従来通り 1 LoRA を fuse、② n>1 なら set_adapters で複数を adapter
    として読み、scales = [args.lora_scale / n] * n で重ね掛けする想定。
    """
    if not compat_loras:
        return []
    # n_min/n_max を [1, len(compat_loras)] にクランプ + n_min <= n_max を保証
    hi = max(1, min(n_max, len(compat_loras)))
    lo = max(1, min(n_min, hi))
    n_target = random.randint(lo, hi)

    # キーワードを大小無視で dedup + シャッフル (build_prompt 側で既に dedup 済みでも安全側に再実行)
    seen: set[str] = set()
    unique_kws: list[str] = []
    for kw in (keywords or []):
        atom = str(kw).strip()
        if not atom:
            continue
        key = atom.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_kws.append(atom)
    random.shuffle(unique_kws)

    picked: list[Path] = []
    picked_stems: set[str] = set()

    if not unique_kws:
        # キーワード無し: 純ランダム pick を n_target 回
        for _ in range(n_target):
            candidates = [l for l in compat_loras if l.stem not in picked_stems]
            if not candidates:
                break
            chosen = random.choice(candidates)
            picked.append(chosen)
            picked_stems.add(chosen.stem)
        return picked

    # 1 pick = 1 kw 消費、kw が尽きたら停止 (実 n は min(n_target, len(unique_kws)))
    n_actual = min(n_target, len(unique_kws))
    for i in range(n_actual):
        candidates = [l for l in compat_loras if l.stem not in picked_stems]
        if not candidates:
            break
        chosen = pick_lora_by_keywords(candidates, [unique_kws[i]], corpus)
        if chosen is None:
            break
        picked.append(chosen)
        picked_stems.add(chosen.stem)
    return picked


def is_lora_skipped(params: dict, lora: Optional[Path]) -> bool:
    """LoRA_param.toml で skip = true がマークされていれば True。
    Target modules 不一致などで読み込み不能と分かった LoRA を恒久的に除外するためのフラグ。
    """
    if lora is None:
        return False
    return bool((params.get(lora.stem) or {}).get("skip", False))


def lora_scale_for(params: dict, lora: Optional[Path], default_scale: float) -> float:
    """LoRA に scale_range = [min, max] が設定されていれば uniform sample、無ければ default を返す。
    slider 系 LoRA で派生ごとに効果量を揺らすのに使う (負値で効果反転、正値で強化)。
    """
    if lora is None:
        return default_scale
    entry = params.get(lora.stem) or {}
    rng = entry.get("scale_range")
    if isinstance(rng, list) and len(rng) == 2:
        try:
            lo, hi = float(rng[0]), float(rng[1])
            return random.uniform(lo, hi)
        except (TypeError, ValueError):
            pass
    return default_scale


def extract_lora_trigger_hints(lora_path: Path) -> list[str]:
    """LoRA safetensors のメタデータから候補 trigger を最大 5 個抽出する。
    取得できなければ空リスト。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(lora_path), framework="pt") as f:
            meta = f.metadata() or {}
    except Exception:
        return []
    candidates: list[str] = []
    # 直接的なフィールド (フォーマットによっては存在)
    for key in ("activation_text", "trigger_words", "ss_keep_tokens"):
        v = meta.get(key)
        if v:
            candidates.append(str(v))
    # ss_tag_frequency: {subset: {tag: count}} を JSON で持つ
    tf_raw = meta.get("ss_tag_frequency")
    if tf_raw:
        try:
            tf_data = json.loads(tf_raw)
            counts: dict[str, int] = {}
            for subset in (tf_data or {}).values():
                if isinstance(subset, dict):
                    for tag, c in subset.items():
                        try:
                            counts[tag] = counts.get(tag, 0) + int(c)
                        except (TypeError, ValueError):
                            continue
            top = sorted(counts.items(), key=lambda x: -x[1])[:5]
            candidates.extend(t for t, _ in top)
        except Exception:
            pass
    # 重複排除しつつ順序保持。数字のみ / "None" / 過度に短い等のノイズは弾く。
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = c.strip().strip('"').strip("'").strip()
        if not c or c in seen:
            continue
        if c.lower() == "none":
            continue
        if c.isdigit():
            continue
        if len(c) < 3:  # '1', 'a' 等の単独文字
            continue
        if len(c) > 60:
            continue
        seen.add(c)
        out.append(c)
    return out[:5]


def update_lora_param_toml(loras: list[Path]) -> int:
    """LoRA_param.toml に未記述の LoRA についてコメントアウトのテンプレートを追記する。
    重複は active entry / `# [stem]` 両方を見て避ける。返り値は追加件数。
    """
    text = LORA_PARAM_TOML.read_text(encoding="utf-8") if LORA_PARAM_TOML.exists() else ""

    def already_present(stem: str) -> bool:
        # active / コメントアウト両方を判定: `[stem]` または `["stem"]` の出現
        return f"[{stem}]" in text or f'["{stem}"]' in text

    missing = [p for p in loras if not already_present(p.stem)]
    if not missing:
        return 0

    new_lines: list[str] = ["", f"# --- auto-added {datetime.now().strftime('%Y-%m-%d')} ---"]
    for lora in sorted(missing, key=lambda p: p.stem.lower()):
        hints = extract_lora_trigger_hints(lora)
        # キーは半角空白や記号があれば quote
        stem = lora.stem
        needs_quote = bool(re.search(r"[^A-Za-z0-9_\-.]", stem))
        key = f'"{stem}"' if needs_quote else stem
        # trigger 値の TOML basic-string エスケープ (\ は \\ に、" は \" に)
        trig_val = ", ".join(hints).replace("\\", "\\\\").replace('"', '\\"')
        new_lines.append(f"# [{key}]")
        new_lines.append(f'# trigger = "{trig_val}"')
        # slider 名なら scale_range のテンプレートも添える (典型 -2..+2)
        if "slider" in stem.lower():
            new_lines.append("# scale_range = [-2.0, 2.0]   # 負: 効果を打ち消す / 正: 効果を強める")
        new_lines.append("")

    with open(LORA_PARAM_TOML, "a", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    return len(missing)


# --------------------------------------------------------------------------- #
# base_setting.toml
# --------------------------------------------------------------------------- #
_SETTING_HEADER = (
    "# base ごとの設定 (max_score / min_score / like / anomaly / upscale_model)。\n"
    "# max_score: 初期 0。生成のたびに `1800-経過秒` を計算し、これより大きければ更新\n"
    "#            (= 最速時の score、1..1800)。0 は未計測フラグ。\n"
    "#            計測は --gear high のときのみ行う (low は Hires Fix / upscale OFF で\n"
    "#            elapsed が代表値にならないのでスキップ)。\n"
    "# min_score: 初期 0。max_score と同時に書き込まれ、生成のたびにこれより小さければ更新\n"
    "#            (= 最遅時の score)。base 内の振れ幅を把握するため。\n"
    "# like:      初期 0。好みで手動加減点 (任意の整数)。\n"
    "# anomaly:   初期 0。奇形が出たら手動加点。抽選には影響せず、生成 step 数に加算\n"
    "#            (use_step = set_step + anomaly)。\n"
    "# upscale_model: \"anime\" / \"real\" / \"mix\" / \"\" のいずれか。\n"
    "#            \"anime\" or \"real\" → アップスケール時に当該モデルを base 別に強制。\n"
    "#            \"mix\" or 未設定 (空文字) → `--upscale-model` 引数に従う (既定 anime)。\n"
    "# 抽選重み = max(1, (max_score + min_score) / 2 + like)。\n"
    "# 抽選パターン: 初回は強制未計測 / 以降 1/3 未計測・2/3 計測済み /\n"
    "#               計測済みを 2 回連続で引いたら次は強制未計測。\n"
    "\n"
)


def load_base_settings() -> dict[str, dict]:
    """base_setting.toml から {stem: {max_score, min_score, like, anomaly}} を返す。
    旧形式の `score` フィールド単体が残っているエントリは自動的に max_score=min_score=score に
    展開して返す (在メモリ変換、ファイルは次の save 時に新形式へ書き換わる)。
    """
    if not BASE_SETTING_TOML.exists():
        return {}
    with open(BASE_SETTING_TOML, "rb") as f:
        data = tomllib.load(f) or {}
    out: dict[str, dict] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        try:
            if "score" in v and "max_score" not in v:
                # 旧形式 → max_score=min_score=score に展開
                s = max(0, min(1800, int(v.get("score", 0))))
                mx, mn = s, s
            else:
                mx = max(0, min(1800, int(v.get("max_score", 0))))
                mn = max(0, min(1800, int(v.get("min_score", 0))))
                if mn > mx:  # 整合性: min > max なら入れ替え
                    mx, mn = mn, mx
        except (TypeError, ValueError):
            mx, mn = 0, 0
        try:
            like = int(v.get("like", 0))
        except (TypeError, ValueError):
            like = 0
        try:
            anomaly = max(0, int(v.get("anomaly", 0)))
        except (TypeError, ValueError):
            anomaly = 0
        # upscale_model は "anime" / "real" / "mix" / "" のみ許容。
        # それ以外 (手動編集ミス、エディタの auto-complete で base 名が入った等) は
        # 空文字に正規化して警告を出す。次の save で正しい値に書き戻される。
        um_raw = str(v.get("upscale_model") or "").strip()
        if um_raw.lower() in ("", "anime", "real", "mix"):
            um = um_raw
        else:
            print(L(f"  [base_setting] {k!r}: upscale_model={um_raw!r} は不正値 → '' に正規化",
                    f"  [base_setting] {k!r}: upscale_model={um_raw!r} is invalid → normalized to ''"))
            um = ""
        out[k] = {
            "max_score": mx, "min_score": mn,
            "like": like, "anomaly": anomaly,
            "upscale_model": um,
        }
    return out


def save_base_settings(settings: dict[str, dict]) -> None:
    """base_setting.toml をフルリライト (header コメント + ソート済み tables、新形式)。

    **書き込み前にディスクから like / anomaly を読み戻し** て in-memory にマージする。
    これにより生成中にユーザが TOML を手動編集して `like` / `anomaly` を変えても、
    次の save で in-memory の値で上書きされて消えるのを防げる。
    `max_score` / `min_score` は in-memory (= 計測値) を信頼してそのまま書き込む。
    in-memory の各 entry の like / anomaly もこの merge でファイル値に同期される (副作用)。
    """
    if BASE_SETTING_TOML.exists():
        on_disk = load_base_settings()
        for stem, entry in settings.items():
            ref = on_disk.get(stem)
            if ref is not None:
                entry["like"] = ref.get("like", entry.get("like", 0))
                entry["anomaly"] = ref.get("anomaly", entry.get("anomaly", 0))
                entry["upscale_model"] = ref.get("upscale_model", entry.get("upscale_model", ""))
    lines: list[str] = [_SETTING_HEADER.rstrip("\n")]
    for stem in sorted(settings.keys(), key=str.lower):
        s = settings[stem]
        needs_quote = bool(re.search(r"[^A-Za-z0-9_\-.]", stem))
        key = f'"{stem}"' if needs_quote else stem
        um = str(s.get("upscale_model") or "").strip()
        lines.append("")
        lines.append(f"[{key}]")
        lines.append(f"max_score = {int(s.get('max_score', 0))}")
        lines.append(f"min_score = {int(s.get('min_score', 0))}")
        lines.append(f"like = {int(s.get('like', 0))}")
        lines.append(f"anomaly = {int(s.get('anomaly', 0))}")
        lines.append(f'upscale_model = "{um}"')
    BASE_SETTING_TOML.write_text("\n".join(lines) + "\n", encoding="utf-8")


def migrate_score_to_minmax() -> int:
    """base_setting.toml 内の旧 score フィールドを max_score / min_score の組に展開する。
    既に新形式 (全エントリに max_score がある) なら何もしない。返り値は変換件数。
    """
    if not BASE_SETTING_TOML.exists():
        return 0
    with open(BASE_SETTING_TOML, "rb") as f:
        raw = tomllib.load(f) or {}
    legacy = sum(
        1 for v in raw.values()
        if isinstance(v, dict) and "score" in v and "max_score" not in v
    )
    if legacy == 0:
        return 0
    # load_base_settings がメモリ上で変換済みのものを返すので、それを save し直す
    save_base_settings(load_base_settings())
    return legacy


def migrate_base_score_toml() -> int:
    """旧 base_score.toml (フラット key=value 形式) を base_setting.toml に変換。
    既に base_setting.toml がある or 旧ファイルが無ければ何もしない。返り値は移行件数。
    """
    if BASE_SETTING_TOML.exists() or not BASE_SCORE_TOML.exists():
        return 0
    try:
        with open(BASE_SCORE_TOML, "rb") as f:
            old = tomllib.load(f) or {}
    except Exception:
        return 0
    settings: dict[str, dict] = {}
    for k, v in old.items():
        if isinstance(v, dict):  # 既に新形式の table が混じっていた場合
            s = max(0, min(1800, int(v.get("score", 0))))
            settings[k] = {
                "max_score": s, "min_score": s,
                "like": int(v.get("like", 0)),
                "anomaly": max(0, int(v.get("anomaly", 0))),
            }
            continue
        try:
            s = max(0, min(1800, int(v)))
        except (TypeError, ValueError):
            continue
        settings[k] = {"max_score": s, "min_score": s, "like": 0, "anomaly": 0}
    save_base_settings(settings)
    BASE_SCORE_TOML.rename(BASE_SCORE_TOML.with_suffix(".toml.bak"))
    return len(settings)


def base_weight(setting: dict) -> int:
    """抽選重み = max(1, (max_score + min_score) / 2 + like)。
    base 内の生成時間振れ幅を中点で代表し、like で好み補正。
    """
    mx = int(setting.get("max_score", 0))
    mn = int(setting.get("min_score", 0))
    like = int(setting.get("like", 0))
    return max(1, (mx + mn) // 2 + like)


def base_upscale_model(setting: dict, default: str) -> list[str]:
    """base 個別の upscale_model 設定を解決し、適用すべきモデル名のリストを返す。
    "anime" / "real" → そのモデルのみ ([それ])
    "mix"            → anime と real の両方をかける ([anime, real]) → 別ファイルで保存される
    "" / 未設定      → [default] (`--upscale-model` 引数に従う)
    """
    v = str((setting or {}).get("upscale_model") or "").strip().lower()
    if v == "mix":
        return ["anime", "real"]
    if v in ("anime", "real"):
        return [v]
    return [default]


def pick_base(bases: list[Path], settings: dict[str, dict],
              state: dict | None = None) -> Path:
    """ベースを抽選する。state を渡すと連続抽選パターンを制御する。

    ルール (state を渡したとき):
      - 最初の呼び出し: 未計測 base があれば必ず未計測を選ぶ (初回計測の優先)。
      - 以降:
          1/3 の確率で未計測 base から (あれば)、2/3 の確率で計測済み base から。
          ただし計測済みを 2 回連続で引いた直後の 3 回めは、未計測があれば強制的に未計測。
      - 計測済み / 未計測 の片方しか無い場合はそちらに寄せる (streak だけ更新)。

    state は呼び出し側が保持する dict (例: `{}` で初期化)。本関数が `scored_streak` を更新する。
    state=None で渡すと連続制御なしの素の 1/3-2/3 抽選になる (後方互換)。
    """
    def is_scored(b: Path) -> bool:
        s = settings.get(b.stem)
        return bool(s and s.get("max_score", 0) > 0)
    scored = [b for b in bases if is_scored(b)]
    unscored = [b for b in bases if not is_scored(b)]

    if not scored:
        if state is not None:
            state["scored_streak"] = 0
        return random.choice(unscored)
    if not unscored:
        if state is not None:
            state["scored_streak"] = state.get("scored_streak", 0) + 1
        return random.choices(scored, weights=[base_weight(settings[b.stem]) for b in scored], k=1)[0]

    # 両方存在する: state があれば streak ルール、無ければ純粋 1/3
    if state is not None:
        # 初回 (キー未設定) は streak=2 とみなして強制 unscored を引かせる
        streak = state.get("scored_streak", 2)
        force_unscored = streak >= 2
    else:
        force_unscored = False

    if force_unscored or random.random() < 1.0 / 3.0:
        if state is not None:
            state["scored_streak"] = 0
        return random.choice(unscored)
    if state is not None:
        state["scored_streak"] = state.get("scored_streak", 0) + 1
    return random.choices(scored, weights=[base_weight(settings[b.stem]) for b in scored], k=1)[0]


# --------------------------------------------------------------------------- #
# safetensors の分類・チェック
# --------------------------------------------------------------------------- #
def classify_tensor(path: Path) -> str:
    """base / lora / embedding / controlnet / inpainting / broken を返す。
    メタデータのみ参照しメモリには展開しない。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            if not keys:
                return "broken"
            # ControlNet 判別 (LoRA より先): diffusers 形式 (controlnet_*) と
            # A1111 / 原実装形式 (control_model.* / zero_convs) と
            # LLLite (sd-forge-controllllite、SDXL 専用) の lllite_* プレフィックス を拾う
            if any(k.startswith("controlnet_") or "controlnet_cond_embedding" in k
                   or "control_model." in k or "zero_convs" in k
                   or k.startswith("lllite_") for k in keys):
                return "controlnet"
            if any("lora_" in k or ".lora_" in k for k in keys):
                return "lora"
            if any(k in {"emb_params", "string_to_param"} or k.startswith("string_to_param") for k in keys):
                return "embedding"
            # Textual Inversion / Embedding: キー数が極端に少ない (base は数百〜数千キー)。
            # 各テンソルは 1D、または 2D なら最終次元が text encoder の hidden dim
            # (768=CLIP-L / 1280=bigG / 1024 / 2048)。
            # 例: ng_deepnegative_v1_75t は [75,768] (prod 57600 で旧 50k 閾値を超えるが TI embedding)。
            if len(keys) <= 8:
                shapes = [list(f.get_slice(k).get_shape()) for k in keys]
                _EMB_DIMS = {768, 1024, 1280, 2048}

                def _is_emb_shape(s: list[int]) -> bool:
                    if len(s) == 1:
                        return True
                    if len(s) == 2:
                        return s[-1] in _EMB_DIMS or math.prod(s) < 50_000
                    return False

                if shapes and all(_is_emb_shape(s) for s in shapes):
                    return "embedding"
        # base 判定後に inpainting checkpoint (UNet 入力 9ch、通常 pipe では使えない) を名前で除外
        if "inpaint" in path.stem.lower():
            return "inpainting"
        return "base"
    except Exception:
        return "broken"


def detect_sd_version(path: Path) -> str:
    """SD15 / SDXL を判別。メタデータのみ参照。"""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        if any("conditioner.embedders" in k or "text_encoder_2" in k for k in keys):
            return "sdxl"
        for k in keys:
            if k.endswith(("to_k.weight", "to_q.weight")):
                shape = list(f.get_slice(k).get_shape())
                if shape and shape[-1] >= 2048:
                    return "sdxl"
    return "sd15"


def detect_controlnet_version(path: Path) -> str | None:
    """ControlNet が SD15 / SDXL のどちら向けか推定。判定不能は None。

    判定基準:
      - cross-attention の to_k / to_q の最終次元 (= cross-attention context dim)
        SDXL: 2048 (CLIP-L 768 + bigG 1280 を concat)
        SD15: 768
      - もしくは SDXL 由来の特徴的キー (text_encoder_2 など) を含む
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            if any("text_encoder_2" in k or "_te2_" in k for k in keys):
                return "sdxl"
            for k in keys:
                # diffusers 形式 / A1111 形式どちらにも対応 (末尾だけ見る)
                if k.endswith(("to_k.weight", "to_q.weight", "to_v.weight")):
                    shape = list(f.get_slice(k).get_shape())
                    if shape:
                        inner = shape[-1]
                        if inner >= 1500:
                            return "sdxl"
                        if 600 <= inner <= 800:
                            return "sd15"
    except Exception:
        return None
    return None


def lora_target_version(path: Path) -> str | None:
    """LoRA が SD15 / SDXL のどちら向けか推定。判定不能は None。

    注意: to_k/to_v は self-attn(attn1) と cross-attn(attn2) が混在し、最初に見つけた
    to_k (self-attn の 320/640/1280 次元) で即断すると SDXL を SD15 と誤判定する。
    そのため **全 to_k/to_v の lora_down 最終次元を集めてから** 判定する:
    cross-attn context 次元が 2048=SDXL / 768=SD15。te2/clip_g キーがあれば即 SDXL。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            if any("text_encoder_2" in k or "_te2_" in k or "lora_te2_" in k
                   or "clip_g" in k.lower() for k in keys):
                return "sdxl"
            dims: set[int] = set()
            for k in keys:
                if ("to_k" in k or "to_v" in k) and ("lora_down.weight" in k or "lora_A.weight" in k):
                    t = f.get_tensor(k)
                    if t.dim() >= 2:
                        dims.add(int(t.shape[-1]))
            if any(d >= 1500 for d in dims):   # cross-attn context 2048 → SDXL
                return "sdxl"
            if 768 in dims:                     # cross-attn context 768 → SD15
                return "sd15"
    except Exception:
        return None
    return None


def detect_embedding_version(path: Path) -> str | None:
    """Textual Inversion Embedding が SD15 / SDXL のどちら向けか推定。判定不能は None。

    判定基準:
      - keys に `clip_g` / `clip_g_text_model` / `text_encoder_2` / `_te2_` 等を含む → SDXL
        (SDXL の TI は CLIP-L (768) + OpenCLIP-bigG (1280) 双方の埋め込みを持つ二段構造)
      - 主テンソルの最終次元が 1280 (= bigG) もしくは embeddings 形式で 2 種類混在 → SDXL
      - 単純な 768 dim の単一テンソル → SD15
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            # SDXL 由来の特徴的キー
            if any("clip_g" in k.lower() or "_te2_" in k.lower() or "text_encoder_2" in k for k in keys):
                return "sdxl"
            # shape ベースで判定: 1280 dim が含まれていれば SDXL
            for k in keys:
                t = f.get_tensor(k)
                if t.dim() >= 1:
                    last = t.shape[-1]
                    if last == 1280:
                        return "sdxl"
                    if last == 768:
                        # 単独 768 → SD15 だが、もう一方の tensor で 1280 を見つけたら SDXL なので
                        # ここでは一旦 break せず、全 keys を確認した後に判定
                        pass
            # 1280 が無く 768 のみだったら SD15
            for k in keys:
                t = f.get_tensor(k)
                if t.dim() >= 1 and t.shape[-1] == 768:
                    return "sd15"
    except Exception:
        return None
    return None


def _flatten_tensors(obj, prefix: str = "") -> dict[str, torch.Tensor]:
    """ネストした dict から Tensor だけを再帰的に拾い、ドット区切りキーで平坦化"""
    out: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        out[prefix or "tensor"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_tensors(v, key))
    return out


def convert_to_safetensors(path: Path) -> Path:
    """.ckpt / .pt を .safetensors に変換。a1111 TI 形式や生 state_dict にも対応。"""
    out = path.with_suffix(".safetensors")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(raw, torch.Tensor):
        sd = {path.stem: raw}
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        sd = {k: v for k, v in raw["state_dict"].items() if isinstance(v, torch.Tensor)}
    elif isinstance(raw, dict) and "string_to_param" in raw:
        # a1111 Textual Inversion 形式: string_to_param['*'] に Tensor
        params = raw.get("string_to_param") or {}
        token = raw.get("name") or path.stem
        tensor = params.get("*") if isinstance(params, dict) else None
        if isinstance(tensor, torch.Tensor):
            sd = {str(token): tensor}
        else:
            sd = _flatten_tensors(params, prefix=str(token))
    elif isinstance(raw, dict):
        # フラットな state_dict
        sd = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
        if not sd:
            sd = _flatten_tensors(raw)
    else:
        raise ValueError(L(f"対応できない形式: {type(raw).__name__}", f"unsupported format: {type(raw).__name__}"))

    if not sd:
        raise ValueError(L("Tensor が見つかりません", "no tensors found"))
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, str(out))
    return out


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(L(f"キャッシュ保存失敗: {e}", f"cache save failed: {e}"), flush=True)


def _is_torch_zip(path: Path) -> bool:
    """torch.save のアーカイブ (data.pkl 入り) は内部 zip。配下に data.pkl があれば真。"""
    try:
        with zipfile.ZipFile(path) as zf:
            return any(n.endswith("data.pkl") or n.endswith("/data.pkl") for n in zf.namelist())
    except Exception:
        return False


def _safe_extract_tensors(zip_path: Path, dest: Path) -> int:
    """zip_path 内の tensor 拡張子のメンバを dest にフラット展開。展開件数を返す。"""
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name  # path traversal 回避でベース名のみ
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


def migrate_high_to_xl(cache: dict | None = None, verbose: bool = False) -> int:
    """旧 2_9_2_hightensors の SDXL ファイルを 5_1_XL_base / 5_2_XL_LoRA / 5_3_XL_Embedding に
    一度だけ振り分ける。`shutil.move` 完了後にキャッシュキーも差し替える。

    cache=None なら load_cache() を使う (このとき save_cache も内部で行う)。
    呼び出し側で cache を持っている場合 (check_tensors) は同じ dict を渡すと整合性が保たれる。
    戻り値は移動件数。
    """
    if not HIGH_DIR.exists():
        return 0
    XL_BASE_DIR.mkdir(exist_ok=True)
    XL_LORA_DIR.mkdir(exist_ok=True)
    XL_EMBED_DIR.mkdir(exist_ok=True)
    save_at_end = False
    if cache is None:
        cache = load_cache()
        save_at_end = True
    moved = 0
    for st in sorted(HIGH_DIR.glob("*.safetensors")):
        key = str(st)
        entry = dict(cache.get(key) or {})
        kind = entry.get("kind")
        if not kind:
            try:
                kind = classify_tensor(st)
            except Exception:
                kind = "broken"
            entry["kind"] = kind
        if kind == "lora":
            target = XL_LORA_DIR
            entry.setdefault("lora_version", "sdxl")
        elif kind == "embedding":
            target = XL_EMBED_DIR
            entry.setdefault("embedding_version", "sdxl")
        elif kind == "base":
            target = XL_BASE_DIR
            entry.setdefault("base_version", "sdxl")
        else:
            if verbose:
                print(f"[migrate] {st.name} kind={kind} → 2_9_1_lowtensors", flush=True)
            try:
                shutil.move(str(st), str(LOW_DIR / st.name))
                cache.pop(key, None)
                moved += 1
            except Exception as e:
                print(L(f"  移動失敗: {e}", f"  move failed: {e}"), flush=True)
            continue
        new_path = target / st.name
        if new_path.exists():
            # 衝突: 旧 2_9_2 側は不要 (5_x が正)
            try:
                shutil.move(str(st), str(LOW_DIR / st.name))
                cache.pop(key, None)
            except Exception as e:
                print(L(f"  移動失敗: {e}", f"  move failed: {e}"), flush=True)
            continue
        try:
            shutil.move(str(st), str(new_path))
            cache[str(new_path)] = entry
            cache.pop(key, None)
            moved += 1
            if verbose:
                print(f"[migrate] {st.name} → {target.name}", flush=True)
        except Exception as e:
            print(L(f"  移動失敗 ({st.name} → {target.name}): {e}",
                    f"  move failed ({st.name} → {target.name}): {e}"), flush=True)
    if save_at_end and moved > 0:
        save_cache(cache)
    return moved


def _classified_destination(kind: str, entry: dict) -> Path:
    """kind + entry から物理移動先 dir を返す。SDXL は 5_x 系へ、broken/inpainting は LOW へ。"""
    if kind == "broken" or kind == "inpainting":
        return LOW_DIR
    if kind == "controlnet":
        # ControlNet は種類 (Union / LLLite / ControlLoRA / depth / canny / softedge / ...) が多く、
        # SD15 / SDXL の自動判別 (detect_controlnet_version) も不安定。
        # ⇒ すべて XL_CONTROLNET_DIR に入れる。混入した SD15 物は手動で取り除く運用。
        return XL_CONTROLNET_DIR
    if kind == "lora":
        return XL_LORA_DIR if entry.get("lora_version") == "sdxl" else LORA_DIR
    if kind == "embedding":
        return XL_EMBED_DIR if entry.get("embedding_version") == "sdxl" else EMBED_DIR
    if kind == "base":
        return XL_BASE_DIR if entry.get("base_version") == "sdxl" else BASE_DIR
    return LOW_DIR  # 念のため


def check_tensors() -> tuple[list[Path], list[tuple[Path, str | None]], list[Path]]:
    """4 ディレクトリ構造 (2_0 受入 / 2_1 base / 2_2 LoRA / 2_3 Embedding) を回って
    分類済みリストを返す。2_0_tensors の未分類は分類して typed dir へ移動する。
    2_9_1_lowtensors / 2_9_2_hightensors は退避先、2_9_3_anomalytensors はノータッチ。
    """
    for d in (BASE_DIR, LORA_DIR, EMBED_DIR, LOW_DIR, HIGH_DIR,
              XL_BASE_DIR, XL_LORA_DIR, XL_EMBED_DIR, XL_CONTROLNET_DIR):
        d.mkdir(exist_ok=True)
    cache = load_cache()

    # 旧 SDXL 隔離 (2_9_2_hightensors) は 5_x に一度だけ移行
    migrate_high_to_xl(cache, verbose=True)

    # ---- Phase 1: 2_0_tensors の zip 展開 / ckpt 変換 ----
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
            if n > 0:
                print(L(f"  展開 {n} 件 → 2_0_tensors / 元は 2_9_1_lowtensors",
                        f"  extracted {n} file(s) → 2_0_tensors / original → 2_9_1_lowtensors"), flush=True)
                shutil.move(str(path), str(LOW_DIR / path.name))
            else:
                print(L(f"  展開対象なし → 2_9_1_lowtensors",
                        f"  nothing to extract → 2_9_1_lowtensors"), flush=True)
                shutil.move(str(path), str(LOW_DIR / path.name))
        except Exception as e:
            print(L(f"  展開失敗 ({e}) → 2_9_1_lowtensors",
                    f"  extraction failed ({e}) → 2_9_1_lowtensors"), flush=True)
            shutil.move(str(path), str(LOW_DIR / path.name))

    for path in sorted([*TENSORS_DIR.glob("*.ckpt"), *TENSORS_DIR.glob("*.pt"), *TENSORS_DIR.glob("*.bin")]):
        print(L(f"[変換] {path.name} ({_human_size(path.stat().st_size)})",
                f"[convert] {path.name} ({_human_size(path.stat().st_size)})"), flush=True)
        try:
            convert_to_safetensors(path)
            path.unlink()
        except Exception as e:
            print(L(f"  変換失敗 ({e}) → 2_9_1_lowtensors",
                    f"  conversion failed ({e}) → 2_9_1_lowtensors"), flush=True)
            shutil.move(str(path), str(LOW_DIR / path.name))

    # ---- Phase 2a: 全 typed dir の safetensors を hash 取得 ----
    bases: list[Path] = []
    loras: list[tuple[Path, str | None]] = []
    embeds: list[Path] = []

    # XL_CONTROLNET_DIR は **scan しない** — ControlNet 種類が多すぎて (Union / LLLite /
    # ControlLoRA / depth / canny / softedge ...) classify_tensor で誤判定しやすく、
    # ユーザが手で揃えた 5_4 を check_tensors が解体する事故が起きる。
    # 2_9_3_anomalytensors と同じ「手動配置、scan しない」規約。
    # 2_0_tensors に新規 DL された ControlNet は classify_tensor → 5_4 に **入れる方向のみ** 行う。
    scan_dirs = [TENSORS_DIR, BASE_DIR, LORA_DIR, EMBED_DIR,
                 XL_BASE_DIR, XL_LORA_DIR, XL_EMBED_DIR]
    all_files: list[Path] = []
    for d in scan_dirs:
        all_files.extend(d.glob("*.safetensors"))
    all_files.sort()

    # hash → [(Path, kind, cache_entry, cache_key), ...] の対応表を作る
    hash_groups: dict[str, list[tuple[Path, str, dict, str]]] = {}
    for st in all_files:
        key = str(st)
        stat = st.stat()
        entry = dict(cache.get(key) or {})
        cached_hit = (
            entry.get("size") == stat.st_size
            and entry.get("mtime") == stat.st_mtime
            and "hash" in entry and "kind" in entry
        )
        if cached_hit:
            digest = entry["hash"]
            kind = entry["kind"]
            # 旧キャッシュ補正: 名前で inpainting と判定し直す
            if kind == "base" and "inpaint" in st.stem.lower():
                kind = "inpainting"
                entry["kind"] = "inpainting"
                cache[key] = entry
        else:
            print(f"[hash] {st.name} ({_human_size(stat.st_size)})", flush=True)
            try:
                digest = file_sha256(st)
            except Exception as e:
                print(L(f"  ハッシュ失敗 ({e}) → 2_9_1_lowtensors",
                        f"  hash failed ({e}) → 2_9_1_lowtensors"), flush=True)
                shutil.move(str(st), str(LOW_DIR / st.name))
                cache.pop(key, None)
                continue
            kind = classify_tensor(st)
            entry = {"size": stat.st_size, "mtime": stat.st_mtime, "hash": digest, "kind": kind}
            cache[key] = entry
        hash_groups.setdefault(digest, []).append((st, kind, entry, key))

    # ---- Phase 2b: 重複解決 (mtime 最新を残し、古いものを 2_9_1_lowtensors へ) ----
    survivors: list[tuple[Path, str, dict, str]] = []
    for digest, group in hash_groups.items():
        if len(group) == 1:
            survivors.append(group[0])
            continue
        # mtime 降順で並べる: 最新が group_sorted[0]
        group_sorted = sorted(
            group,
            key=lambda item: item[0].stat().st_mtime if item[0].exists() else 0,
            reverse=True,
        )
        keeper = group_sorted[0]
        for old_item in group_sorted[1:]:
            old_path, _, _, old_key = old_item
            print(L(f"  重複: {old_path.name} (古い、mtime={old_path.stat().st_mtime:.0f}) "
                    f"← keeper={keeper[0].name} → 2_9_1_lowtensors",
                    f"  duplicate: {old_path.name} (older, mtime={old_path.stat().st_mtime:.0f}) "
                    f"← keeper={keeper[0].name} → 2_9_1_lowtensors"), flush=True)
            try:
                shutil.move(str(old_path), str(LOW_DIR / old_path.name))
            except Exception as e:
                print(L(f"    移動失敗: {e}", f"    move failed: {e}"), flush=True)
            cache.pop(old_key, None)
        survivors.append(keeper)

    # ---- Phase 2c: 生き残りを分類 + 移動 + リストに追加 ----
    for st, kind, entry, key in survivors:
        # version 判定をキャッシュに追加 (LoRA / base 用)
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

        # 移動が必要なら実行 (同一 dir ならスキップ)
        if st.parent != target_dir:
            new_path = target_dir / st.name
            try:
                shutil.move(str(st), str(new_path))
                cache[str(new_path)] = cache.pop(key, entry)
                key = str(new_path)
                st = new_path
                if target_dir in (LOW_DIR, HIGH_DIR):
                    print(f"  {st.name}: {kind} → {target_dir.name}", flush=True)
                else:
                    print(f"  {st.name}: → {target_dir.name}", flush=True)
            except Exception as e:
                print(L(f"  移動失敗 ({st.name} → {target_dir.name}): {e}",
                        f"  move failed ({st.name} → {target_dir.name}): {e}"), flush=True)
                continue

        # 退避先 (LOW / HIGH) は使用リストから除外
        if target_dir == LOW_DIR or target_dir == HIGH_DIR:
            if target_dir == LOW_DIR:
                cache.pop(key, None)  # broken/inpainting はキャッシュからも除外
            continue

        # 採用 (使用するリストに追加)
        # base / embedding の SDXL 物は 5_x dir にあり、SD15 用 returns には含めない
        # (highrender 側が XL_BASE_DIR / XL_EMBED_DIR から直接列挙する)
        # LoRA は version タグ付きでまとめて返し、呼び出し側で filter する既存挙動を維持
        if kind == "base" and target_dir == BASE_DIR:
            bases.append(st)
        elif kind == "lora":
            loras.append((st, entry.get("lora_version")))
        elif kind == "embedding" and target_dir == EMBED_DIR:
            embeds.append(st)

    # 既に存在しないパスのエントリは GC
    for k in list(cache.keys()):
        if not Path(k).exists():
            del cache[k]
    save_cache(cache)
    return bases, loras, embeds


def resolve_tensor_path(name: str) -> Path | None:
    """`--base NAME` / `--lora NAME` 等で渡された名前を分類ディレクトリ群から解決する。
    絶対パス指定 / `name` そのもの / `name.safetensors` のいずれも、TENSORS_DIR + 各 typed dir を
    順に探して最初にヒットしたものを返す。見つからなければ None。
    """
    if not name:
        return None
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    search_dirs = [TENSORS_DIR, BASE_DIR, LORA_DIR, EMBED_DIR, HIGH_DIR,
                   XL_BASE_DIR, XL_LORA_DIR, XL_EMBED_DIR]
    suffixes = ("",) if name.lower().endswith(".safetensors") else ("", ".safetensors")
    for d in search_dirs:
        for suf in suffixes:
            cand = d / f"{name}{suf}"
            if cand.exists():
                return cand
    # 相対パスとして cwd 起点でも一応見る
    if p.exists():
        return p
    return None


# --------------------------------------------------------------------------- #
# パイプライン
# --------------------------------------------------------------------------- #
def _apply_scheduler(pipe, name: str) -> None:
    """name に従って pipe.scheduler を差し替える。'default' は元のまま。"""
    if name == "default":
        return
    from diffusers import (
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        UniPCMultistepScheduler,
    )
    cfg = pipe.scheduler.config
    if name == "dpmpp-2m-karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            cfg, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )
    elif name == "dpmpp-sde-karras":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            cfg, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )
    elif name == "euler-a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
    elif name == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(cfg)


def build_pipeline(base: Path, version: str, device: str, dtype: torch.dtype, vram_gb: float,
                   scheduler: str = "dpmpp-2m-karras", vae_fp32: bool = True,
                   progress: bool = True):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    cls = StableDiffusionXLPipeline if version == "sdxl" else StableDiffusionPipeline
    pipe = cls.from_single_file(str(base), torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=not progress)

    _apply_scheduler(pipe, scheduler)

    # SDXL: pipe.upcast_vae() で VAE 全体 + post_quant_conv / decoder.conv_in /
    # decoder.mid_block の dtype 整合まで取ってくれる (deprecated だが 0.38 系では依然これが安全)。
    # SD15: upcast_vae() は無いため、VAE を fp32 化したうえで decode に来る latents (UNet が fp16
    #       で吐く) を fp32 にキャストするラッパを噛ませる。これをやらないと "Half vs Float" になる。
    #       一部 SD15 base (uberrealisticDreamy_pfg 等) は fp16 VAE で NaN → 灰色画像になるため、
    #       ディテール優先方針からも fp32 decode を既定とする。
    if vae_fp32:
        if version == "sdxl" and hasattr(pipe, "upcast_vae"):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                pipe.upcast_vae()
            # upcast_vae は VAE 自体を fp32 にするが、UNet が吐く bf16 latents を decode に渡す
            # 直前で型を揃えてくれない (diffusers 0.38 の SDXL pipeline 仕様)。明示キャストを挟む。
            pipe.vae.to(dtype=torch.float32)
            _orig_decode_xl = pipe.vae.decode
            def _safe_decode_xl(z, *a, **kw):
                return _orig_decode_xl(z.to(dtype=torch.float32), *a, **kw)
            pipe.vae.decode = _safe_decode_xl
        elif version == "sd15":
            pipe.vae.to(dtype=torch.float32)
            _orig_decode = pipe.vae.decode
            def _safe_decode(z, *a, **kw):
                return _orig_decode(z.to(dtype=torch.float32), *a, **kw)
            pipe.vae.decode = _safe_decode

    pipe_size = PIPE_SIZE_GB.get(version, 2.5)
    need_slicing = vram_gb < pipe_size + 1.5
    need_offload = device == "cuda" and vram_gb < pipe_size

    if need_slicing:
        pipe.enable_attention_slicing("auto")
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
    if need_offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
    return pipe


_CONTROLNET_CACHE: dict = {}


def infer_controlnet_mode(path: Path) -> str:
    """ControlNet ファイル名から推奨 前処理 mode を推定する。

    判定マップ (stem を lowercase してマッチ):
      "canny"           → canny      (Canny edge detection)
      "depth" / "midas" → depth      (Midas depth estimation)
      "openpose"/"pose" → openpose   (Body keypoint)
      "mlsd"            → softedge   (MLSD line segment ≈ softedge 系)
      "hed" / "softedge"/ "soft_edge" → softedge  (HED soft edge detection)
      "tile"/"blur"/"epsblur"/"colorgrid"/"ttplanet" → passthrough (元画像をそのまま入力)
      不明 → passthrough (Tile/Blur 系の方が多いので安全側 default)

    **背景**: ControlNet には種類があり、前処理画像の形式が異なる:
      - canny CN は edge map (白黒線画) を期待
      - softedge CN は HED の soft edge を期待
      - Tile / Blur 系 CN は **元画像そのもの (またはぼかし版)** を期待
      Mismatch すると ControlNet が混乱して線画出力等の事故になる (2026-05-14 実例)。
    """
    stem = path.stem.lower()
    if "canny" in stem:
        return "canny"
    if "depth" in stem or "midas" in stem:
        return "depth"
    if "openpose" in stem or "_pose" in stem or stem.endswith("pose"):
        return "openpose"
    if "mlsd" in stem:
        return "softedge"
    if "hed" in stem or "softedge" in stem or "soft_edge" in stem or "lineart" in stem:
        return "softedge"
    if ("tile" in stem or "blur" in stem or "colorgrid" in stem
            or "ttplanet" in stem or "color_grid" in stem):
        return "passthrough"
    return "passthrough"


def pick_xl_controlnet(tags: list[str]) -> Path | None:
    """5_4_XL_ControlNet/ 配下の safetensors からファイル名抽選する。

    - tags = ["anime"] / ["real"] / ["anime", "real"] (mix) / [other_tag] を受ける
    - ファイル名 stem (lowercase) に tag が含まれるものだけを候補にして `random.choice`
    - マッチが無ければ全 ControlNet からランダムに 1 つ (フォールバック)
    - 5_4_XL_ControlNet が空 or 存在しない → None

    base ごとに `base_upscale_model(setting, default)` で得た tag を渡せば、
    anime ベースには anime 系 ControlNet、real ベースには real 系がかかる。
    """
    if not XL_CONTROLNET_DIR.exists():
        return None
    all_files = sorted(XL_CONTROLNET_DIR.glob("*.safetensors"))
    if not all_files:
        return None
    tags_lower = [t.lower() for t in (tags or []) if t]
    matched = [p for p in all_files
               if any(t in p.stem.lower() for t in tags_lower)] if tags_lower else []
    return random.choice(matched) if matched else random.choice(all_files)


def build_xl_controlnet_pipeline(
    base: Path,
    controlnet: Path,
    device: str,
    dtype: torch.dtype,
    *,
    vram_gb: float = 8.0,
    scheduler: str = "dpmpp-2m-karras",
    vae_fp32: bool = True,
    progress: bool = True,
    enable_tiling: bool = True,
    enable_slicing: bool = True,
    cpu_offload: bool | None = None,
):
    """SDXL base + ControlNet を組み合わせた pipe を構築する (highrender.py 専用)。

    - `base` / `controlnet` は両方ローカル safetensors。`from_single_file` で読む。
      ControlNet は `pick_xl_controlnet(tags)` で `5_4_XL_ControlNet/` から抽選した
      ものを渡す想定 (base の anime/real タグでファイル名フィルタ済み)。
    - CPU 想定なので enable_tiling / enable_slicing は既定 ON にしておく
      (RAM 64GB でも UNet activation が荒れると swap に落ちるため)。
    - vae_fp32 は SDXL の `upcast_vae()` を呼ぶ (deprecated だが 0.38 では依然これが安全)。
    """
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    key = str(controlnet)
    if key in _CONTROLNET_CACHE:
        cnet = _CONTROLNET_CACHE[key]
    else:
        cnet = ControlNetModel.from_single_file(str(controlnet), torch_dtype=dtype)
        _CONTROLNET_CACHE[key] = cnet

    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        str(base), controlnet=cnet, torch_dtype=dtype
    )
    # cpu_offload を使う場合は .to(device) を呼ばない (offload が自前で配置管理する)
    # 既定: cuda + vram < (SDXL UNet 6.5GB + ControlNet 2.5GB ≈ 9GB) なら自動 ON
    if cpu_offload is None:
        cpu_offload = device == "cuda" and vram_gb < 9.0
    if not cpu_offload:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=not progress)
    _apply_scheduler(pipe, scheduler)

    if vae_fp32 and hasattr(pipe, "upcast_vae"):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pipe.upcast_vae()
        # upcast_vae 後も UNet が bf16 latents を吐くので、decode 直前で fp32 にキャストするラッパを噛ませる
        # (build_pipeline 側と同じパターン。これを忘れると 300 step 走り切った最後の decode で
        #  "Input type BFloat16 and bias type float" で落ちる)
        pipe.vae.to(dtype=torch.float32)
        _orig_decode_cn = pipe.vae.decode
        def _safe_decode_cn(z, *a, **kw):
            return _orig_decode_cn(z.to(dtype=torch.float32), *a, **kw)
        pipe.vae.decode = _safe_decode_cn

    if enable_slicing:
        pipe.enable_attention_slicing("auto")
    if enable_tiling and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    # ⚠ 重要: cpu_offload (sequential) は **呼び出し側で LoRA / TI load 完了後に手動で発動**
    # する。ここで先に enable_sequential_cpu_offload() を呼ぶと submodule が meta device に
    # 置かれ、その後の pipe.load_lora_weights() が "META device type not an accelerator" で落ちる。
    # → build_xl_controlnet_pipeline は **CPU 上の生 pipe を返すだけ** にしておき、
    #   caller (highrender.py) が LoRA load → set_adapters → (任意で TI) → pipe.enable_sequential_cpu_offload()
    #   の順で fianlize する。
    return pipe


def build_xl_controlnet_img2img_pipe(pipe, progress: bool = True):
    """build_xl_controlnet_pipeline で作った pipe から ControlNet 付き img2img pipe を派生。
    tiled refinement (tiled_xl_img2img) で使う。"""
    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
    i2i = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(pipe)
    i2i.set_progress_bar_config(disable=not progress)
    return i2i


_CTRL_AUX_CACHE: dict = {}


def prepare_controlnet_image(src_img: "Image.Image", mode: str = "softedge",
                              detect_resolution: int = 1024,
                              image_resolution: int = 1024) -> "Image.Image":
    """src_img を ControlNet 用の制御画像に変換する。

    mode:
        - "canny"     : OpenCV Canny edges
        - "depth"     : Midas depth
        - "softedge"  : HED / PiDi soft edge (既定。線がしっかり乗る)
        - "seg"       : Uniformer semantic segmentation
        - "openpose"  : Body keypoint
        - "passthrough": src_img をそのまま (= ControlNet 入力が既に変換済みのとき)

    controlnet_aux ライブラリで処理。各検出器は初回呼び出し時に DL + キャッシュ。
    """
    if mode == "passthrough":
        return src_img.convert("RGB")
    key = mode
    if key in _CTRL_AUX_CACHE:
        detector = _CTRL_AUX_CACHE[key]
    else:
        if mode == "canny":
            from controlnet_aux import CannyDetector
            detector = CannyDetector()
        elif mode == "depth":
            from controlnet_aux import MidasDetector
            detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        elif mode == "softedge":
            from controlnet_aux import HEDdetector
            detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        elif mode == "seg":
            from controlnet_aux import UniformerDetector
            detector = UniformerDetector.from_pretrained("lllyasviel/Annotators")
        elif mode == "openpose":
            from controlnet_aux import OpenposeDetector
            detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        else:
            raise ValueError(f"unknown controlnet mode: {mode}")
        _CTRL_AUX_CACHE[key] = detector

    out = detector(
        src_img.convert("RGB"),
        detect_resolution=detect_resolution,
        image_resolution=image_resolution,
    )
    # 一部 detector は np.ndarray を返す
    if not isinstance(out, Image.Image):
        out = Image.fromarray(np.asarray(out))
    return out.convert("RGB")


def _get_tile_starts(total: int, tile: int, overlap: int) -> list[int]:
    """[0, ..., total-tile] を tile-overlap 刻みで返す (端点は必ず total-tile)。
    upscale.py の同名関数と同じロジックを共有。
    """
    if total <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, total - tile, step))
    if not starts or starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def tiled_xl_img2img(
    i2i_pipe,
    compel,
    init_img: "Image.Image",
    prompt: str,
    negative: str,
    *,
    ctrl_img: "Image.Image | None" = None,
    target_w: int,
    target_h: int,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    strength: float = 0.3,
    steps: int = 100,
    cfg_scale: float = 7.0,
    controlnet_conditioning_scale: float = 0.7,
    generator=None,
    progress_label: str = "tiled refine",
) -> "Image.Image":
    """init_img を target_w × target_h に LANCZOS で拡大したあと、tile_size タイルで
    SDXL img2img + ControlNet で順次リファインしてつなぎ合わせる。

    - 各タイルは独立に img2img を回し、overlap 領域は cosine ramp で平均合成する
      (`upscale.py` のタイル合成と同じ重み付け)。
    - strength は 0.2-0.35 程度を推奨 (構造を壊さずディテールだけ盛る)。
    - ctrl_img が指定されていれば target に拡大して同じ位置にクロップして条件付け。

    戻り値は RGB PIL Image (target_w × target_h)。
    """
    enlarged = init_img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    if ctrl_img is not None:
        ctrl_full = ctrl_img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    else:
        ctrl_full = None

    arr = np.asarray(enlarged, dtype=np.float32) / 255.0  # H W 3
    out = np.zeros_like(arr)
    weight = np.zeros((target_h, target_w, 1), dtype=np.float32)

    # cosine-edge weight (1.0 中央 → 0 端)
    def _tile_weight(h: int, w: int, ov: int) -> np.ndarray:
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)
        ov_h = min(ov, h // 2)
        ov_w = min(ov, w // 2)
        if ov_h > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov_h, dtype=np.float32))
            wy[:ov_h] = ramp
            wy[-ov_h:] = ramp[::-1]
        if ov_w > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov_w, dtype=np.float32))
            wx[:ov_w] = ramp
            wx[-ov_w:] = ramp[::-1]
        return (wy[:, None] * wx[None, :])[..., None]

    starts_y = _get_tile_starts(target_h, tile_size, tile_overlap)
    starts_x = _get_tile_starts(target_w, tile_size, tile_overlap)
    total = len(starts_y) * len(starts_x)
    print(f"[{progress_label}] {target_w}x{target_h}, "
          f"{len(starts_y)}x{len(starts_x)}={total} tiles, "
          f"tile={tile_size}, overlap={tile_overlap}, strength={strength}, steps={steps}",
          flush=True)

    idx = 0
    for y in starts_y:
        for x in starts_x:
            idx += 1
            tile_pil = enlarged.crop((x, y, x + tile_size, y + tile_size))
            kw = dict(
                image=tile_pil,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )
            if ctrl_full is not None:
                ctile = ctrl_full.crop((x, y, x + tile_size, y + tile_size))
                kw["control_image"] = ctile
                kw["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            print(f"  tile {idx}/{total} @ ({x},{y})", flush=True)
            res = compel_call(i2i_pipe, compel, prompt, negative, "sdxl", **kw)
            tile_arr = np.asarray(res.images[0].convert("RGB"), dtype=np.float32) / 255.0
            tw = _tile_weight(tile_size, tile_size, tile_overlap)
            out[y:y + tile_size, x:x + tile_size] += tile_arr * tw
            weight[y:y + tile_size, x:x + tile_size] += tw

    weight = np.where(weight == 0, 1.0, weight)
    out /= weight
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


_UPSCALE_CACHE: dict = {}


def get_upscale_model(name: str, device: str):
    """Real-ESRGAN モデルを (キャッシュ) ロードして (model, native_scale, dtype) を返す。
    fp16 固定 (Real-ESRGAN は SD UNet と違い fp16 で安全に動く)。
    """
    key = (name, device)
    if key in _UPSCALE_CACHE:
        return _UPSCALE_CACHE[key]
    from upscale import load_model
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, desc = load_model(name, device, dtype)
    _UPSCALE_CACHE[key] = (model, desc.scale, dtype)
    return _UPSCALE_CACHE[key]


def build_img2img_pipe(pipe, version: str, progress: bool = True):
    """build_pipeline で作った txt2img pipe から img2img pipe を派生する。
    UNet/VAE/CLIP を共有するため追加の VRAM はほぼ 0。Hires Fix と ADetailer で使う。
    """
    if version == "sdxl":
        from diffusers import StableDiffusionXLImg2ImgPipeline
        i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
    else:
        from diffusers import StableDiffusionImg2ImgPipeline
        i2i = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
    i2i.set_progress_bar_config(disable=not progress)
    return i2i


def make_compel(pipe, version: str, device: str | torch.device | None = None):
    """pipe に紐づく Compel プロセッサを返す。
    `(word:1.3)` 形式の重み記法を解釈して prompt_embeds に焼き込む役割。
    pipe を `from_pipe` で派生した i2i pipe にも同じ compel を使い回せる (tokenizer / text_encoder を共有しているため)。

    Textual Inversion (Embedding) も対応: `DiffusersTextualInversionManager` を経由して
    `pipe.load_textual_inversion()` 済みのトークン (例: `<easynegative>`, `<bad-hands-5>`) を
    compel に認識させ、`(easynegative:1.3)` のような重みづけも効くようにする。
    呼び出し側は **TI をすべてロード済みの pipe** を渡すこと。

    device 明示指定:
      sequential_cpu_offload を有効化した pipe では `pipe.unet.parameters()` の device が
      `"meta"` になり、そのままだと compel が meta tensor を作ろうとして失敗する。
      offload 利用時は実行先の device ("cuda" / "cpu") を明示的に渡すこと。
    """
    from compel import Compel, ReturnedEmbeddingsType
    from compel.diffusers_textual_inversion_manager import DiffusersTextualInversionManager
    if device is None:
        device = next(pipe.unet.parameters()).device
    ti_manager = DiffusersTextualInversionManager(pipe)
    if version == "sdxl":
        return Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=device,
            textual_inversion_manager=ti_manager,
        )
    return Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        device=device,
        textual_inversion_manager=ti_manager,
    )


def compel_call(pipe, compel, prompt: str, negative: str, version: str, **kw):
    """compel で prompt / negative を embeddings に変換し、pipe を呼び出す。

    pipe.__call__ の `prompt` / `negative_prompt` 引数の代わりに `prompt_embeds` 系を渡す。
    残りの kw (image, width, height, num_inference_steps, generator, strength, ...) は透過する。
    SDXL は pooled embeds も必要。
    長さの異なる positive / negative は `pad_conditioning_tensors_to_same_length` で揃える。
    """
    if version == "sdxl":
        p_emb, p_pool = compel(prompt)
        n_emb, n_pool = compel(negative)
        [p_emb, n_emb] = compel.pad_conditioning_tensors_to_same_length([p_emb, n_emb])
        return pipe(
            prompt_embeds=p_emb,
            negative_prompt_embeds=n_emb,
            pooled_prompt_embeds=p_pool,
            negative_pooled_prompt_embeds=n_pool,
            **kw,
        )
    p_emb = compel(prompt)
    n_emb = compel(negative)
    [p_emb, n_emb] = compel.pad_conditioning_tensors_to_same_length([p_emb, n_emb])
    return pipe(prompt_embeds=p_emb, negative_prompt_embeds=n_emb, **kw)


def generate_image(
    pipe,
    i2i_pipe,
    compel,
    prompt: str,
    negative: str,
    version: str,
    *,
    src_img: "Image.Image | None" = None,
    width: int = 768,
    height: int = 768,
    hires_fix: bool = False,
    hires_base: int = 512,
    hires_strength: float = 0.4,
    strength: float = 0.45,
    steps: int = 50,
    cfg_scale: float = 7.0,
    generator=None,
    adetailer_fn=None,
    adetailer_strength: float = 0.4,
) -> "Image.Image":
    """txt2img / Hires Fix / img2img + ADetailer の 3 生成パスを統合した 1 枚生成関数。

    パスの判定:
      - `src_img is None` かつ `hires_fix=True` かつ `version=="sd15"`:
          **Hires Fix** (Phase1: pipe → hires_base² → Phase2: i2i → width×height with hires_strength)
      - `src_img is None` (それ以外): **txt2img** 単発 (pipe → width×height)
      - `src_img is not None`: **img2img** 単発 (i2i → src_img, strength) — Hires Fix は無視
    最後に `adetailer_fn` が指定されていれば img2img inpainting で修正 (i2i_pipe + adetailer_strength)。

    呼び出し側の責任:
      - `i2i_pipe` は Hires Fix / img2img / ADetailer のいずれかが必要なら渡す (それ以外は None でも可)
      - `compel` は Hires Fix なら共通の compel を pipe / i2i 両方で使えること (`make_compel` が満たす)
      - 例外はこの関数で raise されるので呼び出し側で try/except する
    """
    hires_active = src_img is None and hires_fix and version == "sd15"
    if hires_active:
        # Phase 1: 低解像度生成 (SD15 ネイティブ 512 推奨)
        lo = compel_call(
            pipe, compel, prompt, negative, version,
            width=hires_base, height=hires_base,
            num_inference_steps=steps, guidance_scale=cfg_scale, generator=generator,
        ).images[0]
        # Phase 2: 目標解像度へ img2img で拡大 + denoise
        init = lo.resize((width, height), Image.LANCZOS)
        img = compel_call(
            i2i_pipe, compel, prompt, negative, version,
            image=init, strength=hires_strength,
            num_inference_steps=steps, guidance_scale=cfg_scale, generator=generator,
        ).images[0]
    elif src_img is None:
        # txt2img 単発
        img = compel_call(
            pipe, compel, prompt, negative, version,
            width=width, height=height,
            num_inference_steps=steps, guidance_scale=cfg_scale, generator=generator,
        ).images[0]
    else:
        # img2img 単発 (src_img を初期 latent に)
        img = compel_call(
            i2i_pipe, compel, prompt, negative, version,
            image=src_img, strength=strength,
            num_inference_steps=steps, guidance_scale=cfg_scale, generator=generator,
        ).images[0]

    # ADetailer (任意): 顔/手 検出 → 個別 img2img で修正
    if adetailer_fn is not None:
        img = adetailer_fn(
            i2i_pipe, img, prompt, negative,
            steps=steps, cfg=cfg_scale,
            strength=adetailer_strength, generator=generator,
            compel=compel, version=version,
        )
    return img


def pick_device_dtype(vram_gb: float) -> tuple[str, torch.dtype]:
    """SD パイプラインの weight dtype を決める。
    Ampere (sm_80) 以降は bf16、それ未満の CUDA は fp16、CPU は fp32。

    bf16 は fp32 と同じ指数幅 (8bit) のため、SD15 系で fp16 UNet が overflow して
    灰色画像 (NaN latent → grey) を吐く問題が出ない。Ampere ではスループットも
    fp16 と同等なので、対応 GPU では bf16 を既定とする。
    vram_gb は呼び出し側の build_pipeline 等で slicing/offload 判定に使う。
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


# --------------------------------------------------------------------------- #
# 出力 / 画像判定
# --------------------------------------------------------------------------- #
def unique_output_path(stem: str) -> Path:
    """3_0_generateds/{stem}.png が衝突したら _2, _3 ... を付ける"""
    base = OUT_DIR / f"{stem}.png"
    if not base.exists():
        return base
    for i in range(2, 1000):
        cand = OUT_DIR / f"{stem}_{i}.png"
        if not cand.exists():
            return cand
    raise RuntimeError(L("出力先が枯渇しました", "output path exhausted"))


def save_image_no_metadata(img: Image.Image, path: Path) -> None:
    # tobytes/frombytes 経由で新規 Image を作り、PNG メタデータを完全に除去する
    clean = Image.frombytes(img.mode, img.size, img.tobytes())
    clean.save(str(path), format="PNG")


def detect_degenerate(img: Image.Image) -> Optional[str]:
    """壊れた base/LoRA で出る『灰色画像』『全画素カラーノイズ画像』を判定する。
    返り値は失敗理由 (str) または None (正常)。

    - 灰色: 画素全体の std が極小 (NaN→ゼロや一様色になっている)
    - カラーノイズ: 全体 std は大きいが 16x ダウンサンプル後 std が極小
                  (構造を持たないピクセル単位ノイズで平均すれば中間色になる)
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    overall = float(arr.std())
    if overall < 5.0:
        return f"grey (std={overall:.1f})"
    w, h = img.size
    sw, sh = max(1, w // 16), max(1, h // 16)
    small = np.asarray(img.convert("RGB").resize((sw, sh), Image.BOX), dtype=np.float32)
    small_std = float(small.std())
    if overall > 50.0 and small_std < 10.0:
        return f"noise (std={overall:.1f}, ds_std={small_std:.1f})"
    return None


# --------------------------------------------------------------------------- #
# GPU 温度
# --------------------------------------------------------------------------- #
def current_gpu_temp() -> Optional[int]:
    """nvidia-smi で取れる GPU 温度 (°C)。取得失敗時は None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None
