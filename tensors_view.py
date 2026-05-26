#!/usr/bin/env python3
"""tensors_view.py - safetensors のメタ情報ビューア (Tkinter)。

`--dir` で渡したフォルダを再帰走査し、各 .safetensors の **ヘッダ (JSON)** を
torch / safetensors ライブラリ無しで生読みして表示する。

safetensors のファイル先頭は:
    [ 8 bytes little-endian uint64 = ヘッダ長 N ][ N bytes の JSON ヘッダ ]
JSON ヘッダは tensor名 → {dtype, shape, data_offsets} の辞書 + 特別キー
`__metadata__` (学習メタ等の文字列辞書) で構成される。tensor 本体はロードしない。

表示するもの:
  - テキスト: 種別(base/lora/embedding/controlnet)、base(SD15/SDXL)判定、
    ファイル/テンソル要約(数・パラメータ数・dtype)、LoRA のトリガー語(ss_tag_frequency)、
    学習メタ(ss_* / modelspec.*)、全テンソル一覧(name/shape/dtype)。
  - 画像: __metadata__ に data URI 形式の埋め込みサムネ(modelspec.thumbnail 等)が
    あれば表示 (現状ほぼ無いが将来対応)。

UI 文字列は i18n の L() で英/日切替 (PLAYGROUND_LANG / OS ロケール)。

使い方:
    python tensors_view.py --dir 3_2_SD15_LoRA          # GUI を起動
    python tensors_view.py --dir 3_2_SD15_LoRA --list    # GUI なしで一覧を標準出力
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import queue
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from i18n import L

# Windows console 絵文字落ち防止
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# dtype → ビット数 (バイトサイズ概算・参考用)
_DTYPE_BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8, "BOOL": 8,
}
_EMB_DIMS = {768, 1024, 1280, 2048}


# --------------------------------------------------------------------------- #
# safetensors ヘッダ読み取り (torch / safetensors 不要)
# --------------------------------------------------------------------------- #
def read_header(path: Path) -> tuple[dict, dict]:
    """(tensors, metadata) を返す。tensors: name → {dtype, shape, data_offsets}。"""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError("file too small for safetensors header")
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > 500_000_000:
            raise ValueError(f"implausible header length: {n}")
        raw = f.read(n)
    obj = json.loads(raw)
    meta = obj.pop("__metadata__", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    return obj, meta


def classify(tensors: dict, stem: str) -> str:
    """base / lora / embedding / controlnet / inpainting / broken を判定 (common.py 準拠)。"""
    keys = list(tensors.keys())
    if not keys:
        return "broken"
    if any(k.startswith("controlnet_") or "controlnet_cond_embedding" in k
           or "control_model." in k or "zero_convs" in k or k.startswith("lllite_")
           for k in keys):
        return "controlnet"
    if any("lora_" in k or ".lora_" in k for k in keys):
        return "lora"
    if any(k in {"emb_params", "string_to_param"} or k.startswith("string_to_param")
           for k in keys):
        return "embedding"
    if len(keys) <= 8:
        def _is_emb(s: list[int]) -> bool:
            if len(s) == 1:
                return True
            if len(s) == 2:
                return s[-1] in _EMB_DIMS or math.prod(s) < 50_000
            return False
        shapes = [tensors[k].get("shape", []) for k in keys]
        if shapes and all(_is_emb(s) for s in shapes):
            return "embedding"
    if "inpaint" in stem.lower():
        return "inpainting"
    return "base"


def detect_version(tensors: dict, meta: dict) -> str:
    """SD15 / SDXL を判定。メタ優先 → テンソルの shape ヒューリスティク。不明は '?'。"""
    arch = f"{meta.get('modelspec.architecture', '')} {meta.get('ss_base_model_version', '')}".lower()
    if "xl" in arch:
        return "sdxl"
    if arch.strip() and any(t in arch for t in ("v1-5", "sd_v1", "v1/", "1.5", "sd1")):
        return "sd15"
    keys = list(tensors.keys())
    if any("conditioner.embedders.1" in k or "text_encoder_2" in k or "_te2_" in k
           or "lora_te2_" in k or "clip_g" in k.lower() for k in keys):
        return "sdxl"
    for k in keys:
        if k.endswith(("to_k.weight", "to_q.weight",
                       "to_k.lora_down.weight", "to_k.lora_A.weight")):
            s = tensors[k].get("shape", [])
            if s:
                inner = s[-1]
                if inner >= 1500:
                    return "sdxl"
                if 600 <= inner <= 800:
                    return "sd15"
    last_dims = {s[-1] for k in keys if (s := tensors[k].get("shape", []))}
    if 1280 in last_dims or 2048 in last_dims:
        return "sdxl"
    if 768 in last_dims:
        return "sd15"
    return "?"


def summarize(tensors: dict) -> tuple[int, int, dict]:
    """(テンソル数, 総パラメータ数, {dtype: 個数}) を返す。"""
    n_params = 0
    dtypes: dict[str, int] = {}
    for info in tensors.values():
        s = info.get("shape", [])
        p = math.prod(s) if s else 0
        n_params += p
        d = info.get("dtype", "?")
        dtypes[d] = dtypes.get(d, 0) + 1
    return len(tensors), n_params, dtypes


def extract_triggers(meta: dict, top: int = 20) -> str:
    """ss_tag_frequency から学習タグ (トリガー語候補) を頻度順に抽出する。"""
    raw = meta.get("ss_tag_frequency")
    if not raw:
        return ""
    try:
        tf = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    counts: dict[str, int] = {}
    if isinstance(tf, dict):
        for tags in tf.values():
            if isinstance(tags, dict):
                for tag, c in tags.items():
                    counts[tag] = counts.get(tag, 0) + (c if isinstance(c, int) else 0)
    if not counts:
        return ""
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{t} ({c})" for t, c in ranked)


def find_thumb_uri(meta: dict) -> Optional[str]:
    """__metadata__ から data URI 形式の埋め込みサムネを探す。"""
    for v in meta.values():
        if isinstance(v, str) and v[:11] == "data:image/":
            return v
    return None


def human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


# --------------------------------------------------------------------------- #
# データモデル
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    path: Path
    size: int
    kind: str
    base: str
    n_tensors: int
    n_params: int
    dtypes: dict
    has_thumb: bool
    title: str
    triggers: str
    meta: dict
    preview_path: Optional[Path] = None   # <name>.preview.png サイドカー (あれば)


def load_entry(path: Path) -> Entry:
    """1 つの safetensors を解析して Entry を作る (読めなければ kind='broken')。"""
    size = path.stat().st_size
    try:
        tensors, meta = read_header(path)
    except (OSError, ValueError):
        return Entry(path, size, "broken", "?", 0, 0, {}, False, "", "", {})
    kind = classify(tensors, path.stem)
    base = detect_version(tensors, meta)
    n_t, n_p, dts = summarize(tensors)
    title = meta.get("modelspec.title") or meta.get("ss_output_name") or ""
    sidecar = path.with_suffix(".preview.png")
    preview = sidecar if sidecar.exists() else None
    return Entry(path, size, kind, base, n_t, n_p, dts,
                 bool(find_thumb_uri(meta)) or preview is not None,
                 title, extract_triggers(meta), meta, preview)


def iter_safetensors(directory: Path):
    for p in sorted(directory.rglob("*.safetensors")):
        if p.is_file():
            yield p


def dtypes_str(dtypes: dict) -> str:
    return ", ".join(f"{d}×{c}" for d, c in sorted(dtypes.items(), key=lambda kv: -kv[1]))


# 学習メタの代表キー (概要タブに整形表示する)
_KEY_META = [
    ("ss_base_model_version", "base model"),
    ("modelspec.architecture", "architecture"),
    ("modelspec.resolution", "resolution"),
    ("ss_resolution", "train resolution"),
    ("ss_network_module", "network module"),
    ("ss_network_dim", "network dim"),
    ("ss_network_alpha", "network alpha"),
    ("ss_learning_rate", "learning rate"),
    ("ss_unet_lr", "unet lr"),
    ("ss_text_encoder_lr", "text encoder lr"),
    ("ss_num_train_images", "train images"),
    ("ss_num_epochs", "epochs"),
    ("ss_epoch", "epoch"),
    ("ss_optimizer", "optimizer"),
    ("ss_clip_skip", "clip skip"),
    ("ss_sd_model_name", "sd model name"),
    ("modelspec.date", "date"),
]


# --------------------------------------------------------------------------- #
# LoRA_preview.toml (make_previews のカテゴリ/カスタムプロンプト) の読み書き
# --------------------------------------------------------------------------- #
PREVIEW_TOML = Path(__file__).resolve().parent / "LoRA_preview.toml"
PREVIEW_SETTINGS_TOML = Path(__file__).resolve().parent / "preview_settings.toml"
MAKE_PREVIEWS_PY = Path(__file__).resolve().parent / "make_previews.py"
DEFAULT_CATEGORIES = ["ware", "doing1", "doing2", "doing3", "doingmob",
                      "object", "part", "view", "place", "artstyle", "unknown"]
# カテゴリ別の既定 steps。part は部位が崩れ「クリーチャー」化しやすいので多めにする。
DEFAULT_STEPS = 24
CATEGORY_STEPS = {"part": 40}


def _load_tensors_dirs() -> list[str]:
    """preview_settings.toml [tensors_dirs] list を読む。失敗時は空。"""
    import tomllib
    if not PREVIEW_SETTINGS_TOML.exists():
        return []
    try:
        data = tomllib.loads(PREVIEW_SETTINGS_TOML.read_text(encoding="utf-8"))
        lst = (data.get("tensors_dirs") or {}).get("list") or []
        return [str(x) for x in lst if str(x).strip()]
    except Exception:
        return []


def default_dir() -> Optional[Path]:
    """--dir 省略時に開くディレクトリ。[tensors_dirs] list の先頭 (実在するもの)。"""
    root = Path(__file__).resolve().parent
    for name in _load_tensors_dirs():
        cand = root / name
        if cand.exists():
            return cand.resolve()
    return None


def load_preview_meta() -> tuple[dict, dict, list]:
    """(cats_by_ver, prompts_by_ver, カテゴリ選択肢) を返す。torch 不要。
    cats_by_ver / prompts_by_ver は {"sd15": {stem→...}, "sdxl": {stem→...}}。"""
    import tomllib
    cats_by_ver: dict = {"sd15": {}, "sdxl": {}}
    prompts_by_ver: dict = {"sd15": {}, "sdxl": {}}
    choices = list(DEFAULT_CATEGORIES)
    # 選択肢は preview_template.toml の [LoRA_preview_template] キーから (ハードコード吸収)
    if PREVIEW_SETTINGS_TOML.exists():
        try:
            td = tomllib.loads(PREVIEW_SETTINGS_TOML.read_text(encoding="utf-8"))
            tkeys = list((td.get("LoRA_preview_template") or {}).keys())
            if tkeys:
                choices = tkeys + [c for c in DEFAULT_CATEGORIES if c not in tkeys]
        except Exception:
            pass
    if PREVIEW_TOML.exists():
        try:
            data = tomllib.loads(PREVIEW_TOML.read_text(encoding="utf-8"))
            cats_by_ver["sd15"] = {str(k): str(v) for k, v in (data.get("SD15_categories") or {}).items()}
            cats_by_ver["sdxl"] = {str(k): str(v) for k, v in (data.get("SDXL_categories") or {}).items()}
            prompts_by_ver["sd15"] = {str(k): str(v) for k, v in (data.get("SD15_prompts") or {}).items()}
            prompts_by_ver["sdxl"] = {str(k): str(v) for k, v in (data.get("SDXL_prompts") or {}).items()}
        except Exception:
            pass
    return cats_by_ver, prompts_by_ver, choices


def save_preview_meta(stem: str, version: str, category: str, custom_prompt: str) -> None:
    """LoRA_preview.toml の {SD15,SDXL}_{categories,prompts}[stem] を更新。"""
    import tomllib
    import tomli_w
    if version not in ("sd15", "sdxl"):
        raise ValueError(f"unknown version: {version}")
    data: dict = {}
    if PREVIEW_TOML.exists():
        try:
            data = tomllib.loads(PREVIEW_TOML.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    prefix = "SD15" if version == "sd15" else "SDXL"
    cats = dict(data.get(f"{prefix}_categories") or {})
    prompts = dict(data.get(f"{prefix}_prompts") or {})
    cats[stem] = category
    cp = custom_prompt.strip()
    if cp:
        prompts[stem] = cp
    else:
        prompts.pop(stem, None)
    data[f"{prefix}_categories"] = dict(sorted(cats.items(), key=lambda kv: kv[0].lower()))
    data[f"{prefix}_prompts"] = dict(sorted(prompts.items(), key=lambda kv: kv[0].lower()))
    with open(PREVIEW_TOML, "wb") as f:
        tomli_w.dump(data, f)


def set_preview_categories(stems_by_ver: dict, category: str) -> None:
    """version → [stem] の辞書を受け取り、両 section を 1 度の書き込みで一括更新。"""
    import tomllib
    import tomli_w
    data: dict = {}
    if PREVIEW_TOML.exists():
        try:
            data = tomllib.loads(PREVIEW_TOML.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for ver, stems in stems_by_ver.items():
        if not stems or ver not in ("sd15", "sdxl"):
            continue
        prefix = "SD15" if ver == "sd15" else "SDXL"
        cats = dict(data.get(f"{prefix}_categories") or {})
        for s in stems:
            cats[s] = category
        data[f"{prefix}_categories"] = dict(sorted(cats.items(), key=lambda kv: kv[0].lower()))
    with open(PREVIEW_TOML, "wb") as f:
        tomli_w.dump(data, f)


def _entry_version(entry) -> str:
    """Entry の version を判定 (entry.base が sd15/sdxl ならそれを、不明なら親dirで補完、最終的に sd15)。"""
    base = getattr(entry, "base", "")
    if base in ("sd15", "sdxl"):
        return base
    parent = entry.path.parent.name
    if parent.startswith("3_"):
        return "sd15"
    if parent.startswith("4_"):
        return "sdxl"
    return "sd15"


# --------------------------------------------------------------------------- #
# --list (GUI なし)
# --------------------------------------------------------------------------- #
def run_list(directory: Path) -> None:
    entries = [load_entry(p) for p in iter_safetensors(directory)]
    if not entries:
        print(L(f"safetensors が見つかりません: {directory}", f"No safetensors found: {directory}"))
        return
    entries.sort(key=lambda e: e.path.name.lower())
    print(L(f"=== {directory} : {len(entries)} 個 ===", f"=== {directory} : {len(entries)} files ==="))
    kinds: dict[str, int] = {}
    for e in entries:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    print("  " + " / ".join(f"{k}: {v}" for k, v in sorted(kinds.items())))
    print(f"{'kind':<11} {'base':<5} {'size':>9} {'tensors':>8} {'params':>8}  name")
    print("-" * 100)
    for e in entries:
        print(f"{e.kind:<11} {e.base:<5} {human_size(e.size):>9} {e.n_tensors:>8} "
              f"{human_count(e.n_params):>8}  {e.path.name}")
        if e.triggers:
            print(L(f"    トリガー語: {e.triggers[:120]}", f"    triggers: {e.triggers[:120]}"))


# --------------------------------------------------------------------------- #
# 列定義 (リスト & ソート共有)
# --------------------------------------------------------------------------- #
def _columns() -> list:
    return [
        ("name",    L("名前", "Name"),       260, "w",      lambda e: e.path.name.lower(),  lambda e: e.path.name),
        ("kind",    L("種別", "Kind"),        90, "center", lambda e: e.kind,               lambda e: e.kind),
        ("base",    "Base",                    55, "center", lambda e: e.base,               lambda e: e.base),
        ("size",    L("サイズ", "Size"),       85, "e",      lambda e: e.size,               lambda e: human_size(e.size)),
        ("tensors", L("テンソル", "Tensors"),  75, "e",      lambda e: e.n_tensors,          lambda e: str(e.n_tensors)),
        ("params",  L("パラメータ", "Params"),  80, "e",      lambda e: e.n_params,           lambda e: human_count(e.n_params)),
        ("thumb",   L("画像", "Thumb"),        45, "center", lambda e: e.has_thumb,          lambda e: "✓" if e.has_thumb else ""),
        ("title",   L("タイトル", "Title"),    200, "w",      lambda e: e.title.lower(),      lambda e: e.title),
    ]


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def run_gui(directory: Path) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    from PIL import Image, ImageTk

    PREVIEW_MAX = 320

    class TensorViewApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.directory = directory
            self.title(L(f"テンソル情報ビューア — {directory}", f"Tensor Info Viewer — {directory}"))
            self.geometry("1280x820")

            self.all_entries: list[Entry] = []
            self.tree_iid: dict[str, Entry] = {}
            self.selected: Optional[Entry] = None
            self.selected_entries: list = []      # 複数選択 (一括操作用)
            self._regen_targets: list = []        # 再生成中の対象 (完了時に再読込)
            self.preview_ref = None
            self.loading = True
            self.q: queue.Queue = queue.Queue()
            self.regen_q: queue.Queue = queue.Queue()
            # LoRA_preview.toml (編集対象)。category/custom prompt と選択肢を読み込む
            self.pv_cats, self.pv_prompts, self.pv_choices = load_preview_meta()

            self.columns = _columns()
            self.colkey = {c[0]: c[4] for c in self.columns}
            self.collabel = {c[0]: c[1] for c in self.columns}
            self.label_to_cid = {c[1]: c[0] for c in self.columns}
            self.sort_col = "name"

            self.search_var = tk.StringVar()
            self.search_meta_var = tk.BooleanVar(value=False)   # False=ファイル名のみ / True=メタ込み
            self.kind_var = tk.StringVar(value="All")
            self.base_var = tk.StringVar(value="All")
            self.sort_var = tk.StringVar(value=self.collabel["name"])
            self.desc_var = tk.BooleanVar(value=False)
            self.status_var = tk.StringVar(value=L("読み込み中…", "Loading…"))

            self._search_after = None
            self._build_toolbar()
            self._build_body()

            self.search_var.trace_add("write", self._on_search)
            threading.Thread(target=self._scan_worker, daemon=True).start()
            self.after(80, self._drain_queue)

        # ---- レイアウト ------------------------------------------------- #
        def _build_toolbar(self) -> None:
            bar = ttk.Frame(self, padding=(8, 6))
            bar.pack(side="top", fill="x")

            ttk.Label(bar, text=L("検索:", "Search:")).pack(side="left")
            ttk.Entry(bar, textvariable=self.search_var, width=20).pack(side="left", padx=(4, 4))
            ttk.Checkbutton(bar, text=L("メタ込み", "+meta"), variable=self.search_meta_var,
                            command=self._populate).pack(side="left", padx=(0, 12))

            ttk.Label(bar, text=L("種別:", "Kind:")).pack(side="left")
            self.kind_cb = ttk.Combobox(bar, textvariable=self.kind_var, width=10, state="readonly",
                                        values=["All"])
            self.kind_cb.pack(side="left", padx=(4, 12))
            self.kind_cb.bind("<<ComboboxSelected>>", lambda *_: self._populate())

            ttk.Label(bar, text="Base:").pack(side="left")
            base_cb = ttk.Combobox(bar, textvariable=self.base_var, width=6, state="readonly",
                                   values=["All", "sdxl", "sd15", "?"])
            base_cb.pack(side="left", padx=(4, 12))
            base_cb.bind("<<ComboboxSelected>>", lambda *_: self._populate())

            ttk.Label(bar, text=L("並べ替え:", "Sort:")).pack(side="left")
            sort = ttk.Combobox(bar, textvariable=self.sort_var, width=10, state="readonly",
                                values=[c[1] for c in self.columns])
            sort.pack(side="left", padx=(4, 4))
            sort.bind("<<ComboboxSelected>>", lambda *_: self._on_sort_combo())
            ttk.Checkbutton(bar, text=L("降順", "Desc"), variable=self.desc_var,
                            command=self._populate).pack(side="left", padx=(0, 12))

            # ディレクトリ選択 (project root 直下の安全な候補のみ)
            ttk.Label(bar, text=L("ディレクトリ:", "Dir:")).pack(side="left")
            self.dir_var = tk.StringVar(value=self._dir_label(self.directory))
            dir_cb = ttk.Combobox(bar, textvariable=self.dir_var, width=22, state="readonly",
                                  values=self._dir_choices())
            dir_cb.pack(side="left", padx=(4, 12))
            dir_cb.bind("<<ComboboxSelected>>", lambda *_: self._on_dir_change())

            self.thumb_btn = ttk.Button(bar, text=L("サムネイル生成", "Gen thumbs"),
                                        command=self._batch_missing)
            self.thumb_btn.pack(side="left", padx=(8, 0))
            ttk.Label(bar, textvariable=self.status_var).pack(side="right")

        def _build_body(self) -> None:
            paned = ttk.PanedWindow(self, orient="horizontal")
            paned.pack(side="top", fill="both", expand=True)

            # 左: ファイル一覧 (Treeview)
            left = ttk.Frame(paned)
            paned.add(left, weight=3)
            cids = [c[0] for c in self.columns]
            self.tree = ttk.Treeview(left, columns=cids, show="headings", selectmode="extended")
            for cid, label, width, anchor, *_ in self.columns:
                self.tree.heading(cid, text=label, command=lambda c=cid: self._sort_by_col(c))
                self.tree.column(cid, width=width, anchor=anchor, stretch=(cid in ("name", "title")))
            vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
            hsb = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
            self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            self.tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            left.rowconfigure(0, weight=1)
            left.columnconfigure(0, weight=1)
            for kind, color in (("broken", "#b00020"),
                                ("embedding", "#2e7d32"), ("controlnet", "#8500b0")):
                self.tree.tag_configure(kind, foreground=color)
            self.tree.bind("<<TreeviewSelect>>", self._on_select)

            # 右: 詳細 (サムネ + タブ: 概要 / メタデータ / テンソル)
            right = ttk.Frame(paned, padding=8)
            paned.add(right, weight=2)
            self.preview_lbl = tk.Label(right, background="#2a2a2a",
                                        text=L("(プレビューなし)", "(no preview)"),
                                        fg="#888888", height=4, cursor="hand2")
            self.preview_lbl.pack(side="top", fill="x")
            self.preview_lbl.bind("<Button-1>", lambda *_: self._show_full_preview())

            # プレビュー設定エディタ (LoRA: category/カスタムプロンプト編集 + その場で再生成)
            ed = ttk.LabelFrame(right, text=L("プレビュー設定", "Preview settings"), padding=6)
            ed.pack(side="top", fill="x", pady=(6, 0))
            row = ttk.Frame(ed)
            row.pack(fill="x")
            ttk.Label(row, text=L("カテゴリ:", "Category:")).pack(side="left")
            self.cat_var = tk.StringVar()
            self.cat_cb = ttk.Combobox(row, textvariable=self.cat_var, width=9,
                                       state="disabled", values=self.pv_choices)
            self.cat_cb.pack(side="left", padx=(4, 8))
            self.cat_cb.bind("<<ComboboxSelected>>", lambda *_: self._on_cat_change())
            self.colorful_var = tk.BooleanVar(value=False)
            self.colorful_chk = ttk.Checkbutton(row, text="colorful", variable=self.colorful_var,
                                                state="disabled")
            self.colorful_chk.pack(side="left", padx=(0, 8))
            ttk.Label(row, text="steps:").pack(side="left")
            self.steps_var = tk.IntVar(value=DEFAULT_STEPS)
            self.steps_sb = ttk.Spinbox(row, from_=8, to=120, width=4, textvariable=self.steps_var,
                                        state="disabled")
            self.steps_sb.pack(side="left", padx=(2, 8))
            self.save_btn = ttk.Button(row, text=L("保存", "Save"), command=self._save_editor,
                                       state="disabled")
            self.save_btn.pack(side="left", padx=2)
            self.regen_btn = ttk.Button(row, text=L("再生成", "Regenerate"),
                                        command=self._regenerate, state="disabled")
            self.regen_btn.pack(side="left", padx=2)
            ttk.Label(ed, text=L("カスタムプロンプト (空=カテゴリ雛形):",
                                 "Custom prompt (empty=use template):")).pack(anchor="w", pady=(4, 0))
            self.prompt_text = tk.Text(ed, height=2, wrap="word", font=("Consolas", 9))
            self.prompt_text.pack(fill="x")
            self.editor_status = tk.StringVar(value="")
            ttk.Label(ed, textvariable=self.editor_status, foreground="#1565c0").pack(anchor="w")

            nb = ttk.Notebook(right)
            nb.pack(side="top", fill="both", expand=True, pady=(6, 0))
            self.txt_summary = self._make_text(nb, L("概要", "Summary"))
            self.txt_meta = self._make_text(nb, L("メタデータ", "Metadata"))
            self.txt_tensors = self._make_text(nb, L("テンソル", "Tensors"))

        def _make_text(self, nb, label: str):
            frame = ttk.Frame(nb)
            txt = tk.Text(frame, wrap="word", font=("Consolas", 9), state="disabled")
            sb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            txt.pack(side="left", fill="both", expand=True)
            nb.add(frame, text=label)
            return txt

        # ---- スキャン --------------------------------------------------- #
        def _scan_worker(self) -> None:
            for p in iter_safetensors(self.directory):
                try:
                    self.q.put(load_entry(p))
                except Exception:
                    continue
            self.q.put(None)

        def _drain_queue(self) -> None:
            added = False
            try:
                while True:
                    item = self.q.get_nowait()
                    if item is None:
                        self.loading = False
                        continue
                    self.all_entries.append(item)
                    added = True
            except queue.Empty:
                pass
            if added:
                self._refresh_kind_values()
                self._populate()
            elif not self.loading:
                self._update_status()
            self.after(120 if self.loading else 300, self._drain_queue)

        def _refresh_kind_values(self) -> None:
            kinds = ["All"] + sorted({e.kind for e in self.all_entries})
            if list(self.kind_cb["values"]) != kinds:
                self.kind_cb["values"] = kinds

        # ---- 一覧描画 --------------------------------------------------- #
        def _visible_sorted(self) -> list[Entry]:
            needle = self.search_var.get().strip().lower()
            kind = self.kind_var.get()
            base = self.base_var.get()
            keyfn = self.colkey.get(self.sort_col, lambda e: e.path.name.lower())

            def match(e: Entry) -> bool:
                if kind != "All" and e.kind != kind:
                    return False
                if base != "All" and e.base != base:
                    return False
                if needle:
                    if self.search_meta_var.get():
                        cat = self.pv_cats.get(e.path.stem, "")
                        hay = f"{e.path.name} {e.title} {e.triggers} {e.kind} {e.base} {cat}".lower()
                    else:
                        hay = e.path.name.lower()      # 既定: ファイル名のみ
                    return all(tok in hay for tok in needle.split())
                return True

            vis = [e for e in self.all_entries if match(e)]
            vis.sort(key=keyfn, reverse=self.desc_var.get())
            return vis

        def _on_search(self, *_) -> None:
            # 連続入力をまとめて 250ms 後に1回だけ絞り込み (大量行でも軽い)
            if self._search_after is not None:
                self.after_cancel(self._search_after)
            self._search_after = self.after(250, self._populate)

        def _populate(self) -> None:
            self.tree.delete(*self.tree.get_children())
            self.tree_iid = {}
            for e in self._visible_sorted():
                iid = str(e.path)
                self.tree.insert("", "end", iid=iid, tags=(e.kind,),
                                 values=tuple(disp(e) for *_, disp in self.columns))
                self.tree_iid[iid] = e
            if self.selected and str(self.selected.path) in self.tree_iid:
                self.tree.selection_set(str(self.selected.path))
            self._update_status()

        def _on_sort_combo(self) -> None:
            self.sort_col = self.label_to_cid.get(self.sort_var.get(), "name")
            self._populate()

        def _sort_by_col(self, cid: str) -> None:
            if self.sort_col == cid:
                self.desc_var.set(not self.desc_var.get())
            else:
                self.sort_col = cid
                self.sort_var.set(self.collabel[cid])
            self._populate()

        def _update_status(self) -> None:
            shown = len(self.tree_iid)
            total = len(self.all_entries)
            tail = L("（読み込み中…）", " (loading…)") if self.loading else ""
            self.status_var.set(L(f"表示 {shown} / 全 {total} 個{tail}",
                                  f"shown {shown} / {total} total{tail}"))

        # ---- 選択 → 詳細 ----------------------------------------------- #
        def _on_select(self, _evt=None) -> None:
            sel = self.tree.selection()
            entries = [self.tree_iid[i] for i in sel if i in self.tree_iid]
            self.selected_entries = entries
            if not entries:
                self.selected = None
                self._load_editor()
                return
            # 詳細は focus 行 (最後に触れた行) を表示
            focus_e = self.tree_iid.get(self.tree.focus()) or entries[-1]
            self.selected = focus_e
            self._show_thumb(focus_e)
            self._fill_summary(focus_e)
            self._fill_meta(focus_e)
            self._fill_tensors(focus_e)
            self._load_editor()

        # ---- プレビュー設定エディタ (category/custom prompt + 再生成、複数選択対応) ---- #
        def _load_editor(self) -> None:
            entries = self.selected_entries
            n = len(entries)
            loras = [e for e in entries if e.kind == "lora"]
            regenable = [e for e in entries if e.kind in ("lora", "base")]
            self.editor_status.set("")

            # category: 単一 lora=その値 / 複数 lora=共通なら表示・違えば空
            if loras:
                cset = {self.pv_cats.get(_entry_version(e), {}).get(e.path.stem, "ware") for e in loras}
                self.cat_var.set(next(iter(cset)) if len(cset) == 1 else "")
            else:
                self.cat_var.set("")
            # custom prompt は単一 lora のみ編集可 (複数では意味を持たない)
            self.prompt_text.configure(state="normal")
            self.prompt_text.delete("1.0", "end")
            single_lora = (n == 1 and bool(loras))
            if single_lora:
                v = _entry_version(loras[0])
                self.prompt_text.insert("1.0", self.pv_prompts.get(v, {}).get(loras[0].path.stem, ""))
            self.cat_cb.configure(state="readonly" if loras else "disabled")
            self.prompt_text.configure(state="normal" if single_lora else "disabled")
            self.save_btn.configure(state="normal" if loras else "disabled")
            self.regen_btn.configure(state="normal" if regenable else "disabled")
            self.colorful_chk.configure(state="normal" if regenable else "disabled")
            self.colorful_var.set(False)
            cat_now = self.cat_var.get() if loras else ""
            self.steps_var.set(CATEGORY_STEPS.get(cat_now, DEFAULT_STEPS))
            self.steps_sb.configure(state="normal" if regenable else "disabled")
            if n > 1:
                self.editor_status.set(L(f"{n} 件選択中 (一括: カテゴリ保存 / 再生成)",
                                         f"{n} selected (batch: save category / regenerate)"))

        def _on_cat_change(self) -> None:
            # カテゴリを切り替えたら steps の既定を追従 (part→多め)
            self.steps_var.set(CATEGORY_STEPS.get(self.cat_var.get(), DEFAULT_STEPS))

        def _save_editor(self) -> None:
            loras = [e for e in self.selected_entries if e.kind == "lora"]
            if not loras:
                return
            cat = self.cat_var.get() or "ware"
            try:
                if len(self.selected_entries) == 1:
                    e = loras[0]
                    stem = e.path.stem
                    ver = _entry_version(e)
                    cp = self.prompt_text.get("1.0", "end").strip()
                    save_preview_meta(stem, ver, cat, cp)
                    self.pv_cats.setdefault(ver, {})[stem] = cat
                    if cp:
                        self.pv_prompts.setdefault(ver, {})[stem] = cp
                    else:
                        self.pv_prompts.get(ver, {}).pop(stem, None)
                    self.editor_status.set(L(f"保存 [{ver}]: {cat}" + ("＋custom" if cp else ""),
                                             f"saved [{ver}]: {cat}" + (" +custom" if cp else "")))
                else:
                    stems_by_ver: dict = {}
                    for e in loras:
                        stems_by_ver.setdefault(_entry_version(e), []).append(e.path.stem)
                    set_preview_categories(stems_by_ver, cat)
                    for ver, stems in stems_by_ver.items():
                        for s in stems:
                            self.pv_cats.setdefault(ver, {})[s] = cat
                    total = sum(len(v) for v in stems_by_ver.values())
                    self.editor_status.set(L(f"{total} 件を {cat} に設定",
                                             f"set {total} to {cat}"))
            except Exception as ex:
                self.editor_status.set(L(f"保存失敗: {ex}", f"save failed: {ex}"))

        def _regenerate(self) -> None:
            targets = [e for e in self.selected_entries if e.kind in ("lora", "base")]
            if not targets:
                return
            extra = "colorful clothes" if self.colorful_var.get() else ""
            try:
                steps = max(8, min(120, int(self.steps_var.get())))
            except Exception:
                steps = DEFAULT_STEPS
            self._launch_regen(targets, extra=extra, steps=steps)

        def _batch_missing(self) -> None:
            """サイドカーが無い全テンソル (lora/checkpoint) を一括で焼く。既定設定 (色強制なし, steps 既定)。"""
            missing = [e for e in self.all_entries
                       if e.kind in ("lora", "base") and not e.has_thumb]
            if not missing:
                self.editor_status.set(L("未生成なし (全件サイドカー有)", "no missing previews"))
                return
            self._launch_regen(missing, extra="", steps=DEFAULT_STEPS)

        def _launch_regen(self, targets, extra: str = "", steps: int = DEFAULT_STEPS) -> None:
            """make_previews.py --files をサブプロセスで起動 (cmdline 長制限回避でチャンク分割)。
            進捗は regen_q に流し、_poll_regen が拾って一覧/プレビューを更新する。
            GUI プロセスは torch を読まないので固まらない (閲覧=torch不要 を維持)。"""
            self.regen_btn.configure(state="disabled")
            self.thumb_btn.configure(state="disabled")
            self._regen_targets = list(targets)
            total = len(targets)
            self.editor_status.set(L(f"生成中… 0/{total} (ComfyUI)", f"generating… 0/{total} (ComfyUI)"))
            CHUNK = 100   # Windows cmdline 長制限回避 (1チャンクあたり最大ファイル数)
            chunks = [targets[i:i + CHUNK] for i in range(0, total, CHUNK)]

            def work():
                done = 0
                try:
                    for chunk in chunks:
                        cmd = [sys.executable, str(MAKE_PREVIEWS_PY), "--files"]
                        cmd += [str(e.path.resolve()) for e in chunk]
                        cmd += ["--steps", str(steps)]
                        if extra:
                            cmd += ["--extra", extra]
                        proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            cwd=str(MAKE_PREVIEWS_PY.parent))
                        for line in proc.stdout:
                            if ".preview.png" in line:
                                done += 1
                                self.regen_q.put(("progress", None, None, done, total))
                        rc = proc.wait()
                        if rc != 0:
                            self.regen_q.put(("fatal", None, f"chunk exit rc={rc}", done, total))
                            return
                    self.regen_q.put(("alldone", None, None, 0, total))
                except Exception as ex:
                    self.regen_q.put(("fatal", None, str(ex), done, total))

            threading.Thread(target=work, daemon=True).start()
            self.after(200, self._poll_regen)

        def _poll_regen(self) -> None:
            try:
                while True:
                    status, _e, payload, i, total = self.regen_q.get_nowait()
                    if status == "progress":
                        self.editor_status.set(L(f"生成中… {i}/{total} (ComfyUI)",
                                                 f"generating… {i}/{total} (ComfyUI)"))
                    elif status in ("alldone", "fatal"):
                        self.regen_btn.configure(state="normal")
                        self.thumb_btn.configure(state="normal")
                        if status == "fatal":
                            self.editor_status.set(L(f"エラー: {payload}", f"error: {payload}"))
                            return
                        # サイドカーを再読込して一覧/プレビューを更新
                        n_ok = 0
                        for e in self._regen_targets:
                            sc = e.path.with_suffix(".preview.png")
                            if sc.exists():
                                e.preview_path = sc
                                e.has_thumb = True
                                n_ok += 1
                                iid = str(e.path)
                                if iid in self.tree_iid:
                                    try:
                                        self.tree.set(iid, "thumb", "✓")
                                    except Exception:
                                        pass
                        if self.selected:
                            self._show_thumb(self.selected)
                        self.editor_status.set(L(f"完了 ({n_ok}/{total} 件, rc={i})",
                                                 f"done ({n_ok}/{total}, rc={i})"))
                        return
            except queue.Empty:
                pass
            self.after(200, self._poll_regen)

        def _show_thumb(self, e: Entry) -> None:
            im = None
            # サイドカー (<name>.preview.png) を優先
            if e.preview_path:
                try:
                    im = Image.open(e.preview_path).convert("RGB")
                except Exception:
                    im = None
            # 無ければ __metadata__ の埋め込みサムネ (data URI)
            if im is None:
                uri = find_thumb_uri(e.meta)
                if uri:
                    try:
                        im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
                    except Exception:
                        im = None
            if im is None:
                self.preview_ref = None
                self.preview_lbl.configure(image="", text=L("(プレビューなし)", "(no preview)"), height=4)
                return
            im.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
            self.preview_ref = ImageTk.PhotoImage(im)
            self.preview_lbl.configure(image=self.preview_ref, text="", height=0)

        def _set_text(self, widget, content: str) -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", content)
            widget.configure(state="disabled")

        def _fill_summary(self, e: Entry) -> None:
            lines = [
                e.path.name,
                f"path     : {e.path}",
                f"kind     : {e.kind}",
                f"base     : {e.base}",
                f"size     : {human_size(e.size)}  ({e.size:,} bytes)",
                f"tensors  : {e.n_tensors}",
                f"params   : {human_count(e.n_params)}  ({e.n_params:,})",
                f"dtypes   : {dtypes_str(e.dtypes)}",
            ]
            if e.title:
                lines.append(f"title    : {e.title}")
            if e.preview_path:
                lines.append(f"preview  : {e.preview_path.name} (sidecar)")
            if e.triggers:
                lines += ["", L("[トリガー語 / 学習タグ (頻度順)]", "[trigger words / training tags (by freq)]"),
                          e.triggers]
            present = [(lbl, e.meta[key]) for key, lbl in _KEY_META if key in e.meta]
            if present:
                lines += ["", L("[主要な学習メタ]", "[key training metadata]")]
                lines += [f"  {lbl:<16}: {val}" for lbl, val in present]
            self._set_text(self.txt_summary, "\n".join(lines))

        def _fill_meta(self, e: Entry) -> None:
            if not e.meta:
                self._set_text(self.txt_meta, L("(__metadata__ なし)", "(no __metadata__)"))
                return
            out = []
            for k in sorted(e.meta):
                v = e.meta[k]
                if isinstance(v, str) and v[:11] == "data:image/":
                    out.append(f"{k}: [data URI image, {len(v)} chars]")
                    continue
                # JSON 文字列なら整形して展開
                if isinstance(v, str) and v[:1] in "{[":
                    try:
                        v = json.dumps(json.loads(v), ensure_ascii=False, indent=2)
                    except (ValueError, TypeError):
                        pass
                out.append(f"{k}: {v}")
            self._set_text(self.txt_meta, "\n".join(out))

        def _fill_tensors(self, e: Entry) -> None:
            if e.kind == "broken":
                self._set_text(self.txt_tensors, L("(読み取り失敗)", "(failed to read)"))
                return
            try:
                tensors, _ = read_header(e.path)   # テンソル一覧は選択時に都度読む (省メモリ)
            except (OSError, ValueError) as ex:
                self._set_text(self.txt_tensors, f"ERROR: {ex}")
                return
            rows = [f"{e.n_tensors} tensors / {human_count(e.n_params)} params",
                    "-" * 60]
            for name in sorted(tensors)[:8000]:
                info = tensors[name]
                shape = "×".join(map(str, info.get("shape", []))) or "scalar"
                rows.append(f"{info.get('dtype', '?'):<7} {shape:<20} {name}")
            if len(tensors) > 8000:
                rows.append(L(f"... 他 {len(tensors) - 8000} 個省略", f"... {len(tensors) - 8000} more omitted"))
            self._set_text(self.txt_tensors, "\n".join(rows))

        def _reload(self) -> None:
            self.all_entries = []
            self.tree_iid = {}
            self.selected = None
            self.loading = True
            self.status_var.set(L("読み込み中…", "Loading…"))
            self.q = queue.Queue()
            self._populate()
            self.title(L(f"テンソル情報ビューア — {self.directory}",
                         f"Tensor Info Viewer — {self.directory}"))
            threading.Thread(target=self._scan_worker, daemon=True).start()

        def _project_root(self) -> Path:
            return Path(__file__).resolve().parent

        def _dir_choices(self) -> list[str]:
            """preview_settings.toml [tensors_dirs] list を候補に。--dir で指定した dir
            (候補外でも) は先頭に足して呼び出し履歴を残す。"""
            out = [d for d in _load_tensors_dirs() if (self._project_root() / d).exists()]
            cur = self._dir_label(self.directory)
            if cur not in out:
                out.insert(0, cur)
            return out

        def _dir_label(self, p: Path) -> str:
            try:
                rel = p.resolve().relative_to(self._project_root())
                return str(rel).replace("\\", "/")
            except Exception:
                return str(p)

        def _on_dir_change(self) -> None:
            label = self.dir_var.get()
            new_dir = (self._project_root() / label).resolve()
            if not new_dir.exists() or new_dir == self.directory:
                return
            self.directory = new_dir
            self._reload()

        def _show_full_preview(self) -> None:
            """preview_lbl クリック時に現在エントリのサイドカー (または埋め込みサムネ) をフルビュー表示。"""
            e = self.selected
            if e is None:
                return
            try:
                from PIL import Image, ImageTk
            except Exception:
                return
            im: Optional["Image.Image"] = None
            if e.preview_path and e.preview_path.exists():
                try:
                    im = Image.open(e.preview_path)
                except Exception:
                    im = None
            if im is None:
                uri = find_thumb_uri(e.meta) if e.meta else None
                if uri:
                    try:
                        import base64, io
                        b64 = uri.split(",", 1)[-1]
                        im = Image.open(io.BytesIO(base64.b64decode(b64)))
                    except Exception:
                        im = None
            if im is None:
                return
            # 画面サイズに合わせて等倍だが必要なら縮小
            top = tk.Toplevel(self)
            top.title(e.path.name)
            sw, sh = self.winfo_screenwidth() - 80, self.winfo_screenheight() - 120
            w, h = im.size
            scale = min(1.0, sw / w, sh / h)
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            lbl = tk.Label(top, image=photo, cursor="hand2")
            lbl.image = photo   # type: ignore[attr-defined]
            lbl.pack()
            lbl.bind("<Button-1>", lambda *_: top.destroy())
            top.bind("<Escape>", lambda *_: top.destroy())

    TensorViewApp().mainloop()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=L("safetensors のメタ情報を一覧するビューア (Tkinter)",
                      "Metadata viewer for safetensors files (Tkinter)"))
    ap.add_argument("--dir", type=str, default=None,
                    help=L("走査するディレクトリ (再帰)。省略時は preview_settings.toml [tensors_dirs] list の先頭",
                           "directory to scan (recursive). default = first entry of preview_settings.toml [tensors_dirs] list"))
    ap.add_argument("--list", action="store_true",
                    help=L("GUI を出さず一覧を標準出力", "print a list without GUI"))
    args = ap.parse_args()

    if args.dir:
        directory = Path(args.dir)
    else:
        directory = default_dir()
        if directory is None:
            raise SystemExit(L("--dir 省略時の既定が preview_settings.toml [tensors_dirs] に見つかりません",
                               "no default in preview_settings.toml [tensors_dirs]; pass --dir"))
    if not directory.is_dir():
        raise SystemExit(L(f"ディレクトリが見つかりません: {directory}",
                           f"Directory not found: {directory}"))
    if args.list:
        run_list(directory)
    else:
        run_gui(directory)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(L("\n中断", "\nInterrupted"))
        sys.exit(0)
