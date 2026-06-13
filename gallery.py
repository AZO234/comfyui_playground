#!/usr/bin/env python3
"""gallery.py - 生成済み PNG をサムネイル一覧するデスクトップビューア (Tkinter)。

固定 4 ディレクトリ (SD15 / SDXL × 生成 / アップスケール) を再帰走査し、
A1111 互換メタ (parameters chunk) を読み取ってサムネイル + メタ情報の一覧を表示する。
名前 / 時刻 / アーキ(SD15·SDXL) / Model / Size でソート、テキスト検索でフィルタ、
選択画像の全メタ表示、OS ビューアで開ける。

**閲覧専用**: コピー・削除などの書き込み操作はサポートしない (誤操作防止)。
画像の削除や移動は generate_gui.py のギャラリーやエクスプローラから行う。

メタ読出は pngutil の read_text_chunks / parse_a1111_parameters を流用。
SD15/SDXL の判定は parameters の `Pipeline` フィールド優先 + `Size` で補完。
UI 文字列は i18n の L() で英/日切替 (PLAYGROUND_LANG / OS ロケール)。

使い方:
    python gallery.py                      # GUI ビューアを起動 (固定 4 dir)
    python gallery.py --list               # GUI なしでメタ一覧を標準出力
    python gallery.py --dir PATH [PATH...] # 任意のディレクトリを走査対象に差し替え
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from i18n import L
from pngutil import parse_a1111_parameters, read_text_chunks

# Windows console 絵文字落ち防止
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parent

# 閲覧対象の既定ディレクトリ (SD15 / SDXL × 生成 / upscaled の 4 dir を再帰走査して PNG を集める)
# `--dir PATH [PATH...]` 指定時はこの既定が差し替えられる (main() 参照)
VIEW_DIRS = [
    ROOT / "3_8_SD15_generated",
    ROOT / "3_9_SD15_upscaled",
    ROOT / "5_1_SDXL_generated",
    ROOT / "5_2_SDXL_upscaled",
]

# サムネイル / プレビューの最大辺 (px)
THUMB_SIZE = 200      # アイコン view の格子サムネ
LIST_THUMB = 40       # リスト view の行サムネ (Treeview #0 列)
PREVIEW_MAX = 480
PAD = 6
# アーキ表示色 (caption の文字色)
ARCH_COLOR = {"SD15": "#2e7d32", "SDXL": "#1565c0", "?": "#777777"}


# --------------------------------------------------------------------------- #
# データモデル / メタ抽出
# --------------------------------------------------------------------------- #
@dataclass
class Entry:
    """1 枚の PNG とその抽出済みメタ。thumb / widget は GUI 起動後に埋める。"""
    path: Path
    mtime: float
    arch: str            # "SD15" / "SDXL" / "?"
    model: str
    loras: str
    lora_keywords: str
    pipeline: str
    size_str: str
    seed: str
    steps: str
    sampler: str
    positive: str
    negative: str
    params: dict
    thumb: object = None        # ImageTk.PhotoImage 格子用 (GUI 時のみ)
    pil_thumb: object = None    # 200px PIL 原本 (リスト小サムネをスライダーで作り直す元)
    thumb_sm: object = None     # ImageTk.PhotoImage リスト行用 (GUI 時のみ)
    thumb_sm_size: int = 0      # thumb_sm を生成したときの px (サイズ変更検知用)
    widget: object = None       # サムネイルセルの Frame (GUI 時のみ)


def detect_arch(params: dict) -> str:
    """parameters の Pipeline / Size から SD15 か SDXL かを判定する。

    Pipeline 例: 'SDXL single-pass' / 'SD15 single-pass' /
                 'SD15→SDXL 2-stage (...)' / 'refine (...)'。
    最終画像のアーキで分類するので 2-stage / refine は SDXL 扱い。
    Pipeline 不在時は Size の最大辺 (>=1024 → SDXL) で補完。
    """
    pl = (params.get("Pipeline") or "").upper()
    if pl:
        if "SD15" in pl and "SDXL" not in pl and "2-STAGE" not in pl and "REFINE" not in pl:
            return "SD15"
        if "SDXL" in pl or "2-STAGE" in pl or "REFINE" in pl:
            return "SDXL"
    size = params.get("Size", "")
    try:
        w, h = size.lower().split("x")
        return "SDXL" if max(int(w), int(h)) >= 1024 else "SD15"
    except (ValueError, AttributeError):
        return "?"


def load_entry(path: Path) -> Optional[Entry]:
    """PNG からメタを抽出して Entry を作る。読めない PNG は None。"""
    try:
        chunks = read_text_chunks(path)
        mtime = path.stat().st_mtime
    except (OSError, ValueError):
        return None
    parsed = parse_a1111_parameters(chunks.get("parameters", ""))
    params = parsed.get("params", {})
    return Entry(
        path=path,
        mtime=mtime,
        arch=detect_arch(params),
        model=params.get("Model", ""),
        loras=params.get("Loras", ""),
        lora_keywords=params.get("Lora keywords", ""),
        pipeline=params.get("Pipeline", ""),
        size_str=params.get("Size", ""),
        seed=params.get("Seed", ""),
        steps=params.get("Steps", ""),
        sampler=params.get("Sampler", ""),
        positive=parsed.get("positive", ""),
        negative=parsed.get("negative", ""),
        params=params,
    )


def iter_pngs(directories):
    """渡された複数 dir 配下の .png を再帰で列挙する (順序は (dir 順、その中での sorted))。"""
    if isinstance(directories, Path):
        directories = [directories]
    for d in directories:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.png")):
            if p.is_file():
                yield p


# --------------------------------------------------------------------------- #
# ソート / フィルタ
# --------------------------------------------------------------------------- #
def _intkey(s) -> int:
    """数値文字列 (seed/steps) をソート用 int に。非数値は -1。"""
    try:
        return int(str(s))
    except (ValueError, TypeError):
        return -1


# 列定義: (col_id, 表示ラベル, 幅, 寄せ, ソートキー fn, 表示値 fn)。
# アイコン view の並べ替え combobox と、リスト view (Treeview) の列ヘッダの両方が
# この 1 定義を共有する。
def _columns() -> list:
    fmt_time = lambda e: datetime.fromtimestamp(e.mtime).strftime("%Y-%m-%d %H:%M")
    return [
        ("name",     L("名前", "Name"),     220, "w",      lambda e: e.path.name.lower(),     lambda e: e.path.name),
        ("time",     L("時刻", "Time"),      140, "w",      lambda e: e.mtime,                 fmt_time),
        ("arch",     "Arch",                  55, "center", lambda e: e.arch,                  lambda e: e.arch),
        ("size",     "Size",                  95, "center", lambda e: e.size_str,              lambda e: e.size_str),
        ("model",    "Model",                190, "w",      lambda e: e.model.lower(),         lambda e: e.model),
        ("loras",    "LoRAs",                180, "w",      lambda e: e.loras.lower(),         lambda e: e.loras),
        ("kw",       L("LoRA語", "LoRA kw"), 150, "w",      lambda e: e.lora_keywords.lower(), lambda e: e.lora_keywords),
        ("seed",     "Seed",                  95, "w",      lambda e: _intkey(e.seed),         lambda e: e.seed),
        ("steps",    "Steps",                 55, "center", lambda e: _intkey(e.steps),        lambda e: e.steps),
        ("pipeline", "Pipeline",             150, "w",      lambda e: e.pipeline,              lambda e: e.pipeline),
    ]


def _matches(entry: Entry, needle: str) -> bool:
    """検索語 (小文字) が name/model/loras/keywords/positive/pipeline に含まれるか。"""
    if not needle:
        return True
    hay = " ".join((
        entry.path.name, entry.model, entry.loras,
        entry.lora_keywords, entry.positive, entry.pipeline,
    )).lower()
    return all(tok in hay for tok in needle.split())


# --------------------------------------------------------------------------- #
# --list (GUI なし)
# --------------------------------------------------------------------------- #
def run_list(directories) -> None:
    """GUI を立てずにメタ一覧を標準出力する (ヘッドレス検証 / 確認用)。"""
    entries = [e for e in (load_entry(p) for p in iter_pngs(directories)) if e]
    if not entries:
        print(L(f"PNG が見つかりません: {directories}", f"No PNG found: {directories}"))
        return
    entries.sort(key=lambda e: e.mtime)
    label = ", ".join(d.name for d in directories) if isinstance(directories, list) else str(directories)
    print(L(f"=== {label} : {len(entries)} 枚 ===",
            f"=== {label} : {len(entries)} images ==="))
    n_xl = sum(e.arch == "SDXL" for e in entries)
    n_15 = sum(e.arch == "SD15" for e in entries)
    print(L(f"  SDXL: {n_xl} / SD15: {n_15} / 不明: {len(entries) - n_xl - n_15}",
            f"  SDXL: {n_xl} / SD15: {n_15} / unknown: {len(entries) - n_xl - n_15}"))
    print(f"{'time':<17} {'arch':<4} {'size':<10} {'model':<28} name")
    print("-" * 100)
    for e in entries:
        ts = datetime.fromtimestamp(e.mtime).strftime("%Y-%m-%d %H:%M")
        model = (e.model[:27] + "…") if len(e.model) > 28 else e.model
        print(f"{ts:<17} {e.arch:<4} {e.size_str:<10} {model:<28} {e.path.name}")


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def run_gui(directories) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    from PIL import ImageTk

    class GalleryApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.directories = directories
            label = ", ".join(d.name for d in directories)
            self.title(L(f"画像ギャラリー — {label} (閲覧専用)",
                         f"Image Gallery — {label} (read-only)"))
            self.geometry("1280x820")

            self.all_entries: list[Entry] = []
            self.gridded: list[tk.Widget] = []
            self.selected: Optional[Entry] = None
            self.preview_imgref = None       # プレビュー PhotoImage の参照保持
            self.last_cols = 0
            self.loading = True
            self.q: queue.Queue = queue.Queue()

            self.columns = _columns()
            self.colkey = {c[0]: c[4] for c in self.columns}
            self.collabel = {c[0]: c[1] for c in self.columns}
            self.label_to_cid = {c[1]: c[0] for c in self.columns}
            self.sort_col = "time"
            self.tree_iid: dict[str, Entry] = {}
            self.list_thumb = LIST_THUMB     # リスト行サムネの現在サイズ (スライダーで可変)
            self._thumb_job = None           # スライダー debounce 用 after id

            self.search_var = tk.StringVar()
            self.arch_var = tk.StringVar(value="All")
            self.sort_var = tk.StringVar(value=self.collabel["time"])
            self.view_var = tk.StringVar(value="icon")   # "icon" / "list"
            self.desc_var = tk.BooleanVar(value=True)
            self.status_var = tk.StringVar(value=L("読み込み中…", "Loading…"))

            self._build_toolbar()
            self._build_body()
            self._build_context_menu()

            self.search_var.trace_add("write", lambda *_: self._relayout())
            threading.Thread(target=self._scan_worker, daemon=True).start()
            self.after(80, self._drain_queue)

        # ---- レイアウト構築 ---------------------------------------------- #
        def _build_toolbar(self) -> None:
            bar = ttk.Frame(self, padding=(8, 6))
            bar.pack(side="top", fill="x")

            ttk.Label(bar, text=L("検索:", "Search:")).pack(side="left")
            ent = ttk.Entry(bar, textvariable=self.search_var, width=28)
            ent.pack(side="left", padx=(4, 12))

            ttk.Label(bar, text=L("アーキ:", "Arch:")).pack(side="left")
            arch = ttk.Combobox(bar, textvariable=self.arch_var, width=6, state="readonly",
                                values=["All", "SDXL", "SD15", "?"])
            arch.pack(side="left", padx=(4, 12))
            arch.bind("<<ComboboxSelected>>", lambda *_: self._relayout())

            ttk.Label(bar, text=L("表示:", "View:")).pack(side="left")
            ttk.Radiobutton(bar, text=L("アイコン", "Icon"), value="icon",
                            variable=self.view_var, command=self._switch_view).pack(side="left")
            ttk.Radiobutton(bar, text=L("リスト", "List"), value="list",
                            variable=self.view_var, command=self._switch_view).pack(side="left", padx=(0, 12))

            ttk.Label(bar, text=L("並べ替え:", "Sort:")).pack(side="left")
            sort = ttk.Combobox(bar, textvariable=self.sort_var, width=10, state="readonly",
                                values=[c[1] for c in self.columns])
            sort.pack(side="left", padx=(4, 4))
            sort.bind("<<ComboboxSelected>>", lambda *_: self._on_sort_combo())
            ttk.Checkbutton(bar, text=L("降順", "Desc"), variable=self.desc_var,
                            command=self._refresh_view).pack(side="left", padx=(0, 12))

            ttk.Button(bar, text=L("再読込", "Reload"), command=self._reload).pack(side="left")

            # 行サムネサイズ スライダー (リスト view のときだけ表示。_switch_view で pack/forget)
            self.thumb_ctl = ttk.Frame(bar)
            ttk.Label(self.thumb_ctl, text=L("サムネ:", "Thumb:")).pack(side="left")
            self.thumb_size_lbl = ttk.Label(self.thumb_ctl, text=f"{self.list_thumb}px", width=5)
            self.thumb_scale = ttk.Scale(self.thumb_ctl, from_=24, to=128, orient="horizontal",
                                         length=110, command=self._on_thumb_scale)
            self.thumb_scale.pack(side="left", padx=(4, 2))
            self.thumb_size_lbl.pack(side="left")
            self.thumb_scale.set(self.list_thumb)   # ラベル生成後に set (command が走るため)

            ttk.Label(bar, textvariable=self.status_var).pack(side="right")

        def _build_body(self) -> None:
            paned = ttk.PanedWindow(self, orient="horizontal")
            paned.pack(side="top", fill="both", expand=True)

            # 左: アイコン / リストを切り替えるコンテナ
            left = ttk.Frame(paned)
            paned.add(left, weight=3)

            # --- アイコン表示: スクロール可能なサムネイルグリッド ---
            self.icon_frame = ttk.Frame(left)
            self.canvas = tk.Canvas(self.icon_frame, background="#1e1e1e", highlightthickness=0)
            vsb = ttk.Scrollbar(self.icon_frame, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)
            self.inner = tk.Frame(self.canvas, background="#1e1e1e")
            self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
            self.canvas.bind("<Configure>", self._on_canvas_configure)
            # ホイールはアイコン view のウィジェットにだけ束縛する (bind_all だと
            # 右の情報テキストや Treeview の上で回しても canvas が動いてしまうため)。
            self.canvas.bind("<MouseWheel>", self._on_wheel)
            self.inner.bind("<MouseWheel>", self._on_wheel)

            # --- リスト表示: テキスト重視の Treeview (小サムネ + 列ヘッダソート) ---
            self.list_frame = ttk.Frame(left)
            ttk.Style(self).configure("Treeview", rowheight=self.list_thumb + 4)
            cids = [c[0] for c in self.columns]
            self.tree = ttk.Treeview(self.list_frame, columns=cids, show="tree headings",
                                     selectmode="browse")
            self.tree.heading("#0", text="")          # #0 = 小サムネ列
            self.tree.column("#0", width=self.list_thumb + 12, minwidth=self.list_thumb + 12,
                             anchor="center", stretch=False)
            for cid, label, width, anchor, *_ in self.columns:
                self.tree.heading(cid, text=label, command=lambda c=cid: self._sort_by_col(c))
                # 固定幅 (stretch=False) にして合計幅がペインを超えたら横スクロール可能にする
                self.tree.column(cid, width=width, anchor=anchor, stretch=False)
            tvsb = ttk.Scrollbar(self.list_frame, orient="vertical", command=self.tree.yview)
            thsb = ttk.Scrollbar(self.list_frame, orient="horizontal", command=self.tree.xview)
            self.tree.configure(yscrollcommand=tvsb.set, xscrollcommand=thsb.set)
            # tree + 縦/横スクロールバーの定番 grid 構成
            self.tree.grid(row=0, column=0, sticky="nsew")
            tvsb.grid(row=0, column=1, sticky="ns")
            thsb.grid(row=1, column=0, sticky="ew")
            self.list_frame.rowconfigure(0, weight=1)
            self.list_frame.columnconfigure(0, weight=1)
            self.tree.tag_configure("SD15", foreground="#2e7d32")
            self.tree.tag_configure("SDXL", foreground="#1565c0")
            self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
            self.tree.bind("<Double-Button-1>", lambda _e: self._open_selected())
            self.tree.bind("<Button-3>", self._tree_popup)

            self.icon_frame.pack(fill="both", expand=True)   # 既定はアイコン表示

            # 右: 詳細パネル (プレビュー + メタ + 操作)
            right = ttk.Frame(paned, padding=8)
            self.preview_lbl = tk.Label(right, background="#2a2a2a")
            self.preview_lbl.pack(side="top", fill="x")
            btns = ttk.Frame(right)
            btns.pack(side="top", fill="x", pady=6)
            ttk.Button(btns, text=L("開く", "Open"), command=self._open_selected).pack(side="left")
            self.detail = tk.Text(right, wrap="word", width=44, height=20,
                                  font=("Consolas", 9), state="disabled")
            self.detail.pack(side="top", fill="both", expand=True)
            paned.add(right, weight=2)

        def _build_context_menu(self) -> None:
            self.ctx = tk.Menu(self, tearoff=0)
            self.ctx.add_command(label=L("開く", "Open"), command=self._open_selected)

        # ---- ビュー切替 / 共通リフレッシュ ------------------------------ #
        def _switch_view(self) -> None:
            self.icon_frame.pack_forget()
            self.list_frame.pack_forget()
            if self.view_var.get() == "list":
                self.list_frame.pack(fill="both", expand=True)
                self.thumb_ctl.pack(side="left", padx=(0, 12))   # スライダーを表示
            else:
                self.icon_frame.pack(fill="both", expand=True)
                self.thumb_ctl.pack_forget()                     # アイコン view では隠す
            self._refresh_view()

        def _refresh_view(self) -> None:
            """アクティブなビューだけを再描画 (フィルタ/ソート結果を反映)。"""
            if self.view_var.get() == "list":
                self._populate_tree()
            else:
                self._relayout()
            self._update_status()

        def _on_sort_combo(self) -> None:
            self.sort_col = self.label_to_cid.get(self.sort_var.get(), "time")
            self._refresh_view()

        def _sort_by_col(self, cid: str) -> None:
            """リスト view の列ヘッダクリック。同じ列なら昇降トグル、別列なら切替。"""
            if self.sort_col == cid:
                self.desc_var.set(not self.desc_var.get())
            else:
                self.sort_col = cid
                self.sort_var.set(self.collabel[cid])
            self._refresh_view()

        def _populate_tree(self) -> None:
            self.tree.delete(*self.tree.get_children())
            self.tree_iid = {}
            for e in self._visible_sorted():
                self._ensure_sm(e)
                iid = str(e.path)
                self.tree.insert("", "end", iid=iid, tags=(e.arch,), image=(e.thumb_sm or ""),
                                 values=tuple(disp(e) for *_, disp in self.columns))
                self.tree_iid[iid] = e
            if self.selected and str(self.selected.path) in self.tree_iid:
                self.tree.selection_set(str(self.selected.path))

        def _ensure_sm(self, e: Entry) -> None:
            """現在の list_thumb サイズで行サムネ (thumb_sm) を必要時だけ生成する。"""
            if e.pil_thumb is None:
                return
            if e.thumb_sm is None or e.thumb_sm_size != self.list_thumb:
                im = e.pil_thumb.copy()
                im.thumbnail((self.list_thumb, self.list_thumb), Image.LANCZOS)
                e.thumb_sm = ImageTk.PhotoImage(im)
                e.thumb_sm_size = self.list_thumb

        def _on_thumb_scale(self, _val=None) -> None:
            size = int(round(float(self.thumb_scale.get())))
            self.thumb_size_lbl.configure(text=f"{size}px")
            if size == self.list_thumb:
                return
            self.list_thumb = size
            if self._thumb_job is not None:
                self.after_cancel(self._thumb_job)
            self._thumb_job = self.after(150, self._apply_thumb_size)   # ドラッグ debounce

        def _apply_thumb_size(self) -> None:
            self._thumb_job = None
            ttk.Style(self).configure("Treeview", rowheight=self.list_thumb + 4)
            self.tree.column("#0", width=self.list_thumb + 12, minwidth=self.list_thumb + 12)
            if self.view_var.get() == "list":
                self._populate_tree()

        def _on_tree_select(self, _evt=None) -> None:
            sel = self.tree.selection()
            if sel:
                e = self.tree_iid.get(sel[0])
                if e:
                    self._select(e)

        def _tree_popup(self, event) -> None:
            iid = self.tree.identify_row(event.y)
            if not iid:
                return
            self.tree.selection_set(iid)
            e = self.tree_iid.get(iid)
            if e:
                self._select(e)
                try:
                    self.ctx.tk_popup(event.x_root, event.y_root)
                finally:
                    self.ctx.grab_release()

        # ---- スキャン (バックグラウンド) -------------------------------- #
        def _scan_worker(self) -> None:
            """別スレッドでメタ抽出 + サムネ生成し、結果を queue に流す。"""
            for p in iter_pngs(self.directories):
                e = load_entry(p)
                if e is None:
                    continue
                try:
                    im = Image.open(p)
                    im.draft("RGB", (THUMB_SIZE, THUMB_SIZE))
                    im = im.convert("RGB")
                    im.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                except (OSError, ValueError):
                    continue
                self.q.put((e, im))   # 行サムネは _ensure_sm が現スライダー値で遅延生成
            self.q.put(None)   # 終了マーカー

        def _drain_queue(self) -> None:
            added = False
            try:
                while True:
                    item = self.q.get_nowait()
                    if item is None:
                        self.loading = False
                        continue
                    entry, pil = item
                    entry.thumb = ImageTk.PhotoImage(pil)
                    entry.pil_thumb = pil   # スライダーで行サムネを作り直す元
                    self._make_cell(entry)
                    self.all_entries.append(entry)
                    added = True
            except queue.Empty:
                pass
            if added:
                self._refresh_view()
            elif not self.loading:
                self._update_status()
            if self.loading:
                self.after(80, self._drain_queue)
            else:
                self.after(200, self._drain_queue)   # 操作後の relayout 反映用に継続

        # ---- セル生成 / レイアウト -------------------------------------- #
        def _make_cell(self, entry: Entry) -> None:
            cell = tk.Frame(self.inner, bd=2, relief="flat", background="#1e1e1e",
                            highlightthickness=2, highlightbackground="#1e1e1e")
            img = tk.Label(cell, image=entry.thumb, background="#1e1e1e")
            img.pack()
            cap = (entry.path.name if len(entry.path.name) <= 24
                   else entry.path.name[:22] + "…")
            txt = tk.Label(cell, text=f"{entry.arch}  {cap}", background="#1e1e1e",
                           fg=ARCH_COLOR.get(entry.arch, "#cccccc"),
                           font=("Segoe UI", 8))
            txt.pack()
            entry.widget = cell
            for w in (cell, img, txt):
                w.bind("<Button-1>", lambda _e, en=entry: self._select(en))
                w.bind("<Double-Button-1>", lambda _e, en=entry: (self._select(en), self._open_selected()))
                w.bind("<Button-3>", lambda ev, en=entry: self._popup(ev, en))
                w.bind("<MouseWheel>", self._on_wheel)   # サムネ上でも canvas をスクロール

        def _visible_sorted(self) -> list[Entry]:
            needle = self.search_var.get().strip().lower()
            arch = self.arch_var.get()
            keyfn = self.colkey.get(self.sort_col, lambda e: e.mtime)
            vis = [e for e in self.all_entries
                   if _matches(e, needle) and (arch == "All" or e.arch == arch)]
            vis.sort(key=keyfn, reverse=self.desc_var.get())
            return vis

        def _relayout(self) -> None:
            cols = max(1, (self.canvas.winfo_width() - PAD) // (THUMB_SIZE + 2 * PAD + 8))
            for w in self.gridded:
                w.grid_remove()
            self.gridded = []
            vis = self._visible_sorted()
            for i, e in enumerate(vis):
                r, c = divmod(i, cols)
                e.widget.grid(row=r, column=c, padx=PAD, pady=PAD)
                self.gridded.append(e.widget)
            self.last_cols = cols
            self.inner.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_configure(self, event) -> None:
            self.canvas.itemconfig(self.inner_id, width=event.width)
            if self.view_var.get() != "icon":
                return
            cols = max(1, (event.width - PAD) // (THUMB_SIZE + 2 * PAD + 8))
            if cols != self.last_cols:
                self._relayout()

        def _on_wheel(self, event) -> None:
            if self.view_var.get() == "icon":   # リストは Treeview が自前でスクロール
                self.canvas.yview_scroll(int(-event.delta / 120), "units")

        def _update_status(self) -> None:
            shown = len(self._visible_sorted())
            total = len(self.all_entries)
            tail = L("（読み込み中…）", " (loading…)") if self.loading else ""
            self.status_var.set(L(f"表示 {shown} / 全 {total} 枚{tail}",
                                  f"shown {shown} / {total} total{tail}"))

        # ---- 選択 / 詳細表示 -------------------------------------------- #
        def _select(self, entry: Entry) -> None:
            if self.selected and self.selected.widget:
                self.selected.widget.configure(highlightbackground="#1e1e1e")
            self.selected = entry
            if entry.widget:
                entry.widget.configure(highlightbackground="#e0a000")
            self._show_detail(entry)

        def _popup(self, event, entry: Entry) -> None:
            self._select(entry)
            try:
                self.ctx.tk_popup(event.x_root, event.y_root)
            finally:
                self.ctx.grab_release()

        def _show_detail(self, entry: Entry) -> None:
            # 大きめプレビュー
            try:
                im = Image.open(entry.path)
                im.draft("RGB", (PREVIEW_MAX, PREVIEW_MAX))
                im = im.convert("RGB")
                im.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
                self.preview_imgref = ImageTk.PhotoImage(im)
                self.preview_lbl.configure(image=self.preview_imgref)
            except (OSError, ValueError):
                self.preview_lbl.configure(image="")

            lines = [
                f"{entry.path.name}",
                f"arch : {entry.arch}",
                f"time : {datetime.fromtimestamp(entry.mtime):%Y-%m-%d %H:%M:%S}",
                f"size : {entry.size_str}",
                f"model: {entry.model}",
            ]
            if entry.loras:
                lines.append(f"loras: {entry.loras}")
            if entry.lora_keywords:
                lines.append(f"lora kw: {entry.lora_keywords}")
            if entry.pipeline:
                lines.append(f"pipeline: {entry.pipeline}")
            lines.append(f"seed/steps/sampler: {entry.seed} / {entry.steps} / {entry.sampler}")
            # その他 params (既出を除く)
            shown_keys = {"Size", "Model", "Loras", "Lora keywords", "Pipeline",
                          "Seed", "Steps", "Sampler"}
            for k, v in entry.params.items():
                if k not in shown_keys:
                    lines.append(f"{k}: {v}")
            lines.append("")
            lines.append(L("[positive]", "[positive]"))
            lines.append(entry.positive)
            lines.append("")
            lines.append(L("[negative]", "[negative]"))
            lines.append(entry.negative)

            self.detail.configure(state="normal")
            self.detail.delete("1.0", "end")
            self.detail.insert("1.0", "\n".join(lines))
            self.detail.configure(state="disabled")

        # ---- ファイル操作 ----------------------------------------------- #
        def _open_selected(self) -> None:
            if not self.selected:
                return
            try:
                os.startfile(str(self.selected.path))   # Windows 既定ビューア
            except (OSError, AttributeError) as ex:
                messagebox.showerror(L("エラー", "Error"), str(ex))

        def _reload(self) -> None:
            for e in self.all_entries:
                if e.widget:
                    e.widget.destroy()
            self.all_entries = []
            self.gridded = []
            self.selected = None
            self.loading = True
            self.status_var.set(L("読み込み中…", "Loading…"))
            self.q = queue.Queue()
            self._refresh_view()   # 現ビューを即クリア (アイコン/リスト両対応)
            threading.Thread(target=self._scan_worker, daemon=True).start()

    app = GalleryApp()
    app.mainloop()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=L("生成済み PNG をサムネイル一覧するビューア (Tkinter、閲覧専用)",
                      "Thumbnail gallery viewer for generated PNGs (Tkinter, read-only)"))
    ap.add_argument("--list", action="store_true",
                    help=L("GUI を出さずメタ一覧を標準出力", "print metadata list without GUI"))
    ap.add_argument("--dir", nargs="+", type=Path, metavar="PATH",
                    help=L("走査対象ディレクトリを指定 (複数可、既定の固定 3 dir を差し替え)",
                           "directories to scan (multiple ok, overrides the default 3 fixed dirs)"))
    args = ap.parse_args()

    # --dir 指定時はユーザ指定 dir に差し替え。未指定は VIEW_DIRS (固定 3 dir)
    if args.dir:
        directories = [p.resolve() for p in args.dir]
        missing = [p for p in directories if not p.is_dir()]
        if missing:
            for p in missing:
                print(L(f"  [warn] ディレクトリが見つかりません: {p}",
                        f"  [warn] directory not found: {p}"), file=sys.stderr)
    else:
        directories = VIEW_DIRS

    if args.list:
        run_list(directories)
    else:
        run_gui(directories)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(L("\n中断", "\nInterrupted"))
        sys.exit(0)
