#!/usr/bin/env python3
"""generate_gui.py - 画像生成 GUI (Tkinter, ComfyUI HTTP API 経由)。

generate.py の CLI ループに対する、手動指定・即時可視化版。
- チェックポイント / LoRA / プロンプトを手で選び、1〜300 枚をまとめて生成。
- *word* / **word** / ***word*** の重み記法は normalize_emphasis でそのまま使える。
- 生成画像は版で分かれた dir (SDXL=4_8_SDXL_generated / SD15=3_8_SD15_generated) に
  PNG + A1111 メタ付きで保存し、画面下段の
  サムネイル列に追記。サムネクリックでモーダルフルサイズ表示。

generate.py の build_workflow_txt2img / _submit_and_fetch / save_with_a1111_metadata
/ ensure_comfyui_arch をそのまま import して使う (重複実装しない)。
"""
from __future__ import annotations

import queue
import random
import sys
import threading
import tkinter as tk
import tomllib
import uuid
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import tomli_w

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

from common import (
    _parse_keyword_clauses,
    _text_matches_clauses,
    build_lora_corpus,
    build_prompt,
    load_lora_params,
    load_prompt_config,
    normalize_emphasis,
    pick_n_loras_by_keywords,
)
from pngutil import read_text_chunks, write_text_chunks

# tkinterdnd2 (D&D サポート、無くても起動はする)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    TkinterDnD = None
    DND_FILES = None
    _DND_AVAILABLE = False
from generate import (
    SD15_CHECKPOINT_DIR,
    SD15_EMBED_DIR,
    SD15_GENERATED_DIR,
    SD15_LORA_DIR,
    SD15_UPSCALED_DIR,
    SDXL_CHECKPOINT_DIR,
    SDXL_CONTROLNET_DIR,
    SDXL_EMBED_DIR,
    SDXL_GENERATED_DIR,
    SDXL_LORA_DIR,
    SDXL_UPSCALED_DIR,
    _submit_and_fetch,
    build_workflow_txt2img,
    collect_negative_embeddings,
    ensure_adetailer_model,
    ensure_comfyui_arch,
    ensure_sd15_vae_mse,
    ensure_sdxl_vae_fix,
    ensure_upscale_model,
    infer_controlnet_mode,
    load_checkpoint_toml,
    load_sdxl_lora_subjects,
    prepare_workflow_prompt,
    save_with_a1111_metadata,
    upload_image_to_comfyui,
    write_extra_model_paths,
)

UPSCALE_MODELS = {
    "anime": "RealESRGAN_x4plus_anime_6B.pth",  # 既定: アニメ/イラスト系
    "real":  "RealESRGAN_x4plus.pth",            # 実写系
}
DEFAULT_UPSCALE_STYLE = "anime"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

THUMB_SIZE = (96, 96)
CHECKPOINT_THUMB_SIZE = (160, 160)
GALLERY_THUMB_SIZE = (160, 160)
GUI_SETTINGS_TOML = Path(__file__).parent / "generate_gui.toml"

VERSION_BY_DIR = {
    SD15_CHECKPOINT_DIR: "sd15",
    SDXL_CHECKPOINT_DIR: "sdxl",
}
DEFAULT_RES = {"sd15": 768, "sdxl": 1024}
# 「2 人以上 (ワイド)」ON 時の横長キャンバス (W, H)。SDXL は generate.py の many 既定と揃える
WIDE_RES = {"sd15": (912, 624), "sdxl": (1216, 832)}


def _preview_path(safetensors_path: Path) -> Optional[Path]:
    p = safetensors_path.with_suffix(".preview.png")
    return p if p.exists() else None


def _list_safetensors(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(
        [p for p in d.iterdir() if p.suffix == ".safetensors"],
        key=lambda p: p.stem.lower(),
    )


class GenerateGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("generate GUI - ComfyUI")
        # 左=設定 / 右=ギャラリーの水平 split (1:1)
        root.geometry("1440x1000")

        self._stop_flag = False
        self._worker_thread: Optional[threading.Thread] = None
        self._result_queue: queue.Queue = queue.Queue()
        self._client_id = uuid.uuid4().hex
        # GUI セッション内で ComfyUI を 1 度は確実に再起動して YAML を読ませる
        # (前のセッションのサーバが残っていると extra_model_paths を未ロードのままになる)
        self._server_verified = False

        # Tkinter PhotoImage は強参照が消えると即破棄されるので保持しておく
        self._photo_keep: list = []
        self._gallery_thumbs: list = []
        # ギャラリーに追加された画像パスを順序保持 (モーダルの ←/→ ナビ用)
        self._gallery_paths: list[Path] = []
        self._lora_icon_widgets: list[tk.Widget] = []
        # モーダルは同時 1 枚 (新規表示時に既存を destroy)
        self._modal_win: Optional[tk.Toplevel] = None

        self.checkpoints: list[Path] = []
        self.loras_sd15: list[Path] = []
        self.loras_sdxl: list[Path] = []
        self.controlnets: list[Path] = []
        self.current_loras: list[Path] = []
        self._lora_params_cache: Optional[dict] = None  # LoRA_param.toml の遅延キャッシュ (kw ハイライト用)
        self.selected_lora_indices: set[int] = set()
        self.selected_controlnet_index: int = -1

        # リファレンス画像 (D&D 入力)
        self.reference_image_path: Optional[Path] = None
        self._reference_thumb_keep: Optional[ImageTk.PhotoImage] = None


        self._gallery_col = 0
        self._gallery_row = 0
        # 右ペイン (ギャラリー) は左ペイン (設定) より狭くなる → 列数を控えめに
        self._gallery_max_cols = 3

        self._build_ui()
        self._scan_assets()
        # _build_ui の時点で self.controlnets はまだ空なので、scan 後に listbox を埋める
        self._populate_cn_listbox()
        self._populate_checkpoint_combo()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_app_close)
        self.root.after(150, self._poll_queue)

    # ---------- UI 構築 ---------- #
    def _build_ui(self) -> None:
        # 左=設定 / 右=作成画像ギャラリーの水平 split
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(paned, padding=8)
        paned.add(top, weight=1)

        row = 0
        ttk.Label(top, text="チェックポイント:").grid(row=row, column=0, sticky="ne", padx=4, pady=4)
        ckpt_frame = ttk.Frame(top)
        ckpt_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        # 横方向: column 0 = combobox (伸びる) / column 1 = ランダム chk / column 2 = thumb (固定サイズ)
        ckpt_frame.grid_columnconfigure(0, weight=1)
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(
            ckpt_frame, textvariable=self.checkpoint_var, state="readonly",
        )
        self.checkpoint_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.checkpoint_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_change)

        # ランダムチェック: ON で combobox を無効化、生成時に毎枚 random.choice
        self.checkpoint_random_var = tk.BooleanVar(value=False)
        self.checkpoint_random_check = ttk.Checkbutton(
            ckpt_frame, text="ランダム", variable=self.checkpoint_random_var,
            command=self._on_checkpoint_random_toggle,
        )
        self.checkpoint_random_check.grid(row=0, column=1, sticky="w", padx=(0, 8))

        # 版フィルタ (SDXL / SD15 / 混合 のラジオ)。combobox の表示を絞り込み、ランダム時もこの版から抽選
        # 混合 = 両 dir を統合してリストする (ランダム時は両版から平等抽選、ckpt_random=true 用)
        arch_frame = ttk.Frame(ckpt_frame)
        arch_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.arch_filter_var = tk.StringVar(value="sdxl")
        ttk.Radiobutton(
            arch_frame, text="SDXL", value="sdxl", variable=self.arch_filter_var,
            command=self._on_arch_filter_change,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            arch_frame, text="SD15", value="sd15", variable=self.arch_filter_var,
            command=self._on_arch_filter_change,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            arch_frame, text="混合", value="mix", variable=self.arch_filter_var,
            command=self._on_arch_filter_change,
        ).pack(side=tk.LEFT)

        # thumb は Frame で固定サイズ枠を作り、その中の Label に画像を入れる
        thumb_box = tk.Frame(
            ckpt_frame,
            width=CHECKPOINT_THUMB_SIZE[0], height=CHECKPOINT_THUMB_SIZE[1],
            bg="#eeeeee",
        )
        thumb_box.grid(row=0, column=2, rowspan=2, sticky="ne")
        thumb_box.grid_propagate(False)  # 中の Label のサイズに引っ張られない
        self.checkpoint_thumb_label = tk.Label(
            thumb_box, text="(no thumb)", bg="#eeeeee", cursor="hand2",
        )
        self.checkpoint_thumb_label.place(relx=0.5, rely=0.5, anchor="center")
        self._checkpoint_thumb_path: Optional[Path] = None
        self.checkpoint_thumb_label.bind(
            "<Button-1>",
            lambda e: self._show_modal(self._checkpoint_thumb_path) if self._checkpoint_thumb_path else None,
        )

        row += 1
        ttk.Label(top, text="LoRA キーワード:").grid(
            row=row, column=0, sticky="ne", padx=4, pady=4,
        )
        lora_kw_frame = ttk.Frame(top)
        lora_kw_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        lora_kw_frame.grid_columnconfigure(0, weight=1)
        # キーワードを入れると手動選択が空のとき pick_n_loras_by_keywords で実行時抽選される
        # (手動選択がある場合は通常通り manual 優先、kw は無視 → 動作が混乱しない)
        self.lora_kw_var = tk.StringVar()
        self.lora_kw_entry = ttk.Entry(lora_kw_frame, textvariable=self.lora_kw_var)
        self.lora_kw_entry.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            lora_kw_frame,
            text="(カンマ区切り、手動選択が空のとき毎枚抽選。候補は LoRA 一覧で赤背景表示)",
            foreground="#888",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        # キーワード変更 → LoRA listbox の候補ハイライトを更新
        self.lora_kw_var.trace_add("write", lambda *_: self._refresh_lora_candidate_highlight())

        row += 1
        ttk.Label(top, text="LoRA (Ctrl/Shiftクリックで複数選択):").grid(
            row=row, column=0, sticky="ne", padx=4, pady=4,
        )
        lora_frame = ttk.Frame(top)
        lora_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        self.lora_listbox = tk.Listbox(
            lora_frame, selectmode=tk.EXTENDED, height=7, width=70,
            exportselection=False,
        )
        self.lora_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        lora_scroll = ttk.Scrollbar(lora_frame, orient=tk.VERTICAL,
                                     command=self.lora_listbox.yview)
        lora_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.lora_listbox.config(yscrollcommand=lora_scroll.set)
        self.lora_listbox.bind("<<ListboxSelect>>", self._on_lora_select)

        row += 1
        self.lora_icon_canvas = tk.Canvas(
            top, height=THUMB_SIZE[1] + 28, bg="#f0f0f0", highlightthickness=0,
        )
        self.lora_icon_canvas.grid(row=row, column=1, sticky="ew", padx=4, pady=2)

        # ControlNet (SDXL のみ) — Embedding は CLI と同じ自動投入経路 (helper) に集約済み
        row += 1
        ttk.Label(top, text="ControlNet:").grid(
            row=row, column=0, sticky="ne", padx=4, pady=4)
        cn_emb_frame = ttk.Frame(top)
        cn_emb_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        cn_emb_frame.grid_columnconfigure(0, weight=1)

        ttk.Label(cn_emb_frame, text="ControlNet (SDXL のみ、単一選択):",
                  foreground="#666").grid(row=0, column=0, sticky="w")

        cn_box = ttk.Frame(cn_emb_frame)
        cn_box.grid(row=1, column=0, sticky="ew")
        self.cn_listbox = tk.Listbox(
            cn_box, selectmode=tk.SINGLE, height=4, exportselection=False,
        )
        self.cn_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        cn_scroll = ttk.Scrollbar(cn_box, orient=tk.VERTICAL, command=self.cn_listbox.yview)
        cn_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.cn_listbox.config(yscrollcommand=cn_scroll.set)
        self.cn_listbox.bind("<<ListboxSelect>>", self._on_cn_select)

        # 実際の populate は self.controlnets が埋まる _scan_assets 後に
        # __init__ → _populate_cn_listbox() で実行する

        # リファレンス画像 (D&D で受け取り、ControlNet preprocess に流す)
        row += 1
        ttk.Label(top, text="リファレンス画像:").grid(
            row=row, column=0, sticky="ne", padx=4, pady=4,
        )
        ref_frame = ttk.Frame(top)
        ref_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        ref_frame.grid_columnconfigure(1, weight=1)

        # 左: 固定サイズ枠 (ドロップ先 + サムネ表示)
        REF_THUMB_W, REF_THUMB_H = 180, 180
        ref_drop_box = tk.Frame(
            ref_frame, width=REF_THUMB_W, height=REF_THUMB_H,
            bg="#f5f5f5", relief="ridge", bd=2,
        )
        ref_drop_box.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 8))
        ref_drop_box.grid_propagate(False)
        self.reference_drop_label = tk.Label(
            ref_drop_box,
            text="ここに画像を\nドラッグ&ドロップ",
            bg="#f5f5f5", fg="#888", cursor="hand2",
            wraplength=REF_THUMB_W - 16, justify="center",
        )
        self.reference_drop_label.place(relx=0.5, rely=0.5, anchor="center")
        self.reference_drop_label.bind("<Button-1>", lambda e: self._on_ref_click())

        # D&D ハンドラ登録 (tkinterdnd2 が無いと no-op)
        if _DND_AVAILABLE:
            for w in (ref_drop_box, self.reference_drop_label):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self._on_ref_drop)

        # 右: mode (preprocessor) / strength / クリアボタン
        ttk.Label(ref_frame, text="前処理:").grid(row=0, column=1, sticky="w", padx=(0, 4))
        self.ref_mode_var = tk.StringVar(value="openpose")
        ref_mode_combo = ttk.Combobox(
            ref_frame, textvariable=self.ref_mode_var, state="readonly", width=14,
            values=["openpose", "depth", "canny", "softedge"],
        )
        ref_mode_combo.grid(row=0, column=2, sticky="w")

        ttk.Label(ref_frame, text="強度:").grid(row=1, column=1, sticky="w", padx=(0, 4))
        self.ref_strength_var = tk.DoubleVar(value=0.7)
        ttk.Spinbox(
            ref_frame, from_=0.1, to=1.5, increment=0.05,
            textvariable=self.ref_strength_var, width=6,
        ).grid(row=1, column=2, sticky="w")

        ttk.Button(ref_frame, text="クリア", command=self._on_ref_clear).grid(
            row=2, column=1, columnspan=2, sticky="w", pady=(4, 0),
        )

        ttk.Label(
            ref_frame,
            text="(OpenPose=ポーズのみ / Depth=視点維持 / Canny/SoftEdge=輪郭維持)",
            foreground="#888", wraplength=320,
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(6, 0))

        # プロンプト入力モード: 自由記載 (テキスト欄を使う) / prompt.toml (毎枚 build_prompt 自動)
        row += 1
        ttk.Label(top, text="プロンプト入力:").grid(row=row, column=0, sticky="ne", padx=4, pady=4)
        prompt_mode_frame = ttk.Frame(top)
        prompt_mode_frame.grid(row=row, column=1, sticky="w", padx=4, pady=(4, 0))
        self.prompt_mode_var = tk.StringVar(value="free")
        ttk.Radiobutton(
            prompt_mode_frame, text="自由記載", value="free", variable=self.prompt_mode_var,
            command=self._on_prompt_mode_change,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            prompt_mode_frame, text="prompt.toml (毎枚自動)", value="toml", variable=self.prompt_mode_var,
            command=self._on_prompt_mode_change,
        ).pack(side=tk.LEFT)

        row += 1
        ttk.Label(top, text="プロンプト:").grid(row=row, column=0, sticky="ne", padx=4, pady=4)
        self.prompt_text = tk.Text(top, height=4, width=80, wrap=tk.WORD)
        self.prompt_text.grid(row=row, column=1, sticky="ew", padx=4, pady=4)

        # ポジティブ / ネガティブ は「設定」ダイアログに格納 (文字列バッファとして保持)
        self.positive_value = "natural color"
        self.negative_value = "low quality, worst quality, bad anatomy, text, watermark"

        row += 1
        ctrl = ttk.Frame(top)
        ctrl.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=8)

        # 左端: 枚数 / 推論数 / 設定ボタン
        # 枚数: 0 = 無限 (停止まで)、>0 = その枚数で停止する batch
        # 推論数: 1 枚あたりの denoising step 数 (= steps)。設定ダイアログの Steps と同じ Var
        ttk.Label(ctrl, text="枚数:").pack(side=tk.LEFT, padx=(4, 2))
        self.count_var = tk.IntVar(value=0)
        ttk.Spinbox(ctrl, from_=0, to=300, textvariable=self.count_var, width=5).pack(side=tk.LEFT)
        ttk.Label(ctrl, text="(0=停止まで無限)", foreground="#888").pack(side=tk.LEFT, padx=(2, 8))

        # 推論数 (= steps、設定ダイアログとも共有)
        self.steps_var = tk.IntVar(value=30)
        ttk.Label(ctrl, text="推論数:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(ctrl, from_=1, to=200, textvariable=self.steps_var, width=5).pack(side=tk.LEFT)

        # 2 人以上 (ワイド): ON で横長キャンバス (SDXL=1216x832 / SD15=912x624) に切替え、
        # 複数人物の融合を抑える。OFF は設定ダイアログの幅/高さをそのまま使う
        self.many_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl, text="2人以上 (ワイド)", variable=self.many_var,
        ).pack(side=tk.LEFT, padx=(16, 2))

        # 「設定」ダイアログ: ポジ/ネガ/CFG/AD補正/Hires Fix/幅/高さ をまとめる
        # 関連 Var は __init__ で生成しておき、ダイアログ widget が textvariable で直接束縛する
        self.cfg_var = tk.DoubleVar(value=7.0)
        self.adetailer_var = tk.BooleanVar(value=True)
        # person 系 ADetailer は crop_factor=3.0 で 2 人を 1 bbox にまとめ → 茶色フィルム化
        # する不具合あり (2026-06-05)。既定 OFF とし、必要時のみ ON。
        self.adetailer_person_var = tk.BooleanVar(value=True)
        self.hires_var = tk.BooleanVar(value=True)
        # 各 gen 画像を生成直後に Real-ESRGAN x4 で自動 upscale (既定 ON)
        self.auto_upscale_var = tk.BooleanVar(value=True)
        self.auto_upscale_style_var = tk.StringVar(value=DEFAULT_UPSCALE_STYLE)
        ttk.Button(ctrl, text="設定…", command=self._open_settings).pack(side=tk.LEFT, padx=(16, 2))

        # 右端: 画像生成 / 停止 (pack 順は右から積まれるので 停止→生成 の順で見た目は 生成 停止)
        self.stop_btn = ttk.Button(ctrl, text="停止", command=self._on_stop_clicked,
                                    state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=4)
        self.generate_btn = ttk.Button(ctrl, text="画像生成", command=self._on_generate_clicked)
        self.generate_btn.pack(side=tk.RIGHT, padx=4)

        # 詳細パラメータも「設定」ダイアログに移動 (Var だけ生成しておく)
        # steps_var は上の ctrl row で先行宣言済 (推論数 Spinbox と共有)
        self.width_var = tk.IntVar(value=DEFAULT_RES["sd15"])
        self.height_var = tk.IntVar(value=DEFAULT_RES["sd15"])
        self.seed_var = tk.StringVar(value="-1")
        self.sampler_var = tk.StringVar(value="dpmpp_2m")
        self.scheduler_var = tk.StringVar(value="karras")
        self.hires_scale_var = tk.DoubleVar(value=1.5)
        self.hires_denoise_var = tk.DoubleVar(value=0.35)
        self.hires_steps_var = tk.IntVar(value=20)
        self.lora_total_var = tk.DoubleVar(value=0.8)

        row += 1
        self.status_var = tk.StringVar(value="待機中")
        ttk.Label(top, textvariable=self.status_var, foreground="#444").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2,
        )

        top.grid_columnconfigure(1, weight=1)

        # 右ペイン: 作成画像ギャラリー (左:右 = 1:1)
        bottom = ttk.Frame(paned, padding=8)
        paned.add(bottom, weight=1)

        ttk.Label(bottom, text="作成画像 (クリックでフルサイズ表示):").pack(anchor="w")
        gallery_outer = ttk.Frame(bottom)
        gallery_outer.pack(fill=tk.BOTH, expand=True)

        self.gallery_canvas = tk.Canvas(gallery_outer, bg="#ffffff", highlightthickness=0)
        gv_scroll = ttk.Scrollbar(gallery_outer, orient=tk.VERTICAL,
                                   command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=gv_scroll.set)
        gv_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_inner = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")
        self.gallery_inner.bind(
            "<Configure>",
            lambda e: self.gallery_canvas.configure(
                scrollregion=self.gallery_canvas.bbox("all")
            ),
        )

        # マウスホイールでギャラリーを縦スクロール (Windows: <MouseWheel> delta±120/notch)
        self.gallery_canvas.bind("<MouseWheel>", self._on_gallery_wheel)
        self.gallery_inner.bind("<MouseWheel>", self._on_gallery_wheel)

    # ---------- アセット読み込み ---------- #
    def _scan_assets(self) -> None:
        self.checkpoints = (
            _list_safetensors(SD15_CHECKPOINT_DIR)
            + _list_safetensors(SDXL_CHECKPOINT_DIR)
        )
        self.loras_sd15 = _list_safetensors(SD15_LORA_DIR)
        self.loras_sdxl = _list_safetensors(SDXL_LORA_DIR)
        self.controlnets = _list_safetensors(SDXL_CONTROLNET_DIR)

    def _filtered_checkpoints(self) -> list[Path]:
        """版フィルタ (`self.arch_filter_var`) を適用した checkpoint リスト。
        mix の場合は両 dir をそのまま返す (combobox の [SD15]/[SDXL] tag で区別可能)。"""
        arch = getattr(self, "arch_filter_var", None)
        version = arch.get() if arch else "sdxl"
        if version == "mix":
            return list(self.checkpoints)
        target_dir = SDXL_CHECKPOINT_DIR if version == "sdxl" else SD15_CHECKPOINT_DIR
        return [p for p in self.checkpoints if p.parent == target_dir]

    def _populate_cn_listbox(self) -> None:
        """ControlNet listbox を self.controlnets (= _scan_assets で埋めた SDXL_CONTROLNET_DIR の中身)
        で 1 回だけ埋める。checkpoint 切替では呼ばない (CN は SDXL のときだけ意味があり、
        SD15 切替時は _on_checkpoint_change で選択クリアするのみ)。"""
        if not hasattr(self, "cn_listbox"):
            return
        self.cn_listbox.delete(0, tk.END)
        for p in self.controlnets:
            self.cn_listbox.insert(tk.END, p.stem)

    def _populate_checkpoint_combo(self) -> None:
        ckpts = self._filtered_checkpoints()
        self._visible_checkpoints = ckpts  # current index → Path 解決用
        labels = []
        for p in ckpts:
            tag = "[SD15]" if p.parent == SD15_CHECKPOINT_DIR else "[SDXL]"
            labels.append(f"{tag} {p.stem}")
        self.checkpoint_combo["values"] = labels
        if labels:
            self.checkpoint_combo.current(0)
            self._on_checkpoint_change()
        else:
            self.checkpoint_combo.set("")

    def _current_checkpoint(self) -> Optional[Path]:
        idx = self.checkpoint_combo.current()
        visible = getattr(self, "_visible_checkpoints", self.checkpoints)
        if idx < 0 or idx >= len(visible):
            return None
        return visible[idx]

    def _current_version(self) -> str:
        ckpt = self._current_checkpoint()
        if ckpt is None:
            return "sd15"
        return VERSION_BY_DIR.get(ckpt.parent, "sd15")

    # ---------- イベント ---------- #
    def _on_prompt_mode_change(self) -> None:
        """prompt.toml モードでは prompt_text を読み取り専用にする (毎枚 build_prompt 自動)。"""
        mode = self.prompt_mode_var.get()
        if mode == "toml":
            self.prompt_text.configure(state="disabled", bg="#eeeeee")
        else:
            self.prompt_text.configure(state="normal", bg="white")

    def _on_checkpoint_random_toggle(self) -> None:
        """「ランダム」ON で combobox を無効化 (表示は最後の選択を保持、生成時に毎枚 random.choice)"""
        if self.checkpoint_random_var.get():
            self.checkpoint_combo.configure(state="disabled")
        else:
            self.checkpoint_combo.configure(state="readonly")

    def _on_arch_filter_change(self) -> None:
        """SD15/SDXL ラジオ切替: combobox を版で絞り直し、先頭を選び直す。
        ランダム時はこの版の checkpoint 群からのみ random.choice される。"""
        self._populate_checkpoint_combo()

    def _on_checkpoint_change(self, event=None) -> None:
        ckpt = self._current_checkpoint()
        if ckpt is None:
            return
        prev = _preview_path(ckpt)
        self._checkpoint_thumb_path = prev  # クリック時のモーダル表示用
        if prev and Image is not None:
            try:
                img = Image.open(prev)
                img.thumbnail(CHECKPOINT_THUMB_SIZE)
                photo = ImageTk.PhotoImage(img)
                self.checkpoint_thumb_label.configure(image=photo, text="")
                self._photo_keep.append(photo)
            except Exception:
                self.checkpoint_thumb_label.configure(image="", text="(thumb err)")
        else:
            self.checkpoint_thumb_label.configure(image="", text="(no thumb)")

        version = self._current_version()
        self.current_loras = self.loras_sd15 if version == "sd15" else self.loras_sdxl
        self.lora_listbox.delete(0, tk.END)
        for p in self.current_loras:
            self.lora_listbox.insert(tk.END, p.stem)
        self.selected_lora_indices.clear()
        self._refresh_lora_icons()
        # 版切替 → LoRA リストが入れ替わったので候補ハイライトを再計算
        self._refresh_lora_candidate_highlight()

        # Embedding は worker 側で helper が CLI と同じ自動投入 (UI 不要)
        # SD15 を選んだら ControlNet は使えないので選択解除
        if version == "sd15" and hasattr(self, "cn_listbox"):
            self.cn_listbox.selection_clear(0, tk.END)
            self.selected_controlnet_index = -1

        # 解像度を版に合わせて自動セット (ユーザが弄った後は次の checkpoint 切替で上書きされる点に注意)
        if hasattr(self, "width_var"):
            self.width_var.set(DEFAULT_RES[version])
            self.height_var.set(DEFAULT_RES[version])

    def _on_lora_select(self, event=None) -> None:
        self.selected_lora_indices = set(self.lora_listbox.curselection())
        self._refresh_lora_icons()

    def _refresh_lora_candidate_highlight(self) -> None:
        """LoRA キーワード変更/版切替時に呼び出す。pick_lora_by_keywords でマッチしうる
        LoRA を listbox で赤背景 (#ffcccc) に染め、それ以外は白に戻す。"""
        if not hasattr(self, "lora_listbox"):
            return
        raw = self.lora_kw_var.get() or ""
        keywords = [k.strip() for k in raw.split(",") if k.strip()]
        n = self.lora_listbox.size()
        if not keywords:
            for i in range(n):
                try:
                    self.lora_listbox.itemconfig(i, background="white")
                except Exception:
                    pass
            return
        try:
            if self._lora_params_cache is None:
                self._lora_params_cache = load_lora_params()
        except Exception:
            self._lora_params_cache = {}
        corpus = build_lora_corpus(self.current_loras, self._lora_params_cache or {})
        clauses = _parse_keyword_clauses(keywords)
        for i, lora in enumerate(self.current_loras):
            if i >= n:
                break
            text = corpus.get(lora.stem) or lora.stem.lower()
            bg = "#ffcccc" if _text_matches_clauses(text, clauses) else "white"
            try:
                self.lora_listbox.itemconfig(i, background=bg)
            except Exception:
                pass

    def _on_cn_select(self, event=None) -> None:
        sel = self.cn_listbox.curselection()
        self.selected_controlnet_index = sel[0] if sel else -1

    # ---------- リファレンス画像 D&D ---------- #
    def _on_ref_drop(self, event) -> None:
        """tkinterdnd2 の <<Drop>> イベント。event.data は brace-quoted 可能性ありの空白区切り。"""
        raw = (event.data or "").strip()
        if not raw:
            return
        # {C:\path with spaces.png} C:\other.png のような形式を分解 → 先頭 1 件のみ採用
        path_str = self._parse_dnd_paths(raw)
        if not path_str:
            return
        self._set_reference_image(Path(path_str))

    def _parse_dnd_paths(self, raw: str) -> Optional[str]:
        """DND_FILES の data string から最初のパスを取り出す。"""
        s = raw.strip()
        if s.startswith("{"):
            end = s.find("}")
            if end > 0:
                return s[1:end]
        # 空白で分かれた素のパス (スペース無しの場合)
        return s.split()[0] if s else None

    def _on_ref_click(self) -> None:
        """ドロップ枠をクリックでフォーカス (将来的にファイルダイアログを開く余地)。"""
        if self.reference_image_path is not None:
            # 設定済みなら原寸モーダルを開く
            self._show_modal(self.reference_image_path)

    def _on_ref_clear(self) -> None:
        self.reference_image_path = None
        self._reference_thumb_keep = None
        self.reference_drop_label.configure(
            image="", text="ここに画像を\nドラッグ&ドロップ", fg="#888",
        )

    def _auto_pick_controlnet(self, mode: str) -> Optional[Path]:
        """mode (openpose/depth/canny/softedge) に対応する ControlNet を 4_3_SDXL_ControlNet から検索。
        手動選択が無く + ref 画像があるときの自動選択用。"""
        mode_keys = {
            "openpose": ("openpose", "_pose", "pose"),
            "depth":    ("depth", "midas"),
            "canny":    ("canny",),
            "softedge": ("softedge", "soft_edge", "hed"),
        }
        keys = mode_keys.get(mode.lower(), ())
        if not keys:
            return None
        for cn in self.controlnets:
            s = cn.stem.lower()
            if any(k in s for k in keys):
                return cn
        return None

    def _set_reference_image(self, path: Path) -> None:
        """ドロップされた画像をサムネ表示しパスを保持。"""
        if not path.is_file():
            messagebox.showerror("エラー", f"ファイルが見つかりません: {path}")
            return
        if Image is None:
            messagebox.showerror("エラー", "Pillow が無いため画像を表示できません")
            return
        try:
            img = Image.open(path)
            img.thumbnail((172, 172))
            photo = ImageTk.PhotoImage(img)
        except Exception as e:
            messagebox.showerror("エラー", f"画像を開けません: {e}")
            return
        self.reference_image_path = path
        self._reference_thumb_keep = photo
        self.reference_drop_label.configure(image=photo, text="", fg="#000")

    def _refresh_lora_icons(self) -> None:
        for w in self._lora_icon_widgets:
            w.destroy()
        self._lora_icon_widgets.clear()

        x = 8
        for i in sorted(self.selected_lora_indices):
            lora = self.current_loras[i]
            frame = tk.Frame(self.lora_icon_canvas, bg="#f0f0f0")
            prev = _preview_path(lora)
            if prev and Image is not None:
                try:
                    img = Image.open(prev)
                    img.thumbnail(THUMB_SIZE)
                    photo = ImageTk.PhotoImage(img)
                    lbl = tk.Label(frame, image=photo, bg="#f0f0f0",
                                    cursor="hand2" if prev else "")
                    lbl.image = photo
                except Exception:
                    lbl = tk.Label(frame, text="(thumb)", width=12, height=6, bg="#dddddd")
            else:
                lbl = tk.Label(frame, text="(no thumb)", width=12, height=6, bg="#dddddd")
            if prev:
                lbl.bind("<Button-1>", lambda e, p=prev: self._show_modal(p))
            lbl.pack()
            name_lbl = tk.Label(
                frame, text=lora.stem[:18], bg="#f0f0f0",
                font=("TkDefaultFont", 8), width=14,
            )
            name_lbl.pack()
            self.lora_icon_canvas.create_window(x, 4, anchor="nw", window=frame)
            self._lora_icon_widgets.append(frame)
            x += THUMB_SIZE[0] + 16
        self.lora_icon_canvas.configure(scrollregion=(0, 0, max(x, 100), THUMB_SIZE[1] + 28))

    # ---------- ジョブ排他 ヘルパ ---------- #
    def _gen_alive(self) -> bool:
        return bool(self._worker_thread and self._worker_thread.is_alive())

    # ---------- 生成 ---------- #
    def _on_generate_clicked(self) -> None:
        if self._gen_alive():
            return
        ckpt_random = bool(self.checkpoint_random_var.get())
        ckpt = self._current_checkpoint()
        random_pool = self._filtered_checkpoints()
        if not ckpt_random and ckpt is None:
            messagebox.showerror("エラー", "チェックポイントが選択されていません")
            return
        if ckpt_random and not random_pool:
            arch_label = self.arch_filter_var.get().upper() if self.arch_filter_var.get() != "mix" else "混合 (SD15+SDXL)"
            messagebox.showerror(
                "エラー",
                f"選択中の版 ({arch_label}) に "
                "チェックポイントが 1 つも見つかりません",
            )
            return
        prompt_mode = self.prompt_mode_var.get()  # "free" / "toml"
        positive_extra = self.positive_value.strip()
        negative = self.negative_value.strip()
        if prompt_mode == "toml":
            # prompt.toml モード: worker 側で毎枚 build_prompt するので、UI 入力は空でも OK
            # positive_extra は free モード時のみ前置 (toml モードはエントリそのものをそのまま使う)
            positive = ""
        else:
            prompt_body = self.prompt_text.get("1.0", "end").strip()
            if not prompt_body and not positive_extra:
                messagebox.showerror("エラー", "プロンプトかポジティブを入力してください")
                return
            positive = ", ".join(p for p in (prompt_body, positive_extra) if p)
            positive = normalize_emphasis(positive)
        negative = normalize_emphasis(negative)

        version = self._current_version()
        loras = [self.current_loras[i] for i in sorted(self.selected_lora_indices)]
        lora_total = float(self.lora_total_var.get())
        scale = lora_total / max(1, len(loras))
        # Path で運ぶ (helper の family-gate / pose-gate で stem を参照するため)
        loras_with_strength = [(p, scale) for p in loras]
        # negative embedding は worker 側で helper が CLI と同じ neg-gate + append を行う

        # 選択された ControlNet (SDXL のみ)。リファレンス画像が D&D されている場合のみ実配線される
        # 手動選択優先 / 無ければ ref_mode に対応する ControlNet を auto pick
        cn_path: Optional[Path] = None
        if version == "sdxl":
            if 0 <= self.selected_controlnet_index < len(self.controlnets):
                cn_path = self.controlnets[self.selected_controlnet_index]
            elif self.reference_image_path is not None:
                cn_path = self._auto_pick_controlnet(self.ref_mode_var.get())

        ref_mode = str(self.ref_mode_var.get() or "openpose")
        ref_strength = float(self.ref_strength_var.get() or 0.7)
        ref_path = self.reference_image_path

        try:
            seed_input = int(self.seed_var.get().strip())
        except ValueError:
            seed_input = -1

        # LoRA キーワード (カンマ区切り→list[str])。手動選択が空 + ckpt 確定 のとき毎枚抽選に使う
        lora_kw_raw = self.lora_kw_var.get() or ""
        lora_keywords = [k.strip() for k in lora_kw_raw.split(",") if k.strip()]

        params = {
            "checkpoint": ckpt,
            "checkpoint_random": ckpt_random,
            "checkpoint_random_pool": random_pool,
            "version": version,
            "prompt_mode": prompt_mode,
            "positive": positive,
            "positive_extra": positive_extra,  # toml モードで追加前置するときに使う
            "negative": negative,
            "loras": loras_with_strength,
            "lora_keywords": lora_keywords,
            "count": int(self.count_var.get()),
            "cfg": float(self.cfg_var.get()),
            "adetailer": bool(self.adetailer_var.get()),
            "adetailer_person": bool(self.adetailer_person_var.get()),
            "hires_fix": bool(self.hires_var.get()),
            "width": int(self.width_var.get()),
            "height": int(self.height_var.get()),
            "many": bool(self.many_var.get()),
            "steps": int(self.steps_var.get()),
            "sampler": self.sampler_var.get(),
            "scheduler": self.scheduler_var.get(),
            "seed": seed_input,
            "hires_scale": float(self.hires_scale_var.get()),
            "hires_denoise": float(self.hires_denoise_var.get()),
            "hires_steps": int(self.hires_steps_var.get()),
            "controlnet": cn_path.name if cn_path else None,
            "controlnet_mode": ref_mode,
            "controlnet_strength": ref_strength,
            "reference_image_path": ref_path,
            "auto_upscale": bool(self.auto_upscale_var.get()),
            "auto_upscale_style": str(self.auto_upscale_style_var.get() or DEFAULT_UPSCALE_STYLE),
        }

        self._stop_flag = False
        self._save_settings()  # 走行直前の状態を永続化
        self.generate_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        cnt = params["count"]
        self.status_var.set(
            "開始準備中 (停止まで無限)" if cnt == 0 else f"開始準備中 (0/{cnt})"
        )

        self._worker_thread = threading.Thread(
            target=self._worker, args=(params,), daemon=True,
        )
        self._worker_thread.start()

    def _on_stop_clicked(self) -> None:
        self._stop_flag = True
        self.status_var.set("停止要求受信 - 現在の枚を完了後停止")

    # ---------- 設定 TOML の永続化 ---------- #
    def _gather_settings(self) -> dict:
        """保存対象の値を dict に詰める。Var/Text 由来の生の値を入れる (型は数値/bool/str)。"""
        try:
            prompt_body = self.prompt_text.get("1.0", "end").rstrip("\n")
        except tk.TclError:
            prompt_body = ""
        return {
            "prompt":            prompt_body,
            "prompt_mode":       str(self.prompt_mode_var.get()),
            "count":             int(self.count_var.get()),
            "checkpoint_random": bool(self.checkpoint_random_var.get()),
            "arch_filter":       str(self.arch_filter_var.get()),
            "lora_keywords":     self.lora_kw_var.get() or "",
            "many":              bool(self.many_var.get()),
            "positive":          self.positive_value,
            "negative":          self.negative_value,
            # 生成パラメータ
            "cfg":            float(self.cfg_var.get()),
            "steps":          int(self.steps_var.get()),
            "seed":           str(self.seed_var.get()),
            "width":          int(self.width_var.get()),
            "height":         int(self.height_var.get()),
            "sampler":        str(self.sampler_var.get()),
            "scheduler":      str(self.scheduler_var.get()),
            "lora_total":     float(self.lora_total_var.get()),
            # 品質補正
            "adetailer":        bool(self.adetailer_var.get()),
            "adetailer_person": bool(self.adetailer_person_var.get()),
            "hires_fix":        bool(self.hires_var.get()),
            "hires_scale":    float(self.hires_scale_var.get()),
            "hires_denoise":  float(self.hires_denoise_var.get()),
            "hires_steps":    int(self.hires_steps_var.get()),
            "auto_upscale":       bool(self.auto_upscale_var.get()),
            "auto_upscale_style": str(self.auto_upscale_style_var.get()),
        }

    def _save_settings(self) -> None:
        """書込前に再読込してマージ → ファイル即クローズ (TOML I/O ハイジーン規約)。"""
        data = self._gather_settings()
        merged: dict = {}
        if GUI_SETTINGS_TOML.exists():
            try:
                merged = tomllib.loads(GUI_SETTINGS_TOML.read_text(encoding="utf-8")) or {}
            except Exception:
                merged = {}
        merged.update(data)
        try:
            GUI_SETTINGS_TOML.write_text(tomli_w.dumps(merged), encoding="utf-8")
        except Exception as e:
            print(f"[settings] 保存失敗: {e}", flush=True)

    def _load_settings(self) -> None:
        """起動時に TOML を読んで Var/Text に流し込む。読込直後にクローズ。"""
        if not GUI_SETTINGS_TOML.exists():
            return
        try:
            data = tomllib.loads(GUI_SETTINGS_TOML.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[settings] 読込失敗: {e}", flush=True)
            return

        def _try(setter, key, conv):
            if key in data:
                try:
                    setter(conv(data[key]))
                except Exception:
                    pass

        if "prompt" in data:
            try:
                self.prompt_text.delete("1.0", "end")
                self.prompt_text.insert("1.0", str(data["prompt"]))
            except tk.TclError:
                pass
        if "positive" in data:
            self.positive_value = str(data["positive"])
        if "negative" in data:
            self.negative_value = str(data["negative"])

        _try(self.count_var.set,        "count",         int)
        if "prompt_mode" in data and str(data["prompt_mode"]) in ("free", "toml"):
            self.prompt_mode_var.set(str(data["prompt_mode"]))
            self._on_prompt_mode_change()
        # 版フィルタを先に復元 → combobox を絞った状態で他の Var を入れる
        if "arch_filter" in data and str(data["arch_filter"]) in ("sd15", "sdxl", "mix"):
            self.arch_filter_var.set(str(data["arch_filter"]))
            self._populate_checkpoint_combo()
        _try(self.checkpoint_random_var.set, "checkpoint_random", bool)
        _try(self.many_var.set,              "many",              bool)
        # ランダムチェック復元後に combobox の disabled 状態を反映
        self._on_checkpoint_random_toggle()
        if "lora_keywords" in data:
            try:
                self.lora_kw_var.set(str(data["lora_keywords"]))
            except Exception:
                pass
        _try(self.cfg_var.set,          "cfg",           float)
        _try(self.steps_var.set,        "steps",         int)
        _try(self.seed_var.set,         "seed",          str)
        _try(self.width_var.set,        "width",         int)
        _try(self.height_var.set,       "height",        int)
        _try(self.sampler_var.set,      "sampler",       str)
        _try(self.scheduler_var.set,    "scheduler",     str)
        _try(self.lora_total_var.set,   "lora_total",    float)
        _try(self.adetailer_var.set,        "adetailer",        bool)
        _try(self.adetailer_person_var.set, "adetailer_person", bool)
        _try(self.hires_var.set,            "hires_fix",        bool)
        _try(self.hires_scale_var.set,  "hires_scale",   float)
        _try(self.hires_denoise_var.set,"hires_denoise", float)
        _try(self.hires_steps_var.set,  "hires_steps",   int)
        _try(self.auto_upscale_var.set,       "auto_upscale",       bool)
        _try(self.auto_upscale_style_var.set, "auto_upscale_style", str)

    def _on_app_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    # ---------- 設定ダイアログ ---------- #
    def _open_settings(self) -> None:
        """ポジティブ・ネガティブ・CFG・AD補正・Hires Fix・幅・高さ をまとめた設定ダイアログ。
        spinbox/checkbutton は self.*_var を textvariable=/variable= で直接束縛するので、
        ダイアログを閉じれば即反映 (Cancel 概念なし)。テキスト 2 つだけ閉じる時に self に書き戻す。"""
        if getattr(self, "_settings_win", None) is not None:
            try:
                if self._settings_win.winfo_exists():
                    self._settings_win.lift()
                    self._settings_win.focus_set()
                    return
            except tk.TclError:
                pass

        win = tk.Toplevel(self.root)
        self._settings_win = win
        win.title("設定")
        win.geometry("780x720")
        win.transient(self.root)

        pad = {"padx": 8, "pady": 4}

        # ---- プロンプト群 ----
        ttk.Label(win, text="ポジティブ:").grid(row=0, column=0, sticky="ne", **pad)
        pos_text = tk.Text(win, height=4, width=70, wrap=tk.WORD)
        pos_text.grid(row=0, column=1, columnspan=5, sticky="ew", **pad)
        pos_text.insert("1.0", self.positive_value)

        ttk.Label(win, text="ネガティブ:").grid(row=1, column=0, sticky="ne", **pad)
        neg_text = tk.Text(win, height=4, width=70, wrap=tk.WORD)
        neg_text.grid(row=1, column=1, columnspan=5, sticky="ew", **pad)
        neg_text.insert("1.0", self.negative_value)

        # ---- 解像度・主要パラメータ ----
        gen_frame = ttk.LabelFrame(win, text="生成パラメータ", padding=6)
        gen_frame.grid(row=2, column=0, columnspan=6, sticky="ew", **pad)

        ttk.Label(gen_frame, text="CFG:").grid(row=0, column=0, sticky="e", padx=(4, 2), pady=2)
        ttk.Spinbox(gen_frame, from_=1.0, to=20.0, increment=0.5,
                     textvariable=self.cfg_var, width=7).grid(row=0, column=1, sticky="w")
        ttk.Label(gen_frame, text="Steps:").grid(row=0, column=2, sticky="e", padx=(12, 2))
        ttk.Spinbox(gen_frame, from_=1, to=200,
                     textvariable=self.steps_var, width=6).grid(row=0, column=3, sticky="w")
        ttk.Label(gen_frame, text="Seed:").grid(row=0, column=4, sticky="e", padx=(12, 2))
        ttk.Entry(gen_frame, textvariable=self.seed_var, width=12).grid(
            row=0, column=5, sticky="w")
        ttk.Label(gen_frame, text="(-1=ランダム)", foreground="#888").grid(
            row=0, column=6, sticky="w", padx=(2, 4))

        ttk.Label(gen_frame, text="幅:").grid(row=1, column=0, sticky="e", padx=(4, 2), pady=2)
        ttk.Spinbox(gen_frame, from_=256, to=2048, increment=64,
                     textvariable=self.width_var, width=7).grid(row=1, column=1, sticky="w")
        ttk.Label(gen_frame, text="高さ:").grid(row=1, column=2, sticky="e", padx=(12, 2))
        ttk.Spinbox(gen_frame, from_=256, to=2048, increment=64,
                     textvariable=self.height_var, width=7).grid(row=1, column=3, sticky="w")

        ttk.Label(gen_frame, text="Sampler:").grid(row=2, column=0, sticky="e", padx=(4, 2), pady=2)
        ttk.Combobox(gen_frame, textvariable=self.sampler_var, state="readonly", width=18,
                      values=[
                          "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde",
                          "dpmpp_sde", "dpmpp_2s_ancestral",
                          "euler", "euler_ancestral",
                          "ddim", "uni_pc", "lcm",
                      ]).grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Label(gen_frame, text="Scheduler:").grid(row=2, column=3, sticky="e", padx=(12, 2))
        ttk.Combobox(gen_frame, textvariable=self.scheduler_var, state="readonly", width=14,
                      values=["karras", "normal", "exponential", "sgm_uniform",
                              "simple", "ddim_uniform"]).grid(row=2, column=4, columnspan=2, sticky="w")

        ttk.Label(gen_frame, text="LoRA 合計強度:").grid(row=3, column=0, sticky="e", padx=(4, 2), pady=2)
        ttk.Spinbox(gen_frame, from_=0.1, to=2.0, increment=0.05,
                     textvariable=self.lora_total_var, width=7).grid(row=3, column=1, sticky="w")
        ttk.Label(gen_frame, text="(n 個で割って各 LoRA に配分)", foreground="#888").grid(
            row=3, column=2, columnspan=5, sticky="w", padx=(2, 4))

        # ---- 品質補正 ----
        boost_frame = ttk.LabelFrame(win, text="品質補正", padding=6)
        boost_frame.grid(row=3, column=0, columnspan=6, sticky="ew", **pad)

        ttk.Checkbutton(boost_frame, text="AD補正 (face/hand)", variable=self.adetailer_var).grid(
            row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(boost_frame, text="AD補正 (person)", variable=self.adetailer_person_var).grid(
            row=0, column=1, sticky="w", padx=(12, 4), pady=2)
        ttk.Checkbutton(boost_frame, text="Hires Fix", variable=self.hires_var).grid(
            row=0, column=2, sticky="w", padx=(12, 4), pady=2)

        ttk.Label(boost_frame, text="Hires Scale:").grid(row=1, column=0, sticky="e", padx=(4, 2), pady=2)
        ttk.Spinbox(boost_frame, from_=1.0, to=2.5, increment=0.1,
                     textvariable=self.hires_scale_var, width=7).grid(row=1, column=1, sticky="w")
        ttk.Label(boost_frame, text="Hires Denoise:").grid(row=1, column=2, sticky="e", padx=(12, 2))
        ttk.Spinbox(boost_frame, from_=0.1, to=0.9, increment=0.05,
                     textvariable=self.hires_denoise_var, width=7).grid(row=1, column=3, sticky="w")
        ttk.Label(boost_frame, text="Hires Steps:").grid(row=1, column=4, sticky="e", padx=(12, 2))
        ttk.Spinbox(boost_frame, from_=1, to=100,
                     textvariable=self.hires_steps_var, width=6).grid(row=1, column=5, sticky="w")

        # 自動アップスケール (Real-ESRGAN x4)
        ttk.Checkbutton(
            boost_frame, text="アップスケール (Real-ESRGAN x4)",
            variable=self.auto_upscale_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))
        ttk.Label(boost_frame, text="スタイル:").grid(row=2, column=3, sticky="e", padx=(12, 2))
        ttk.Combobox(
            boost_frame, textvariable=self.auto_upscale_style_var, state="readonly",
            values=list(UPSCALE_MODELS.keys()), width=8,
        ).grid(row=2, column=4, columnspan=2, sticky="w")

        # ---- 閉じるボタン ----
        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=4, column=0, columnspan=6, sticky="ew", **pad)

        def close_and_apply():
            self.positive_value = pos_text.get("1.0", "end").rstrip("\n")
            self.negative_value = neg_text.get("1.0", "end").rstrip("\n")
            win.destroy()
            self._settings_win = None
            self._save_settings()

        ttk.Button(btn_frame, text="閉じる", command=close_and_apply).pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", close_and_apply)
        win.grid_columnconfigure(1, weight=1)
        win.grid_columnconfigure(3, weight=1)
        win.grid_columnconfigure(5, weight=1)
        win.grid_rowconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)

    def _worker(self, params: dict) -> None:
        # version / ckpt は ckpt_random 時に毎枚再決定するため、ここでは取得しない
        try:
            # extra_model_paths.yaml で 3_x / 4_x の model dir を ComfyUI に登録
            # (これを書いておかないと ckpt/LoRA リストが空のまま 400 エラーになる)
            self._result_queue.put({"status": "ComfyUI 起動確認中..."})
            yaml_changed = write_extra_model_paths()
            # GUI セッション初回は YAML 既存でも force_restart → 古いサーバが残ってる場合に新 YAML を読ませる
            ensure_comfyui_arch("cuda", force_restart=yaml_changed or not self._server_verified)
            self._server_verified = True
        except Exception as e:
            self._result_queue.put({"error": f"ComfyUI 起動失敗: {e}"})
            return

        # ADetailer 用 YOLO 重みを必要なら HF から DL (初回のみ)
        # person 系は crop_factor=3.0 で 2 人を 1 bbox にまとめ → 茶色フィルム化する問題があるため
        # adetailer_person トグルが ON のときだけ DL + 配線する
        face_model = hand_model = person_model = None
        if params["adetailer"]:
            self._result_queue.put({"status": "ADetailer モデル確認中..."})
            try:
                face_model = ensure_adetailer_model("bbox/face_yolov8s.pt")
                hand_model = ensure_adetailer_model("bbox/hand_yolov8s.pt")
                if params.get("adetailer_person"):
                    person_model = ensure_adetailer_model("segm/person_yolov8s-seg.pt")
            except Exception as e:
                self._result_queue.put({"status": f"ADetailer モデル準備失敗 ({e}) → スキップ"})

        # 各版の安定 VAE を worker 起動時 1 回 DL + 解決
        # SDXL: madebyollin/sdxl-vae-fp16-fix (茶色フィルム対策)
        # SD15: stabilityai/sd-vae-ft-mse-original (グレースケール脱色 + マゼンタスペック対策)
        try:
            sdxl_vae_name = ensure_sdxl_vae_fix()
        except Exception:
            sdxl_vae_name = None
        try:
            sd15_vae_name = ensure_sd15_vae_mse()
        except Exception:
            sd15_vae_name = None

        base_w = int(params["width"])
        base_h = int(params["height"])
        steps = int(params["steps"])
        sampler = params["sampler"]
        scheduler = params["scheduler"]
        cfg = float(params["cfg"])
        count = int(params["count"])
        seed_input = int(params["seed"])
        hires_scale = float(params["hires_scale"])
        ckpt_random = bool(params.get("checkpoint_random"))
        lora_keywords: list[str] = list(params.get("lora_keywords") or [])
        manual_loras = params["loras"] or []
        lora_total = float(self.lora_total_var.get())
        prompt_mode = params.get("prompt_mode", "free")
        positive_extra = str(params.get("positive_extra") or "").strip()

        # prompt.toml モード: load_prompt_config を 1 回だけキャッシュし、毎枚 build_prompt で抽選
        prompt_cfg = None
        if prompt_mode == "toml":
            try:
                prompt_cfg = load_prompt_config()
            except Exception as e:
                self._result_queue.put({"error": f"prompt.toml 読み込み失敗: {e}"})
                return

        # キーワード抽選で使う LoRA corpus は ckpt 確定後に組む必要があるが、
        # LoRA_param.toml は 1 回だけ読めば良いので先にキャッシュ
        try:
            lora_params_cache = load_lora_params()
        except Exception:
            lora_params_cache = {}

        # CLI と同じプロンプト augmentation (prepare_workflow_prompt) に渡す共通 context
        # score prefix / kw append / neg embed 自動投入 / family-gate / pose-gate を一括処理
        try:
            checkpoint_data = load_checkpoint_toml()
        except Exception:
            checkpoint_data = {}
        try:
            sdxl_lora_subjects = load_sdxl_lora_subjects()
        except Exception:
            sdxl_lora_subjects = {}
        try:
            neg_embed_by_ver = {
                "sd15": collect_negative_embeddings(SD15_EMBED_DIR),
                "sdxl": collect_negative_embeddings(SDXL_EMBED_DIR),
            }
        except Exception:
            neg_embed_by_ver = {"sd15": [], "sdxl": []}

        # リファレンス画像 (ControlNet) を 1 回だけアップロード。失敗時は ControlNet を OFF
        cn_name: Optional[str] = params.get("controlnet")
        cn_mode: str = params.get("controlnet_mode") or "openpose"
        cn_strength: float = float(params.get("controlnet_strength") or 0.7)
        ref_path: Optional[Path] = params.get("reference_image_path")
        ref_uploaded_name: Optional[str] = None
        if ref_path and cn_name:
            try:
                self._result_queue.put({
                    "status": f"リファレンス画像をアップロード中 ({ref_path.name})...",
                })
                ref_uploaded_name = upload_image_to_comfyui(ref_path)
            except Exception as e:
                self._result_queue.put({
                    "status": f"リファレンス画像アップロード失敗 ({e}) → ControlNet OFF",
                })
                ref_uploaded_name = None
        # ref が無いか upload 失敗時は cn_name を捨てて pipe にも渡さない
        if not ref_uploaded_name:
            cn_name = None

        # count == 0 は無限ループ (停止ボタン押下まで)。count > 0 はその回数で終了
        i = 0
        while not self._stop_flag:
            if count > 0 and i >= count:
                break
            # seed_input -1 = 毎回ランダム / それ以外 = seed_input から +i (batch ごとに変化)
            if seed_input < 0:
                seed = random.randint(0, 2**31 - 1)
            else:
                seed = (seed_input + i) & 0x7FFFFFFF

            # prompt.toml モード: 毎枚 build_prompt で抽選 (positive/negative/kw/many を上書き)
            # 自由記載モード: gather 時点で確定済の params["positive"] 等をそのまま使う
            if prompt_mode == "toml":
                iter_pos, iter_neg, iter_kws, iter_many = build_prompt(prompt_cfg)
                if positive_extra:
                    iter_pos = f"{positive_extra}, {iter_pos}" if iter_pos else positive_extra
                iter_pos = normalize_emphasis(iter_pos)
                iter_neg = normalize_emphasis(iter_neg)
            else:
                iter_pos = params["positive"]
                iter_neg = params["negative"]
                iter_kws = lora_keywords
                iter_many = bool(params.get("many"))

            # チェックポイントが「ランダム」なら毎枚抽選し、version / pool を再決定
            if ckpt_random:
                # SD15/SDXL/混合 ラジオで絞った候補のみから抽選
                # 混合のときは generate.pick_checkpoint と同じく 25/75 (SD15/SDXL) で先に版を決める
                random_pool = params.get("checkpoint_random_pool") or self.checkpoints
                sd15_sub = [p for p in random_pool if VERSION_BY_DIR.get(p.parent) == "sd15"]
                sdxl_sub = [p for p in random_pool if VERSION_BY_DIR.get(p.parent) == "sdxl"]
                if sd15_sub and sdxl_sub:
                    lane_pool = sd15_sub if random.random() < 0.25 else sdxl_sub
                else:
                    lane_pool = random_pool
                this_ckpt = random.choice(lane_pool)
                this_version = VERSION_BY_DIR.get(this_ckpt.parent, "sd15")
                this_pool = (
                    self.loras_sd15 if this_version == "sd15" else self.loras_sdxl
                )
                # ラジオで版固定なので手動 LoRA も版マッチしている前提でそのまま使う
                this_manual = manual_loras
            else:
                this_ckpt = params["checkpoint"]
                this_version = params["version"]
                this_pool = (
                    self.loras_sd15 if this_version == "sd15" else self.loras_sdxl
                )
                this_manual = manual_loras

            # LoRA 決定:
            #   ① 手動選択あり → そのまま
            #   ② 手動選択なし & キーワードあり → pick_n_loras_by_keywords で 1-8 個抽選
            #   ③ 手動選択なし & キーワードなし → LoRA 無し
            if this_manual:
                this_loras = this_manual
            elif iter_kws:
                corpus = build_lora_corpus(this_pool, lora_params_cache)
                picked = pick_n_loras_by_keywords(
                    this_pool, iter_kws, corpus, n_max=8, n_min=1,
                )
                scale = lora_total / max(1, len(picked))
                this_loras = [(p, scale) for p in picked]
            else:
                this_loras = []

            # 2 人以上 (ワイド) ON → 版に応じた横長プリセットで base 解像度を上書き
            # (人物融合を抑える。OFF は設定ダイアログの width/height をそのまま使う)
            # toml モードでは build_prompt の many フラグも OR で考慮する
            if iter_many:
                this_w, this_h = WIDE_RES.get(this_version, (base_w, base_h))
            else:
                this_w, this_h = base_w, base_h

            ckpt_label = this_ckpt.stem if ckpt_random else this_ckpt.name
            # count=0 (無限) は分母なし。count>0 のときだけ "/N" を出す
            progress = f"{i+1}/{count}" if count > 0 else f"{i+1}"
            self._result_queue.put({
                "status": f"生成中 {progress} (seed={seed}, ckpt={ckpt_label})",
            })

            # SDXL のみ ControlNet 配線 (SD15 ckpt が混ざる場合は cn 引数を捨てる)
            this_cn_name = cn_name if this_version == "sdxl" else None
            this_cn_image = ref_uploaded_name if this_cn_name else None
            effective_cn_mode = cn_mode if this_cn_name else ""

            # CLI と同じ helper で augmentation 集約: score prefix / kw append / neg embed / family-gate / pose-gate
            positive_aug, negative_aug, this_loras_filtered, gate_logs = prepare_workflow_prompt(
                iter_pos, iter_neg,
                lora_keywords=iter_kws,
                picked_loras=this_loras,
                checkpoint_path=this_ckpt,
                checkpoint_data=checkpoint_data,
                neg_embed_stems=neg_embed_by_ver.get(this_version, []),
                explicit_quality_prefix="",
                force_pony_prefix=False,
                controlnet_mode=effective_cn_mode,
                sdxl_lora_subjects=sdxl_lora_subjects,
                lora_total=lora_total,
            )
            for line in gate_logs:
                print(line, flush=True)
            workflow_loras = [(p.name, s) for p, s in this_loras_filtered]

            workflow = build_workflow_txt2img(
                checkpoint=this_ckpt.name,
                positive=positive_aug,
                negative=negative_aug,
                seed=seed,
                steps=steps,
                cfg=cfg,
                width=this_w,
                height=this_h,
                sampler_name=sampler,
                scheduler=scheduler,
                loras=workflow_loras or None,
                adetailer=params["adetailer"],
                adetailer_face_model=face_model or "bbox/face_yolov8s.pt",
                adetailer_hand_model=hand_model,
                adetailer_person_model=person_model,
                hires_fix=params["hires_fix"],
                hires_scale=hires_scale,
                hires_denoise=float(params["hires_denoise"]),
                hires_steps=int(params["hires_steps"]),
                controlnet_name=this_cn_name,
                controlnet_mode=cn_mode,
                controlnet_image=this_cn_image,
                controlnet_strength=cn_strength,
                # SDXL のときだけ安定 VAE を被せる (SD15 ckpt には流用不可)
                vae_override=sdxl_vae_name if this_version == "sdxl" else sd15_vae_name,
            )

            try:
                img_bytes, _info, _outs = _submit_and_fetch(workflow, self._client_id)
            except Exception as e:
                self._result_queue.put({"error": f"生成失敗 ({progress}): {e}"})
                return

            if img_bytes is None:
                self._result_queue.put({"status": f"{progress}: 画像未取得、スキップ"})
                i += 1
                continue

            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            # 出力 dir は ckpt の版で振り分け (SD15=3_8 / SDXL=5_1)
            gen_dir = SDXL_GENERATED_DIR if this_version == "sdxl" else SD15_GENERATED_DIR
            gen_dir.mkdir(exist_ok=True)
            out_path = gen_dir / f"gui_{ts}_{i+1:04d}.png"
            final_w = int(this_w * hires_scale) if params["hires_fix"] else this_w
            final_h = int(this_h * hires_scale) if params["hires_fix"] else this_h

            try:
                save_with_a1111_metadata(
                    img_bytes, out_path,
                    positive=positive_aug, negative=negative_aug, seed=seed,
                    steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,
                    width=final_w, height=final_h,
                    checkpoint=this_ckpt.name,
                    lora_keywords=iter_kws,
                    loras=workflow_loras or None,
                    adetailer=params["adetailer"],
                    pipeline="GUI",
                )
            except Exception as e:
                self._result_queue.put({"error": f"保存失敗 ({progress}): {e}"})
                return

            # 自動アップスケール: gen ループ内で同期実行 (ComfyUI への submit を直列化)
            # auto_upscale 時は upscale 完了画像だけギャラリーに追加し、gen は status だけ
            auto_up = bool(params.get("auto_upscale"))
            if auto_up:
                self._result_queue.put({
                    "status": f"生成完了 ({progress}): {out_path.name} → アップスケール中…",
                })
            else:
                self._result_queue.put({
                    "image": out_path, "iter": i + 1, "total": count,
                })

            if auto_up:
                try:
                    up_path = self._do_upscale_sync(
                        out_path,
                        params.get("auto_upscale_style") or DEFAULT_UPSCALE_STYLE,
                        client_id=self._client_id,
                    )
                    if up_path is not None:
                        self._result_queue.put({
                            "image": up_path, "iter": i + 1, "total": count,
                            "job": "upscale_inline",
                        })
                except Exception as e:
                    # アップスケール失敗時は gen 画像をギャラリーに残す
                    self._result_queue.put({
                        "status": f"アップスケール失敗 ({out_path.name}): {e}",
                    })
                    self._result_queue.put({
                        "image": out_path, "iter": i + 1, "total": count,
                    })

            i += 1

        self._result_queue.put({"done": True})

    # ---------- queue / ギャラリー ---------- #
    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._result_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        # 走行中ジョブの状態に合わせて生成ボタンを自動更新
        # (upscale はスレッド終了時に明示的な "done" を送らないので poll 側でケアする)
        self._refresh_busy_state()
        self.root.after(150, self._poll_queue)

    def _refresh_busy_state(self) -> None:
        """gen ボタンの enable/disable を gen 走行状態に合わせて反映。
        gen 走行中は disable、止まっていれば normal。"""
        if self._gen_alive():
            self.generate_btn.configure(state=tk.DISABLED)
        else:
            self.generate_btn.configure(state=tk.NORMAL)

    def _handle_message(self, msg: dict) -> None:
        # job="upscale" は gen の状態 (status バー / 生成ボタン) を巻き込まないように分離
        job = msg.get("job", "gen")
        if "error" in msg:
            if job == "upscale":
                # gen 走行中でも upscale の失敗は status だけ更新、エラーダイアログのみ。
                # 生成ボタンや stop ボタンは触らない (gen の進行を妨げない)
                self.status_var.set(f"アップスケールエラー: {msg['error']}")
                messagebox.showerror("アップスケールエラー", msg["error"])
                return
            self.status_var.set(f"エラー: {msg['error']}")
            messagebox.showerror("生成エラー", msg["error"])
            self._reset_buttons()
            return
        if "status" in msg:
            # gen 走行中の upscale status は混乱の元なので gen 状態が空のときだけ表示
            # (gen の進捗 = "生成中 N/M" が走っているならそれを上書きしない)
            if job == "upscale" and self._worker_thread and self._worker_thread.is_alive():
                # gen 走行中: 末尾に併記して両方見えるようにする
                cur = self.status_var.get()
                self.status_var.set(f"{cur}  /  {msg['status']}")
            else:
                self.status_var.set(msg["status"])
            return
        if "image" in msg:
            if job == "upscale":
                self.status_var.set(f"アップスケール完了: {Path(msg['image']).name}")
            elif msg.get("total"):
                # 有限 batch: "完了 N/M"
                self.status_var.set(f"完了 {msg['iter']}/{msg['total']}: {Path(msg['image']).name}")
            elif msg.get("iter"):
                # 無限ループ: "完了 N枚" (分母なし)
                self.status_var.set(f"完了 {msg['iter']}枚: {Path(msg['image']).name}")
            else:
                self.status_var.set(f"完了: {Path(msg['image']).name}")
            self._add_gallery_thumbnail(msg["image"])
            return
        if "done" in msg:
            # done は gen ワーカーのみ送る (upscale は status 完了で代替)
            self.status_var.set("全完了")
            self._reset_buttons()
            return

    def _reset_buttons(self) -> None:
        self.generate_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def _add_gallery_thumbnail(self, path: Path) -> None:
        if Image is None:
            return
        try:
            img = Image.open(path)
            img.thumbnail(GALLERY_THUMB_SIZE)
            photo = ImageTk.PhotoImage(img)
        except Exception:
            return
        self._gallery_thumbs.append(photo)
        self._gallery_paths.append(path)

        cell = tk.Frame(self.gallery_inner, padx=4, pady=4)
        lbl = tk.Label(cell, image=photo, cursor="hand2", relief="solid", borderwidth=1)
        lbl.image = photo
        lbl.pack()
        name_lbl = tk.Label(cell, text=path.name, font=("TkDefaultFont", 8))
        name_lbl.pack()
        lbl.bind("<Button-1>", lambda e, p=path: self._show_modal(p))
        # 右クリックメニュー (削除 / アップスケール)
        lbl.bind("<Button-3>", lambda e, p=path, c=cell: self._show_gallery_menu(e, p, c))
        name_lbl.bind("<Button-3>", lambda e, p=path, c=cell: self._show_gallery_menu(e, p, c))
        # ホイールイベントもサムネ上で受けて Canvas に転送
        for w in (cell, lbl, name_lbl):
            w.bind("<MouseWheel>", self._on_gallery_wheel)
        cell.grid(row=self._gallery_row, column=self._gallery_col, sticky="nw")
        self._gallery_col += 1
        if self._gallery_col >= self._gallery_max_cols:
            self._gallery_col = 0
            self._gallery_row += 1

        self.gallery_inner.update_idletasks()
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))

    # ---------- 右クリックメニュー ---------- #
    def _show_gallery_menu(self, event, path: Path, cell: tk.Widget) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="削除", command=lambda: self._delete_gallery_item(path, cell))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _delete_gallery_item(self, path: Path, cell: tk.Widget) -> None:
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            messagebox.showerror("削除失敗", f"{path.name}: {e}")
            return
        cell.destroy()
        # モーダルナビ用の順序保持リストからも除外 (空き grid セルは許容、_gallery_paths だけ整合させる)
        try:
            self._gallery_paths.remove(path)
        except ValueError:
            pass
        self.status_var.set(f"削除: {path.name}")

    def _do_upscale_sync(
        self, path: Path, style: str, client_id: str,
    ) -> Optional[Path]:
        """1 枚を Real-ESRGAN x4 アップスケールして保存先 dir は path の親 dir から判別:
            3_8_SD15_generated → 3_9_SD15_upscaled
            4_8_SDXL_generated → 4_9_SDXL_upscaled
            それ以外 (旧 5_1_generated 等の手動入力含む) は SDXL_UPSCALED_DIR にフォールバック。
        失敗時は raise (caller がメッセージ処理)。worker (gen ループ内同期) で利用。
        ソース PNG (path) に A1111 parameters chunk があれば upscale 後に複写
        (Size フィールドのみ新解像度に書き換え)。"""
        model_name = UPSCALE_MODELS.get(style, UPSCALE_MODELS[DEFAULT_UPSCALE_STYLE])
        resolved = ensure_upscale_model(model_name)
        if not resolved:
            raise RuntimeError(f"アップスケールモデル {model_name} を取得できません")
        model_name = resolved
        uploaded = upload_image_to_comfyui(path)
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": uploaded}},
            "2": {"class_type": "UpscaleModelLoader",
                  "inputs": {"model_name": model_name}},
            "3": {"class_type": "ImageUpscaleWithModel",
                  "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"images": ["3", 0],
                             "filename_prefix": f"gui_upscale_{style}"}},
        }
        img_bytes, _info, _ = _submit_and_fetch(workflow, client_id)
        if img_bytes is None:
            return None
        # 親 dir から版を判別 (SD15_GENERATED_DIR or 旧 3_9_SD15_rough → SD15、それ以外 → SDXL)
        if path.parent == SD15_GENERATED_DIR:
            up_dir = SD15_UPSCALED_DIR
        else:
            up_dir = SDXL_UPSCALED_DIR
        up_dir.mkdir(exist_ok=True)
        out_path = up_dir / path.name
        out_path.write_bytes(img_bytes)

        # ソース PNG の A1111 parameters chunk を複写。Size だけ新解像度に書き換える
        try:
            src_chunks = read_text_chunks(path)
        except Exception:
            src_chunks = {}
        if src_chunks:
            try:
                from PIL import Image as _Image
                with _Image.open(out_path) as _im:
                    new_w, new_h = _im.size
                params_text = src_chunks.get("parameters")
                if params_text:
                    # "Size: WxH" を書き換え (見つからなければ末尾追加)
                    import re
                    new_size = f"Size: {new_w}x{new_h}"
                    if re.search(r"Size:\s*\d+x\d+", params_text):
                        params_text = re.sub(r"Size:\s*\d+x\d+", new_size, params_text)
                    else:
                        params_text = f"{params_text}, {new_size}"
                    src_chunks["parameters"] = params_text
                write_text_chunks(out_path, src_chunks)
            except Exception as e:
                # メタ複写失敗は致命的ではない (画像本体は保存済み)
                print(f"[upscale] metadata copy failed: {e}", flush=True)
        return out_path

    def _on_gallery_wheel(self, event) -> str:
        """ホイール delta±120/notch をスクロール単位 (units) に変換して Canvas を縦スクロール。"""
        self.gallery_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _show_modal(self, path: Path) -> None:
        """モーダルプレビュー (singleton)。既存があれば破棄して新規 1 枚だけ表示。
        ←/→ でギャラリー内の前後画像に遷移 (ラップアラウンド)。ギャラリー外の
        画像 (checkpoint thumb / reference image) の場合は ←/→ は no-op。"""
        if Image is None:
            return
        # 既存モーダルを必ず閉じる
        if self._modal_win is not None:
            try:
                if self._modal_win.winfo_exists():
                    self._modal_win.destroy()
            except tk.TclError:
                pass
            self._modal_win = None

        top = tk.Toplevel(self.root)
        self._modal_win = top
        try:
            orig_img = Image.open(path)
        except Exception as e:
            top.title(path.name)
            tk.Label(top, text=f"open error: {e}").pack()
            return

        ow, oh = orig_img.size
        sw = self.root.winfo_screenwidth() - 80
        sh = self.root.winfo_screenheight() - 160  # title バー分余裕
        # 初期スケール = 画面に収まるサイズ (原寸=100%。原寸が画面より小さければ 100%)
        init_scale = min(sw / ow, sh / oh, 1.0)
        state = {"scale": init_scale}

        # Canvas + 2 軸スクロール (拡大時に画像が画面外にはみ出るため)
        body = tk.Frame(top)
        body.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(body, bg="#222222", highlightthickness=0)
        hsb = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=canvas.xview)
        vsb = ttk.Scrollbar(body, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        image_id = canvas.create_image(0, 0, anchor="nw")
        photo_holder = {"photo": None}  # GC 回避用

        def _render() -> None:
            scale = max(0.05, min(8.0, state["scale"]))
            state["scale"] = scale
            new_w = max(1, int(ow * scale))
            new_h = max(1, int(oh * scale))
            try:
                resized = orig_img.resize((new_w, new_h), Image.LANCZOS)
            except Exception:
                resized = orig_img
            photo = ImageTk.PhotoImage(resized)
            photo_holder["photo"] = photo
            canvas.itemconfigure(image_id, image=photo)
            canvas.configure(scrollregion=(0, 0, new_w, new_h))
            top.title(f"{path.name}  —  {int(scale * 100)}%")

        _render()

        # Modal 初期サイズ: フィット時のサイズ + scrollbar 分
        init_w = min(sw, int(ow * init_scale) + 24)
        init_h = min(sh, int(oh * init_scale) + 48)
        top.geometry(f"{init_w}x{init_h}")

        # ←/→ でナビ
        def _navigate(delta: int) -> None:
            if path not in self._gallery_paths or len(self._gallery_paths) < 2:
                return
            idx = self._gallery_paths.index(path)
            new_idx = (idx + delta) % len(self._gallery_paths)
            self._show_modal(self._gallery_paths[new_idx])

        # Ctrl+ホイール で拡大縮小 (event.state & 0x4 が Ctrl)。1 notch で 1.1x
        def _on_wheel(event) -> str:
            if event.state & 0x4:  # Ctrl
                factor = 1.1 if event.delta > 0 else (1 / 1.1)
                state["scale"] *= factor
                _render()
                return "break"
            # Ctrl 無しは縦スクロールに振る (拡大時の画像内移動)
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        top.bind("<Escape>", lambda e: top.destroy())
        # クリック閉じは中で拡大表示のとき扱いづらいので右ボタンを閉じる用に振る
        top.bind("<Button-3>", lambda e: top.destroy())
        top.bind("<Left>",  lambda e: _navigate(-1))
        top.bind("<Right>", lambda e: _navigate(+1))
        canvas.bind("<MouseWheel>", _on_wheel)
        top.bind("<MouseWheel>", _on_wheel)

        # ホイールプッシュ (ミドルボタン) + 移動で画像をパン (Canvas.scan_mark/scan_dragto)
        def _pan_start(event) -> None:
            canvas.config(cursor="fleur")
            canvas.scan_mark(event.x, event.y)

        def _pan_move(event) -> None:
            # gain=1 でマウスドラッグ量とパン量が 1:1
            canvas.scan_dragto(event.x, event.y, gain=1)

        def _pan_end(event) -> None:
            canvas.config(cursor="")

        canvas.bind("<ButtonPress-2>", _pan_start)
        canvas.bind("<B2-Motion>", _pan_move)
        canvas.bind("<ButtonRelease-2>", _pan_end)

        # destroy 時に self._modal_win の参照を片付ける
        top.bind("<Destroy>", lambda e: self._on_modal_destroyed(e, top))
        top.focus_set()

    def _on_modal_destroyed(self, event, top) -> None:
        # Toplevel と子 widget 両方から Destroy が来るので Toplevel 本体のみ反応
        if event.widget is top and self._modal_win is top:
            self._modal_win = None


def main() -> None:
    # tkinterdnd2 があれば TkinterDnD.Tk() を使う (ファイル D&D を受け付けるため)
    root = TkinterDnD.Tk() if _DND_AVAILABLE else tk.Tk()
    GenerateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
