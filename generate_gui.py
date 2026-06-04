#!/usr/bin/env python3
"""generate_gui.py - 画像生成 GUI (Tkinter, ComfyUI HTTP API 経由)。

generate.py の CLI ループに対する、手動指定・即時可視化版。
- チェックポイント / LoRA / プロンプトを手で選び、1〜300 枚をまとめて生成。
- *word* / **word** / ***word*** の重み記法は normalize_emphasis でそのまま使える。
- 生成画像は 5_1_generated に PNG + A1111 メタ付きで保存し、画面下段の
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

from common import normalize_emphasis
from generate import (
    GENERATED_DIR,
    SD15_CHECKPOINT_DIR,
    SD15_EMBED_DIR,
    SD15_LORA_DIR,
    SDXL_CHECKPOINT_DIR,
    SDXL_CONTROLNET_DIR,
    SDXL_EMBED_DIR,
    SDXL_LORA_DIR,
    UPSCALED_DIR,
    _submit_and_fetch,
    build_workflow_txt2img,
    ensure_adetailer_model,
    ensure_comfyui_arch,
    ensure_upscale_model,
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
        root.geometry("1280x920")

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
        self._lora_icon_widgets: list[tk.Widget] = []

        self.checkpoints: list[Path] = []
        self.loras_sd15: list[Path] = []
        self.loras_sdxl: list[Path] = []
        self.embeds_sd15: list[Path] = []
        self.embeds_sdxl: list[Path] = []
        self.controlnets: list[Path] = []
        self.current_loras: list[Path] = []
        self.current_embeds: list[Path] = []
        self.selected_lora_indices: set[int] = set()
        self.selected_embed_indices: set[int] = set()
        self.selected_controlnet_index: int = -1

        self._gallery_col = 0
        self._gallery_row = 0
        self._gallery_max_cols = 7

        self._build_ui()
        self._scan_assets()
        self._populate_checkpoint_combo()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_app_close)
        self.root.after(150, self._poll_queue)

    # ---------- UI 構築 ---------- #
    def _build_ui(self) -> None:
        paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(paned, padding=8)
        paned.add(top, weight=3)

        row = 0
        ttk.Label(top, text="チェックポイント:").grid(row=row, column=0, sticky="ne", padx=4, pady=4)
        ckpt_frame = ttk.Frame(top)
        ckpt_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        # 横方向: column 0 = combobox (伸びる) / column 1 = thumb (固定サイズ)
        ckpt_frame.grid_columnconfigure(0, weight=1)
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(
            ckpt_frame, textvariable=self.checkpoint_var, state="readonly",
        )
        self.checkpoint_combo.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.checkpoint_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_change)
        # thumb は Frame で固定サイズ枠を作り、その中の Label に画像を入れる
        thumb_box = tk.Frame(
            ckpt_frame,
            width=CHECKPOINT_THUMB_SIZE[0], height=CHECKPOINT_THUMB_SIZE[1],
            bg="#eeeeee",
        )
        thumb_box.grid(row=0, column=1, rowspan=2, sticky="ne")
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

        # ControlNet (SDXL のみ) + Embedding (版でフィルタ) を横並び
        row += 1
        ttk.Label(top, text="ControlNet / Embedding:").grid(
            row=row, column=0, sticky="ne", padx=4, pady=4)
        cn_emb_frame = ttk.Frame(top)
        cn_emb_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        cn_emb_frame.grid_columnconfigure(0, weight=1)
        cn_emb_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(cn_emb_frame, text="ControlNet (SDXL のみ、単一選択):",
                  foreground="#666").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Label(cn_emb_frame, text="Embedding (negative に自動追加):",
                  foreground="#666").grid(row=0, column=1, sticky="w", padx=(4, 0))

        cn_box = ttk.Frame(cn_emb_frame)
        cn_box.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self.cn_listbox = tk.Listbox(
            cn_box, selectmode=tk.SINGLE, height=4, exportselection=False,
        )
        self.cn_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        cn_scroll = ttk.Scrollbar(cn_box, orient=tk.VERTICAL, command=self.cn_listbox.yview)
        cn_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.cn_listbox.config(yscrollcommand=cn_scroll.set)
        self.cn_listbox.bind("<<ListboxSelect>>", self._on_cn_select)

        emb_box = ttk.Frame(cn_emb_frame)
        emb_box.grid(row=1, column=1, sticky="ew", padx=(4, 0))
        self.embed_listbox = tk.Listbox(
            emb_box, selectmode=tk.EXTENDED, height=4, exportselection=False,
        )
        self.embed_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        emb_scroll = ttk.Scrollbar(emb_box, orient=tk.VERTICAL, command=self.embed_listbox.yview)
        emb_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.embed_listbox.config(yscrollcommand=emb_scroll.set)
        self.embed_listbox.bind("<<ListboxSelect>>", self._on_embed_select)

        # ControlNet を populate (1 回だけ、checkpoint 切替で消さない)
        for p in self.controlnets:
            self.cn_listbox.insert(tk.END, p.stem)

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

        # 左端: 推論数 / OpenPose / 設定ボタン
        ttk.Label(ctrl, text="推論数:").pack(side=tk.LEFT, padx=(4, 2))
        self.count_var = tk.IntVar(value=50)
        ttk.Spinbox(ctrl, from_=1, to=300, textvariable=self.count_var, width=5).pack(side=tk.LEFT)

        self.openpose_var = tk.BooleanVar(value=True)
        self.openpose_check = ttk.Checkbutton(ctrl, text="OpenPose", variable=self.openpose_var)
        self.openpose_check.pack(side=tk.LEFT, padx=(16, 2))

        # 「設定」ダイアログ: ポジ/ネガ/CFG/AD補正/Hires Fix/幅/高さ をまとめる
        # 関連 Var は __init__ で生成しておき、ダイアログ widget が textvariable で直接束縛する
        self.cfg_var = tk.DoubleVar(value=7.0)
        self.adetailer_var = tk.BooleanVar(value=True)
        self.hires_var = tk.BooleanVar(value=True)
        ttk.Button(ctrl, text="設定…", command=self._open_settings).pack(side=tk.LEFT, padx=(16, 2))

        # 右端: 画像生成 / 停止 (pack 順は右から積まれるので 停止→生成 の順で見た目は 生成 停止)
        self.stop_btn = ttk.Button(ctrl, text="停止", command=self._on_stop_clicked,
                                    state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=4)
        self.generate_btn = ttk.Button(ctrl, text="画像生成", command=self._on_generate_clicked)
        self.generate_btn.pack(side=tk.RIGHT, padx=4)

        # 詳細パラメータも「設定」ダイアログに移動 (Var だけ生成しておく)
        self.width_var = tk.IntVar(value=DEFAULT_RES["sd15"])
        self.height_var = tk.IntVar(value=DEFAULT_RES["sd15"])
        self.steps_var = tk.IntVar(value=30)
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

        # 下段: 作成画像ギャラリー
        bottom = ttk.Frame(paned, padding=8)
        paned.add(bottom, weight=2)

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
        self.embeds_sd15 = _list_safetensors(SD15_EMBED_DIR)
        self.embeds_sdxl = _list_safetensors(SDXL_EMBED_DIR)
        self.controlnets = _list_safetensors(SDXL_CONTROLNET_DIR)

    def _populate_checkpoint_combo(self) -> None:
        labels = []
        for p in self.checkpoints:
            tag = "[SD15]" if p.parent == SD15_CHECKPOINT_DIR else "[SDXL]"
            labels.append(f"{tag} {p.stem}")
        self.checkpoint_combo["values"] = labels
        if labels:
            self.checkpoint_combo.current(0)
            self._on_checkpoint_change()

    def _current_checkpoint(self) -> Optional[Path]:
        idx = self.checkpoint_combo.current()
        if idx < 0 or idx >= len(self.checkpoints):
            return None
        return self.checkpoints[idx]

    def _current_version(self) -> str:
        ckpt = self._current_checkpoint()
        if ckpt is None:
            return "sd15"
        return VERSION_BY_DIR.get(ckpt.parent, "sd15")

    # ---------- イベント ---------- #
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

        # Embedding は版で切り替え (LoRA と同じパターン)
        if hasattr(self, "embed_listbox"):
            self.current_embeds = (
                self.embeds_sd15 if version == "sd15" else self.embeds_sdxl
            )
            self.embed_listbox.delete(0, tk.END)
            for p in self.current_embeds:
                self.embed_listbox.insert(tk.END, p.stem)
            self.selected_embed_indices.clear()
            # SD15 を選んだら ControlNet は使えないので選択解除
            if version == "sd15" and hasattr(self, "cn_listbox"):
                self.cn_listbox.selection_clear(0, tk.END)
                self.selected_controlnet_index = -1

        # OpenPose も ControlNet 系なので SD15 では無効化 (4_3 は SDXL のみ)
        if hasattr(self, "openpose_check"):
            if version == "sd15":
                self.openpose_var.set(False)
                self.openpose_check.state(["disabled"])
            else:
                self.openpose_check.state(["!disabled"])

        # 解像度を版に合わせて自動セット (ユーザが弄った後は次の checkpoint 切替で上書きされる点に注意)
        if hasattr(self, "width_var"):
            self.width_var.set(DEFAULT_RES[version])
            self.height_var.set(DEFAULT_RES[version])

    def _on_lora_select(self, event=None) -> None:
        self.selected_lora_indices = set(self.lora_listbox.curselection())
        self._refresh_lora_icons()

    def _on_cn_select(self, event=None) -> None:
        sel = self.cn_listbox.curselection()
        self.selected_controlnet_index = sel[0] if sel else -1

    def _on_embed_select(self, event=None) -> None:
        self.selected_embed_indices = set(self.embed_listbox.curselection())

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

    # ---------- 生成 ---------- #
    def _on_generate_clicked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        ckpt = self._current_checkpoint()
        if ckpt is None:
            messagebox.showerror("エラー", "チェックポイントが選択されていません")
            return
        prompt_body = self.prompt_text.get("1.0", "end").strip()
        positive_extra = self.positive_value.strip()
        negative = self.negative_value.strip()
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
        loras_with_strength = [(p.name, scale) for p in loras]

        # 選択された Embedding stem 一覧 (negative に embedding:<stem> として埋め込む)
        embeds = [self.current_embeds[i] for i in sorted(self.selected_embed_indices)]
        embed_stems = [p.stem for p in embeds]
        if embed_stems:
            embed_prefix = ", ".join(f"embedding:{s}" for s in embed_stems)
            negative = f"{embed_prefix}, {negative}" if negative else embed_prefix

        # 選択された ControlNet (SDXL のみ)。controlnet_image 未供給なので workflow は素通しになる点に注意
        cn_path: Optional[Path] = None
        if version == "sdxl" and 0 <= self.selected_controlnet_index < len(self.controlnets):
            cn_path = self.controlnets[self.selected_controlnet_index]

        try:
            seed_input = int(self.seed_var.get().strip())
        except ValueError:
            seed_input = -1

        params = {
            "checkpoint": ckpt,
            "version": version,
            "positive": positive,
            "negative": negative,
            "loras": loras_with_strength,
            "count": int(self.count_var.get()),
            "cfg": float(self.cfg_var.get()),
            "adetailer": bool(self.adetailer_var.get()),
            "hires_fix": bool(self.hires_var.get()),
            "openpose": bool(self.openpose_var.get()),
            "width": int(self.width_var.get()),
            "height": int(self.height_var.get()),
            "steps": int(self.steps_var.get()),
            "sampler": self.sampler_var.get(),
            "scheduler": self.scheduler_var.get(),
            "seed": seed_input,
            "hires_scale": float(self.hires_scale_var.get()),
            "hires_denoise": float(self.hires_denoise_var.get()),
            "hires_steps": int(self.hires_steps_var.get()),
            "controlnet": cn_path.name if cn_path else None,
        }

        self._stop_flag = False
        self._save_settings()  # 走行直前の状態を永続化
        self.generate_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"開始準備中 (0/{params['count']})")

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
            "prompt":         prompt_body,
            "count":          int(self.count_var.get()),
            "openpose":       bool(self.openpose_var.get()),
            "positive":       self.positive_value,
            "negative":       self.negative_value,
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
            "adetailer":      bool(self.adetailer_var.get()),
            "hires_fix":      bool(self.hires_var.get()),
            "hires_scale":    float(self.hires_scale_var.get()),
            "hires_denoise":  float(self.hires_denoise_var.get()),
            "hires_steps":    int(self.hires_steps_var.get()),
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
        _try(self.openpose_var.set,     "openpose",      bool)
        _try(self.cfg_var.set,          "cfg",           float)
        _try(self.steps_var.set,        "steps",         int)
        _try(self.seed_var.set,         "seed",          str)
        _try(self.width_var.set,        "width",         int)
        _try(self.height_var.set,       "height",        int)
        _try(self.sampler_var.set,      "sampler",       str)
        _try(self.scheduler_var.set,    "scheduler",     str)
        _try(self.lora_total_var.set,   "lora_total",    float)
        _try(self.adetailer_var.set,    "adetailer",     bool)
        _try(self.hires_var.set,        "hires_fix",     bool)
        _try(self.hires_scale_var.set,  "hires_scale",   float)
        _try(self.hires_denoise_var.set,"hires_denoise", float)
        _try(self.hires_steps_var.set,  "hires_steps",   int)

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

        ttk.Checkbutton(boost_frame, text="AD補正", variable=self.adetailer_var).grid(
            row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(boost_frame, text="Hires Fix", variable=self.hires_var).grid(
            row=0, column=1, sticky="w", padx=(12, 4), pady=2)

        ttk.Label(boost_frame, text="Hires Scale:").grid(row=1, column=0, sticky="e", padx=(4, 2), pady=2)
        ttk.Spinbox(boost_frame, from_=1.0, to=2.5, increment=0.1,
                     textvariable=self.hires_scale_var, width=7).grid(row=1, column=1, sticky="w")
        ttk.Label(boost_frame, text="Hires Denoise:").grid(row=1, column=2, sticky="e", padx=(12, 2))
        ttk.Spinbox(boost_frame, from_=0.1, to=0.9, increment=0.05,
                     textvariable=self.hires_denoise_var, width=7).grid(row=1, column=3, sticky="w")
        ttk.Label(boost_frame, text="Hires Steps:").grid(row=1, column=4, sticky="e", padx=(12, 2))
        ttk.Spinbox(boost_frame, from_=1, to=100,
                     textvariable=self.hires_steps_var, width=6).grid(row=1, column=5, sticky="w")

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
        version = params["version"]
        ckpt_name = params["checkpoint"].name
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
        face_model = hand_model = person_model = None
        if params["adetailer"]:
            self._result_queue.put({"status": "ADetailer モデル確認中..."})
            try:
                face_model = ensure_adetailer_model("bbox/face_yolov8n.pt")
                hand_model = ensure_adetailer_model("bbox/hand_yolov8n.pt")
                person_model = ensure_adetailer_model("segm/person_yolov8n-seg.pt")
            except Exception as e:
                self._result_queue.put({"status": f"ADetailer モデル準備失敗 ({e}) → スキップ"})

        base_w = int(params["width"])
        base_h = int(params["height"])
        steps = int(params["steps"])
        sampler = params["sampler"]
        scheduler = params["scheduler"]
        cfg = float(params["cfg"])
        count = int(params["count"])
        seed_input = int(params["seed"])
        hires_scale = float(params["hires_scale"])

        for i in range(count):
            if self._stop_flag:
                break
            # seed_input -1 = 毎回ランダム / それ以外 = seed_input から +i (batch ごとに変化)
            if seed_input < 0:
                seed = random.randint(0, 2**31 - 1)
            else:
                seed = (seed_input + i) & 0x7FFFFFFF
            self._result_queue.put({"status": f"生成中 {i+1}/{count} (seed={seed})"})

            workflow = build_workflow_txt2img(
                checkpoint=ckpt_name,
                positive=params["positive"],
                negative=params["negative"],
                seed=seed,
                steps=steps,
                cfg=cfg,
                width=base_w,
                height=base_h,
                sampler_name=sampler,
                scheduler=scheduler,
                loras=params["loras"] or None,
                adetailer=params["adetailer"],
                adetailer_face_model=face_model or "bbox/face_yolov8n.pt",
                adetailer_hand_model=hand_model,
                adetailer_person_model=person_model,
                hires_fix=params["hires_fix"],
                hires_scale=hires_scale,
                hires_denoise=float(params["hires_denoise"]),
                hires_steps=int(params["hires_steps"]),
            )

            try:
                img_bytes, _info, _outs = _submit_and_fetch(workflow, self._client_id)
            except Exception as e:
                self._result_queue.put({"error": f"生成失敗 ({i+1}/{count}): {e}"})
                return

            if img_bytes is None:
                self._result_queue.put({"status": f"{i+1}: 画像未取得、スキップ"})
                continue

            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            out_path = GENERATED_DIR / f"gui_{ts}_{i+1:04d}.png"
            final_w = int(base_w * hires_scale) if params["hires_fix"] else base_w
            final_h = int(base_h * hires_scale) if params["hires_fix"] else base_h

            try:
                save_with_a1111_metadata(
                    img_bytes, out_path,
                    positive=params["positive"], negative=params["negative"], seed=seed,
                    steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,
                    width=final_w, height=final_h,
                    checkpoint=ckpt_name,
                    lora_keywords=[],
                    loras=params["loras"] or None,
                    adetailer=params["adetailer"],
                    pipeline="GUI",
                )
            except Exception as e:
                self._result_queue.put({"error": f"保存失敗 ({i+1}/{count}): {e}"})
                return

            self._result_queue.put({"image": out_path, "iter": i + 1, "total": count})

        self._result_queue.put({"done": True})

    # ---------- queue / ギャラリー ---------- #
    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._result_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _handle_message(self, msg: dict) -> None:
        if "error" in msg:
            self.status_var.set(f"エラー: {msg['error']}")
            messagebox.showerror("生成エラー", msg["error"])
            self._reset_buttons()
            return
        if "status" in msg:
            self.status_var.set(msg["status"])
            return
        if "image" in msg:
            if msg.get("total"):
                self.status_var.set(f"完了 {msg['iter']}/{msg['total']}: {Path(msg['image']).name}")
            else:
                self.status_var.set(f"完了: {Path(msg['image']).name}")
            self._add_gallery_thumbnail(msg["image"])
            return
        if "done" in msg:
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

        upscale_menu = tk.Menu(menu, tearoff=0)
        upscale_menu.add_command(
            label="アニメ (既定)",
            command=lambda: self._upscale_gallery_item(path, "anime"),
        )
        upscale_menu.add_command(
            label="実写",
            command=lambda: self._upscale_gallery_item(path, "real"),
        )
        menu.add_cascade(label="アップスケール (Real-ESRGAN x4)", menu=upscale_menu)

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
        self.status_var.set(f"削除: {path.name}")

    def _upscale_gallery_item(self, path: Path, style: str = DEFAULT_UPSCALE_STYLE) -> None:
        if not path.exists():
            messagebox.showerror("エラー", f"{path.name} が見つかりません")
            return
        # 別スレッドで ComfyUI に upscale 専用 workflow を投入
        threading.Thread(target=self._upscale_worker, args=(path, style), daemon=True).start()
        self.status_var.set(f"アップスケール投入 ({style}): {path.name}")

    def _upscale_worker(self, path: Path, style: str = DEFAULT_UPSCALE_STYLE) -> None:
        model_name = UPSCALE_MODELS.get(style, UPSCALE_MODELS[DEFAULT_UPSCALE_STYLE])
        try:
            write_extra_model_paths()
            ensure_comfyui_arch("cuda",
                                  force_restart=not self._server_verified)
            self._server_verified = True
            # 必要なら upscale モデルを公式 release から DL (~17-64 MB)
            self._result_queue.put({"status": f"アップスケールモデル確認中: {model_name}"})
            resolved = ensure_upscale_model(model_name)
            if not resolved:
                self._result_queue.put({"error": f"アップスケールモデル {model_name} を取得できません"})
                return
            model_name = resolved
            uploaded = upload_image_to_comfyui(path)
            workflow = {
                "1": {"class_type": "LoadImage", "inputs": {"image": uploaded}},
                "2": {"class_type": "UpscaleModelLoader",
                      "inputs": {"model_name": model_name}},
                "3": {"class_type": "ImageUpscaleWithModel",
                      "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
                "7": {"class_type": "SaveImage",
                      "inputs": {"images": ["3", 0], "filename_prefix": f"gui_upscale_{style}"}},
            }
            img_bytes, _info, _ = _submit_and_fetch(workflow, self._client_id)
        except Exception as e:
            self._result_queue.put({"error": f"アップスケール失敗 ({path.name}): {e}"})
            return
        if img_bytes is None:
            self._result_queue.put({"status": f"アップスケール結果未取得: {path.name}"})
            return
        UPSCALED_DIR.mkdir(exist_ok=True)
        out_path = UPSCALED_DIR / path.name
        try:
            out_path.write_bytes(img_bytes)
        except Exception as e:
            self._result_queue.put({"error": f"アップスケール保存失敗: {e}"})
            return
        self._result_queue.put({"image": out_path, "iter": 0, "total": 0})
        self._result_queue.put({"status": f"アップスケール完了: {out_path.name}"})

    def _on_gallery_wheel(self, event) -> str:
        """ホイール delta±120/notch をスクロール単位 (units) に変換して Canvas を縦スクロール。"""
        self.gallery_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _show_modal(self, path: Path) -> None:
        if Image is None:
            return
        top = tk.Toplevel(self.root)
        top.title(path.name)
        try:
            img = Image.open(path)
        except Exception as e:
            tk.Label(top, text=f"open error: {e}").pack()
            return
        sw = self.root.winfo_screenwidth() - 80
        sh = self.root.winfo_screenheight() - 140
        img.thumbnail((sw, sh))
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(top, image=photo)
        lbl.image = photo
        lbl.pack()
        top.bind("<Escape>", lambda e: top.destroy())
        top.bind("<Button-1>", lambda e: top.destroy())
        top.focus_set()


def main() -> None:
    root = tk.Tk()
    GenerateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
