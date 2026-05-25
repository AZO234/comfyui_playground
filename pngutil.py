#!/usr/bin/env python3
"""pngutil.py - メタ入り PNG (A1111 互換) の文章プロンプト・LoRA キーワード操作。

playground では「メタ入り PNG = プロンプト標準」という位置付け。
このスクリプトは PNG の text chunk (`parameters` フィールド) を A1111 互換で読み書きする。

A1111 形式の `parameters` text chunk:
    <positive prompt>
    Negative prompt: <negative>
    Steps: 30, Sampler: dpmpp_2m_karras, CFG scale: 7, Seed: 12345, Size: 1024x1024, ...

LoRA キーワードは playground 独自フィールドとして params 行末に追加:
    Steps: ..., Lora keywords: "school wear, jewel, finger"
A1111 / CivitAI 等の他ツールは未知フィールドを無視するため round-trip 互換。

使い方:
    python pngutil.py <PNG file>                          # 確認 (規定)
    python pngutil.py <PNG file> --sentence "new prompt"  # 文章プロンプトを上書き
    python pngutil.py <PNG file> --lora "kw1, kw2"        # LoRA キーワードを上書き (空文字でクリア)
    python pngutil.py <PNG file> --erase                  # 全 text chunk を削除
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from i18n import L

# Windows console 絵文字落ち防止
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# --------------------------------------------------------------------------- #
# A1111 parameters parser / serializer
# --------------------------------------------------------------------------- #
# params 行のキーは `Key: value` 形式、value にはクォート可能な文字列含む。
# 例: `Steps: 30, Sampler: Euler a, Model: "foo, bar", Lora keywords: "kw1, kw2"`
_PARAMS_KV_RE = re.compile(
    r'(?P<key>[A-Za-z][A-Za-z0-9_ ]*?):\s*'
    r'(?P<val>"[^"]*"|[^,]+)'
)


def parse_a1111_parameters(text: str) -> dict:
    """A1111 `parameters` text chunk を {positive, negative, params} に分解する。

    戻り値: {
        "positive": str,
        "negative": str,
        "params":   {key: value, ...},   # OrderedDict 風 (insertion order)
        "_params_line": str,             # 元の params 行 (round-trip 用)
    }
    """
    if not text:
        return {"positive": "", "negative": "", "params": {}, "_params_line": ""}

    # 「Negative prompt:」で split。無ければ positive のみ
    if "\nNegative prompt:" in text:
        positive_part, rest = text.split("\nNegative prompt:", 1)
        rest = rest.lstrip(" ").lstrip("\n")
    else:
        positive_part = text
        rest = ""

    # rest を行で分けて、最後の「params 行」(= "Steps:" 等で始まる、または `Key: value, ` パターンの行)
    # を探す。それより前は negative。
    negative_part = ""
    params_line = ""
    if rest:
        lines = rest.split("\n")
        # 末尾から params 行を探す (最後の "Key: value" CSV パターンを持つ行)
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx].strip()
            if not line:
                continue
            # params 行の判定: "Steps:" など A1111 標準キーで始まる、または `Key: value,` パターン
            if (re.match(r'^[A-Za-z][A-Za-z0-9_ ]*?:\s*"?[^,]', line)
                    and "," in line):
                params_line = line
                negative_part = "\n".join(lines[:idx]).rstrip()
                break
        if not params_line:
            # params 行不在、rest 全部が negative
            negative_part = rest.rstrip()

    params: dict[str, str] = {}
    if params_line:
        for m in _PARAMS_KV_RE.finditer(params_line):
            key = m.group("key").strip()
            val = m.group("val").strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            params[key] = val

    return {
        "positive": positive_part,
        "negative": negative_part,
        "params": params,
        "_params_line": params_line,
    }


def serialize_a1111_parameters(parsed: dict) -> str:
    """`parse_a1111_parameters` の dict を A1111 形式テキストに戻す。

    params が存在する場合は **必ず `Negative prompt:` 行を入れる** (空でも)。
    パーサが positive / negative / params の境界を「Negative prompt:」で判別するため。
    """
    positive = parsed.get("positive", "") or ""
    negative = parsed.get("negative", "") or ""
    params = parsed.get("params") or {}

    parts = [positive]
    if negative or params:
        parts.append(f"Negative prompt: {negative}")
    if params:
        kv_strs = []
        for key, val in params.items():
            sval = str(val)
            # comma / quote 含む値はクォート
            if "," in sval or '"' in sval:
                sval = '"' + sval.replace('"', '\\"') + '"'
            kv_strs.append(f"{key}: {sval}")
        parts.append(", ".join(kv_strs))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# PNG text chunk I/O
# --------------------------------------------------------------------------- #
def read_text_chunks(png_path: Path) -> dict[str, str]:
    """PNG の全 text chunk を {key: value} で返す。"""
    img = Image.open(png_path)
    # img.info は tEXt / iTXt / zTXt 全部を含む
    chunks = {}
    for k, v in img.info.items():
        if isinstance(v, str):
            chunks[k] = v
    return chunks


def write_text_chunks(png_path: Path, chunks: dict[str, str], out_path: Optional[Path] = None) -> Path:
    """PNG の text chunk を上書きして保存。out_path 省略時は in-place。"""
    img = Image.open(png_path)
    img.load()
    metadata = PngInfo()
    for k, v in chunks.items():
        if v is None:
            continue
        metadata.add_text(k, v)
    target = out_path or png_path
    img.save(str(target), format="PNG", pnginfo=metadata)
    return target


def erase_text_chunks(png_path: Path, out_path: Optional[Path] = None) -> Path:
    """PNG から全 text chunk を削除して保存 (in-place 既定)。"""
    img = Image.open(png_path)
    img.load()
    # 新規 PngInfo を渡さず、メタデータなしで保存
    target = out_path or png_path
    img.save(str(target), format="PNG")  # pnginfo 渡さない = メタ無し
    return target


# --------------------------------------------------------------------------- #
# 表示
# --------------------------------------------------------------------------- #
def show_info(png_path: Path) -> None:
    """PNG 内のメタ情報を全て表示する。"""
    chunks = read_text_chunks(png_path)
    if not chunks:
        print(L("(text chunk なし)", "(no text chunks)"))
        return

    print(f"--- {png_path.name} ---")
    print(f"text chunks: {sorted(chunks.keys())}")
    print()

    # A1111 parameters chunk を主視点で表示
    if "parameters" in chunks:
        parsed = parse_a1111_parameters(chunks["parameters"])
        print("[A1111 parameters]")
        print(f"positive : {parsed['positive']}")
        print(f"negative : {parsed['negative']}")
        if parsed["params"]:
            print("params:")
            for k, v in parsed["params"].items():
                print(f"  {k} = {v}")
        # LoRA キーワード (playground 独自)
        lora_kw = parsed["params"].get("Lora keywords", "")
        if lora_kw:
            print(L(f"\nLoRA キーワード: {lora_kw}", f"\nLoRA keywords: {lora_kw}"))
        print()

    # ComfyUI workflow / prompt chunk (JSON) も表示 (概要のみ)
    for key in ("workflow", "prompt"):
        if key in chunks:
            val = chunks[key]
            preview = val[:200].replace("\n", " ")
            print(f"[ComfyUI {key}] ({len(val)} chars): {preview}...")
            print()

    # その他 chunk
    other = [k for k in chunks if k not in ("parameters", "workflow", "prompt")]
    for key in other:
        val = chunks[key]
        preview = val[:200].replace("\n", " ")
        print(f"[{key}] ({len(val)} chars): {preview}")


# --------------------------------------------------------------------------- #
# 編集
# --------------------------------------------------------------------------- #
def set_sentence(png_path: Path, new_positive: str) -> None:
    """`parameters` chunk の positive (文章プロンプト) 部分を上書き。
    negative / params は維持。`parameters` 自体が無ければ作る。"""
    chunks = read_text_chunks(png_path)
    parsed = parse_a1111_parameters(chunks.get("parameters", ""))
    parsed["positive"] = new_positive
    chunks["parameters"] = serialize_a1111_parameters(parsed)
    write_text_chunks(png_path, chunks)
    print(L(f"OK: 文章プロンプトを更新 ({png_path.name})", f"OK: sentence prompt updated ({png_path.name})"))


def set_lora_keywords(png_path: Path, keywords: str) -> None:
    """`parameters` chunk の params 行に `Lora keywords: "..."` を追加/上書き。
    空文字を渡したらフィールド自体を削除。"""
    chunks = read_text_chunks(png_path)
    parsed = parse_a1111_parameters(chunks.get("parameters", ""))
    if keywords.strip():
        parsed["params"]["Lora keywords"] = keywords.strip()
    else:
        parsed["params"].pop("Lora keywords", None)
    chunks["parameters"] = serialize_a1111_parameters(parsed)
    write_text_chunks(png_path, chunks)
    _action = L("更新", "updated") if keywords.strip() else L("削除", "cleared")
    print(L(f"OK: LoRA キーワードを {_action} ({png_path.name})", f"OK: LoRA keywords {_action} ({png_path.name})"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=L(
        "メタ入り PNG (A1111 互換) の文章プロンプト・LoRA キーワード操作",
        "Inspect/edit sentence prompt and LoRA keywords in A1111-compatible PNG metadata"))
    ap.add_argument("png", type=str, help=L("対象 PNG ファイル", "target PNG file"))
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--sentence", type=str, default=None,
                       help=L("文章プロンプトを上書き", "replace the sentence (positive) prompt"))
    group.add_argument("--lora", type=str, default=None,
                       help=L('LoRA キーワードを上書き (例: "school wear, jewel")。空文字でクリア',
                               'replace LoRA keywords (e.g. "school wear, jewel"); empty string to clear'))
    group.add_argument("--erase", action="store_true",
                       help=L("全 text chunk を削除", "strip all text chunks"))
    args = ap.parse_args()

    png_path = Path(args.png)
    if not png_path.exists():
        raise SystemExit(L(f"PNG が見つかりません: {png_path}", f"PNG file not found: {png_path}"))
    if png_path.suffix.lower() != ".png":
        raise SystemExit(L(f"PNG ファイルを指定してください: {png_path}", f"Please specify a PNG file: {png_path}"))

    if args.sentence is not None:
        set_sentence(png_path, args.sentence)
    elif args.lora is not None:
        set_lora_keywords(png_path, args.lora)
    elif args.erase:
        erase_text_chunks(png_path)
        print(L(f"OK: 全 text chunk を削除 ({png_path.name})", f"OK: all text chunks stripped ({png_path.name})"))
    else:
        show_info(png_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(L("\n中断", "\nInterrupted"))
        sys.exit(0)
