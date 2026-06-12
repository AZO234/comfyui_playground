#!/usr/bin/env python3
"""lora_chance_ui.py - LoRA 選択確率の対話的確認 TUI (comfyui_playground 用)。

`pick_n_loras_by_keywords` を N 回 (規定 300) 回して、選ばれた LoRA の Top 30 を
バーグラフ表示する。LoRA 採用バイアスや keyword マッチ範囲のデバッグに使う。

機能 (3 モード):
    - random       : prompt.toml の語句をランダムに選択 (build_prompt を毎回回す)
    - manual       : prompt.toml の各セクションをユーザが対話選択 (矢印+Enter)
    - lora_keyword : LoRA キーワードをカンマ区切りで直接入力

使い方:
    python lora_chance_ui.py
    python lora_chance_ui.py --trials 500 --top 50

依存: questionary (pip install questionary)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Windows console での絵文字 / 全角落ち防止
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import questionary

from common import (
    L,
    build_prompt,
    load_prompt_config,
    pick_n_loras_by_keywords,
)
# generate.py との重複定義を撤廃し、定数/ヘルパは generate.py を正本として import
from generate import (
    SDXL_LORA_DIR as LORA_DIR,
    build_lora_corpus_for_playground,
    load_lora_keywords_toml,
)


# --------------------------------------------------------------------------- #
# prompt.toml の who エントリから kw を取り出すヘルパ (v01/v02/v03 後方互換)
# --------------------------------------------------------------------------- #
def get_entry_keywords(entry: list, section: str) -> list[str]:
    """エントリの kw フィールドを atomic kw list で返す。

    who: [char, has_wearing, has_motion?, has_where?, many?, kw?]  (型で判別)
    その他: [value, weight, kw?]
    """
    if section == "who":
        kw_str = ""
        # 6 要素 (v04): [char, has_wearing, has_motion, has_where, many, kw] (index4=bool)
        if len(entry) >= 6 and isinstance(entry[2], bool) and isinstance(entry[3], bool) and isinstance(entry[4], bool):
            kw_str = str(entry[5] or "")
        # 5 要素 (v03): [char, has_wearing, has_motion, has_where, kw] (index4=str)
        elif len(entry) >= 5 and isinstance(entry[2], bool) and isinstance(entry[3], bool):
            kw_str = str(entry[4] or "")
        # 4 要素 (v02): [char, has_wearing, has_motion, kw]
        elif len(entry) >= 4 and isinstance(entry[2], bool):
            kw_str = str(entry[3] or "")
        # 3 要素 (v01): [char, has_motion, kw]
        elif len(entry) >= 3 and isinstance(entry[2], str):
            kw_str = str(entry[2] or "")
    else:
        # wearing / with_items / motion: [value, weight, kw?]
        kw_str = str(entry[2] or "") if len(entry) >= 3 else ""

    return [k.strip() for k in kw_str.split(",") if k.strip()]


# --------------------------------------------------------------------------- #
# 抽選試行
# --------------------------------------------------------------------------- #
def run_trials(loras: list[Path], corpus: dict, keywords: list[str],
                n_trials: int, n_max: int = 5, n_min: int = 3) -> Counter:
    counter: Counter = Counter()
    for _ in range(n_trials):
        picked = pick_n_loras_by_keywords(loras, keywords, corpus,
                                            n_max=n_max, n_min=n_min)
        for p in picked:
            counter[p.stem] += 1
    return counter


# --------------------------------------------------------------------------- #
# 結果表示 (Top N バーグラフ)
# --------------------------------------------------------------------------- #
def display_top(counter: Counter, n_top: int, n_trials: int) -> None:
    if not counter:
        print("\n(no picks)")
        return
    total_picks = sum(counter.values())
    max_count = counter.most_common(1)[0][1]
    bar_width = 40
    avg_n = total_picks / n_trials
    print(f"\n=== Top {n_top} of {len(counter)} unique LoRAs ===")
    print(f"trials={n_trials}, picks={total_picks} (avg {avg_n:.2f} LoRA/trial)\n")
    for stem, count in counter.most_common(n_top):
        pct = 100 * count / n_trials
        bar_len = max(1, int(count / max_count * bar_width))
        bar = "#" * bar_len
        print(f"  {count:>4} ({pct:>5.1f}%) {bar:<{bar_width}} {stem}")


# --------------------------------------------------------------------------- #
# モード: random
# --------------------------------------------------------------------------- #
def mode_random(cfg: dict, loras: list[Path], corpus: dict, n_trials: int,
                 n_max: int = 5, n_min: int = 3) -> Counter:
    """build_prompt をランダムに回して、その都度の lora_keywords で抽選を蓄積。"""
    counter: Counter = Counter()
    for _ in range(n_trials):
        _pos, _neg, kws, _many = build_prompt(cfg)
        picked = pick_n_loras_by_keywords(loras, kws, corpus,
                                            n_max=n_max, n_min=n_min)
        for p in picked:
            counter[p.stem] += 1
    return counter


# --------------------------------------------------------------------------- #
# モード: manual
# --------------------------------------------------------------------------- #
def mode_manual(cfg: dict, loras: list[Path], corpus: dict, n_trials: int,
                 n_max: int = 5, n_min: int = 3) -> Counter:
    """各セクションをユーザ選択 → 集めた kw で抽選。"""
    sections = [
        ("who",        cfg.get("who") or []),
        ("wearing",    cfg.get("wearing") or []),
        ("with_items", cfg.get("with_items") or []),
        ("motion",     cfg.get("motion") or []),
    ]
    all_kws: list[str] = []
    selected_summary: list[str] = []
    for sec_name, entries in sections:
        if not entries:
            continue
        labels: list[str] = [L("(このセクションをスキップ)", "(skip this section)")]
        kws_per_label: list[list[str]] = [[]]
        for ent in entries:
            value = str(ent[0])
            kws = get_entry_keywords(ent, sec_name)
            tag = f"  [kw: {', '.join(kws)}]" if kws else L("  [kw 無し]", "  [no kw]")
            labels.append(f"{value}{tag}")
            kws_per_label.append(kws)
        choice = questionary.select(
            L(f"[{sec_name}] を選択 (Enter で次へ):",
              f"[{sec_name}] select (Enter to continue):"),
            choices=labels, default=labels[0],
        ).ask()
        if choice is None:
            return Counter()  # キャンセル
        idx = labels.index(choice)
        if idx > 0:
            all_kws.extend(kws_per_label[idx])
            selected_summary.append(f"[{sec_name}] {entries[idx-1][0]}")
    if selected_summary:
        print(L("\n選択:", "\nSelection:"))
        for s in selected_summary:
            print(f"  {s}")
    print(L(f"  → 集約 kw: {', '.join(all_kws) if all_kws else '(空)'}",
            f"  → aggregated kw: {', '.join(all_kws) if all_kws else '(empty)'}"))
    return run_trials(loras, corpus, all_kws, n_trials=n_trials,
                       n_max=n_max, n_min=n_min)


# --------------------------------------------------------------------------- #
# モード: lora_keyword
# --------------------------------------------------------------------------- #
def mode_lora_keyword(loras: list[Path], corpus: dict, n_trials: int,
                       n_max: int = 5, n_min: int = 3) -> Counter:
    """ユーザ入力 kw 文字列で抽選。"""
    text = questionary.text(
        L("LoRA キーワードをカンマ区切りで入力 (空 = キャンセル):",
          "Enter LoRA keywords comma-separated (empty = cancel):"),
    ).ask()
    if not text:
        return Counter()
    kws = [k.strip() for k in text.split(",") if k.strip()]
    print(L(f"\n入力 kw: {', '.join(kws)}",
            f"\ninput kw: {', '.join(kws)}"))
    return run_trials(loras, corpus, kws, n_trials=n_trials,
                       n_max=n_max, n_min=n_min)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=L("LoRA 選択確率の対話的確認 TUI",
                      "Interactive TUI to inspect LoRA pick probability"))
    ap.add_argument("--trials", type=int, default=300,
                    help=L("抽選試行回数 (既定 300)",
                           "number of draw trials (default 300)"))
    ap.add_argument("--top", type=int, default=30,
                    help=L("バーグラフに表示する上位件数 (既定 30)",
                           "top N entries to show in the bar graph (default 30)"))
    ap.add_argument("--lora-stack-min", type=int, default=3,
                    help=L("1 試行あたりの重ね掛け LoRA 最小数 (既定 3、generate.py と同既定)",
                           "minimum number of stacked LoRAs per trial (default 3, same as generate.py)"))
    ap.add_argument("--lora-stack-max", type=int, default=5,
                    help=L("1 試行あたりの重ね掛け LoRA 最大数 (既定 5、generate.py と同既定)",
                           "maximum number of stacked LoRAs per trial (default 5, same as generate.py)"))
    args = ap.parse_args()

    print(L("LoRA 列挙中...", "Enumerating LoRAs..."), end=" ", flush=True)
    loras = sorted(LORA_DIR.glob("*.safetensors"))
    print(L(f"{len(loras)} 件", f"{len(loras)} found"))
    if not loras:
        raise SystemExit(L(f"{LORA_DIR} に LoRA がありません",
                           f"No LoRAs found in {LORA_DIR}"))

    print(L("LoRA_keywords.toml + corpus 構築...", "Building LoRA_keywords.toml + corpus..."),
          end=" ", flush=True)
    kw_data = load_lora_keywords_toml()
    corpus = build_lora_corpus_for_playground(loras, kw_data)
    print(L(f"({len(kw_data)} 件登録)", f"({len(kw_data)} entries registered)"))

    cfg = load_prompt_config()

    MODES = {
        L("random       (prompt.toml をランダム抽選で 300 回)",
          "random       (draw from prompt.toml randomly, 300 times)"):   "random",
        L("manual       (prompt.toml の各セクションをユーザ選択)",
          "manual       (user selects each section of prompt.toml)"):    "manual",
        L("lora_keyword (LoRA キーワードを直接入力)",
          "lora_keyword (enter LoRA keywords directly)"):                "lora_keyword",
        L("終了", "Quit"):                                               None,
    }

    while True:
        print()
        choice = questionary.select(
            L("モード選択 (↑↓ + Enter):", "Select mode (↑↓ + Enter):"),
            choices=list(MODES.keys()),
        ).ask()
        if choice is None or MODES[choice] is None:
            print(L("終了", "Quit"))
            return
        mode = MODES[choice]
        if mode == "random":
            counter = mode_random(cfg, loras, corpus, n_trials=args.trials,
                                    n_max=args.lora_stack_max, n_min=args.lora_stack_min)
        elif mode == "manual":
            counter = mode_manual(cfg, loras, corpus, n_trials=args.trials,
                                    n_max=args.lora_stack_max, n_min=args.lora_stack_min)
        else:  # lora_keyword
            counter = mode_lora_keyword(loras, corpus, n_trials=args.trials,
                                          n_max=args.lora_stack_max, n_min=args.lora_stack_min)
        display_top(counter, n_top=args.top, n_trials=args.trials)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(L("\n中断", "\nInterrupted"))
        sys.exit(0)
