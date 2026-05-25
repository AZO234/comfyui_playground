#!/usr/bin/env python3
"""i18n.py - コンソール出力の言語切替 (英 / 日)。

依存は標準ライブラリ (os / locale / warnings) のみの軽量モジュール。torch 等の重い
パッケージを読み込まないので、check_env.py のように「重い依存を遅延させたい」スクリプト
からも安全に import できる。common.py はこれを再エクスポートする (from common import L 可)。

言語の決定順:
    1. 環境変数 PLAYGROUND_LANG  (ja / en。jp / japanese / english も受け付ける)
    2. OS ロケール               (日本語環境なら ja)
    3. どちらも取れなければ en

使い方:
    from i18n import L
    print(L("日本語の文章", "English text"))
    # f-string を両方そのまま渡せば文字列補間も両言語で効く:
    print(L(f"{n} 枚生成", f"generated {n} images"))
"""
from __future__ import annotations

import locale
import os
import warnings


def _resolve_lang() -> str:
    v = os.environ.get("PLAYGROUND_LANG", "").strip().lower()
    if v in ("ja", "jp", "ja_jp", "japanese"):
        return "ja"
    if v in ("en", "en_us", "english"):
        return "en"
    # 環境変数が無ければ OS ロケールから推定する
    loc = ""
    try:
        loc = locale.getlocale()[0] or ""
    except Exception:
        loc = ""
    if not loc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # getdefaultlocale は 3.11+ で DeprecationWarning
            try:
                loc = locale.getdefaultlocale()[0] or ""
            except Exception:
                loc = ""
    if not loc:
        loc = (os.environ.get("LANG") or os.environ.get("LC_ALL")
               or os.environ.get("LC_MESSAGES") or "")
    loc = loc.lower()
    return "ja" if (loc.startswith("ja") or "japanese" in loc) else "en"


LANG: str = _resolve_lang()


def L(ja: str, en: str) -> str:
    """LANG=='ja' なら ja を、それ以外は en を返す (コンソール出力の言語切替)。"""
    return ja if LANG == "ja" else en


def set_lang(lang: str) -> str:
    """LANG を明示的に上書きする (主にテスト用)。'ja' / 'en' を受け付け確定値を返す。"""
    global LANG
    LANG = "ja" if str(lang).strip().lower().startswith("ja") else "en"
    return LANG
