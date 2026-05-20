# ComfyUI SDXL 人物画像生成環境

ComfyUIの人物画像制作環境です。SDXLベース。

GeForce/Quadro 16GB VRAM推奨。
遅いが、CPU+16GB RAMでも動作する。

## 初期設定

### Windows

``` powershell
cd ~
git clone https://github.com/AZO234/comfyui_playground
python -m venv .venv
.\.venv\Scripts\activate
# PyTorch を環境に合わせて先に入れる（いずれか）
#pip install --index-url https://download.pytorch.org/whl/cpu  torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# 残りの依存
pip install -r requirements.txt
```

### Linux/macOS

``` bash
$ cd ~
$ git clone https://github.com/AZO234/comfyui_playground
$ python -m venv .venv
# PyTorch を環境に合わせて先に入れる（いずれか）
#$ pip install --index-url https://download.pytorch.org/whl/cpu  torch torchvision
$ pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# 残りの依存
$ pip install -r requirements.txt
```

## person_yolov8n-seg.pt 配置

[person_yolov8n-seg.pt](https://huggingface.co/Bingsu/adetailer/blob/main/person_yolov8n-seg.pt) をダウンロードし、
`ComfyUI/models/ultralytics/segs/` ディレクトリに配置。

## ディレクトリ構成

- `./` : スクリプト・設定ファイル
- `1_prompts` ： プロンプトの PNG画像（メタ入り） をユーザが配置する
- `2_tensors` ： 種別不明なテンソルをユーザーが配置する
- `2_1_checkpoint` ： checkpoint テンソル
- `2_2_LoRA` ： LoRA テンソル
- `2_3_Embedding` ： Embedding テンソル
- `2_4_ControlNet` ： ControlNet テンソル
- `2_8_SD15` ： SD15など下位テンソル
- `2_9_error` ： 不正なテンソル
- `3_1_generated` ： 生成された PNG画像（メタ入り） が出力される
- `3_2_upscaled` ： アップスケールされた PNG画像（メタ入り） が出力される

## 一番簡単な使い方

```
python generate.py --sentence "a girl walking with umbrella in outside"
```
と入力すると、
`3_1_generated`・`3_2_upscaled`ディレクトリに画像が生成されます。


## プロンプト設定ファイル prompt.toml

プロンプトを生成するキーワードを記述する。
重み記法あり。 '*～*'（1.1倍）、'**～**'（1.3倍）、'***～***'（1.5倍）。

- who だれが
  - 「どこであれを着た誰かがなにをしている」という記法も可能（後の要素を飛ばせる）

`["**a girl**", false, false, false, ""],`
`["**a school wear girl**", true, false, false, "school wear"],`
`["**a school wear girl running**", true, true, false, "school wear"],`

  - 着ているもの・している事・場所 を、含んでいればtrue、抽選にする場合はfalse
  - LoRAキーワードは後述

- wearing 着ているもの

「wearing ～」に変換される。（""や"nothing"は、"naked"に変換）

`["dress", 10, "dress"],`

  - 抽選の重み（重み/総重み）とLoRAキーワードを記載。

- with_items 装飾品・状況

「with ～」に変換される。（""や"nothing"は、空文字に変換）

`["earring", 10, "jewel"],`

  - 抽選の重みとLoRAキーワードを記載。

- motion している事・動作

`["sitting", 50, ""],`
`["standing", 50, ""],`

  - 抽選の重みとLoRAキーワードを記載。


## スクリプト

### PNGプロンプト ユーティリティ pngutil.py

```
python pngutil.py <PNG file>
```

- 確認
規定。PNG画像ファイル内にある文章プロンプト・LoRAキーワード（後述）を確認。

- `--sentence`

PNG画像ファイル内にある文章プロンプトを変更。

- `--lora`

PNG画像ファイル内にあるLoRAキーワードを変更。

- `--erase`

PNG画像ファイル内のテキスト情報を削除。

#### プロンプト入りPNGビューア

[stable-diffusion-prompt-reader](https://github.com/receyuki/stable-diffusion-prompt-reader) が使いやすい。

### テンソル振り分け＆チェック tensors.py

```
python tensors.py
```

`2_tensors` ディレクトリにあるテンソルを振り分ける。
ハッシュチェックを行い、すでに同内容のテンソルがある場合は、時刻の新しいテンソルを残す。古い方は削除。
格納ハッシュ情報を `tensors.toml` に記載する。`LoRA_keywords.toml` に初期値 `keyword`は`""` として追記。

`tensors.toml` には、テンソルファイル名をグループ名とした値がある。
`hash` ： ハッシュ値

`LoRA_keywords.toml` には、LoRAテンソルファイル名をグループ名とした値がある。
`keyword` ： LoRAキーワード

### 生成 generate.py

```
python generate.py
```

画像を連続で生成する。
生成前に、tensors.py が実行される。

メタ情報入り画像 `YYYYMMDDHHMMSS.png` が `3_1_generated`・`3_2_upscaled` ディレクトリに出力される。  

画像生成後、(デバイス温度-50)秒、冷却インターバル。  

使用したチェックポイントが `checkpoint.toml` に無ければ追記。（後述）


Ctrl+Cで終了。

#### プロンプト

`--prompt auto` ： prompt.toml からプロンプト文章・LoRAキーワードを作成し、画像生成する。読出画像なし
`--prompt sentence <"prompt"> ["LoRA keyword", ...]` ： 文章・LoRAキーワード（後述）で画像生成する。読出画像なし
`--prompt png <PNG file>` ： メタ入りPNG画像 からプロンプト文章・LoRAキーワードを読み出し、画像生成する。
`--prompt original <PNG file>` ： メタ入りPNG画像 のチェックポイント・LoRA・プロンプト文章を使用して、画像生成する。

`--prompt auto` が規定。

##### チェックポイント抽選方式

`--checkpoint <checkpoint name>` ： チェックポイントを指定する。

既定では、
１度めは `checkpoint.toml` にないチェックポイントをランダムに使用し、
２度め以降は、2/3は `checkpoint.toml`（後述） にあるもの、1/3はないものを抽選する。

`checkpoint.toml` にあるものからは、
((`checkpoint.toml`内の`slow`最大時間*2)-(`fast`+`slow`))/2+`like` を重みとして、抽選する。

##### checkpoint.toml

`checkpoint.toml` には、チェックポイントの補足情報を記述する
- `slow` ： １枚の生成にかかった最大時間(s)
- `fast` ： １枚の生成にかかった最小時間(s)
- `like` ： 好み、正負値
- `inference` ： 追加する推論ステップ数、正負値
- `style` ： "anime" or "real" or "mix" or ""

`--gear high`にて、画像が生成されたあと、
チェックポイント名とするグループ名がなければ、
`slow` = `fast` = 実測値(s)、`like = 0`、`inference = 0`、`style = ""`
の内容で追記する。

##### LoRAキーワード・抽選方式

LoRAキーワードは独自規格。
文章とは別に、LoRAを抽選するワード列である。

大小文字の区別なしで、スペース区切りはAND、コンマ区切りはOR、で検索。

LoRA数決定 （規定は1～3） →
LoRAキーワード抽選（重複なし） →
90％の確率で、LoRAキーワードで、LoRAのファイル名・メタ情報・`LoRA_keywords.toml`の`keyword`を検索 ＆
10％の確率で、全LoRA+なし から検索

プロンプトの組み合わせやLoRAキーワードで、どのLoRAが抽選されるかを、
`lora_chance_ui.py` で確認可能。

LoRAキーワードは、(0.8/LoRAキーワード数)*LoRAキーワード、という重みでプロンプト文章に加えられる。

#### ギア

`--gear low` ： ラフ画生成。推論数30・ADetailerなし・LoRA無し・ControlNet無し・アップスケールなし
`--gear high` ： 本番生成。推論数50・ADetailerあり・LoRA抽選・ControlNet抽選・アップスケールあり

`--gear high` が規定。

#### アーキテクチャ

`--arch cuda` ： CUDA+VRAM を使用
`--arch cpu` ： CPU+RAM を使用

`--arch cuda` が規定。

#### LoRA

`--lora prompt` ： プロンプトにあるLoRAを使用
`--lora manual <lora name, ...>` ： LoRAを指定する 
`--lora keyword <word, ...>` ： キーワードを素にLoRAを抽選する

`--lora prompt` が規定。

#### ControlNet
`--controlnet <controlnet name>` ： ControlNetを指定する。

ControlNetは、既定では、`checkpoint.toml` の `style` が、
`anime` であれば ControlNetファイル名にanimeとあるもの、
`real` であれば ControlNetファイル名にrealとあるもの、
の中からランダムに１つ抽選される。
`mix`や空文字の場合はランダムに抽選する。

また、ControlNetのファイル名から、ソース画像ルーティングを正しく選択する。
(canny → canny edge 抽出、tile → 元画像そのまま、等)

### LoRA選択確率確認 lora_chance_ui.py

```
python lora_chance_ui.py
```

300回の抽選で選択されるLoRAの確率Top30を、グラフ表示。

- `random` ： prompt.toml の語句をランダムに選択した場合
- `manual` ： prompt.toml の語句をユーザが選択した場合
- `lora_keyword` ： 入力したlora_keywordの場合

## ライセンス

GPL-3.0
