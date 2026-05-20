# ComfyUI SDXL Portrait Image Generation Environment

A ComfyUI-based portrait image generation environment, built on SDXL.

[日本語はこちら](README_ja.md)

GeForce/Quadro 16GB VRAM recommended.
Works on CPU + 16GB RAM as well (slower).

## Setup

### Windows

``` powershell
cd ~
git clone https://github.com/AZO234/comfyui_playground
python -m venv .venv
.\.venv\Scripts\activate
# Install PyTorch matching your environment first (pick one)
#pip install --index-url https://download.pytorch.org/whl/cpu  torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# Remaining dependencies
pip install -r requirements.txt
```

### Linux/macOS

``` bash
$ cd ~
$ git clone https://github.com/AZO234/comfyui_playground
$ python -m venv .venv
# Install PyTorch matching your environment first (pick one)
#$ pip install --index-url https://download.pytorch.org/whl/cpu  torch torchvision
$ pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# Remaining dependencies
$ pip install -r requirements.txt
```

## Directory Layout

- `./` : Scripts and configuration files
- `1_prompts` : Place PNG images (with embedded metadata) used as prompts
- `2_tensors` : Drop unclassified tensors here
- `2_1_checkpoint` : Checkpoint tensors
- `2_2_LoRA` : LoRA tensors
- `2_3_Embedding` : Embedding tensors
- `2_4_ControlNet` : ControlNet tensors
- `2_8_SD15` : Lower-tier tensors such as SD15
- `2_9_error` : Invalid tensors
- `3_1_generated` : Generated PNGs (with metadata) are written here
- `3_2_upscaled` : Upscaled PNGs (with metadata) are written here

## Quickstart

```
python generate.py --sentence "a girl walking with umbrella in outside"
```

Running the command above produces images in the `3_1_generated` and `3_2_upscaled` directories.


## Prompt Config: prompt.toml

Defines the keywords used to build prompts.
Emphasis syntax is supported: `*word*` (1.1x), `**word**` (1.3x), `***word***` (1.5x).

- `who` — Who is in the scene
  - The notation "someone wearing X doing Y at Z" is also valid (later fields can be omitted)

`["**a girl**", false, false, false, false, ""],`
`["**a school wear girl**", true, false, false, false, "school wear"],`
`["**a school wear girl running**", true, true, false, false, "school wear"],`
`["**2 girls kissing**", true, true, false, true, "kiss"],`

  - The first three booleans flag whether the entry already contains the wearing / doing / location (true) or should draw them from the section below (false)
  - The fourth boolean (`many`) marks a multi-person entry (e.g. "2 women"). When true, the image is generated on a landscape canvas (`--many-width` × `--many-height`, default 1216×832) to reduce fusion between figures
  - Backward compatible: a 5-element entry without `many` (a string at index 4) is treated as `many=false`
  - LoRA keywords (last field) are described later

- `wearing` — What is being worn

Converted to `wearing X`. (`""` or `"nothing"` becomes `naked`.)

`["dress", 10, "dress"],`

  - Each entry has a weight (weight / total) and an optional LoRA keyword.

- `with_items` — Accessories or contextual items

Converted to `with X`. (`""` or `"nothing"` becomes empty.)

`["earring", 10, "jewel"],`

  - Each entry has a weight and an optional LoRA keyword.

- `motion` — Actions or poses

`["sitting", 50, ""],`
`["standing", 50, ""],`

  - Each entry has a weight and an optional LoRA keyword.


## Scripts

### PNG Prompt Utility: pngutil.py

```
python pngutil.py <PNG file>
```

- Default (no flag)
Inspect the prompt text and LoRA keywords (described below) embedded in the PNG.

- `--sentence`

Replace the embedded prompt text.

- `--lora`

Replace the embedded LoRA keywords.

- `--erase`

Strip all text metadata from the PNG.

#### Recommended PNG metadata viewer

[stable-diffusion-prompt-reader](https://github.com/receyuki/stable-diffusion-prompt-reader) works well.

### Tensor Triage and Check: tensors.py

```
python tensors.py
```

Sorts the tensors in `2_tensors` into the appropriate directories.
A hash check is performed; when an identical tensor already exists, the newest one is kept and the older copy is removed.
Hash information is written to `tensors.toml`. New LoRA entries are appended to `LoRA_keywords.toml` with `keyword` initialised to `""`.

`tensors.toml` groups entries by tensor filename:
- `hash` : Hash value

`LoRA_keywords.toml` groups entries by LoRA filename:
- `keyword` : LoRA keyword

### Generation: generate.py

```
python generate.py
```

Generates images continuously.
`tensors.py` is run first.

PNGs with embedded metadata, named `YYYYMMDDHHMMSS.png`, are written to `3_1_generated` and `3_2_upscaled`.

After each image, there is a cooldown interval of (device temperature − 50) seconds.

If the checkpoint used is not listed in `checkpoint.toml`, an entry is appended (see below).


Press Ctrl+C to stop.

#### Prompt source

`--prompt auto` : Build the prompt text and LoRA keywords from `prompt.toml`. No reference image.
`--prompt sentence <"prompt"> ["LoRA keyword", ...]` : Use the supplied text and LoRA keywords (see below). No reference image.
`--prompt png <PNG file>` : Read prompt text and LoRA keywords from a metadata-embedded PNG.
`--prompt original <PNG file>` : Reuse the checkpoint, LoRAs, and prompt text from a metadata-embedded PNG.

`--prompt auto` is the default.

##### Checkpoint selection

`--checkpoint <checkpoint name>` : Pin a specific checkpoint.

By default,
the first run picks a checkpoint **not** listed in `checkpoint.toml` at random,
and subsequent runs pick from the listed checkpoints with probability 2/3 and from the unlisted ones with probability 1/3.

For entries in `checkpoint.toml`, the weight is
((`slow` max in `checkpoint.toml` × 2) − (`fast` + `slow`)) / 2 + `like`.

##### checkpoint.toml

`checkpoint.toml` stores supplementary information per checkpoint:
- `slow` : Maximum observed generation time per image (s)
- `fast` : Minimum observed generation time per image (s)
- `like` : User preference, positive or negative
- `inference` : Extra inference steps to add, positive or negative
- `style` : `"anime"`, `"real"`, `"mix"`, or `""`

When `--gear high` finishes generating an image, if the checkpoint is not yet listed,
an entry is added with `slow` = `fast` = measured time (s), `like = 0`, `inference = 0`, `style = ""`.

##### LoRA keywords and selection

LoRA keywords are a playground-specific convention.
They are a word list used purely to pick LoRAs, separate from the prompt text.

Matching is case-insensitive: whitespace = AND, commas = OR.

Decide the number of LoRAs (default 1–3) →
pick LoRA keywords (no duplicates) →
with 90% probability match the keyword against each LoRA's filename, embedded metadata, and the `keyword` field in `LoRA_keywords.toml`, &
with 10% probability pick from all LoRAs + "none".

Use `lora_chance_ui.py` to inspect which LoRAs are likely to be picked for a given prompt or keyword set.

LoRA keywords are also appended to the prompt text with weight `(0.8 / number_of_keywords) × keyword`.

#### Gear

`--gear low` : Rough generation. 30 inference steps, no ADetailer, no LoRA, no ControlNet, no upscaling.
`--gear high` : Production generation. 50 inference steps, ADetailer on, LoRA selection, ControlNet selection, upscaling on.

`--gear high` is the default.

#### Architecture

`--arch cuda` : Use CUDA + VRAM
`--arch cpu` : Use CPU + RAM

`--arch cuda` is the default.

#### LoRA

`--lora prompt` : Use LoRAs picked from the prompt
`--lora manual <lora name, ...>` : Pin specific LoRAs
`--lora keyword <word, ...>` : Pick LoRAs based on the given keywords

`--lora prompt` is the default.

#### ControlNet
`--controlnet <controlnet name>` : Pin a specific ControlNet.

By default the ControlNet is chosen based on the `style` field in `checkpoint.toml`:
- `anime` → ControlNets whose filename contains `anime`
- `real` → ControlNets whose filename contains `real`

One is picked at random from the matching set.
For `mix` or empty, the pick is uniformly random.

The source-image routing is also selected correctly from the ControlNet filename
(e.g. `canny` → canny edge extraction, `tile` → pass the source image through unchanged).

### LoRA Selection Probability: lora_chance_ui.py

```
python lora_chance_ui.py
```

Runs 300 draws and prints the Top 30 LoRA selection probabilities as a bar chart.

- `random` : Each draw picks `prompt.toml` entries randomly
- `manual` : Each draw uses entries chosen interactively from `prompt.toml`
- `lora_keyword` : Each draw uses the LoRA keywords you type

## Auto-placement of face_yolov8n.pt, hand_yolov8n.pt, person_yolov8n-seg.pt

The following detection models are downloaded and placed automatically if missing:
- `ComfyUI/models/ultralytics/bbox/face_yolov8n.pt`
- `ComfyUI/models/ultralytics/bbox/hand_yolov8n.pt`
- `ComfyUI/models/ultralytics/segm/person_yolov8n-seg.pt`

## License

GPL-3.0
