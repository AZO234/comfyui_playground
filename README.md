# ComfyUI SDXL Portrait Image Generation Environment

A ComfyUI-based portrait image generation environment.

[日本語はこちら](README_ja.md)

GeForce/Quadro 16GB VRAM recommended.
Works on CPU + 16GB RAM as well (slower).

## Console language

Console output (logs, progress, `--help`) is available in English and Japanese.
The language is chosen from the `PLAYGROUND_LANG` environment variable (`en` / `ja`);
if unset, it is auto-detected from the OS locale (Japanese locale → `ja`, otherwise `en`).

``` powershell
$env:PLAYGROUND_LANG = "en"   # force English
$env:PLAYGROUND_LANG = "ja"   # force Japanese
```

Image metadata (the PNG `parameters` chunk, e.g. the `Pipeline:` field) is always written in English regardless of this setting.

## Setup

### Windows

``` powershell
cd ~
git clone https://github.com/AZO234/comfyui_playground
python -m venv .venv
.\.venv\Scripts\Activate.ps1
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
$ source .venv/bin/activate
# Install PyTorch matching your environment first (pick one)
#$ pip install --index-url https://download.pytorch.org/whl/cpu  torch torchvision
$ pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
# Remaining dependencies
$ pip install -r requirements.txt
```

## Directory Layout

- `./` : Scripts and configuration files
- `1_0_prompts` : Place PNG images (with embedded metadata) used as prompts
- `2_0_tensors` : Drop unclassified tensors here (intake tray)
- `2_1_errortensors` : Invalid / broken / duplicate tensors
- `2_2_lowtensors` : Manual quarantine for tensors you judge bad by eye (no color, buggy output, etc.) — excluded from both scanning and generation
- `2_3_hightensors` : Manual quarantine for tensors out of scope — different architecture (AuraFlow / Flux / SD3 etc.) or simply too heavy for the current SDXL/SD15 pipeline — excluded from both scanning and generation
- `3_1_SD15_checkpoint` : SD15 checkpoint tensors (rough / high-volume lane)
- `3_2_SD15_LoRA` : SD15 LoRA tensors
- `3_3_SD15_Embedding` : SD15 embedding tensors
- `3_9_SD15_rough` : SD15 drafts from the two-stage chain (rough output before the clean pass) are written here
- `4_1_SDXL_checkpoint` : SDXL checkpoint tensors (production lane)
- `4_2_SDXL_LoRA` : SDXL LoRA tensors
- `4_3_SDXL_ControlNet` : SDXL ControlNet tensors
- `4_4_SDXL_Embedding` : SDXL embedding tensors
- `5_1_generated` : Generated PNGs (with metadata) are written here
- `5_2_upscaled` : Upscaled PNGs (with metadata) are written here

Tensors are auto-sorted by `tensors.py` into the SD15 (`3_x`) or SDXL (`4_x`) lane
based on the model architecture detected from each file.

## Quickstart

1. Put checkpoint tensors into `2_0_tensors`.

2. Run the tensor-sorting script `tensors.py`.

```
python tensors.py
```

3. Generate images.

```
python generate.py --sentence "a girl walking with umbrella in outside"
```

Standard images are written to `5_1_generated`,
upscaled images to `5_2_upscaled`.


## Generation Flow (Overview)

A single image from `generate.py` roughly follows the flow below (terms are detailed later).

```mermaid
flowchart TD
  P["Get prompt<br/>--prompt auto / sentence / png / original"]
  P --> CK["Unified checkpoint draw<br/>pool = --version (auto:3_1+4_1 / sd15 / sdxl)<br/>weight = checkpoint.toml (slow/fast/like)"]
  CK --> V{"Version of the drawn checkpoint<br/>(its directory)"}

  V -->|"SD15 / gear high"| DR1
  V -->|"SD15 / gear low"| S1["SD15 single pass / rough (raw)"]
  V -->|"SDXL"| S2["SDXL single pass"]

  subgraph DR["Two-stage chain (draft → clean)"]
    direction TB
    DR1["1. Draft: SD15 ckpt+LoRA<br/>512 / no ADetailer/CN/upscale"]
    DR1 --> DR2["2. Clean: draw an SDXL separately<br/>img2img from the draft<br/>1024 / denoise 0.45"]
  end

  DR2 --> FAM
  S2 --> FAM
  FAM{"family?<br/>checkpoint.toml / filename"}
  FAM -->|Pony| FP["auto-prepend score_9…<br/>allow Pony neg embeds"]
  FAM -->|non-Pony| FN["no score prefix<br/>drop Pony neg embeds"]

  FP --> HG{"gear high?"}
  FN --> HG
  HG -->|high| AD["ADetailer<br/>person→face→hand→NSFW parts<br/>→ upscale x4"]
  HG -->|low| OUT
  AD --> OUT["Output 5_1_generated / 5_2_upscaled<br/>meta: Pipeline / Draft model / Draft loras"]
  S1 --> OUT
```

In words, the key points are:

- **Checkpoints are drawn from a single pool**, and **the drawn version (SD15/SDXL) together with `--gear` decides the route.**
- Only when an **SD15 checkpoint is drawn with `--gear high`** does the **two-stage chain** (SD15 draft → SDXL clean) kick in. Everything else is a single pass.
- **In the SDXL stage the `family` is inspected**: Pony gets a score prefix + Pony neg embeds; non-Pony drops them.
- **`--gear high`** runs ADetailer (person→face→hand→parts) and upscaling.

(This diagram renders as-is on GitHub. It is Mermaid, so just edit this block when the behavior changes.)

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

Sorts the tensors in `2_0_tensors` into the appropriate directories.
A hash check is performed; when an identical tensor already exists, the newest one is kept and the older copy is removed.
Hash information is written to `tensors.toml`. New LoRA entries are appended to `LoRA_keywords.toml` with `keyword` initialised to `""`.

`tensors.toml` groups entries by tensor filename:
- `hash` : Hash value

`LoRA_keywords.toml` groups entries by LoRA filename:
- `keyword` : LoRA keyword

`SDXL_LoRA.toml` records the subject of each SDXL LoRA (`4_2_SDXL_LoRA`); `tensors.py` appends entries automatically with `subject` initialised to empty.
- `subject` : `object` / `accessory` / `ware` / `facial` / `pose`, etc., filled in by you.
  - Only **`subject="pose"`** is functionally meaningful. **When OpenPose is in use, pose LoRAs are automatically excluded** (letting OpenPose and a LoRA fight over the pose breaks the image).
  - The `# hint:` at the end of each line is an auto-generated hint derived from the filename (a clue for "is this an object? an accessory?"); it is regenerated every time and need not be edited. Your `subject` values are preserved.

### Generation: generate.py

```
python generate.py
```

Generates images continuously.
`tensors.py` is run first.

PNGs with embedded metadata, named `YYYYMMDDHHMMSS.png`, are written to `5_1_generated` and `5_2_upscaled`.

After each image, there is a cooldown interval of (device temperature − 50) seconds.

If the checkpoint used is not listed in `checkpoint.toml`, an entry is appended (see below).


Press Ctrl+C to stop.

#### Prompt (input source)

The input source determines the mode automatically (you do not need to pass `--prompt` explicitly).

- No flag (`--prompt auto`, default) : Build prompt text and LoRA keywords from `prompt.toml` and generate continuously. No reference image.
- `--sentence "<text>" [--lora-keywords "kw,..."]` : Generate continuously from the supplied text + LoRA keywords (see below). No reference image.
- `--png <PNG file>` : **Quality-up refine.** Repaint the *image* of that PNG with SDXL img2img, lifting it from SD15-grade toward SDXL-grade quality (**stops after one image**, see below).
- `--png-sentence <PNG file>` : Generate continuously from the *prompt text* embedded in the PNG (the image itself is not used). The unified draw routes SD15 → two-stage or SDXL → single pass.
- `--prompt original --png-sentence <PNG file>` : Reuse the checkpoint, LoRAs, and prompt text from the PNG metadata.

**Note (behavior change):** the old `--png` meant "read text from a PNG and generate anew", but the roles have changed to
**`--png` = image refine / `--png-sentence` = text generation**. For the text use case, use `--png-sentence`.

##### Quality-up refine (`--png`)

A one-shot mode that lifts older SD15 images (and the like) toward SDXL-grade quality.

- The given PNG's image is used as the init for **SDXL img2img** repainting. The checkpoint is drawn from the SDXL-only pool (or `--checkpoint`), with family handling (Pony → score prefix / Pony neg).
- LoRAs are drawn from the SDXL LoRAs using the PNG's LoRA keywords (or `--lora-keywords`).
- `--refine-denoise <0.0–1.0>` : img2img denoise (default 0.5; low = faithful to the original / high = SDXL repaints more aggressively).
- The resolution is scaled to roughly 1MP (SDXL-native) while preserving the original PNG's aspect ratio. `--gear high` adds ADetailer + upscaling; `--gear low` does refine only.
- Output goes to `5_1_generated` / `5_2_upscaled`, and the `Pipeline` metadata records `refine (src:…, denoise…)`. **Generates one image and exits.**

##### Checkpoint selection

`--checkpoint <checkpoint name>` : Pin a specific checkpoint.

The draw pool is determined by `--version` (see below). By default (`auto`) the draw is from a single pool that
merges SD15 + SDXL, and the drawn version automatically decides single-stage vs. two-stage.

Within the pool,
the first run picks a checkpoint **not** in `checkpoint.toml` at random,
and subsequent runs pick from the listed checkpoints (see below) with probability 2/3 and from the unlisted ones with 1/3.

For listed checkpoints, the weight is
((max `slow` in `checkpoint.toml` × 2) − (`fast` + `slow`)) / 2 + `like`.

##### checkpoint.toml

`checkpoint.toml` stores supplementary information per checkpoint:
- `slow` : Maximum observed generation time per image (s)
- `fast` : Minimum observed generation time per image (s)
- `like` : User preference, positive or negative
- `inference` : Extra inference steps to add, positive or negative
- `style` : `"anime"`, `"real"`, `"mix"`, or `""` (used to pick the ControlNet / upscaling model)
- `family` : Lineage `"pony"` / `"illustrious"` / `"real"` / `""` (used for Pony handling; see "Lineage-aware handling" below)

When `--gear high` finishes generating an image, if the checkpoint is not yet listed,
an entry is added with `slow` = `fast` = measured time (s), `like = 0`, `inference = 0`, `style = ""`,
and `family =` a guess from the filename (pony/pdxl/pny → pony, etc.; `""` when undecidable).
`family` can be wrong, so fix it by eye (Illustrious-family models in particular are hard to tell from the filename).

##### Lineage-aware handling (family-aware neg / score)

Because lineage (Pony / Illustrious) cannot be told from metadata, it is held in the `family` field of `checkpoint.toml`.
Based on the drawn checkpoint's `family` (or its filename when absent), the SDXL stage automatically:

- **Pony** → prepends `score_9, score_8_up, …` to the front of the positive prompt, and allows Pony neg embeddings (**up to 3**, to curb over-negation).
- **non-Pony** → **automatically excludes** Pony-only neg embeddings (names containing `pony` / `pdxl` / `pny`), because applying them to non-Pony models breaks colors and suppresses the subject. General-purpose neg embeddings are used regardless of lineage.
- No Pony-related additions are made to the SD15 draft stage.

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

`--gear low` : Rough generation. Single pass, 30 inference steps, no ADetailer, no LoRA, no ControlNet, no upscaling. Even if the drawn checkpoint is SD15, it stays a single pass (for draft inspection).
`--gear high` : Production generation. 50 inference steps, ADetailer on, LoRA selection, ControlNet selection, upscaling on. **It branches in two by the version of the drawn checkpoint:**
- **SDXL drawn** → finish as a straight SDXL single pass.
- **SD15 drawn** → draw a draft in SD15, then have SDXL repaint it via img2img — a two-stage process (see "Two-stage chain" below).

`--gear high` is the default.

You do not need to think about SD15 vs. SDXL; just choose "rough (low)" or "finish (high)".
If SD15 is drawn during a finish, it automatically becomes "SD15 draft → SDXL clean".

#### Architecture

`--arch cuda` : Use CUDA + VRAM
`--arch cpu` : Use CPU + RAM

`--arch cuda` is the default.

#### Version (checkpoint draw pool)

`--version` only narrows the checkpoint draw pool; it does not fix the generation lane.
The version of the drawn checkpoint (its directory) automatically decides single-stage vs. two-stage.

`--version auto` : Draw from a single pool merging SD15 (`3_1`) + SDXL (`4_1`), weighted by `checkpoint.toml` (default).
`--version sdxl` : Draw only `4_1` SDXL.
`--version sd15` : Draw only `3_1` SD15.

`--version auto` is the default.

The merged-pool weights use the `checkpoint.toml` values as-is, without distinguishing SD15 / SDXL.
SD15 generates fast and tends to get a higher weight, so use `like` to adjust manually if you want to reduce the bias.

##### Two-stage chain (SD15 draft → SDXL clean)

A two-stage pipeline triggered when an **SD15 checkpoint is drawn** under `--gear high`.
It mass-produces roughs with SD15's vast LoRA assets and finishes them with SDXL's rendering power — a "draft → clean" workflow.

1. **Draft:** generate with the drawn SD15 checkpoint + SD15 LoRAs (512, a raw rough with no ADetailer / ControlNet / upscaling).
2. **Clean:** hand the draft to ComfyUI and repaint it via img2img with a separately drawn **SDXL** checkpoint + SDXL LoRAs (1024, with ADetailer / upscaling).

`--chain-denoise <0.0–1.0>` : denoise of the clean img2img (default 0.45). Lower preserves the draft's composition and colors; higher lets SDXL repaint more.
`--save-draft` / `--no-save-draft` : Save the intermediate SD15 draft to `3_9_SD15_rough` (**default ON**; `--no-save-draft` to turn it off).

Resolution defaults follow the version (SD15 = 512 / SDXL = 1024; landscape `many` is SD15 = 768×512 / SDXL = 1216×832).
Override with `--width` / `--height` / `--many-width` / `--many-height`.

Which route was taken is visible in the `path:` line of the run log and the `Pipeline:` field of the output PNG metadata
(e.g. `Pipeline: SD15→SDXL 2-stage (draft: … / clean: …, denoise 0.45)` for the chain, `Pipeline: SDXL single-pass` for a single pass).

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

#### Visualizing the workflow (debug)

Dump the ComfyUI API workflow (node graph) that `generate.py` builds to a JSON file.
**Drag and drop** the resulting JSON onto the blank canvas of the ComfyUI WebUI (http://127.0.0.1:8188)
and it expands into a node graph with automatic layout, letting you see the workflow the code actually built.

`--dump-workflow` : Also save each submitted workflow to `workflow_dump/<timestamp>_<kind>.json` (**generation still runs as usual**). In the two-stage chain both the draft (`draft_sd15`) and the clean pass (`chain_clean`) are written; for refine, `refine` is written too.
`--dump-only` : Only dump the workflow JSON; **do not** submit it to ComfyUI (inspect the graph without using the GPU). Writes one single-pass workflow and exits immediately (chain / refine are disabled).

```
# Dump one SDXL single-pass graph without using the GPU
python generate.py --dump-only --version sdxl
# Adding --gear low drops ADetailer/upscale, giving the minimal graph (txt2img + LoRA) that is easiest to read
```

Kinds are `sdxl_single` / `sd15_single` / `chain_clean` / `draft_sd15` / `refine`. `workflow_dump/` is gitignored.

### Tensor Info Viewer: tensors_view.py

```
python tensors_view.py [--dir <directory>] [--list]
```

A Tkinter viewer that reads `safetensors` headers raw and lists metadata, tensor counts, dtypes, and file size. No torch dependency (header scanning only).

- When `--dir` is omitted, the first directory in `[tensors_dirs].list` of `preview_settings.toml` is opened.
- The directory combo on the toolbar switches between the candidates from `[tensors_dirs].list` and reloads immediately.
- Search / kind / base filters and sorting are available. The "+meta" toggle extends search to metadata text.
- If a sidecar preview (`<name>.preview.png`, produced by `make_previews.py`) exists, it is shown in the top-right pane. Click the preview to open a full-size view (click or Esc to close).
- Selecting a LoRA row opens the "Preview settings" editor on the right: edit the category (`ware` / `doing1-3` / `doingmob` / `object` / `part` / `view` / `place` / `artstyle` / `unknown`, ...) and an optional custom prompt. Multi-selection batch-sets the category. SD15 vs. SDXL routing is detected per row automatically.
- "Regenerate" launches `make_previews.py --files` in a subprocess to re-render sidecars only for the selected tensors.
- "Gen thumbs" batch-renders sidecars for every tensor that does not yet have one.
- Pass `--list` to skip the GUI and print the listing to the terminal instead.

#### Settings file: preview_settings.toml

Shared between the viewer and `make_previews.py`.

- `[tensors_dirs]` : An ordered list (`list = [...]`) of candidate directories the viewer switches between. The first entry is the default when `--dir` is omitted.
- `[LoRA_preview_template]` : The minimal prompt scaffold per LoRA category.
- `[checkpoint_preview_template]` : Scaffold for checkpoint previews (`default` is the generic key; add `pony` and other family keys to switch per family).

#### Sidecar metadata: LoRA_preview.toml

Per-LoRA category and custom-prompt overrides, split between SD15 and SDXL (so that stems that collide between the two versions stay disambiguated):

- `[SD15_categories]` / `[SDXL_categories]` : `stem = "<category>"`
- `[SD15_prompts]` / `[SDXL_prompts]` : `stem = "<custom positive>"` (LoRAs without an entry are rendered with the category scaffold alone)

The audit performed at `tensors.py` startup auto-appends new LoRAs as `ware` and warns about stale entries whose source file no longer exists (manual category edits are preserved — nothing is deleted automatically). Run `python make_previews.py --init-categories` to fully reconcile the file.

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
