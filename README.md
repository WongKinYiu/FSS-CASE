# CASE: Concept-Aware Segmentation from Positive and Negative Visual Exemplars

Official code and benchmark release for **CASE**, including the **CASE benchmark** and **Concept Prompt Adaptation (CPA)** built on SAM 3.

CASE studies segmentation from positive and negative visual exemplars. The benchmark contains six task families spanning category-level and attribute-level compositional visual prompts, and CPA adapts SAM 3 with lightweight LoRA while preserving positive-only behavior.

> **Paper:** publication link will be added here.  
> **CASE_DATA:** dataset download link will be added here.

---

## Overview

CASE contains six benchmark tasks:

| Chapter | Task | Description |
| --- | --- | --- |
| Category | SuperEx | Segment a super-category while excluding a specified category |
| Category | FineIso | Isolate a fine-grained category from related distractors |
| Category | Union | Segment the union of multiple positive categories |
| Attribute | AttrEx | Segment objects while excluding a specified attribute |
| Attribute | AttrIso | Isolate objects with a target attribute from attribute distractors |
| Attribute | AttrUnion | Segment the union defined by a target attribute |

The released runtime metadata contains frozen query masks, visual prompt boxes, fixed 1-shot/5-shot evaluation supports, and training support pools. COCO/PACO annotations are therefore **not required at training or CASE evaluation time**.

Source images are not redistributed by this repository. Users must obtain the corresponding COCO images separately.

---

## Benchmark Statistics

| Task | Train episodes | Eval episodes / shot |
| --- | ---: | ---: |
| SuperEx | 156,000 | 4,502 |
| FineIso | 140,727 | 4,331 |
| Union | 50,000 | 903 |
| AttrEx | 12,909 | 764 |
| AttrIso | 66,942 | 550 |
| AttrUnion | 34,812 | 558 |
| **Total** | **461,390** | **11,608** |

Evaluation uses nested support sets: the official 1-shot support is the first support of the corresponding 5-shot episode.

---

## Repository Structure

```text
CASE/
├── README.md
├── requirements.txt
├── environment.txt
├── data/
│   ├── case_dataset.py
│   └── coco20i_dataset.py
├── model/
│   ├── train_cpa.py
│   ├── eval_case.py
│   ├── eval_coco20i.py
│   └── sam3/
│       └── ...
└── tools/
    └── dataset_builder/
        ├── build_superex.py
        ├── build_fineiso.py
        ├── build_union.py
        ├── build_attrex.py
        ├── build_attriso.py
        ├── build_attrunion.py
        ├── freeze_eval_supports.py
        └── compile_case_data.py
```

---

## Installation

The release was validated with:

- Python 3.12.12
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0
- NVIDIA GPU with BF16 support

Create an environment:

```bash
conda create -n case python=3.12.12 -y
conda activate case
```

Install the tested CUDA 12.8 PyTorch build:

```bash
pip install torch==2.11.0 torchvision==0.26.0   --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### SAM 3 checkpoint access

This repository vendors the SAM 3 implementation used by CASE under `model/sam3/`, but **does not redistribute the SAM 3 model weights**.

Before the first run:

1. Request access to the official `facebook/sam3` model on Hugging Face.
2. Authenticate locally:

```bash
hf auth login
```

The model builder downloads `sam3.pt` from Hugging Face automatically when no local SAM 3 checkpoint is specified.

The vendored SAM 3 implementation remains subject to its license in:

```text
model/sam3/LICENSE
```

---

## CASE_DATA

### Where to place it

After downloading and extracting the benchmark, the simplest local layout is:

```text
CASE/
├── CASE_DATA/
├── data/
├── model/
├── tools/
└── ...
```

`CASE_DATA/` is excluded by `.gitignore`, so it can safely live inside the repository working directory without being committed.

All examples below therefore use:

```bash
--case-data-root ./CASE_DATA
```

### Public CASE_DATA layout

```text
CASE_DATA/
├── manifest.json
├── train/
│   ├── vc_cat/
│   │   ├── SuperEx/episodes.sqlite
│   │   ├── FineIso/episodes.sqlite
│   │   └── Union/episodes.sqlite
│   └── vc_attr/
│       ├── AttrEx/episodes.sqlite
│       ├── AttrIso/episodes.sqlite
│       └── AttrUnion/episodes.sqlite
└── eval/
    ├── 1shot/
    │   ├── vc_cat/<task>/*.json.gz
    │   └── vc_attr/<task>/*.json.gz
    └── 5shot/
        ├── vc_cat/<task>/*.json.gz
        └── vc_attr/<task>/*.json.gz
```

Training metadata is stored in random-access SQLite databases with independently compressed episode payloads. This keeps memory usage approximately constant instead of expanding all 461k training episodes into RAM.

### Source images

CASE metadata stores relative image paths. For the official benchmark these paths resolve against COCO 2017 train/validation images.

A typical image root is:

```text
/path/to/COCO2017/
├── train2017/
└── val2017/
```

Use that directory as:

```bash
--image-root /path/to/COCO2017
```

PACO-based CASE tasks use the corresponding COCO images; PACO annotation files are needed only when reconstructing the benchmark.

---

## Train CPA

CPA trains LoRA adapters on frozen SAM 3.

The paper configuration uses:

- LoRA rank 16
- LoRA alpha 32
- LoRA dropout 0.05
- AdamW, learning rate 1e-4
- weight decay 1e-4
- preservation weight 1.0
- 10 epochs
- 2,000 iterations per epoch
- seed 123
- batch size 1
- BF16 autocast on CUDA

### 1-shot

```bash
python model/train_cpa.py   --case-data-root ./CASE_DATA   --image-root /path/to/COCO2017   --output-dir ./checkpoints/cpa_1shot   --k-shot 1   --num-epochs 10   --epoch-size 2000   --num-workers 2   --seed 123
```

The final PEFT checkpoint is written to:

```text
checkpoints/cpa_1shot/epoch_10/
```

### 5-shot

```bash
python model/train_cpa.py   --case-data-root ./CASE_DATA   --image-root /path/to/COCO2017   --output-dir ./checkpoints/cpa_5shot   --k-shot 5   --num-epochs 10   --epoch-size 2000   --num-workers 2   --seed 123
```

The final PEFT checkpoint is written to:

```text
checkpoints/cpa_5shot/epoch_10/
```

Do not enable `--padding` when reproducing the default paper configuration.

---

## Evaluate on CASE

`model/eval_case.py` supports both positive-negative and positive-only prompting.

### Example: 1-shot SuperEx, positive-negative prompts

```bash
python model/eval_case.py   --checkpoint ./checkpoints/cpa_1shot/epoch_10   --chapter vc_cat   --task SuperEx   --k-shot 1   --prompt-mode posneg   --case-data-root ./CASE_DATA   --image-root /path/to/COCO2017   --output-json ./results/superex_1shot_posneg.json
```

### Positive-only evaluation

Use the same checkpoint with:

```bash
--prompt-mode pos
```

For example:

```bash
python model/eval_case.py   --checkpoint ./checkpoints/cpa_1shot/epoch_10   --chapter vc_cat   --task SuperEx   --k-shot 1   --prompt-mode pos   --case-data-root ./CASE_DATA   --image-root /path/to/COCO2017   --output-json ./results/superex_1shot_pos.json
```

Task/chapter mapping:

```text
vc_cat  : SuperEx, FineIso, Union
vc_attr : AttrEx, AttrIso, AttrUnion
```

For 5-shot evaluation, use the 5-shot CPA checkpoint and `--k-shot 5`.

---

## Evaluate on COCO-20i

COCO-20i evaluation uses COCO 2014 validation images and annotations.

Expected layout:

```text
/path/to/COCO2014/
├── val2014/
└── annotations/
    └── instances_val2014.json
```

### CPA 1-shot

```bash
python model/eval_coco20i.py   --checkpoint ./checkpoints/cpa_1shot/epoch_10   --shot 1   --data-root /path/to/COCO2014   --seed 0   --folds 0 1 2 3   --output-json ./results/coco20i_1shot.json
```

### CPA 5-shot

```bash
python model/eval_coco20i.py   --checkpoint ./checkpoints/cpa_5shot/epoch_10   --shot 5   --data-root /path/to/COCO2014   --seed 0   --folds 0 1 2 3   --output-json ./results/coco20i_5shot.json
```

To evaluate frozen SAM 3, omit `--checkpoint`.

---

## Rebuilding CASE from COCO and PACO

Most users should use the released `CASE_DATA`. The following utilities are provided for benchmark reconstruction and reproducibility.

The reconstruction pipeline is:

```text
COCO / PACO annotations
        │
        ▼
build_<task>.py
        │
        ▼
intermediate CASE metadata
        │
        ▼
freeze_eval_supports.py
        │
        ▼
fixed nested 1-shot / 5-shot evaluation splits
        │
        ▼
compile_case_data.py
        │
        ▼
public CASE_DATA
```

### 1. Build category-level metadata

COCO 2017 training:

```bash
for TASK in superex fineiso union; do
  python "tools/dataset_builder/build_${TASK}.py"     --split train     --annotation-file /path/to/annotations/instances_train2017.json     --output-root ./CASE_METADATA     --seed 2026
done
```

COCO 2017 validation:

```bash
for TASK in superex fineiso union; do
  python "tools/dataset_builder/build_${TASK}.py"     --split val     --annotation-file /path/to/annotations/instances_val2017.json     --output-root ./CASE_METADATA
done
```

### 2. Build attribute-level metadata

PACO training:

```bash
for TASK in attrex attriso attrunion; do
  python "tools/dataset_builder/build_${TASK}.py"     --split train     --annotation-file /path/to/paco_lvis_v1_train.json     --output-root ./CASE_METADATA     --seed 2026
done
```

PACO validation:

```bash
for TASK in attrex attriso attrunion; do
  python "tools/dataset_builder/build_${TASK}.py"     --split val     --annotation-file /path/to/paco_lvis_v1_val.json     --output-root ./CASE_METADATA
done
```

### 3. Freeze the official nested evaluation supports

```bash
python tools/dataset_builder/freeze_eval_supports.py   --metadata-root ./CASE_METADATA   --output-root ./CASE_OFFICIAL_SPLITS   --seed 42
```

The 1-shot and 5-shot sets are nested: 1-shot is obtained from the first item of the same sampled 5-shot support set.

### 4. Compile the public CASE_DATA release

```bash
python tools/dataset_builder/compile_case_data.py   --train-metadata-root ./CASE_METADATA   --eval-splits-root ./CASE_OFFICIAL_SPLITS   --coco-train-ann /path/to/annotations/instances_train2017.json   --coco-val-ann /path/to/annotations/instances_val2017.json   --paco-train-ann /path/to/paco_lvis_v1_train.json   --paco-val-ann /path/to/paco_lvis_v1_val.json   --output-root ./CASE_DATA
```

A successful full build validates the formal train/evaluation episode counts.

The compiler freezes query masks using the exact target-mask semantics used by the submitted CASE experiments. It may report informational differences between historical builder-side `gt_foreground_pixels` values and the runtime mask area; the released `CASE_DATA` rewrites the foreground area from the frozen runtime mask.

---

## Reproducibility Notes

- Evaluation support sampling uses seed 42.
- The official 1-shot support is nested inside the corresponding 5-shot support set.
- Training benchmark construction uses seed 2026.
- CPA training uses seed 123.
- COCO-20i evaluation uses seed 0 by default.
- CASE evaluation masks are frozen as COCO RLE and do not require COCO/PACO annotations at runtime.
- Support prompt boxes are preserved from the benchmark metadata and are not truncated during public release compilation.
- The public training loader samples a CASE task uniformly and then samples an episode uniformly within that task, matching the submitted CPA training protocol.

---

## Data and Third-Party Assets

This repository does not redistribute:

- COCO images
- COCO annotations
- PACO annotations
- SAM 3 model weights

Users are responsible for obtaining these resources from their respective official sources and complying with their terms.

The vendored SAM 3 source under `model/sam3/` is distributed under the SAM license included in `model/sam3/LICENSE`.

---

## Citation

Citation metadata for the CASE paper will be added when the final publication metadata is available.
