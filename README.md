# CASE: Concept-Aware Segmentation from Positive and Negative Visual Exemplars

**Accepted at ACCV 2026**

**Paper:** Coming soon
**CASE_DATA:** [GitHub Release v1.0](https://github.com/WongKinYiu/FSS-CASE/releases/tag/v1.0)

This repository contains the official implementation and benchmark release for **CASE: Concept-Aware Segmentation from Positive and Negative Visual Exemplars**.

Few-shot semantic segmentation typically identifies target objects from positive support examples of a single semantic class. However, positive cues alone can be underspecified when the intended target depends on exclusion, fine-grained discrimination, functional grouping, or visual attributes.

We formulate this setting as **Specified Visual Prompt Segmentation (SVPS)**, where the target concept is specified jointly by **positive and negative visual prompts**. In CASE, visual exemplars are support-image regions specified by bounding-box prompts: positive exemplars provide inclusion evidence, while negative exemplars act as intent-defining constraints for exclusion and disambiguation.

To study SVPS, we introduce **CASE**, a benchmark built on COCO and PACO, and **Concept Prompt Adaptation (CPA)**, a conservative LoRA-based adaptation of SAM 3 designed to improve positive-negative prompt interpretation while preserving competitive positive-only few-shot segmentation performance.

## CASE Benchmark

CASE contains six task families covering category-level and attribute-level concept specification.

| Task          | Target concept                                                 | Positive exemplars                                                  | Negative exemplars                                                   |
| ------------- | -------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **SuperEx**   | Supercategory excluding one class, e.g., vehicles except cars  | Included classes from the supercategory, e.g., buses and trains     | The excluded class, e.g., cars                                       |
| **FineIso**   | One fine-grained class within a supercategory, e.g., dogs      | Target class instances, e.g., dogs                                  | Related distractor classes, e.g., cats or sheep                      |
| **Union**     | Classes sharing a common function, e.g., furniture for resting | Functionally related classes, e.g., beds and sofas                  | Related classes with different functions, e.g., chairs or tables     |
| **AttrEx**    | A category excluding one attribute, e.g., non-red cars         | Instances without the excluded attribute, e.g., black or white cars | Instances with the excluded attribute, e.g., red cars                |
| **AttrIso**   | A category with one specified attribute, e.g., red cars        | Target attribute instances, e.g., red cars                          | Same category with other attributes, e.g., blue or white cars        |
| **AttrUnion** | An attribute shared across categories, e.g., wooden objects    | Diverse classes with the attribute, e.g., wooden chairs or bowls    | Similar classes without the attribute, e.g., plastic chairs or bowls |

The category-level tasks (**SuperEx, FineIso, Union**) are constructed from COCO. The attribute-level tasks (**AttrEx, AttrIso, AttrUnion**) are constructed from PACO.

Each CASE episode contains a support set with positive and negative bounding-box prompts, a query image, and a binary ground-truth query mask.

### Benchmark Statistics

| Task        | Source      | Training episodes | Evaluation episodes | Concepts |  Avg. Pos / Neg |
| ----------- | ----------- | ----------------: | ------------------: | -------: | --------------: |
| SuperEx     | COCO        |           156,000 |               9,004 |       78 |     6.71 / 2.31 |
| FineIso     | COCO        |           140,727 |               8,662 |       79 |     2.43 / 6.17 |
| Union       | COCO        |            50,000 |               1,806 |       25 |     5.04 / 3.96 |
| AttrEx      | PACO        |            12,909 |               1,528 |      109 |     2.09 / 1.04 |
| AttrIso     | PACO        |            66,942 |               1,100 |       94 |     1.04 / 2.09 |
| AttrUnion   | PACO        |            34,812 |               1,116 |       21 |     2.15 / 1.64 |
| **Overall** | COCO / PACO |       **461,390** |          **23,216** |  **406** | **3.94 / 3.56** |

The training split contains candidate episodes from which support examples are sampled during training. Evaluation episodes are fixed for deterministic comparison.

CASE provides both **1-shot and 5-shot** evaluation protocols. For each evaluation target, the 5-shot support set is sampled first and the corresponding 1-shot support set uses its first support image. Thus, the 1-shot support set is nested within the 5-shot support set, ensuring that both protocols evaluate the same target episodes while differing only in the amount of support evidence.

## Concept Prompt Adaptation

**Concept Prompt Adaptation (CPA)** adapts SAM 3 to SVPS while keeping the pretrained SAM 3 weights frozen.

Following the shared-canvas formulation of FSS-SAM3, support and query images are composed into a single **1008 × 1008** canvas. Positive and negative support boxes are transformed into the canvas coordinate system and provided through the native SAM 3 box-prompt interface.

CPA inserts LoRA adapters into transformer encoder and decoder layers. Training uses two prompting conditions:

* **Positive-negative condition:** the adapted model receives both positive and negative exemplars and is supervised by the CASE target mask.
* **Positive-only preservation condition:** only positive exemplars are provided, and the adapted model is regularized to remain close to the frozen SAM 3 prediction under the same positive-only input.

The training objective is

```
L = L_svps^(±) + λ_pres L_pres^(+),
```

with `λ_pres = 1.0`.

### CPA Configuration

| Component            | Setting                                                        |
| -------------------- | -------------------------------------------------------------- |
| Base model           | SAM 3                                                          |
| Input canvas         | 1008 × 1008                                                    |
| 1-shot layout        | Vertical support-query split                                   |
| 5-shot layout        | L-shaped support-query canvas                                  |
| Prompt type          | Support-image bounding boxes                                   |
| Trainable modules    | LoRA adapters                                                  |
| Frozen modules       | All pretrained SAM 3 weights                                   |
| LoRA target modules  | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `linear1`, `linear2` |
| LoRA rank            | 16                                                             |
| LoRA scaling factor  | 32                                                             |
| LoRA dropout         | 0.05                                                           |
| Trainable parameters | 1.13M                                                          |
| Optimizer            | AdamW                                                          |
| Learning rate        | 1 × 10^-4                                                      |
| Weight decay         | 1 × 10^-4                                                      |
| Batch size           | 1                                                              |
| Dataloader workers   | 2                                                              |
| Training epochs      | 10                                                             |
| Random seed          | 123                                                            |
| Mixed precision      | bfloat16 autocast on CUDA                                      |
| Gradient clipping    | Maximum norm 1.0                                               |
| Preservation weight  | 1.0                                                            |

For 1-shot episodes, the support image occupies the top half of the canvas and the query image occupies the bottom half.

For 5-shot episodes, CPA uses an L-shaped layout with three support images in the top row, two in the left column, and the query image in the bottom-right region.

## Main Results

### CASE

The main paper reports the following aggregated results for CPA with positive-negative prompts.

| Setting     | Category mIoU | Attribute mIoU | Mean mIoU |
| ----------- | ------------: | -------------: | --------: |
| CPA, 1-shot |         48.63 |          23.04 | **35.83** |
| CPA, 5-shot |         57.15 |          27.14 | **42.15** |

For comparison, frozen FSS-SAM3 with positive-negative prompts obtains **18.62** mean mIoU in 1-shot and **34.14** in 5-shot.

### Conventional Positive-Only FSS

To examine whether adaptation preserves conventional positive-only few-shot segmentation ability, CPA is also evaluated on COCO-20i under the standard positive-only protocol.

| Method               | 1-shot mIoU | 5-shot mIoU |
| -------------------- | ----------: | ----------: |
| FSS-SAM3             |        66.6 |        71.8 |
| CPA w/o Preservation |        50.8 |        66.8 |
| **CPA**              |    **61.6** |    **74.8** |

These results illustrate the role of positive-only preservation: CPA improves positive-negative prompt interpretation on CASE while retaining strong conventional few-shot segmentation performance.

## Repository Structure

```
FSS-CASE/
├── README.md
├── requirements.txt
├── data/
│   ├── case_dataset.py
│   └── coco20i_dataset.py
├── model/
│   ├── train_cpa.py
│   ├── eval_case.py
│   ├── eval_coco20i.py
│   └── sam3/
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

## Installation

Clone the repository:

```
git clone https://github.com/WongKinYiu/FSS-CASE.git
cd FSS-CASE
```

Create a Python environment and install a PyTorch build appropriate for your system. Then install the remaining dependencies:

```
pip install -r requirements.txt
```

### SAM 3 Checkpoint

CPA uses SAM 3 as its base model. The pretrained SAM 3 weights are **not redistributed** in this repository.

Please obtain access to the official SAM 3 checkpoint and authenticate with Hugging Face before running training or evaluation.

The SAM 3 source code required by this project is included under `model/sam3/`. Its original license is provided in `model/sam3/LICENSE`.

## CASE_DATA

CASE_DATA contains the released benchmark metadata used by the public loader.

Download it from:

**[CASE_DATA v1.0](https://github.com/WongKinYiu/FSS-CASE/releases/tag/v1.0)**

The release provides six training archives, one evaluation archive, the benchmark manifest, and SHA256 checksums:

```
CASE_DATA_train_SuperEx.tar.gz
CASE_DATA_train_FineIso.tar.gz
CASE_DATA_train_Union.tar.gz
CASE_DATA_train_AttrEx.tar.gz
CASE_DATA_train_AttrIso.tar.gz
CASE_DATA_train_AttrUnion.tar.gz
CASE_DATA_eval.tar.gz
CASE_DATA_manifest.json
SHA256SUMS
```

CASE_DATA distributes benchmark metadata, visual prompt annotations, fixed evaluation supports, and frozen query masks. Source images are not redistributed and must be obtained from the corresponding source datasets.

After extraction, organize the benchmark as:

```
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
    └── 5shot/
```

Training episodes are stored in task-specific SQLite databases for random-access sampling. Fixed evaluation episodes are distributed separately for the 1-shot and 5-shot protocols.

The released query masks are stored as COCO-style RLE so that evaluation does not require regenerating masks from the original source annotations.

## Training CPA

CPA is trained separately for the requested shot setting.

### 1-shot

```
python model/train_cpa.py \
    --shot 1 \
    --case-data-root ./CASE_DATA \
    --image-root /path/to/source/images \
    --save-dir ./checkpoints/cpa_1shot
```

### 5-shot

```
python model/train_cpa.py \
    --shot 5 \
    --case-data-root ./CASE_DATA \
    --image-root /path/to/source/images \
    --save-dir ./checkpoints/cpa_5shot
```

The default release configuration follows the CPA settings reported in the paper: LoRA rank 16, scaling factor 32, learning rate `1e-4`, weight decay `1e-4`, batch size 1, 10 training epochs, random seed 123, and positive-only preservation weight 1.0.

## Evaluation on CASE

CASE supports both positive-negative and positive-only evaluation.

### Positive-Negative

```
python model/eval_case.py \
    --shot 1 \
    --prompt-mode posneg \
    --checkpoint /path/to/cpa_checkpoint \
    --case-data-root ./CASE_DATA \
    --image-root /path/to/source/images
```

For 5-shot evaluation, use:

```
--shot 5
```

### Positive-Only

```
python model/eval_case.py \
    --shot 1 \
    --prompt-mode pos \
    --checkpoint /path/to/cpa_checkpoint \
    --case-data-root ./CASE_DATA \
    --image-root /path/to/source/images
```

The evaluator reports task-wise mIoU and the equal-weight mean across the six CASE task families.

## Evaluation on COCO-20i

CPA is evaluated on COCO-20i using the standard positive-only FSS protocol. Support masks are converted into tight bounding boxes and used as positive visual prompts; no negative prompts are provided.

Use:

```
python model/eval_coco20i.py --help
```

for the available dataset, fold, shot, and checkpoint arguments.

The checkpoint argument can be omitted when evaluating the frozen SAM 3 baseline.

## Benchmark Reconstruction

The released CASE_DATA files are sufficient for normal CPA training and CASE evaluation.

The scripts under `tools/dataset_builder/` are provided for reproducibility of benchmark construction from the original COCO and PACO annotations.

The reconstruction pipeline consists of:

1. building task-specific candidate training and validation episodes;
2. freezing nested 1-shot and 5-shot evaluation supports;
3. compiling the runtime release representation with frozen RLE target masks.

Task builders share the interface:

```
python tools/dataset_builder/build_superex.py --help
python tools/dataset_builder/build_fineiso.py --help
python tools/dataset_builder/build_union.py --help
python tools/dataset_builder/build_attrex.py --help
python tools/dataset_builder/build_attriso.py --help
python tools/dataset_builder/build_attrunion.py --help
```

Evaluation support freezing and release compilation are handled by:

```
python tools/dataset_builder/freeze_eval_supports.py --help
python tools/dataset_builder/compile_case_data.py --help
```

## Reproducibility Notes

* CASE evaluation episodes are fixed for deterministic comparison.
* The 1-shot support set is nested within the corresponding 5-shot support set.
* Category-level tasks are constructed from COCO, while attribute-level tasks are constructed from PACO.
* Training uses mixed CASE episodes across all six task families.
* Standard FSS evaluation uses positive prompts only.
* CASE evaluation uses the released frozen target masks rather than dynamically regenerating query ground truth.

## Citation

The citation will be added when the ACCV 2026 paper is publicly available.

## Acknowledgements

This project builds on SAM 3 and uses annotations from COCO and PACO. Please follow the original licenses and terms of use of these projects and datasets.
