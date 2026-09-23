#!/usr/bin/env python3
"""
Compile the internal CASE benchmark metadata into the public CASE_DATA format.

The compiler consumes the exact intermediate metadata used by the CASE paper
and produces self-contained benchmark metadata for CPA training/evaluation.

Public CASE_DATA runtime metadata contains:
  - source image paths
  - frozen target masks as COCO RLE
  - positive/negative support prompts
  - fixed evaluation supports
  - dynamic training support pools

COCO/PACO annotation files are required only while compiling CASE_DATA.
They are not required by data/case_dataset.py at runtime.

Output layout
-------------
CASE_DATA/
├── train/
│   ├── vc_cat/
│   │   ├── SuperEx/episodes.sqlite
│   │   ├── FineIso/episodes.sqlite
│   │   └── Union/episodes.sqlite
│   └── vc_attr/
│       ├── AttrEx/episodes.sqlite
│       ├── AttrIso/episodes.sqlite
│       └── AttrUnion/episodes.sqlite
├── eval/
│   ├── 1shot/
│   │   ├── vc_cat/<task>/*.json.gz
│   │   └── vc_attr/<task>/*.json.gz
│   └── 5shot/
│       ├── vc_cat/<task>/*.json.gz
│       └── vc_attr/<task>/*.json.gz
└── manifest.json

Important reproducibility rule
------------------------------
Support prompt boxes are copied exactly from the submitted benchmark metadata.
The compiler does NOT truncate the number of positive/negative prompt boxes.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import shutil
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

try:
    import ijson
except ImportError as exc:
    raise ImportError(
        "compile_case_data.py requires `ijson` for streaming the large "
        "training metadata files.\n"
        "Install it with:\n"
        "  pip install ijson"
    ) from exc


# ======================================================================
# Formal CASE benchmark definition
# ======================================================================

CATEGORY_TASKS = ("SuperEx", "FineIso", "Union")
ATTRIBUTE_TASKS = ("AttrEx", "AttrIso", "AttrUnion")
ALL_TASKS = CATEGORY_TASKS + ATTRIBUTE_TASKS

TASK_TO_CHAPTER = {
    **{task: "vc_cat" for task in CATEGORY_TASKS},
    **{task: "vc_attr" for task in ATTRIBUTE_TASKS},
}

# Formal counts reported for the CASE benchmark.
EXPECTED_TRAIN_COUNTS = {
    "SuperEx": 156_000,
    "FineIso": 140_727,
    "Union": 50_000,
    "AttrEx": 12_909,
    "AttrIso": 66_942,
    "AttrUnion": 34_812,
}

# Evaluation totals reported in the paper include 1-shot + 5-shot.
EXPECTED_EVAL_TOTAL_COUNTS = {
    "SuperEx": 9_004,
    "FineIso": 8_662,
    "Union": 1_806,
    "AttrEx": 1_528,
    "AttrIso": 1_100,
    "AttrUnion": 1_116,
}

EXPECTED_EVAL_PER_SHOT = {
    task: total // 2
    for task, total in EXPECTED_EVAL_TOTAL_COUNTS.items()
}


# ======================================================================
# JSON streaming
# ======================================================================

def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_json_array(path: Path) -> Iterator[dict]:
    """Stream a top-level JSON array without loading the full file into RAM."""
    with open_text(path, "rt") as handle:
        for item in ijson.items(handle, "item", use_float=True):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"Expected JSON objects in {path}, got {type(item).__name__}"
                )
            yield dict(item)


class GzipJsonArrayWriter:
    """Streaming writer for a valid gzip-compressed JSON array."""

    def __init__(self, path: Path, compression_level: int = 6):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(
            path,
            "wt",
            encoding="utf-8",
            compresslevel=compression_level,
        )
        self.handle.write("[")
        self.count = 0
        self.closed = False

    def append(self, obj: dict) -> None:
        if self.closed:
            raise RuntimeError(f"Writer already closed: {self.path}")

        if self.count:
            self.handle.write(",")

        json.dump(
            obj,
            self.handle,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.count += 1

    def close(self) -> None:
        if self.closed:
            return
        self.handle.write("]")
        self.handle.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SQLiteTrainWriter:
    """Random-access CASE training episode store.

    Each row contains one independently zlib-compressed compact JSON episode.
    Integer row IDs are contiguous from 1..N, allowing uniform random episode
    sampling without loading the full training release into memory.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        path: Path,
        task: str,
        compression_level: int = 6,
        commit_every: int = 1000,
    ):
        self.path = path
        self.task = task
        self.compression_level = compression_level
        self.commit_every = commit_every
        self.count = 0
        self.closed = False

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.path.exists():
            self.path.unlink()

        self.connection = sqlite3.connect(
            str(self.path)
        )

        # This database is a generated artifact. NORMAL synchronization keeps
        # compilation reasonably fast while still producing a normal SQLite DB.
        self.connection.execute(
            "PRAGMA synchronous=NORMAL"
        )
        self.connection.execute(
            "PRAGMA temp_store=MEMORY"
        )

        self.connection.execute(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY,
                episode_id TEXT NOT NULL,
                payload BLOB NOT NULL
            )
            """
        )

        metadata = {
            "format": "CASE_TRAIN_SQLITE",
            "format_version": str(self.FORMAT_VERSION),
            "task": task,
            "codec": "zlib-json",
        }

        self.connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            list(metadata.items()),
        )

        self.connection.commit()

    def append(self, episode: dict) -> None:
        if self.closed:
            raise RuntimeError(
                f"Writer already closed: {self.path}"
            )

        episode_id = str(
            episode.get(
                "episode_id",
                f"episode_{self.count}",
            )
        )

        raw = json.dumps(
            episode,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        compressed = zlib.compress(
            raw,
            level=self.compression_level,
        )

        self.connection.execute(
            """
            INSERT INTO episodes(episode_id, payload)
            VALUES (?, ?)
            """,
            (
                episode_id,
                sqlite3.Binary(compressed),
            ),
        )

        self.count += 1

        if (
            self.commit_every > 0
            and self.count % self.commit_every == 0
        ):
            self.connection.commit()

    def close(self) -> None:
        if self.closed:
            return

        self.connection.commit()

        # Record the final episode count inside the DB as well.
        self.connection.execute(
            """
            INSERT OR REPLACE INTO metadata(key, value)
            VALUES ('episodes', ?)
            """,
            (str(self.count),),
        )

        self.connection.commit()
        self.connection.execute("PRAGMA optimize")
        self.connection.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ======================================================================
# Annotation utilities
# ======================================================================
# Annotation utilities
# ======================================================================

def is_valid_annotation(
    ann: Mapping,
    valid_cat_ids: Optional[set] = None,
) -> bool:
    if ann.get("iscrowd", 0):
        return False

    if valid_cat_ids is not None:
        if ann.get("category_id") not in valid_cat_ids:
            return False

    return True


def get_object_level_cat_ids(coco: COCO) -> set:
    """
    PACO contains object and part categories.

    CASE attribute tasks use object-level PACO categories only. Part category
    names contain ':' in the released PACO annotations.
    """
    result = set()
    for cat in coco.loadCats(coco.getCatIds()):
        if ":" not in cat.get("name", ""):
            result.add(cat["id"])
    return result


def mask_to_rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(
        np.asfortranarray(mask.astype(np.uint8))
    )

    if isinstance(encoded["counts"], bytes):
        encoded["counts"] = encoded["counts"].decode("utf-8")

    return {
        "size": [int(v) for v in encoded["size"]],
        "counts": encoded["counts"],
    }


# ======================================================================
# CASE target-mask definitions
# ======================================================================

def generate_target_mask(
    coco: COCO,
    episode: Mapping,
    valid_attr_cat_ids: Optional[set] = None,
) -> np.ndarray:
    """
    Materialize the CASE target mask using the exact GT semantics of the
    dataset loader used for the submitted CASE/CPA experiments.

    IMPORTANT:
    The public frozen-RLE release intentionally follows the original runtime
    `_generate_dynamic_gt_mask()` behavior, rather than re-deriving masks from
    later/cleaned builder conventions.

    `valid_attr_cat_ids` is retained in the function signature for API
    compatibility with the compiler, but the original runtime did not apply
    an object-level PACO category filter to AttrUnion or AttrEx.
    """
    del valid_attr_cat_ids

    task = episode["task"]

    if "dynamic_gt_config" not in episode:
        raise KeyError(
            f"{episode.get('episode_id')}: missing dynamic_gt_config"
        )

    config = episode["dynamic_gt_config"]
    target = episode["target"]
    image_id = target["image_id"]

    image_info = coco.loadImgs(image_id)
    if not image_info:
        raise KeyError(
            f"{episode.get('episode_id')}: image_id {image_id} "
            "not found in annotation file"
        )

    image_info = image_info[0]
    height = int(image_info["height"])
    width = int(image_info["width"])

    gt_mask = np.zeros((height, width), dtype=bool)

    # --------------------------------------------------------------
    # SuperEx
    #
    # Original runtime behavior:
    #   - no explicit iscrowd=False filter
    # --------------------------------------------------------------
    if task == "SuperEx":
        mask_base = np.zeros((height, width), dtype=bool)
        mask_neg = np.zeros((height, width), dtype=bool)

        for ann in coco.loadAnns(
            coco.getAnnIds(
                imgIds=image_id,
                catIds=config["base_supercategory_ids"],
            )
        ):
            mask_base = np.logical_or(
                mask_base,
                coco.annToMask(ann).astype(bool),
            )

        for ann in coco.loadAnns(
            coco.getAnnIds(
                imgIds=image_id,
                catIds=[config["negative_category_id"]],
            )
        ):
            mask_neg = np.logical_or(
                mask_neg,
                coco.annToMask(ann).astype(bool),
            )

        return np.logical_and(
            mask_base,
            np.logical_not(mask_neg),
        )

    # --------------------------------------------------------------
    # FineIso
    #
    # Original runtime behavior:
    #   - no explicit iscrowd=False filter
    # --------------------------------------------------------------
    if task == "FineIso":
        for ann in coco.loadAnns(
            coco.getAnnIds(
                imgIds=image_id,
                catIds=[config["target_category_id"]],
            )
        ):
            gt_mask = np.logical_or(
                gt_mask,
                coco.annToMask(ann).astype(bool),
            )

        return gt_mask

    # --------------------------------------------------------------
    # Union
    #
    # Original runtime behavior:
    #   - no explicit iscrowd=False filter
    # --------------------------------------------------------------
    if task == "Union":
        for ann in coco.loadAnns(
            coco.getAnnIds(
                imgIds=image_id,
                catIds=config["positive_category_ids"],
            )
        ):
            gt_mask = np.logical_or(
                gt_mask,
                coco.annToMask(ann).astype(bool),
            )

        return gt_mask

    # --------------------------------------------------------------
    # AttrUnion
    #
    # Original runtime behavior:
    #   - iterate over every PACO annotation in the image
    #   - no object-level-category filter
    #   - no iscrowd filter
    # --------------------------------------------------------------
    if task == "AttrUnion":
        target_attr_id = config["target_attribute_id"]

        for ann in coco.loadAnns(
            coco.getAnnIds(imgIds=image_id)
        ):
            if (
                "attribute_ids" in ann
                and target_attr_id in ann["attribute_ids"]
            ):
                gt_mask = np.logical_or(
                    gt_mask,
                    coco.annToMask(ann).astype(bool),
                )

        return gt_mask

    # --------------------------------------------------------------
    # AttrIso
    #
    # Exact original runtime filtering:
    #   target attribute
    #   + not crowd
    #   + bbox exists
    # --------------------------------------------------------------
    if task == "AttrIso":
        target_attr_id = config["target_attribute_id"]
        cat_ids = config.get("supercategory_ids", [])

        for ann in coco.loadAnns(
            coco.getAnnIds(
                imgIds=image_id,
                catIds=cat_ids,
            )
        ):
            if (
                "attribute_ids" in ann
                and target_attr_id in ann["attribute_ids"]
            ):
                if (
                    not ann.get("iscrowd", 0)
                    and "bbox" in ann
                ):
                    gt_mask = np.logical_or(
                        gt_mask,
                        coco.annToMask(ann).astype(bool),
                    )

        return gt_mask

    # --------------------------------------------------------------
    # AttrEx
    #
    # Original runtime behavior:
    #   - no object-level-category filter
    #   - no iscrowd filter
    #   - annotations with attribute_ids=[] still contribute to base
    #   - only absence of the attribute_ids key is skipped
    # --------------------------------------------------------------
    if task == "AttrEx":
        supercat_ids = config["supercategory_ids"]
        exclude_attr_id = config["excluded_attribute_id"]

        mask_base = np.zeros((height, width), dtype=bool)
        mask_neg = np.zeros((height, width), dtype=bool)

        for ann in coco.loadAnns(
            coco.getAnnIds(
                imgIds=image_id,
                catIds=supercat_ids,
            )
        ):
            if "attribute_ids" not in ann:
                continue

            mask = coco.annToMask(ann).astype(bool)

            mask_base = np.logical_or(
                mask_base,
                mask,
            )

            if exclude_attr_id in ann["attribute_ids"]:
                mask_neg = np.logical_or(
                    mask_neg,
                    mask,
                )

        return np.logical_and(
            mask_base,
            np.logical_not(mask_neg),
        )

    raise ValueError(
        f"{episode.get('episode_id')}: unsupported CASE task {task!r}"
    )


# ======================================================================
# Public schema conversion
# ======================================================================

def copy_boxes(boxes) -> List[List[float]]:
    if boxes is None:
        return []

    result = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError(f"Invalid bounding box: {box}")
        result.append([float(v) for v in box])

    return result


def compile_support(support: Mapping, episode_id: str) -> dict:
    """
    Convert an internal support entry to the public CASE schema.

    Prompt boxes are preserved exactly. No prompt-count truncation is applied.
    """
    if "image_id" not in support:
        raise KeyError(f"{episode_id}: support missing image_id")

    if "image_path" not in support:
        raise KeyError(f"{episode_id}: support missing image_path")

    prompt_source = support.get("prompts", support)

    pos_boxes = copy_boxes(
        prompt_source.get("pos_bboxes", [])
    )
    neg_boxes = copy_boxes(
        prompt_source.get("neg_bboxes", [])
    )

    if not pos_boxes:
        raise ValueError(
            f"{episode_id}: support image {support.get('image_id')} "
            "has no positive prompt boxes"
        )

    return {
        "image_id": support["image_id"],
        "image_path": support["image_path"],
        "prompts": {
            "pos_bboxes": pos_boxes,
            "neg_bboxes": neg_boxes,
        },
    }


def compile_target(
    target: Mapping,
    target_mask: np.ndarray,
    episode_id: str,
) -> dict:
    if "image_id" not in target:
        raise KeyError(f"{episode_id}: target missing image_id")

    if "image_path" not in target:
        raise KeyError(f"{episode_id}: target missing image_path")

    result = {
        "image_id": target["image_id"],
        "image_path": target["image_path"],
        # Recompute this value from the exact paper-runtime mask.
        # The legacy builder value can differ because some builders used
        # stricter annotation filtering than the runtime dataset loader.
        "gt_foreground_pixels": int(target_mask.sum()),
        "gt_mask_rle": mask_to_rle(target_mask),
    }

    # Target boxes are retained as benchmark diagnostics only.
    if "pos_bboxes" in target:
        result["pos_bboxes"] = copy_boxes(
            target.get("pos_bboxes", [])
        )

    if "neg_bboxes" in target:
        result["neg_bboxes"] = copy_boxes(
            target.get("neg_bboxes", [])
        )

    return result


def compile_episode(
    episode: Mapping,
    coco: COCO,
    split: str,
    valid_attr_cat_ids: Optional[set] = None,
    strict_gt_area: bool = False,
) -> tuple[dict, bool]:
    episode_id = str(
        episode.get("episode_id", "unknown_episode")
    )

    task = episode.get("task")
    if task not in ALL_TASKS:
        raise ValueError(
            f"{episode_id}: unsupported CASE task {task!r}"
        )

    mask = generate_target_mask(
        coco,
        episode,
        valid_attr_cat_ids=valid_attr_cat_ids,
    )

    area_mismatch = False
    expected_area = (
        episode.get("target", {})
        .get("gt_foreground_pixels")
    )

    if expected_area is not None:
        actual_area = int(mask.sum())
        expected_area = int(expected_area)

        if actual_area != expected_area:
            area_mismatch = True

            message = (
                f"{episode_id}: legacy builder area differs from paper-runtime mask: "
                f"legacy={expected_area}, runtime={actual_area}"
            )

            if strict_gt_area:
                raise ValueError(message)

    # Preserve lightweight concept-level metadata while removing the dynamic
    # mask recipe and the old support representation.
    output = {
        key: value
        for key, value in episode.items()
        if key
        not in {
            "dynamic_gt_config",
            "target",
            "support",
            "support_pool",
        }
    }

    output["target"] = compile_target(
        episode["target"],
        mask,
        episode_id,
    )

    if split == "train":
        support_pool = episode.get("support_pool")

        if not isinstance(support_pool, list) or not support_pool:
            raise ValueError(
                f"{episode_id}: training episode has no support_pool"
            )

        output["support_pool"] = [
            compile_support(support, episode_id)
            for support in support_pool
        ]

    elif split == "eval":
        support = episode.get("support")

        if not isinstance(support, list) or not support:
            raise ValueError(
                f"{episode_id}: evaluation episode has no fixed support"
            )

        output["support"] = [
            compile_support(item, episode_id)
            for item in support
        ]

    else:
        raise ValueError(f"Unknown split: {split}")

    return output, area_mismatch


# ======================================================================
# Source discovery
# ======================================================================

def source_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []

    files = list(directory.glob("*.json"))
    files += list(directory.glob("*.json.gz"))
    return sorted(set(files))


def find_eval_source(
    eval_root: Path,
    chapter: str,
    task: str,
    shot: int,
) -> Path:
    directory = eval_root / chapter / task

    candidates = sorted(
        list(directory.glob(f"*_{shot}shot_*.json"))
        + list(directory.glob(f"*_{shot}shot_*.json.gz"))
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {shot}-shot evaluation file for "
            f"{chapter}/{task}, found {len(candidates)} in {directory}: "
            f"{[p.name for p in candidates]}"
        )

    return candidates[0]


# ======================================================================
# Compilation
# ======================================================================

def compile_train_task(
    task: str,
    metadata_root: Path,
    output_root: Path,
    coco: COCO,
    valid_attr_cat_ids: Optional[set],
    compression_level: int,
    strict_gt_area: bool,
    max_episodes: int,
    progress_every: int,
) -> dict:
    chapter = TASK_TO_CHAPTER[task]

    input_dir = (
        metadata_root
        / chapter
        / task
        / "train"
    )

    files = source_json_files(input_dir)

    if not files:
        raise FileNotFoundError(
            f"No training metadata files found for {task}: {input_dir}"
        )

    output_dir = (
        output_root
        / "train"
        / chapter
        / task
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_path = (
        output_dir
        / "episodes.sqlite"
    )

    total = 0
    area_mismatches = 0
    start = time.time()

    print()
    print("=" * 80)
    print(f"[train] {task}")
    print("=" * 80)
    print(f"source files : {len(files)}")
    print(f"output       : {database_path}")

    with SQLiteTrainWriter(
        path=database_path,
        task=task,
        compression_level=compression_level,
    ) as writer:
        stop = False

        for source_path in files:
            if stop:
                break

            print(f"  reading {source_path.name}")

            for episode in iter_json_array(
                source_path
            ):
                compiled, mismatch = compile_episode(
                    episode,
                    coco=coco,
                    split="train",
                    valid_attr_cat_ids=valid_attr_cat_ids,
                    strict_gt_area=strict_gt_area,
                )

                writer.append(compiled)

                total += 1
                area_mismatches += int(mismatch)

                if (
                    progress_every > 0
                    and total % progress_every == 0
                ):
                    elapsed = time.time() - start
                    rate = (
                        total
                        / max(elapsed, 1e-9)
                    )

                    print(
                        f"    {total:>9,} episodes "
                        f"| {rate:>7.1f} ep/s "
                        f"| legacy-area mismatches="
                        f"{area_mismatches}"
                    )

                if (
                    max_episodes
                    and total >= max_episodes
                ):
                    stop = True
                    break

    elapsed = time.time() - start

    print(
        f"[done] {task}: {total:,} episodes, "
        f"{elapsed / 60.0:.1f} min"
    )

    return {
        "episodes": total,
        "area_mismatches": area_mismatches,
        # Only portable names are written to the public manifest.
        "source_files": [
            path.name
            for path in files
        ],
        "database": str(
            database_path.relative_to(
                output_root
            )
        ),
        "storage": "sqlite+zlib-json",
    }


def compile_eval_task(
    task: str,
    eval_root: Path,
    output_root: Path,
    coco: COCO,
    valid_attr_cat_ids: Optional[set],
    compression_level: int,
    strict_gt_area: bool,
    max_episodes: int,
    progress_every: int,
) -> dict:
    chapter = TASK_TO_CHAPTER[task]
    result = {}

    for shot in (1, 5):
        source_path = find_eval_source(
            eval_root,
            chapter,
            task,
            shot,
        )

        output_dir = (
            output_root
            / "eval"
            / f"{shot}shot"
            / chapter
            / task
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        source_name = source_path.name
        if source_name.endswith(".json.gz"):
            output_name = source_name
        elif source_name.endswith(".json"):
            output_name = source_name + ".gz"
        else:
            raise ValueError(source_name)

        output_path = output_dir / output_name

        total = 0
        area_mismatches = 0
        start = time.time()

        print()
        print("=" * 80)
        print(f"[eval] {task} {shot}-shot")
        print("=" * 80)
        print(f"source : {source_path}")
        print(f"output : {output_path}")

        with GzipJsonArrayWriter(
            output_path,
            compression_level=compression_level,
        ) as writer:
            for episode in iter_json_array(source_path):
                compiled, mismatch = compile_episode(
                    episode,
                    coco=coco,
                    split="eval",
                    valid_attr_cat_ids=valid_attr_cat_ids,
                    strict_gt_area=strict_gt_area,
                )

                support = compiled["support"]
                if len(support) != shot:
                    raise ValueError(
                        f"{compiled.get('episode_id')}: "
                        f"{shot}-shot file contains "
                        f"{len(support)} supports"
                    )

                writer.append(compiled)
                total += 1
                area_mismatches += int(mismatch)

                if (
                    progress_every > 0
                    and total % progress_every == 0
                ):
                    print(
                        f"    {total:>7,} episodes "
                        f"| area mismatches={area_mismatches}"
                    )

                if max_episodes and total >= max_episodes:
                    break

        elapsed = time.time() - start

        print(
            f"[done] {task} {shot}-shot: "
            f"{total:,} episodes, "
            f"{elapsed:.1f} sec"
        )

        result[f"{shot}shot"] = {
            "episodes": total,
            "area_mismatches": area_mismatches,
            "source_file": source_path.name,
            "output_file": str(output_path.relative_to(output_root)),
        }

    return result


# ======================================================================
# Validation
# ======================================================================

def validate_formal_counts(
    train_results: Mapping,
    eval_results: Mapping,
    selected_tasks: Sequence[str],
    compiled_splits: str,
    max_episodes: int,
) -> None:
    if max_episodes:
        print(
            "\n[count validation skipped] "
            "--max-episodes-per-task was used."
        )
        return

    errors = []

    if compiled_splits in {"all", "train"}:
        for task in selected_tasks:
            actual = train_results[task]["episodes"]
            expected = EXPECTED_TRAIN_COUNTS[task]

            if actual != expected:
                errors.append(
                    f"train/{task}: expected {expected:,}, "
                    f"got {actual:,}"
                )

    if compiled_splits in {"all", "eval"}:
        for task in selected_tasks:
            for shot in (1, 5):
                actual = eval_results[task][f"{shot}shot"]["episodes"]
                expected = EXPECTED_EVAL_PER_SHOT[task]

                if actual != expected:
                    errors.append(
                        f"eval/{shot}shot/{task}: "
                        f"expected {expected:,}, got {actual:,}"
                    )

    if errors:
        raise RuntimeError(
            "Formal CASE episode-count validation failed:\n  - "
            + "\n  - ".join(errors)
        )

    print()
    print("=" * 80)
    print("Formal CASE episode counts: PASS")
    print("=" * 80)


def validate_area_mismatches(
    train_results: Mapping,
    eval_results: Mapping,
) -> int:
    total = 0

    for task_result in train_results.values():
        total += int(
            task_result.get("area_mismatches", 0)
        )

    for task_result in eval_results.values():
        for shot_result in task_result.values():
            total += int(
                shot_result.get("area_mismatches", 0)
            )

    if total:
        print(
            f"\n[info] {total:,} episodes differ between the legacy builder "
            "gt_foreground_pixels and the exact paper-runtime GT mask area."
        )
        print(
            "This is retained as a provenance diagnostic only. "
            "Public CASE_DATA rewrites gt_foreground_pixels from the frozen "
            "paper-runtime mask."
        )
    else:
        print("\nLegacy builder/runtime GT-area consistency: PASS")

    return total


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile internal CASE metadata into the public frozen-RLE "
            "CASE_DATA release."
        )
    )

    parser.add_argument(
        "--train-metadata-root",
        type=Path,
        required=True,
        help=(
            "Root containing vc_cat/<task>/train and "
            "vc_attr/<task>/train metadata."
        ),
    )

    parser.add_argument(
        "--eval-splits-root",
        type=Path,
        required=True,
        help=(
            "Root containing the fixed nested evaluation splits: "
            "vc_cat/<task> and vc_attr/<task>."
        ),
    )

    parser.add_argument(
        "--coco-train-ann",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--coco-val-ann",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--paco-train-ann",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--paco-val-ann",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output CASE_DATA directory.",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS,
        default=list(ALL_TASKS),
        help="Subset of CASE tasks to compile.",
    )

    parser.add_argument(
        "--splits",
        choices=("all", "train", "eval"),
        default="all",
    )

    parser.add_argument(
        "--compression-level",
        type=int,
        default=6,
        choices=range(1, 10),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N episodes. Set 0 to disable.",
    )

    parser.add_argument(
        "--strict-gt-area",
        action="store_true",
        help=(
            "Diagnostic only: fail if the legacy builder "
            "gt_foreground_pixels differs from the exact paper-runtime "
            "mask area. Such differences may exist in legacy metadata."
        ),
    )

    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=0,
        help=(
            "Compile only the first N episodes per task/shot. "
            "Intended for smoke testing; 0 means all episodes."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root before compiling.",
    )

    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def prepare_output_root(
    output_root: Path,
    overwrite: bool,
) -> None:
    if output_root.exists():
        nonempty = any(output_root.iterdir())

        if nonempty and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}\n"
                "Use --overwrite only when you intentionally want to "
                "replace that CASE_DATA directory."
            )

        if nonempty and overwrite:
            shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    if args.max_episodes_per_task < 0:
        raise ValueError(
            "--max-episodes-per-task must be >= 0"
        )

    args.tasks = list(dict.fromkeys(args.tasks))

    require_path(
        args.train_metadata_root,
        "training metadata root",
    )
    require_path(
        args.eval_splits_root,
        "evaluation split root",
    )
    require_path(
        args.coco_train_ann,
        "COCO train annotation",
    )
    require_path(
        args.coco_val_ann,
        "COCO val annotation",
    )
    require_path(
        args.paco_train_ann,
        "PACO train annotation",
    )
    require_path(
        args.paco_val_ann,
        "PACO val annotation",
    )

    prepare_output_root(
        args.output_root,
        overwrite=args.overwrite,
    )

    train_results: Dict[str, dict] = {}
    eval_results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Category training
    # ------------------------------------------------------------------
    category_selected = [
        task
        for task in args.tasks
        if task in CATEGORY_TASKS
    ]

    attribute_selected = [
        task
        for task in args.tasks
        if task in ATTRIBUTE_TASKS
    ]

    if args.splits in {"all", "train"} and category_selected:
        print("\nLoading COCO train annotations...")
        coco_train = COCO(str(args.coco_train_ann))

        for task in category_selected:
            train_results[task] = compile_train_task(
                task=task,
                metadata_root=args.train_metadata_root,
                output_root=args.output_root,
                coco=coco_train,
                valid_attr_cat_ids=None,
                compression_level=args.compression_level,
                strict_gt_area=args.strict_gt_area,
                max_episodes=args.max_episodes_per_task,
                progress_every=args.progress_every,
            )

        del coco_train
        gc.collect()

    # ------------------------------------------------------------------
    # Attribute training
    # ------------------------------------------------------------------
    if args.splits in {"all", "train"} and attribute_selected:
        print("\nLoading PACO train annotations...")
        paco_train = COCO(str(args.paco_train_ann))
        train_valid_cat_ids = get_object_level_cat_ids(
            paco_train
        )

        for task in attribute_selected:
            train_results[task] = compile_train_task(
                task=task,
                metadata_root=args.train_metadata_root,
                output_root=args.output_root,
                coco=paco_train,
                valid_attr_cat_ids=train_valid_cat_ids,
                compression_level=args.compression_level,
                strict_gt_area=args.strict_gt_area,
                max_episodes=args.max_episodes_per_task,
                progress_every=args.progress_every,
            )

        del paco_train
        del train_valid_cat_ids
        gc.collect()

    # ------------------------------------------------------------------
    # Category evaluation
    # ------------------------------------------------------------------
    if args.splits in {"all", "eval"} and category_selected:
        print("\nLoading COCO val annotations...")
        coco_val = COCO(str(args.coco_val_ann))

        for task in category_selected:
            eval_results[task] = compile_eval_task(
                task=task,
                eval_root=args.eval_splits_root,
                output_root=args.output_root,
                coco=coco_val,
                valid_attr_cat_ids=None,
                compression_level=args.compression_level,
                strict_gt_area=args.strict_gt_area,
                max_episodes=args.max_episodes_per_task,
                progress_every=args.progress_every,
            )

        del coco_val
        gc.collect()

    # ------------------------------------------------------------------
    # Attribute evaluation
    # ------------------------------------------------------------------
    if args.splits in {"all", "eval"} and attribute_selected:
        print("\nLoading PACO val annotations...")
        paco_val = COCO(str(args.paco_val_ann))
        val_valid_cat_ids = get_object_level_cat_ids(
            paco_val
        )

        for task in attribute_selected:
            eval_results[task] = compile_eval_task(
                task=task,
                eval_root=args.eval_splits_root,
                output_root=args.output_root,
                coco=paco_val,
                valid_attr_cat_ids=val_valid_cat_ids,
                compression_level=args.compression_level,
                strict_gt_area=args.strict_gt_area,
                max_episodes=args.max_episodes_per_task,
                progress_every=args.progress_every,
            )

        del paco_val
        del val_valid_cat_ids
        gc.collect()

    # ------------------------------------------------------------------
    # Formal validation
    # ------------------------------------------------------------------
    validate_formal_counts(
        train_results=train_results,
        eval_results=eval_results,
        selected_tasks=args.tasks,
        compiled_splits=args.splits,
        max_episodes=args.max_episodes_per_task,
    )

    area_mismatches = validate_area_mismatches(
        train_results,
        eval_results,
    )

    manifest = {
        "format": "CASE_DATA",
        "format_version": 1,
        "tasks": args.tasks,
        "compiled_splits": args.splits,
        "training_storage": "SQLite with one zlib-compressed JSON episode per row",
        "evaluation_storage": "gzip-compressed JSON arrays",
        "compression_level": args.compression_level,
        "support_prompt_policy": (
            "All positive/negative prompt boxes are preserved exactly "
            "from the formal benchmark metadata."
        ),
        "target_mask_policy": (
            "Frozen COCO-style RLE generated from dynamic_gt_config "
            "using the exact GT semantics of the dataset loader used "
            "for the submitted CASE/CPA experiments."
        ),
        "expected_train_counts": {
            task: EXPECTED_TRAIN_COUNTS[task]
            for task in args.tasks
        },
        "expected_eval_per_shot_counts": {
            task: EXPECTED_EVAL_PER_SHOT[task]
            for task in args.tasks
        },
        "train": train_results,
        "eval": eval_results,
        "gt_area_mismatches": area_mismatches,
        "smoke_test_limit": args.max_episodes_per_task,
    }

    manifest_path = (
        args.output_root / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("CASE_DATA compilation complete")
    print("=" * 80)
    print(f"Output   : {args.output_root}")
    print(f"Manifest : {manifest_path}")

    if args.max_episodes_per_task:
        print(
            "Mode     : SMOKE TEST "
            f"({args.max_episodes_per_task} episodes/task)"
        )
    else:
        print("Mode     : FORMAL FULL RELEASE")


if __name__ == "__main__":
    main()
