"""CASE benchmark dataset loader.

The public CASE release contains complete episode metadata for the six benchmark
tasks, including support prompts and frozen target masks. Runtime training and
evaluation therefore do not require COCO/PACO annotation files.

Source COCO/PACO annotations are only needed by the separate benchmark
construction tools used to regenerate the released metadata.

Expected release layout
-----------------------
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
└── eval/
    ├── 1shot/
    │   ├── vc_cat/<task>/
    │   └── vc_attr/<task>/
    └── 5shot/
        ├── vc_cat/<task>/
        └── vc_attr/<task>/

Training uses one random-access SQLite database per task, with each episode
stored as an independently compressed JSON payload. Evaluation directories use
.json or .json.gz files containing the fixed official episode sets.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sqlite3
import zlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


CATEGORY_TASKS = ("SuperEx", "FineIso", "Union")
ATTRIBUTE_TASKS = ("AttrEx", "AttrIso", "AttrUnion")
CASE_TASKS = CATEGORY_TASKS + ATTRIBUTE_TASKS

TASK_TO_CHAPTER = {
    **{task: "vc_cat" for task in CATEGORY_TASKS},
    **{task: "vc_attr" for task in ATTRIBUTE_TASKS},
}


class CASEDataset(Dataset):
    """CASE episodic dataset for CPA training and evaluation.

    Parameters
    ----------
    benchmark_root:
        Root of the released CASE metadata.
    image_root:
        Root used to resolve episode ``image_path`` entries. The benchmark
        release provides metadata and masks, but not third-party source images.
    split:
        ``"train"`` or ``"eval"``.
    k_shot:
        Number of support images. The paper reports 1-shot and 5-shot.
    epoch_size:
        Virtual number of iterations per training epoch.
    task:
        Optional single-task filter. ``None`` uses all six tasks for training.
        Evaluation requires a task.
    """

    def __init__(
        self,
        benchmark_root,
        image_root,
        split: str,
        k_shot: int,
        epoch_size: int = 2000,
        task: str | None = None,
    ) -> None:
        super().__init__()

        if split not in {"train", "eval"}:
            raise ValueError(
                f"split must be 'train' or 'eval'; got {split!r}"
            )
        if k_shot not in {1, 5}:
            raise ValueError(
                f"k_shot must be 1 or 5 for the paper protocol; got {k_shot}"
            )
        if epoch_size <= 0:
            raise ValueError(
                f"epoch_size must be positive; got {epoch_size}"
            )
        if task is not None and task not in CASE_TASKS:
            raise ValueError(
                f"Unknown CASE task {task!r}. Expected one of {CASE_TASKS}."
            )
        if split == "eval" and task is None:
            raise ValueError("Evaluation requires an explicit CASE task.")

        self.benchmark_root = Path(benchmark_root).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.split = split
        self.k_shot = int(k_shot)
        self.epoch_size = int(epoch_size)
        self.task_filter = task

        if not self.benchmark_root.exists():
            raise FileNotFoundError(
                f"CASE benchmark root not found: {self.benchmark_root}"
            )
        if not self.image_root.exists():
            raise FileNotFoundError(
                f"CASE image root not found: {self.image_root}"
            )

        self.episodes: List[dict] = []

        # Training metadata is random-access and never expanded into memory.
        self.training_db_by_task: Dict[str, Path] = {}
        self.training_counts_by_task: Dict[str, int] = {}

        # SQLite connections are opened lazily inside the process that calls
        # __getitem__. This is safe for PyTorch DataLoader workers.
        self._training_connections: Dict[str, sqlite3.Connection] = {}
        self._training_connection_pid = None

        if self.split == "train":
            self._load_training_index()
        else:
            self._load_evaluation_episodes()

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------

    @staticmethod
    def _json_files(directory: Path) -> List[Path]:
        if not directory.exists():
            return []

        files = list(directory.glob("*.json"))
        files += list(directory.glob("*.json.gz"))
        return sorted(set(files))

    @staticmethod
    def _read_json(path: Path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def _read_episode_file(cls, path: Path) -> List[dict]:
        payload = cls._read_json(path)

        if isinstance(payload, list):
            episodes = payload
        elif isinstance(payload, Mapping):
            episodes = payload.get(
                "episodes",
                payload.get("queries", []),
            )
        else:
            raise ValueError(
                f"Unsupported JSON payload in {path}: "
                f"{type(payload).__name__}"
            )

        if not isinstance(episodes, list):
            raise ValueError(
                f"Episode payload must be a list in {path}"
            )

        return episodes

    def _load_training_index(self) -> None:
        """Register the six random-access SQLite training stores.

        No training episode JSON is loaded here. Only episode counts and
        database paths are retained in memory.
        """
        tasks = (
            [self.task_filter]
            if self.task_filter is not None
            else list(CASE_TASKS)
        )

        for task in tasks:
            chapter = TASK_TO_CHAPTER[task]

            database_path = (
                self.benchmark_root
                / "train"
                / chapter
                / task
                / "episodes.sqlite"
            )

            if not database_path.exists():
                raise FileNotFoundError(
                    f"No CASE training database found for {task}: "
                    f"{database_path}"
                )

            uri = (
                database_path.as_uri()
                + "?mode=ro"
            )

            with sqlite3.connect(
                uri,
                uri=True,
            ) as connection:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(*),
                        MIN(id),
                        MAX(id)
                    FROM episodes
                    """
                ).fetchone()

                if row is None:
                    raise RuntimeError(
                        f"Could not read training database: "
                        f"{database_path}"
                    )

                count, min_id, max_id = row

                count = int(count or 0)

                if count <= 0:
                    raise RuntimeError(
                        f"CASE training database is empty: "
                        f"{database_path}"
                    )

                # Uniform row-id sampling requires a contiguous 1..N keyspace.
                if int(min_id) != 1 or int(max_id) != count:
                    raise RuntimeError(
                        f"Training row IDs are not contiguous in "
                        f"{database_path}: "
                        f"count={count}, min={min_id}, max={max_id}"
                    )

                try:
                    db_task_row = connection.execute(
                        """
                        SELECT value
                        FROM metadata
                        WHERE key='task'
                        """
                    ).fetchone()
                except sqlite3.Error:
                    db_task_row = None

                if (
                    db_task_row is not None
                    and db_task_row[0] != task
                ):
                    raise RuntimeError(
                        f"Training database task mismatch: "
                        f"expected {task}, found {db_task_row[0]}"
                    )

            self.training_db_by_task[
                task
            ] = database_path

            self.training_counts_by_task[
                task
            ] = count

        self.available_tasks = sorted(
            self.training_db_by_task
        )

        print("Loaded CASE training release:")
        for task in self.available_tasks:
            print(
                f"  {task:<10} "
                f"{self.training_counts_by_task[task]:>8,} episodes"
            )

    def _close_training_connections(self) -> None:
        for connection in self._training_connections.values():
            try:
                connection.close()
            except Exception:
                pass

        self._training_connections = {}
        self._training_connection_pid = None

    def _get_training_connection(
        self,
        task: str,
    ) -> sqlite3.Connection:
        """Return a read-only SQLite connection local to this process."""
        pid = os.getpid()

        # Handles forked DataLoader workers safely if the parent happened to
        # create a connection before worker processes were started.
        if self._training_connection_pid != pid:
            self._close_training_connections()
            self._training_connection_pid = pid

        if task not in self._training_connections:
            path = self.training_db_by_task[task]

            uri = (
                path.as_uri()
                + "?mode=ro"
            )

            self._training_connections[
                task
            ] = sqlite3.connect(
                uri,
                uri=True,
            )

        return self._training_connections[task]

    def _read_training_episode(
        self,
        task: str,
        row_id: int,
    ) -> dict:
        connection = self._get_training_connection(
            task
        )

        row = connection.execute(
            """
            SELECT payload
            FROM episodes
            WHERE id=?
            """,
            (int(row_id),),
        ).fetchone()

        if row is None:
            raise IndexError(
                f"Training episode row {row_id} "
                f"not found for {task}"
            )

        compressed = row[0]

        if isinstance(
            compressed,
            memoryview,
        ):
            compressed = compressed.tobytes()

        raw = zlib.decompress(
            compressed
        )

        episode = json.loads(
            raw.decode("utf-8")
        )

        if episode.get("task") != task:
            raise RuntimeError(
                f"Training DB returned task "
                f"{episode.get('task')!r}; expected {task!r}"
            )

        return episode

    def __getstate__(self):
        """Do not pickle live SQLite connections into worker processes."""
        state = self.__dict__.copy()
        state["_training_connections"] = {}
        state["_training_connection_pid"] = None
        return state

    def __del__(self):
        try:
            self._close_training_connections()
        except Exception:
            pass

    def _load_evaluation_episodes(self) -> None:
        task = self.task_filter
        chapter = TASK_TO_CHAPTER[task]

        task_dir = (
            self.benchmark_root
            / "eval"
            / f"{self.k_shot}shot"
            / chapter
            / task
        )

        files = self._json_files(task_dir)
        if not files:
            raise FileNotFoundError(
                f"No CASE evaluation metadata found: {task_dir}"
            )

        episodes = []
        for path in files:
            episodes.extend(self._read_episode_file(path))

        self.episodes = [
            episode
            for episode in episodes
            if episode.get("task") == task
        ]

        if not self.episodes:
            raise RuntimeError(
                f"No valid {task} evaluation episodes loaded from {task_dir}"
            )

        print(
            f"Loaded CASE {self.k_shot}-shot evaluation: "
            f"{task} ({len(self.episodes):,} episodes)"
        )

    # ------------------------------------------------------------------
    # PyTorch dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.split == "train":
            return self.epoch_size
        return len(self.episodes)

    def __getitem__(self, index):
        if self.split == "train":
            # Exact submitted sampling semantics:
            #   1. sample a CASE task uniformly;
            #   2. sample an episode uniformly within that task.
            #
            # SQLite row IDs are contiguous 1..N, so this remains uniform
            # without loading all training episodes into memory.
            task = random.choice(
                self.available_tasks
            )

            episode_count = (
                self.training_counts_by_task[
                    task
                ]
            )

            row_id = (
                random.randrange(
                    episode_count
                )
                + 1
            )

            episode = (
                self._read_training_episode(
                    task,
                    row_id,
                )
            )
        else:
            episode = self.episodes[int(index)]

        self._validate_episode(episode)

        support_entries = self._select_supports(episode)
        support_data = [
            self._load_support(entry)
            for entry in support_entries
        ]
        target_data = self._load_target(episode["target"])
        meta_info = self._build_meta_info(episode, index)

        if self.k_shot == 1:
            return support_data[0], target_data, meta_info

        return support_data, target_data, meta_info

    # ------------------------------------------------------------------
    # Episode parsing
    # ------------------------------------------------------------------

    def _validate_episode(self, episode: dict) -> None:
        task = episode.get("task")
        if task not in CASE_TASKS:
            raise ValueError(
                f"Episode {episode.get('episode_id')} has unsupported "
                f"task {task!r}."
            )

        target = episode.get("target")
        if not isinstance(target, Mapping):
            raise ValueError(
                f"Episode {episode.get('episode_id')} is missing target."
            )

        if "gt_mask_rle" not in target:
            raise ValueError(
                f"Episode {episode.get('episode_id')} is missing "
                "target.gt_mask_rle. Public CASE metadata must contain "
                "a frozen target mask."
            )

    def _select_supports(self, episode: dict) -> List[dict]:
        if self.split == "train":
            pool = episode.get(
                "support_pool",
                episode.get("support", []),
            )
            if not pool:
                raise ValueError(
                    f"Training episode {episode.get('episode_id')} "
                    "has no support pool."
                )

            if len(pool) >= self.k_shot:
                return random.sample(pool, self.k_shot)

            # Preserve the original CPA training behavior for small pools.
            return random.choices(pool, k=self.k_shot)

        support = episode.get("support", [])
        if len(support) < self.k_shot:
            raise ValueError(
                f"Evaluation episode {episode.get('episode_id')} has "
                f"{len(support)} supports; expected {self.k_shot}."
            )

        # Official evaluation supports are fixed. For nested 1/5-shot
        # protocols, the order in the released metadata is authoritative.
        return list(support[: self.k_shot])

    def _load_support(self, support: dict) -> dict:
        image = self._load_image(support["image_path"])

        prompts = support.get("prompts", support)
        pos_boxes = list(prompts.get("pos_bboxes", []))
        neg_boxes = list(prompts.get("neg_bboxes", []))

        if not pos_boxes:
            raise ValueError(
                f"Support image {support.get('image_id')} contains "
                "no positive prompt."
            )

        return {
            "image": image,
            "boxes": pos_boxes + neg_boxes,
            "labels": (
                [1] * len(pos_boxes)
                + [0] * len(neg_boxes)
            ),
            "mask": None,
            "image_id": support.get("image_id"),
        }

    def _load_target(self, target: dict) -> dict:
        image = self._load_image(target["image_path"])
        mask = self._decode_rle(target["gt_mask_rle"])

        expected_size = (image.height, image.width)
        if mask.shape != expected_size:
            raise ValueError(
                f"Target mask shape {mask.shape} does not match image "
                f"shape {expected_size} for image {target.get('image_id')}."
            )

        return {
            "image": image,
            "mask": mask,
            "pos_bboxes": list(
                target.get("pos_bboxes", [])
            ),
            "neg_bboxes": list(
                target.get("neg_bboxes", [])
            ),
            "image_id": target.get("image_id"),
        }

    def _load_image(self, relative_path) -> Image.Image:
        path = self.image_root / str(relative_path).lstrip("/\\")
        if not path.exists():
            raise FileNotFoundError(
                f"CASE source image not found: {path}"
            )
        return Image.open(path).convert("RGB")

    @staticmethod
    def _decode_rle(rle: Mapping) -> np.ndarray:
        encoded = dict(rle)

        # Compressed COCO RLE is serialized as text in JSON but decoded by
        # pycocotools from bytes.
        if isinstance(encoded.get("counts"), str):
            encoded["counts"] = encoded["counts"].encode("utf-8")

        mask = mask_utils.decode(encoded)

        if mask.ndim == 3:
            mask = np.any(mask, axis=2)

        return np.asarray(mask, dtype=np.uint8)

    @staticmethod
    def _build_meta_info(episode: dict, index) -> dict:
        return {
            "episode_id": episode.get(
                "episode_id",
                f"episode_{index}",
            ),
            "task": episode["task"],
            "intent": episode.get(
                "intent",
                "unknown_intent",
            ),
        }
