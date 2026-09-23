"""COCO-20i episodic dataset for the CASE paper FSS evaluation.

This file intentionally preserves the sampling protocol used by the submitted
CPA experiments:

- validation uses COCO val2014,
- held-out classes satisfy ``class_index % 4 == fold``,
- each fold contains 1000 sampled episodes,
- episode sampling uses ``random.Random(seed + fold)``,
- support instances require area > 1000 pixels,
- query masks union all non-crowd instances of the target class.

Do not change the validation seed rule or support/query sampling policy when
reproducing the reported COCO-20i numbers.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


COCO_ID_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}

SORTED_COCO_IDS = sorted(COCO_ID_TO_NAME)
NUM_EVAL_EPISODES = 1000
MIN_SUPPORT_AREA = 1000


class COCOFSSDataset(Dataset):
    """Fixed COCO-20i validation episodes used for CPA preservation evaluation."""

    def __init__(
        self,
        lists_root,
        data_root,
        fold: int = 0,
        k_shot: int = 1,
        mode: str = "val",
        seed: int = 0,
    ) -> None:
        super().__init__()

        del lists_root

        if fold not in {0, 1, 2, 3}:
            raise ValueError(f"fold must be 0, 1, 2, or 3; got {fold}")
        if k_shot not in {1, 5}:
            raise ValueError(
                f"k_shot must be 1 or 5 for the paper evaluation; got {k_shot}"
            )
        if mode != "val":
            raise ValueError(
                "Release COCOFSSDataset implements the paper's validation "
                "protocol only; expected mode='val'."
            )

        self.data_root = Path(data_root).expanduser().resolve()
        self.img_dir = self.data_root / "val2014"
        self.ann_file = (
            self.data_root / "annotations" / "instances_val2014.json"
        )
        self.fold = int(fold)
        self.k_shot = int(k_shot)
        self.seed = int(seed)

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"COCO val2014 directory not found: {self.img_dir}"
            )
        if not self.ann_file.exists():
            raise FileNotFoundError(
                f"COCO val2014 annotation file not found: {self.ann_file}"
            )

        self.coco = COCO(str(self.ann_file))

        self.active_cat_ids = [
            category_id
            for index, category_id in enumerate(SORTED_COCO_IDS)
            if index % 4 == self.fold
        ]

        self.target_pool: Dict[int, List[int]] = {}
        self.support_pool: Dict[int, List[int]] = {}

        valid_categories = []
        for category_id in self.active_cat_ids:
            target_ids = []
            support_ids = []

            for image_id in self.coco.getImgIds(catIds=[category_id]):
                annotations = self.coco.loadAnns(
                    self.coco.getAnnIds(
                        imgIds=[image_id],
                        catIds=[category_id],
                    )
                )

                has_target = any(
                    not ann.get("iscrowd", 0)
                    for ann in annotations
                )
                has_support = any(
                    ann.get("area", 0) > MIN_SUPPORT_AREA
                    and not ann.get("iscrowd", 0)
                    for ann in annotations
                )

                if has_target:
                    target_ids.append(image_id)
                if has_support:
                    support_ids.append(image_id)

            if target_ids and len(support_ids) >= self.k_shot + 1:
                self.target_pool[category_id] = target_ids
                self.support_pool[category_id] = support_ids
                valid_categories.append(category_id)

        self.active_cat_ids = valid_categories
        if not self.active_cat_ids:
            raise RuntimeError(
                f"No valid COCO-20i classes for fold={fold}, k_shot={k_shot}."
            )

        # Exact validation seed rule used by the submitted evaluator.
        rng = random.Random(self.seed + self.fold)

        self.test_items: List[Tuple[int, int, List[int]]] = []
        for _ in range(NUM_EVAL_EPISODES):
            category_id = rng.choice(self.active_cat_ids)
            target_image_id = rng.choice(
                self.target_pool[category_id]
            )
            support_candidates = [
                image_id
                for image_id in self.support_pool[category_id]
                if image_id != target_image_id
            ]

            if len(support_candidates) < self.k_shot:
                raise RuntimeError(
                    "Support pool became invalid after excluding the query "
                    f"image for class {category_id}."
                )

            support_image_ids = rng.sample(
                support_candidates,
                self.k_shot,
            )
            self.test_items.append(
                (
                    category_id,
                    target_image_id,
                    support_image_ids,
                )
            )

        print(
            f"COCO-20i Fold-{self.fold}: "
            f"{len(self.active_cat_ids)} classes, "
            f"{len(self.test_items)} episodes, "
            f"{self.k_shot}-shot, seed={self.seed}"
        )

    def __len__(self) -> int:
        return len(self.test_items)

    def __getitem__(self, idx):
        if not self.test_items:
            raise RuntimeError("COCO-20i episode list is empty.")

        for offset in range(len(self.test_items)):
            current_idx = (int(idx) + offset) % len(self.test_items)
            category_id, target_image_id, support_image_ids = (
                self.test_items[current_idx]
            )

            target_data = self._load_target(
                target_image_id,
                category_id,
            )
            if target_data is None:
                continue

            support_data = []
            for image_id in support_image_ids:
                item = self._load_support(
                    image_id,
                    category_id,
                )
                if item is None:
                    break
                support_data.append(item)

            if len(support_data) != self.k_shot:
                continue

            return (
                support_data,
                [target_data],
                category_id,
                COCO_ID_TO_NAME[category_id],
            )

        raise RuntimeError(
            "Unable to load a valid COCO-20i episode. "
            "Check the COCO val2014 files and annotations."
        )

    def _load_support(self, image_id: int, category_id: int):
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.img_dir / image_info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"COCO image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        annotations = self.coco.loadAnns(
            self.coco.getAnnIds(
                imgIds=[image_id],
                catIds=[category_id],
            )
        )
        valid_annotations = [
            ann
            for ann in annotations
            if ann.get("area", 0) > MIN_SUPPORT_AREA
            and not ann.get("iscrowd", 0)
        ]

        if not valid_annotations:
            return None

        # Keep the original process-level RNG behavior.
        target_ann = random.choice(valid_annotations)

        return {
            "image": image,
            "box": target_ann["bbox"],
            "name": image_info["file_name"],
            "image_id": image_id,
            "cat_id": category_id,
            "ann_id": target_ann.get("id"),
        }

    def _load_target(self, image_id: int, category_id: int):
        image_info = self.coco.loadImgs([image_id])[0]
        image_path = self.img_dir / image_info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(
                f"COCO image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")
        width = int(image_info["width"])
        height = int(image_info["height"])
        target_mask = np.zeros((height, width), dtype=np.uint8)

        annotations = self.coco.loadAnns(
            self.coco.getAnnIds(
                imgIds=[image_id],
                catIds=[category_id],
            )
        )
        if not annotations:
            return None

        for ann in annotations:
            if ann.get("iscrowd", 0):
                continue
            target_mask = np.maximum(
                target_mask,
                self.coco.annToMask(ann),
            )

        if target_mask.max() == 0:
            return None

        return {
            "image": image,
            "mask": target_mask,
            "name": image_info["file_name"],
            "image_id": image_id,
            "cat_id": category_id,
        }
