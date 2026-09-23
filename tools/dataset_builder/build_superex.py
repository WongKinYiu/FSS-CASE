#!/usr/bin/env python3
"""Build intermediate CASE SuperEx metadata.

This public builder contains the exact validation and training construction
implementations used for the CASE benchmark. The two historical implementations
are intentionally kept logically separate inside this module so that combining
the CLI does not change benchmark construction semantics.

Examples
--------
Validation:
    python tools/dataset_builder/build_superex.py \\
        --split val \\
        --annotation-file /path/to/annotation.json \\
        --output-root /path/to/metadata

Training:
    python tools/dataset_builder/build_superex.py \\
        --split train \\
        --annotation-file /path/to/annotation.json \\
        --output-root /path/to/metadata \\
        --seed 2026
"""

import argparse
from pathlib import Path
import os
import json
import numpy as np
from pycocotools.coco import COCO
from collections import defaultdict
import random
import sys


# ======================================================================
# Validation implementation
# Mechanically namespaced from the verified historical validation builder.
# ======================================================================

# BEGIN CASE LEGACY VAL IMPLEMENTATION
_val_ANN_FILE = None
_val_EPISODES_DIR = None
_val_MIN_FG_PIXELS = 500
_val_MIN_DIVERSITY_SCORE = 2

def _val_calculate_iou(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2 = (x1_1 + w1, y1_1 + h1)
    x2_1, y2_1, w2, h2 = box2
    x2_2, y2_2 = (x2_1 + w2, y2_1 + h2)
    xi_1, yi_1 = (max(x1_1, x2_1), max(y1_1, y2_1))
    xi_2, yi_2 = (min(x1_2, x2_2), min(y1_2, y2_2))
    inter_area = max(0, xi_2 - xi_1) * max(0, yi_2 - yi_1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def _val_is_duplicate(new_box, existing_boxes, iou_thresh=0.75):
    for box in existing_boxes:
        if _val_calculate_iou(new_box, box) > iou_thresh:
            return True
    return False

def _val_calculate_gt_area(coco, image_id, supercat_ids, neg_cat_id):
    img_info = coco.loadImgs(image_id)[0]
    height, width = (img_info['height'], img_info['width'])
    mask_base = np.zeros((height, width), dtype=bool)
    mask_neg = np.zeros((height, width), dtype=bool)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=supercat_ids, iscrowd=False)):
        mask_base = np.logical_or(mask_base, coco.annToMask(ann).astype(bool))
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=[neg_cat_id], iscrowd=False)):
        mask_neg = np.logical_or(mask_neg, coco.annToMask(ann).astype(bool))
    return np.sum(np.logical_and(mask_base, np.logical_not(mask_neg)))

def _val_parse_args():
    """Parse public benchmark-construction paths without changing task logic."""
    parser = argparse.ArgumentParser(description='Build intermediate CASE SuperEx val metadata.')
    parser.add_argument('--annotation-file', type=Path, required=True, help='COCO 2017 annotation JSON used by this builder.')
    parser.add_argument('--output-root', type=Path, required=True, help='Intermediate CASE metadata root. Outputs are written under vc_cat/SuperEx' + ('/train.' if 'val' == 'train' else '.'))
    return parser.parse_args()

def _val_main():
    global _val_ANN_FILE, _val_EPISODES_DIR
    args = _val_parse_args()
    _val_ANN_FILE = str(args.annotation_file.expanduser().resolve())
    _val_EPISODES_DIR = str(args.output_root.expanduser().resolve() / 'vc_cat' / 'SuperEx')
    os.makedirs(_val_EPISODES_DIR, exist_ok=True)
    if not os.path.exists(_val_ANN_FILE):
        print(f'錯誤：找不到 {_val_ANN_FILE}')
        return
    print(f'載入 SuperEx Val 標註檔: {_val_ANN_FILE}')
    coco = COCO(_val_ANN_FILE)
    cats = coco.loadCats(coco.getCatIds())
    supercat_to_cids = defaultdict(list)
    id_to_name = {}
    for cat in cats:
        supercat_to_cids[cat['supercategory']].append(cat['id'])
        id_to_name[cat['id']] = cat['name']
    CatToImgs = {cat['id']: set() for cat in cats}
    ImgToCats = defaultdict(set)
    for ann in coco.dataset['annotations']:
        if ann.get('iscrowd', 0):
            continue
        if 'bbox' not in ann or len(ann['bbox']) != 4:
            continue
        CatToImgs[ann['category_id']].add(ann['image_id'])
        ImgToCats[ann['image_id']].add(ann['category_id'])
    benchmark_episodes = []
    episode_count = 0
    print('\n開始生成 SuperEx 驗證集 (嚴格 Pos/Neg 防守與 BBox 碰撞檢測)...')
    for supercat, all_cids in supercat_to_cids.items():
        if len(all_cids) < 3:
            continue
        for neg_id in all_cids:
            concept_name = f'All {supercat} MINUS {id_to_name[neg_id]}'
            valid_pos_cids = set(all_cids) - {neg_id}
            for target_img_id in CatToImgs[neg_id]:
                if not ImgToCats[target_img_id].intersection(valid_pos_cids):
                    continue
                fg_pixels = _val_calculate_gt_area(coco, target_img_id, all_cids, neg_id)
                if fg_pixels < _val_MIN_FG_PIXELS:
                    continue
                pos_tgt_bboxes = []
                neg_tgt_bboxes = []
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                    if ann['category_id'] == neg_id and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _val_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.7):
                            neg_tgt_bboxes.append(bbox)
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                    if ann['category_id'] in valid_pos_cids and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _val_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.5) and (not _val_is_duplicate(bbox, pos_tgt_bboxes, iou_thresh=0.7)):
                            pos_tgt_bboxes.append(bbox)
                if not pos_tgt_bboxes or not neg_tgt_bboxes:
                    continue
                candidate_supp_ids = CatToImgs[neg_id] - {target_img_id}
                qualified_supports = []
                for supp_id in candidate_supp_ids:
                    pos_bboxes = []
                    neg_bboxes = []
                    pos_cats = set()
                    for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                        if ann['category_id'] in valid_pos_cids and len(ann['bbox']) == 4:
                            bbox = ann['bbox']
                            if not _val_is_duplicate(bbox, pos_bboxes, iou_thresh=0.7):
                                pos_bboxes.append(bbox)
                                pos_cats.add(ann['category_id'])
                    for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                        if ann['category_id'] == neg_id and len(ann['bbox']) == 4:
                            bbox = ann['bbox']
                            if not _val_is_duplicate(bbox, pos_bboxes, iou_thresh=0.5) and (not _val_is_duplicate(bbox, neg_bboxes, iou_thresh=0.7)):
                                neg_bboxes.append(bbox)
                    diversity_score = len(pos_cats)
                    if diversity_score >= _val_MIN_DIVERSITY_SCORE and len(neg_bboxes) >= 1:
                        qualified_supports.append({'image_id': supp_id, 'image_path': f"val2017/{coco.loadImgs(supp_id)[0]['file_name']}", 'diversity_score': diversity_score, 'pos_bboxes': pos_bboxes, 'neg_bboxes': neg_bboxes, 'present_pos_categories': [id_to_name[pid] for pid in pos_cats], 'present_neg_categories': [id_to_name[neg_id]]})
                if not qualified_supports:
                    continue
                qualified_supports.sort(key=lambda x: x['diversity_score'], reverse=True)
                episode_id = f'SuperEx_{supercat}_{id_to_name[neg_id]}_{episode_count:04d}'.replace(' ', '_')
                benchmark_episodes.append({'episode_id': episode_id, 'task': 'SuperEx', 'supercategory': supercat, 'intent': concept_name, 'target': {'image_id': target_img_id, 'image_path': f"val2017/{coco.loadImgs(target_img_id)[0]['file_name']}", 'gt_foreground_pixels': int(fg_pixels), 'pos_bboxes': pos_tgt_bboxes, 'neg_bboxes': neg_tgt_bboxes}, 'dynamic_gt_config': {'base_supercategory_ids': all_cids, 'negative_category_id': neg_id}, 'support_pool': qualified_supports})
                episode_count += 1
    out_json = os.path.join(_val_EPISODES_DIR, 'val.json')
    with open(out_json, 'w') as f:
        json.dump(benchmark_episodes, f, indent=4)
    print(f' SuperEx Val 生成完畢！共 {episode_count} 筆 -> {out_json}')
# END CASE LEGACY VAL IMPLEMENTATION


# ======================================================================
# Training implementation
# Mechanically namespaced from the verified historical training builder.
# ======================================================================

# BEGIN CASE LEGACY TRAIN IMPLEMENTATION
_train_ANN_FILE = None
_train_OUTPUT_DB_DIR = None
_train_MIN_FG_PIXELS = 500
_train_MIN_DIVERSITY_SCORE = 2
_train_CANDIDATE_POOL_SIZE = 300
_train_MAX_SUPPORT_POOL_SIZE = 100
_train_MAX_TARGETS_PER_CONCEPT = 2000

def _train_calculate_iou(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2 = (x1_1 + w1, y1_1 + h1)
    x2_1, y2_1, w2, h2 = box2
    x2_2, y2_2 = (x2_1 + w2, y2_1 + h2)
    xi_1, yi_1 = (max(x1_1, x2_1), max(y1_1, y2_1))
    xi_2, yi_2 = (min(x1_2, x2_2), min(y1_2, y2_2))
    inter_area = max(0, xi_2 - xi_1) * max(0, yi_2 - yi_1)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def _train_is_duplicate(new_box, existing_boxes, iou_thresh=0.75):
    for box in existing_boxes:
        if _train_calculate_iou(new_box, box) > iou_thresh:
            return True
    return False

def _train_calculate_gt_area(coco, image_id, supercat_ids, neg_cat_id):
    img_info = coco.loadImgs(image_id)[0]
    height, width = (img_info['height'], img_info['width'])
    mask_base = np.zeros((height, width), dtype=bool)
    mask_neg = np.zeros((height, width), dtype=bool)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=supercat_ids, iscrowd=False)):
        mask_base = np.logical_or(mask_base, coco.annToMask(ann).astype(bool))
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=[neg_cat_id], iscrowd=False)):
        mask_neg = np.logical_or(mask_neg, coco.annToMask(ann).astype(bool))
    return np.sum(np.logical_and(mask_base, np.logical_not(mask_neg)))

def _train_parse_args():
    """Parse public benchmark-construction paths without changing task logic."""
    parser = argparse.ArgumentParser(description='Build intermediate CASE SuperEx train metadata.')
    parser.add_argument('--annotation-file', type=Path, required=True, help='COCO 2017 annotation JSON used by this builder.')
    parser.add_argument('--output-root', type=Path, required=True, help='Intermediate CASE metadata root. Outputs are written under vc_cat/SuperEx' + ('/train.' if 'train' == 'train' else '.'))
    parser.add_argument('--seed', type=int, default=2026, help='Training metadata construction seed. The formal CASE benchmark uses 2026.')
    return parser.parse_args()

def _train_main():
    global _train_ANN_FILE, _train_OUTPUT_DB_DIR
    args = _train_parse_args()
    _train_ANN_FILE = str(args.annotation_file.expanduser().resolve())
    _train_OUTPUT_DB_DIR = str(args.output_root.expanduser().resolve() / 'vc_cat' / 'SuperEx' / 'train')
    os.makedirs(_train_OUTPUT_DB_DIR, exist_ok=True)
    random.seed(args.seed)
    if not os.path.exists(_train_ANN_FILE):
        print(f'錯誤：找不到 {_train_ANN_FILE}，請確認 COCO train2017 JSON 已放置定位。')
        return
    print(f'載入 SuperEx Train 標註檔: {_train_ANN_FILE}')
    coco = COCO(_train_ANN_FILE)
    cats = coco.loadCats(coco.getCatIds())
    supercat_to_cids = defaultdict(list)
    id_to_name = {}
    for cat in cats:
        supercat_to_cids[cat['supercategory']].append(cat['id'])
        id_to_name[cat['id']] = cat['name']
    CatToImgs = {cat['id']: set() for cat in cats}
    for ann in coco.dataset['annotations']:
        if ann.get('iscrowd', 0):
            continue
        if 'bbox' not in ann or len(ann['bbox']) != 4:
            continue
        CatToImgs[ann['category_id']].add(ann['image_id'])
    total_episodes = 0
    print('\n[Step 2] 開始生成 SuperEx 訓練集 (Target 放寬 + 隨機採樣池 + 嚴格防守)...')
    for supercat, all_cids in supercat_to_cids.items():
        if len(all_cids) < 3:
            continue
        supercat_episodes = []
        for neg_id in all_cids:
            concept_name = f'All {supercat} MINUS {id_to_name[neg_id]}'
            valid_pos_cids = set(all_cids) - {neg_id}
            potential_targets = set()
            for cid in valid_pos_cids:
                potential_targets.update(CatToImgs[cid])
            concept_target_count = 0
            for target_img_id in potential_targets:
                if concept_target_count >= _train_MAX_TARGETS_PER_CONCEPT:
                    break
                fg_pixels = _train_calculate_gt_area(coco, target_img_id, all_cids, neg_id)
                if fg_pixels < _train_MIN_FG_PIXELS:
                    continue
                pos_tgt_bboxes = []
                neg_tgt_bboxes = []
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                    if ann['category_id'] == neg_id and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _train_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.7):
                            neg_tgt_bboxes.append(bbox)
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                    if ann['category_id'] in valid_pos_cids and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _train_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.5) and (not _train_is_duplicate(bbox, pos_tgt_bboxes, iou_thresh=0.7)):
                            pos_tgt_bboxes.append(bbox)
                if not pos_tgt_bboxes:
                    continue
                candidate_supp_ids = CatToImgs[neg_id] - {target_img_id}
                qualified_supports = []
                for supp_id in candidate_supp_ids:
                    pos_bboxes = []
                    neg_bboxes = []
                    pos_cats = set()
                    for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                        if ann['category_id'] in valid_pos_cids and len(ann['bbox']) == 4:
                            bbox = ann['bbox']
                            if not _train_is_duplicate(bbox, pos_bboxes, iou_thresh=0.7):
                                pos_bboxes.append(bbox)
                                pos_cats.add(ann['category_id'])
                    for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                        if ann['category_id'] == neg_id and len(ann['bbox']) == 4:
                            bbox = ann['bbox']
                            if not _train_is_duplicate(bbox, pos_bboxes, iou_thresh=0.5) and (not _train_is_duplicate(bbox, neg_bboxes, iou_thresh=0.7)):
                                neg_bboxes.append(bbox)
                    diversity_score = len(pos_cats)
                    if diversity_score >= _train_MIN_DIVERSITY_SCORE and len(neg_bboxes) >= 1:
                        qualified_supports.append({'image_id': supp_id, 'image_path': f"train2017/{coco.loadImgs(supp_id)[0]['file_name']}", 'diversity_score': diversity_score, 'pos_bboxes': pos_bboxes, 'neg_bboxes': neg_bboxes, 'present_pos_categories': [id_to_name[pid] for pid in pos_cats], 'present_neg_categories': [id_to_name[neg_id]]})
                if not qualified_supports:
                    continue
                qualified_supports.sort(key=lambda x: x['diversity_score'], reverse=True)
                top_candidates = qualified_supports[:_train_CANDIDATE_POOL_SIZE]
                if len(top_candidates) > _train_MAX_SUPPORT_POOL_SIZE:
                    final_supports = random.sample(top_candidates, _train_MAX_SUPPORT_POOL_SIZE)
                else:
                    final_supports = top_candidates
                random.shuffle(final_supports)
                episode_id = f'Train_SuperEx_{supercat}_{id_to_name[neg_id]}_{total_episodes:06d}'.replace(' ', '_')
                supercat_episodes.append({'episode_id': episode_id, 'task': 'SuperEx', 'supercategory': supercat, 'intent': concept_name, 'target': {'image_id': target_img_id, 'image_path': f"train2017/{coco.loadImgs(target_img_id)[0]['file_name']}", 'gt_foreground_pixels': int(fg_pixels), 'pos_bboxes': pos_tgt_bboxes, 'neg_bboxes': neg_tgt_bboxes}, 'dynamic_gt_config': {'base_supercategory_ids': all_cids, 'negative_category_id': neg_id}, 'support_pool': final_supports})
                total_episodes += 1
                concept_target_count += 1
        if supercat_episodes:
            out_json = os.path.join(_train_OUTPUT_DB_DIR, f'{supercat}.json')
            with open(out_json, 'w') as f:
                json.dump(supercat_episodes, f, indent=4)
            print(f'  - 生成 {len(supercat_episodes)} 筆 -> {out_json}')
    print(f'\n SuperEx 訓練集構建完成，共生成 {total_episodes} 筆。')
# END CASE LEGACY TRAIN IMPLEMENTATION


# ======================================================================
# Unified public CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build intermediate CASE SuperEx train or validation metadata."
        )
    )

    parser.add_argument(
        "--split",
        choices=("train", "val"),
        required=True,
        help="Metadata split to construct.",
    )

    parser.add_argument(
        "--annotation-file",
        type=Path,
        required=True,
        help="COCO 2017 annotation JSON for the selected split.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Intermediate CASE metadata root. "
            "Outputs are written under vc_cat/SuperEx."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Training metadata construction seed. "
            "Used only with --split train. "
            "The formal CASE benchmark uses 2026."
        ),
    )

    return parser.parse_args()


def _run_historical_builder(
    split,
    annotation_file,
    output_root,
    seed,
):
    """Dispatch to the unchanged namespaced train/val implementation.

    Each historical implementation already owns its exact path setup and
    training RNG initialization. We only translate the unified public CLI into
    the original argument interface.
    """
    forwarded = [
        sys.argv[0],
        "--annotation-file",
        str(
            annotation_file
            .expanduser()
            .resolve()
        ),
        "--output-root",
        str(
            output_root
            .expanduser()
            .resolve()
        ),
    ]

    if split == "train":
        forwarded.extend(
            [
                "--seed",
                str(seed),
            ]
        )

    original_argv = sys.argv

    try:
        sys.argv = forwarded

        if split == "train":
            _train_main()
        else:
            _val_main()

    finally:
        sys.argv = original_argv


def main():
    args = parse_args()

    _run_historical_builder(
        split=args.split,
        annotation_file=args.annotation_file,
        output_root=args.output_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
