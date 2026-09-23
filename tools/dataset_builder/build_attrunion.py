#!/usr/bin/env python3
"""Build intermediate CASE AttrUnion metadata.

This public builder contains the exact validation and training construction
implementations used for the CASE benchmark. The two historical implementations
are intentionally kept logically separate inside this module so that combining
the CLI does not change benchmark construction semantics.

Examples
--------
Validation:
    python tools/dataset_builder/build_attrunion.py \\
        --split val \\
        --annotation-file /path/to/annotation.json \\
        --output-root /path/to/metadata

Training:
    python tools/dataset_builder/build_attrunion.py \\
        --split train \\
        --annotation-file /path/to/annotation.json \\
        --output-root /path/to/metadata \\
        --seed 2026
"""

import argparse
from pathlib import Path
import os
import json
import re
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
_val_PACO_ANN_FILE = None
_val_OUTPUT_DB_DIR = None
_val_MIN_FG_PIXELS = 500
_val_TARGET_COLORS = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
_val_TARGET_MATERIALS = ['stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic']

def _val_safe_filename(name):
    return re.sub('[^a-zA-Z0-9_]', '_', name.lower().strip())

def _val_is_valid_ann(ann, valid_cat_ids):
    if ann.get('iscrowd', 0):
        return False
    if 'bbox' not in ann or len(ann['bbox']) != 4:
        return False
    if not ann.get('attribute_ids'):
        return False
    if ann['category_id'] not in valid_cat_ids:
        return False
    return True

def _val_extract_clean_attr_name(raw_name):
    return raw_name.lower().split(':')[-1].strip()

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

def _val_calculate_union_area(coco, img_id, target_attr_id, valid_cat_ids):
    img_info = coco.loadImgs(img_id)[0]
    gt_mask = np.zeros((img_info['height'], img_info['width']), dtype=bool)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        if _val_is_valid_ann(ann, valid_cat_ids) and target_attr_id in ann['attribute_ids']:
            gt_mask = np.logical_or(gt_mask, coco.annToMask(ann).astype(bool))
    return np.sum(gt_mask)

def _val_parse_args():
    """Parse public benchmark-construction paths without changing task logic."""
    parser = argparse.ArgumentParser(description='Build intermediate CASE AttrUnion val metadata.')
    parser.add_argument('--annotation-file', type=Path, required=True, help='PACO annotation JSON used by this builder.')
    parser.add_argument('--output-root', type=Path, required=True, help='Intermediate CASE metadata root. Outputs are written under vc_attr/AttrUnion' + ('/train.' if 'val' == 'train' else '.'))
    return parser.parse_args()

def _val_main():
    global _val_PACO_ANN_FILE, _val_OUTPUT_DB_DIR
    args = _val_parse_args()
    _val_PACO_ANN_FILE = str(args.annotation_file.expanduser().resolve())
    _val_OUTPUT_DB_DIR = str(args.output_root.expanduser().resolve() / 'vc_attr' / 'AttrUnion')
    os.makedirs(_val_OUTPUT_DB_DIR, exist_ok=True)
    if not os.path.exists(_val_PACO_ANN_FILE):
        print(f'錯誤：找不到 {_val_PACO_ANN_FILE}')
        return
    print(f'載入 PACO 標註檔 (Val): {_val_PACO_ANN_FILE}')
    with open(_val_PACO_ANN_FILE, 'r') as f:
        paco_data = json.load(f)
    coco = COCO(_val_PACO_ANN_FILE)
    cat_to_name = {c['id']: c['name'] for c in paco_data['categories'] if ':' not in c['name']}
    valid_cat_ids = set(cat_to_name.keys())
    attr_to_name = {a['id']: a['name'] for a in paco_data['attributes']}
    color_ids = {a['id'] for a in paco_data['attributes'] if _val_extract_clean_attr_name(a['name']) in _val_TARGET_COLORS}
    mat_ids = {a['id'] for a in paco_data['attributes'] if _val_extract_clean_attr_name(a['name']) in _val_TARGET_MATERIALS}
    AttrToImgs = defaultdict(set)
    for ann in paco_data['annotations']:
        if _val_is_valid_ann(ann, valid_cat_ids):
            for attr_id in ann['attribute_ids']:
                AttrToImgs[attr_id].add(ann['image_id'])
    benchmark_episodes = []
    episode_count = 0
    print('\n[Step 2] 開始生成 AttrUnion 驗證集 (目標聯集 + BBox + 提示拒馬防守)...')
    for target_attr_id, potential_targets in AttrToImgs.items():
        is_col = target_attr_id in color_ids
        is_mat = target_attr_id in mat_ids
        if not (is_col or is_mat):
            continue
        attr_pool = color_ids if is_col else mat_ids
        attr_name = attr_to_name[target_attr_id]
        for tgt_id in potential_targets:
            fg_pixels = _val_calculate_union_area(coco, tgt_id, target_attr_id, valid_cat_ids)
            if fg_pixels < _val_MIN_FG_PIXELS:
                continue
            pos_tgt_bboxes = []
            pos_tgt_cats = set()
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=tgt_id)):
                if not _val_is_valid_ann(ann, valid_cat_ids):
                    continue
                bbox = ann['bbox']
                if target_attr_id in ann['attribute_ids']:
                    if not _val_is_duplicate(bbox, pos_tgt_bboxes, iou_thresh=0.7):
                        pos_tgt_bboxes.append(bbox)
                        pos_tgt_cats.add(ann['category_id'])
            if len(pos_tgt_cats) < 2:
                continue
            qualified_supports = []
            for supp_id in set(potential_targets) - {tgt_id}:
                pos_bboxes = []
                neg_bboxes = []
                pos_cats = set()
                neg_cats = set()
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id)):
                    if not _val_is_valid_ann(ann, valid_cat_ids):
                        continue
                    if target_attr_id in ann['attribute_ids']:
                        if not _val_is_duplicate(ann['bbox'], pos_bboxes, iou_thresh=0.7):
                            pos_bboxes.append(ann['bbox'])
                            pos_cats.add(ann['category_id'])
                if len(pos_cats) < 2:
                    continue
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id)):
                    if not _val_is_valid_ann(ann, valid_cat_ids):
                        continue
                    attrs = set(ann['attribute_ids'])
                    bbox = ann['bbox']
                    if target_attr_id not in attrs and attrs.intersection(attr_pool):
                        if _val_is_duplicate(bbox, pos_bboxes, iou_thresh=0.5):
                            continue
                        if not _val_is_duplicate(bbox, neg_bboxes, iou_thresh=0.7):
                            neg_bboxes.append(bbox)
                            neg_cats.add(ann['category_id'])
                if not neg_bboxes:
                    continue
                diversity_score = len(pos_cats) + len(neg_cats)
                qualified_supports.append({'image_id': supp_id, 'image_path': coco.loadImgs(supp_id)[0]['file_name'], 'diversity_score': diversity_score, 'pos_bboxes': pos_bboxes, 'neg_bboxes': neg_bboxes, 'present_pos_categories': [cat_to_name[cid] for cid in pos_cats], 'present_neg_categories': [cat_to_name[cid] for cid in neg_cats]})
            if not qualified_supports:
                continue
            qualified_supports.sort(key=lambda x: x['diversity_score'], reverse=True)
            episode_id = f'AttrUnion_{_val_safe_filename(attr_name)}_{episode_count:04d}'
            benchmark_episodes.append({'episode_id': episode_id, 'task': 'AttrUnion', 'intent': f'ALL {attr_name} objects', 'target': {'image_id': tgt_id, 'image_path': coco.loadImgs(tgt_id)[0]['file_name'], 'gt_foreground_pixels': int(fg_pixels), 'pos_bboxes': pos_tgt_bboxes}, 'dynamic_gt_config': {'target_attribute_id': target_attr_id}, 'support_pool': qualified_supports})
            episode_count += 1
    out_json = os.path.join(_val_OUTPUT_DB_DIR, 'val.json')
    with open(out_json, 'w') as f:
        json.dump(benchmark_episodes, f, indent=4)
    print(f'\n AttrUnion 驗證集 (Val) 構建完成！共產出 {episode_count} 筆 -> {out_json}')
# END CASE LEGACY VAL IMPLEMENTATION


# ======================================================================
# Training implementation
# Mechanically namespaced from the verified historical training builder.
# ======================================================================

# BEGIN CASE LEGACY TRAIN IMPLEMENTATION
_train_PACO_ANN_FILE = None
_train_OUTPUT_DB_DIR = None
_train_MIN_FG_PIXELS = 500
_train_CANDIDATE_POOL_SIZE = 300
_train_MAX_SUPPORT_POOL_SIZE = 100
_train_MAX_TARGETS_PER_CONCEPT = 2000
_train_TARGET_COLORS = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
_train_TARGET_MATERIALS = ['stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic']

def _train_safe_filename(name):
    return re.sub('[^a-zA-Z0-9_]', '_', name.lower().strip())

def _train_extract_clean_attr_name(raw_name):
    return raw_name.lower().split(':')[-1].strip()

def _train_is_valid_ann(ann, valid_cat_ids):
    if ann.get('iscrowd', 0):
        return False
    if 'bbox' not in ann or len(ann['bbox']) != 4:
        return False
    if not ann.get('attribute_ids'):
        return False
    if ann['category_id'] not in valid_cat_ids:
        return False
    return True

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

def _train_calculate_union_area(coco, img_id, target_attr_id, valid_cat_ids):
    img_info = coco.loadImgs(img_id)[0]
    gt_mask = np.zeros((img_info['height'], img_info['width']), dtype=bool)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        if _train_is_valid_ann(ann, valid_cat_ids) and target_attr_id in ann['attribute_ids']:
            gt_mask = np.logical_or(gt_mask, coco.annToMask(ann).astype(bool))
    return np.sum(gt_mask)

def _train_parse_args():
    """Parse public benchmark-construction paths without changing task logic."""
    parser = argparse.ArgumentParser(description='Build intermediate CASE AttrUnion train metadata.')
    parser.add_argument('--annotation-file', type=Path, required=True, help='PACO annotation JSON used by this builder.')
    parser.add_argument('--output-root', type=Path, required=True, help='Intermediate CASE metadata root. Outputs are written under vc_attr/AttrUnion' + ('/train.' if 'train' == 'train' else '.'))
    parser.add_argument('--seed', type=int, default=2026, help='Training metadata construction seed. The formal CASE benchmark uses 2026.')
    return parser.parse_args()

def _train_main():
    global _train_PACO_ANN_FILE, _train_OUTPUT_DB_DIR
    args = _train_parse_args()
    _train_PACO_ANN_FILE = str(args.annotation_file.expanduser().resolve())
    _train_OUTPUT_DB_DIR = str(args.output_root.expanduser().resolve() / 'vc_attr' / 'AttrUnion' / 'train')
    os.makedirs(_train_OUTPUT_DB_DIR, exist_ok=True)
    random.seed(args.seed)
    if not os.path.exists(_train_PACO_ANN_FILE):
        print(f'錯誤：找不到 {_train_PACO_ANN_FILE}')
        return
    print(f'載入 PACO 標註檔 (Train): {_train_PACO_ANN_FILE}')
    with open(_train_PACO_ANN_FILE, 'r') as f:
        paco_data = json.load(f)
    coco = COCO(_train_PACO_ANN_FILE)
    cat_to_name = {c['id']: c['name'] for c in paco_data['categories'] if ':' not in c['name']}
    valid_cat_ids = set(cat_to_name.keys())
    attr_to_name = {a['id']: a['name'] for a in paco_data['attributes']}
    color_ids = {a['id'] for a in paco_data['attributes'] if _train_extract_clean_attr_name(a['name']) in _train_TARGET_COLORS}
    mat_ids = {a['id'] for a in paco_data['attributes'] if _train_extract_clean_attr_name(a['name']) in _train_TARGET_MATERIALS}
    AttrToImgs = defaultdict(set)
    for ann in paco_data['annotations']:
        if _train_is_valid_ann(ann, valid_cat_ids):
            for attr_id in ann['attribute_ids']:
                AttrToImgs[attr_id].add(ann['image_id'])
    total_episodes = 0
    print('\n[Step 2] 開始生成 AttrUnion 訓練集 (Target 放寬 + 2Pos/1Neg 嚴防)...')
    for target_attr_id, potential_targets in AttrToImgs.items():
        is_col = target_attr_id in color_ids
        is_mat = target_attr_id in mat_ids
        if not (is_col or is_mat):
            continue
        attr_pool = color_ids if is_col else mat_ids
        attr_name = attr_to_name[target_attr_id]
        clean_attr_name = _train_extract_clean_attr_name(attr_name)
        concept_target_count = 0
        attr_episodes = []
        for tgt_id in potential_targets:
            if concept_target_count >= _train_MAX_TARGETS_PER_CONCEPT:
                break
            fg_pixels = _train_calculate_union_area(coco, tgt_id, target_attr_id, valid_cat_ids)
            if fg_pixels < _train_MIN_FG_PIXELS:
                continue
            pos_tgt_bboxes = []
            neg_tgt_bboxes = []
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=tgt_id)):
                if not _train_is_valid_ann(ann, valid_cat_ids):
                    continue
                bbox = ann['bbox']
                attrs = set(ann['attribute_ids'])
                if target_attr_id in attrs:
                    if not _train_is_duplicate(bbox, pos_tgt_bboxes, iou_thresh=0.7):
                        pos_tgt_bboxes.append(bbox)
                elif attrs.intersection(attr_pool):
                    if not _train_is_duplicate(bbox, pos_tgt_bboxes, iou_thresh=0.5) and (not _train_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.7)):
                        neg_tgt_bboxes.append(bbox)
            if not pos_tgt_bboxes:
                continue
            qualified_supports = []
            for supp_id in set(potential_targets) - {tgt_id}:
                pos_bboxes = []
                neg_bboxes = []
                pos_cats = set()
                neg_cats = set()
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id)):
                    if not _train_is_valid_ann(ann, valid_cat_ids):
                        continue
                    if target_attr_id in ann['attribute_ids']:
                        if not _train_is_duplicate(ann['bbox'], pos_bboxes, iou_thresh=0.7):
                            pos_bboxes.append(ann['bbox'])
                            pos_cats.add(ann['category_id'])
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id)):
                    if not _train_is_valid_ann(ann, valid_cat_ids):
                        continue
                    attrs = set(ann['attribute_ids'])
                    bbox = ann['bbox']
                    if target_attr_id not in attrs and attrs.intersection(attr_pool):
                        if _train_is_duplicate(bbox, pos_bboxes, iou_thresh=0.5):
                            continue
                        if not _train_is_duplicate(bbox, neg_bboxes, iou_thresh=0.7):
                            neg_bboxes.append(bbox)
                            neg_cats.add(ann['category_id'])
                if len(pos_cats) < 2 or not neg_bboxes:
                    continue
                diversity_score = len(pos_cats) + len(neg_cats)
                qualified_supports.append({'image_id': supp_id, 'image_path': coco.loadImgs(supp_id)[0]['file_name'], 'diversity_score': diversity_score, 'pos_bboxes': pos_bboxes, 'neg_bboxes': neg_bboxes, 'present_pos_categories': [cat_to_name[cid] for cid in pos_cats], 'present_neg_categories': [cat_to_name[cid] for cid in neg_cats]})
            if not qualified_supports:
                continue
            qualified_supports.sort(key=lambda x: x['diversity_score'], reverse=True)
            top_candidates = qualified_supports[:_train_CANDIDATE_POOL_SIZE]
            if len(top_candidates) > _train_MAX_SUPPORT_POOL_SIZE:
                final_supports = random.sample(top_candidates, _train_MAX_SUPPORT_POOL_SIZE)
            else:
                final_supports = top_candidates
            random.shuffle(final_supports)
            episode_id = f'Train_AttrUnion_{_train_safe_filename(clean_attr_name)}_{total_episodes:06d}'
            attr_episodes.append({'episode_id': episode_id, 'task': 'AttrUnion', 'intent': f'ALL {clean_attr_name} objects', 'target': {'image_id': tgt_id, 'image_path': coco.loadImgs(tgt_id)[0]['file_name'], 'gt_foreground_pixels': int(fg_pixels), 'pos_bboxes': pos_tgt_bboxes, 'neg_bboxes': neg_tgt_bboxes}, 'dynamic_gt_config': {'target_attribute_id': target_attr_id}, 'support_pool': final_supports})
            total_episodes += 1
            concept_target_count += 1
        if attr_episodes:
            out_json = os.path.join(_train_OUTPUT_DB_DIR, f'{_train_safe_filename(clean_attr_name)}.json')
            with open(out_json, 'w') as f:
                json.dump(attr_episodes, f, indent=4)
            print(f'  - 生成 {len(attr_episodes)} 筆 -> {out_json}')
    print(f'\n AttrUnion 訓練集構建完成，共生成 {total_episodes} 筆。')
# END CASE LEGACY TRAIN IMPLEMENTATION


# ======================================================================
# Unified public CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build intermediate CASE AttrUnion train or validation metadata."
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
        help="PACO annotation JSON for the selected split.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Intermediate CASE metadata root. "
            "Outputs are written under vc_attr/AttrUnion."
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
