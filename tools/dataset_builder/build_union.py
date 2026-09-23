#!/usr/bin/env python3
"""Build intermediate CASE Union metadata.

This public builder contains the exact validation and training construction
implementations used for the CASE benchmark. The two historical implementations
are intentionally kept logically separate inside this module so that combining
the CLI does not change benchmark construction semantics.

Examples
--------
Validation:
    python tools/dataset_builder/build_union.py \\
        --split val \\
        --annotation-file /path/to/annotation.json \\
        --output-root /path/to/metadata

Training:
    python tools/dataset_builder/build_union.py \\
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
import re
import sys


# ======================================================================
# Validation implementation
# Mechanically namespaced from the verified historical validation builder.
# ======================================================================

# BEGIN CASE LEGACY VAL IMPLEMENTATION
_val_ANN_FILE = None
_val_EPISODES_DIR = None
_val_MIN_FG_PIXELS = 500
_val_UNION_INTENTS = [{'intent_name': 'Motorized Vehicles', 'pos_categories': ['car', 'motorcycle', 'bus', 'truck', 'train', 'airplane'], 'neg_categories': ['bicycle', 'skateboard']}, {'intent_name': 'Public Transportation', 'pos_categories': ['bus', 'train'], 'neg_categories': ['car', 'motorcycle']}, {'intent_name': 'Human Powered Mobility', 'pos_categories': ['bicycle', 'skateboard'], 'neg_categories': ['car', 'motorcycle']}, {'intent_name': 'Human Weight-Bearing Furniture', 'pos_categories': ['chair', 'couch', 'bed'], 'neg_categories': ['dining table', 'toilet']}, {'intent_name': 'Rest Furniture', 'pos_categories': ['bed', 'couch'], 'neg_categories': ['chair', 'dining table']}, {'intent_name': 'Indoor Decorations', 'pos_categories': ['vase', 'clock', 'potted plant'], 'neg_categories': ['chair', 'bed']}, {'intent_name': 'Domestic Pets', 'pos_categories': ['dog', 'cat'], 'neg_categories': ['cow', 'horse', 'zebra']}, {'intent_name': 'Herbivorous Mammals', 'pos_categories': ['cow', 'horse', 'sheep', 'zebra', 'giraffe', 'elephant'], 'neg_categories': ['dog', 'cat', 'bear']}, {'intent_name': 'Predatory Or Omnivorous Mammals', 'pos_categories': ['dog', 'cat', 'bear'], 'neg_categories': ['cow', 'horse', 'sheep', 'giraffe']}, {'intent_name': 'Large Wild Mammals', 'pos_categories': ['bear', 'elephant', 'giraffe'], 'neg_categories': ['dog', 'cat']}, {'intent_name': 'Portable Electronics', 'pos_categories': ['cell phone', 'laptop', 'remote'], 'neg_categories': ['tv', 'microwave', 'oven']}, {'intent_name': 'Interactive Computing Devices', 'pos_categories': ['keyboard', 'mouse', 'laptop'], 'neg_categories': ['tv', 'remote', 'cell phone']}, {'intent_name': 'Display Electronics', 'pos_categories': ['tv', 'laptop'], 'neg_categories': ['keyboard', 'mouse']}, {'intent_name': 'Eating Tools', 'pos_categories': ['fork', 'knife', 'spoon', 'bowl'], 'neg_categories': ['cup', 'wine glass']}, {'intent_name': 'Drink Containers', 'pos_categories': ['cup', 'wine glass', 'bottle'], 'neg_categories': ['fork', 'knife']}, {'intent_name': 'Cooking Appliances', 'pos_categories': ['microwave', 'oven'], 'neg_categories': ['cup', 'fork']}, {'intent_name': 'Staple Meals', 'pos_categories': ['pizza', 'sandwich', 'hot dog'], 'neg_categories': ['broccoli', 'carrot', 'cake', 'donut']}, {'intent_name': 'Fresh Produce', 'pos_categories': ['apple', 'banana', 'orange', 'broccoli', 'carrot'], 'neg_categories': ['pizza', 'cake']}, {'intent_name': 'Desserts', 'pos_categories': ['cake', 'donut'], 'neg_categories': ['apple', 'banana']}, {'intent_name': 'Portable Storage Accessories', 'pos_categories': ['backpack', 'handbag', 'suitcase'], 'neg_categories': ['tie', 'umbrella']}, {'intent_name': 'Wearable Accessories', 'pos_categories': ['tie', 'umbrella'], 'neg_categories': ['backpack', 'suitcase']}, {'intent_name': 'Ball And Racket Sports', 'pos_categories': ['sports ball', 'baseball bat', 'tennis racket'], 'neg_categories': ['skateboard', 'surfboard']}, {'intent_name': 'Board Sports', 'pos_categories': ['skateboard', 'surfboard'], 'neg_categories': ['sports ball', 'baseball bat', 'tennis racket']}, {'intent_name': 'Airborne Objects', 'pos_categories': ['airplane', 'bird', 'kite', 'frisbee'], 'neg_categories': ['car', 'bus', 'person', 'traffic light']}, {'intent_name': 'Wheeled Mobility', 'pos_categories': ['car', 'bus', 'train', 'truck', 'motorcycle', 'bicycle', 'skateboard'], 'neg_categories': ['boat', 'airplane', 'horse']}, {'intent_name': 'Street & Traffic Infrastructure', 'pos_categories': ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'], 'neg_categories': ['car', 'truck', 'person']}, {'intent_name': 'Electronic Displays', 'pos_categories': ['tv', 'laptop', 'cell phone'], 'neg_categories': ['remote', 'keyboard', 'mouse', 'microwave']}, {'intent_name': 'Personal Hygiene', 'pos_categories': ['toilet', 'sink', 'toothbrush', 'hair drier'], 'neg_categories': ['bed', 'chair', 'teddy bear']}, {'intent_name': 'Cutlery', 'pos_categories': ['fork', 'knife', 'spoon'], 'neg_categories': ['bowl', 'cup', 'wine glass', 'dining table']}, {'intent_name': 'Meal Setup', 'pos_categories': ['pizza', 'hot dog', 'sandwich', 'cup', 'wine glass', 'bowl'], 'neg_categories': ['dining table', 'chair', 'person']}]

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

def _val_calculate_union_gt_area(coco, image_id, pos_cat_ids):
    img_info = coco.loadImgs(image_id)[0]
    height, width = (img_info['height'], img_info['width'])
    gt_mask = np.zeros((height, width), dtype=bool)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=pos_cat_ids, iscrowd=False)):
        gt_mask = np.logical_or(gt_mask, coco.annToMask(ann).astype(bool))
    return np.sum(gt_mask)

def _val_parse_args():
    """Parse public benchmark-construction paths without changing task logic."""
    parser = argparse.ArgumentParser(description='Build intermediate CASE Union val metadata.')
    parser.add_argument('--annotation-file', type=Path, required=True, help='COCO 2017 annotation JSON used by this builder.')
    parser.add_argument('--output-root', type=Path, required=True, help='Intermediate CASE metadata root. Outputs are written under vc_cat/Union' + ('/train.' if 'val' == 'train' else '.'))
    return parser.parse_args()

def _val_main():
    global _val_ANN_FILE, _val_EPISODES_DIR
    args = _val_parse_args()
    _val_ANN_FILE = str(args.annotation_file.expanduser().resolve())
    _val_EPISODES_DIR = str(args.output_root.expanduser().resolve() / 'vc_cat' / 'Union')
    os.makedirs(_val_EPISODES_DIR, exist_ok=True)
    if not os.path.exists(_val_ANN_FILE):
        print(f'錯誤：找不到 {_val_ANN_FILE}')
        return
    print(f'載入 Union Val 標註檔: {_val_ANN_FILE}')
    coco = COCO(_val_ANN_FILE)
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {cat['name']: cat['id'] for cat in cats}
    id_to_name = {cat['id']: cat['name'] for cat in cats}
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
    print('\n開始生成 Union 驗證集 (嚴格 2 Pos / 1 Neg 鑑別度防守)...')
    for intent in _val_UNION_INTENTS:
        try:
            pos_cids = [name_to_id[n] for n in intent['pos_categories']]
            neg_cids = [name_to_id[n] for n in intent['neg_categories']]
        except KeyError:
            continue
        potential_target_ids = set()
        for cid in pos_cids:
            potential_target_ids.update(CatToImgs[cid])
        for target_img_id in potential_target_ids:
            img_cats = ImgToCats[target_img_id]
            if len(img_cats.intersection(pos_cids)) < 2 or not img_cats.intersection(neg_cids):
                continue
            fg_pixels = _val_calculate_union_gt_area(coco, target_img_id, pos_cids)
            if fg_pixels < _val_MIN_FG_PIXELS:
                continue
            pos_tgt_bboxes = []
            neg_tgt_bboxes = []
            pos_tgt_cats = set()
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                if ann['category_id'] in neg_cids and len(ann['bbox']) == 4:
                    bbox = ann['bbox']
                    if not _val_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.7):
                        neg_tgt_bboxes.append(bbox)
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                if ann['category_id'] in pos_cids and len(ann['bbox']) == 4:
                    bbox = ann['bbox']
                    if not _val_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.5) and (not _val_is_duplicate(bbox, pos_tgt_bboxes, iou_thresh=0.7)):
                        pos_tgt_bboxes.append(bbox)
                        pos_tgt_cats.add(ann['category_id'])
            if len(pos_tgt_cats) < 2 or not neg_tgt_bboxes:
                continue
            candidate_supp_ids = set()
            for cid in pos_cids + neg_cids:
                candidate_supp_ids.update(CatToImgs[cid])
            candidate_supp_ids -= {target_img_id}
            qualified_supports = []
            for supp_id in candidate_supp_ids:
                pos_bboxes = []
                neg_bboxes = []
                pos_cats = set()
                neg_cats = set()
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                    if ann['category_id'] in pos_cids and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _val_is_duplicate(bbox, pos_bboxes, iou_thresh=0.7):
                            pos_bboxes.append(bbox)
                            pos_cats.add(ann['category_id'])
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                    if ann['category_id'] in neg_cids and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _val_is_duplicate(bbox, pos_bboxes, iou_thresh=0.5) and (not _val_is_duplicate(bbox, neg_bboxes, iou_thresh=0.7)):
                            neg_bboxes.append(bbox)
                            neg_cats.add(ann['category_id'])
                if len(pos_cats) >= 2 and len(neg_bboxes) >= 1:
                    diversity_score = len(pos_cats) + len(neg_cats)
                    qualified_supports.append({'image_id': supp_id, 'image_path': f"val2017/{coco.loadImgs(supp_id)[0]['file_name']}", 'diversity_score': diversity_score, 'pos_bboxes': pos_bboxes, 'neg_bboxes': neg_bboxes, 'present_pos_categories': [id_to_name[pid] for pid in pos_cats], 'present_neg_categories': [id_to_name[nid] for nid in neg_cats]})
            if not qualified_supports:
                continue
            qualified_supports.sort(key=lambda x: x['diversity_score'], reverse=True)
            episode_id = f"Union_{intent['intent_name'].replace(' ', '_')}_{episode_count:04d}"
            benchmark_episodes.append({'episode_id': episode_id, 'task': 'Union', 'supercategory': 'Mixed', 'intent': intent['intent_name'], 'target': {'image_id': target_img_id, 'image_path': f"val2017/{coco.loadImgs(target_img_id)[0]['file_name']}", 'gt_foreground_pixels': int(fg_pixels), 'pos_bboxes': pos_tgt_bboxes, 'neg_bboxes': neg_tgt_bboxes}, 'dynamic_gt_config': {'positive_category_ids': pos_cids}, 'support_pool': qualified_supports})
            episode_count += 1
    out_json = os.path.join(_val_EPISODES_DIR, 'val.json')
    with open(out_json, 'w') as f:
        json.dump(benchmark_episodes, f, indent=4)
    print(f'✅ Union Val 生成完畢！共 {episode_count} 筆 -> {out_json}')
# END CASE LEGACY VAL IMPLEMENTATION


# ======================================================================
# Training implementation
# Mechanically namespaced from the verified historical training builder.
# ======================================================================

# BEGIN CASE LEGACY TRAIN IMPLEMENTATION
_train_ANN_FILE = None
_train_OUTPUT_DB_DIR = None
_train_MIN_FG_PIXELS = 500
_train_CANDIDATE_POOL_SIZE = 300
_train_MAX_SUPPORT_POOL_SIZE = 100
_train_MAX_TARGETS_PER_CONCEPT = 2000
_train_UNION_INTENTS = [{'intent_name': 'Motorized Vehicles', 'pos_categories': ['car', 'motorcycle', 'bus', 'truck', 'train', 'airplane'], 'neg_categories': ['bicycle', 'skateboard']}, {'intent_name': 'Public Transportation', 'pos_categories': ['bus', 'train'], 'neg_categories': ['car', 'motorcycle']}, {'intent_name': 'Human Powered Mobility', 'pos_categories': ['bicycle', 'skateboard'], 'neg_categories': ['car', 'motorcycle']}, {'intent_name': 'Human Weight-Bearing Furniture', 'pos_categories': ['chair', 'couch', 'bed'], 'neg_categories': ['dining table', 'toilet']}, {'intent_name': 'Rest Furniture', 'pos_categories': ['bed', 'couch'], 'neg_categories': ['chair', 'dining table']}, {'intent_name': 'Indoor Decorations', 'pos_categories': ['vase', 'clock', 'potted plant'], 'neg_categories': ['chair', 'bed']}, {'intent_name': 'Domestic Pets', 'pos_categories': ['dog', 'cat'], 'neg_categories': ['cow', 'horse', 'zebra']}, {'intent_name': 'Herbivorous Mammals', 'pos_categories': ['cow', 'horse', 'sheep', 'zebra', 'giraffe', 'elephant'], 'neg_categories': ['dog', 'cat', 'bear']}, {'intent_name': 'Predatory Or Omnivorous Mammals', 'pos_categories': ['dog', 'cat', 'bear'], 'neg_categories': ['cow', 'horse', 'sheep', 'giraffe']}, {'intent_name': 'Large Wild Mammals', 'pos_categories': ['bear', 'elephant', 'giraffe'], 'neg_categories': ['dog', 'cat']}, {'intent_name': 'Portable Electronics', 'pos_categories': ['cell phone', 'laptop', 'remote'], 'neg_categories': ['tv', 'microwave', 'oven']}, {'intent_name': 'Interactive Computing Devices', 'pos_categories': ['keyboard', 'mouse', 'laptop'], 'neg_categories': ['tv', 'remote', 'cell phone']}, {'intent_name': 'Display Electronics', 'pos_categories': ['tv', 'laptop'], 'neg_categories': ['keyboard', 'mouse']}, {'intent_name': 'Eating Tools', 'pos_categories': ['fork', 'knife', 'spoon', 'bowl'], 'neg_categories': ['cup', 'wine glass']}, {'intent_name': 'Drink Containers', 'pos_categories': ['cup', 'wine glass', 'bottle'], 'neg_categories': ['fork', 'knife']}, {'intent_name': 'Cooking Appliances', 'pos_categories': ['microwave', 'oven'], 'neg_categories': ['cup', 'fork']}, {'intent_name': 'Staple Meals', 'pos_categories': ['pizza', 'sandwich', 'hot dog'], 'neg_categories': ['broccoli', 'carrot', 'cake', 'donut']}, {'intent_name': 'Fresh Produce', 'pos_categories': ['apple', 'banana', 'orange', 'broccoli', 'carrot'], 'neg_categories': ['pizza', 'cake']}, {'intent_name': 'Desserts', 'pos_categories': ['cake', 'donut'], 'neg_categories': ['apple', 'banana']}, {'intent_name': 'Portable Storage Accessories', 'pos_categories': ['backpack', 'handbag', 'suitcase'], 'neg_categories': ['tie', 'umbrella']}, {'intent_name': 'Wearable Accessories', 'pos_categories': ['tie', 'umbrella'], 'neg_categories': ['backpack', 'suitcase']}, {'intent_name': 'Ball And Racket Sports', 'pos_categories': ['sports ball', 'baseball bat', 'tennis racket'], 'neg_categories': ['skateboard', 'surfboard']}, {'intent_name': 'Board Sports', 'pos_categories': ['skateboard', 'surfboard'], 'neg_categories': ['sports ball', 'baseball bat', 'tennis racket']}, {'intent_name': 'Airborne Objects', 'pos_categories': ['airplane', 'bird', 'kite', 'frisbee'], 'neg_categories': ['car', 'bus', 'person', 'traffic light']}, {'intent_name': 'Wheeled Mobility', 'pos_categories': ['car', 'bus', 'train', 'truck', 'motorcycle', 'bicycle', 'skateboard'], 'neg_categories': ['boat', 'airplane', 'horse']}, {'intent_name': 'Street & Traffic Infrastructure', 'pos_categories': ['traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'], 'neg_categories': ['car', 'truck', 'person']}, {'intent_name': 'Electronic Displays', 'pos_categories': ['tv', 'laptop', 'cell phone'], 'neg_categories': ['remote', 'keyboard', 'mouse', 'microwave']}, {'intent_name': 'Personal Hygiene', 'pos_categories': ['toilet', 'sink', 'toothbrush', 'hair drier'], 'neg_categories': ['bed', 'chair', 'teddy bear']}, {'intent_name': 'Cutlery', 'pos_categories': ['fork', 'knife', 'spoon'], 'neg_categories': ['bowl', 'cup', 'wine glass', 'dining table']}, {'intent_name': 'Meal Setup', 'pos_categories': ['pizza', 'hot dog', 'sandwich', 'cup', 'wine glass', 'bowl'], 'neg_categories': ['dining table', 'chair', 'person']}]

def _train_safe_filename(name):
    return re.sub('[^a-zA-Z0-9_]', '_', name.lower().strip())

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

def _train_calculate_union_gt_area(coco, image_id, pos_cat_ids):
    img_info = coco.loadImgs(image_id)[0]
    height, width = (img_info['height'], img_info['width'])
    gt_mask = np.zeros((height, width), dtype=bool)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=pos_cat_ids, iscrowd=False)):
        gt_mask = np.logical_or(gt_mask, coco.annToMask(ann).astype(bool))
    return np.sum(gt_mask)

def _train_parse_args():
    """Parse public benchmark-construction paths without changing task logic."""
    parser = argparse.ArgumentParser(description='Build intermediate CASE Union train metadata.')
    parser.add_argument('--annotation-file', type=Path, required=True, help='COCO 2017 annotation JSON used by this builder.')
    parser.add_argument('--output-root', type=Path, required=True, help='Intermediate CASE metadata root. Outputs are written under vc_cat/Union' + ('/train.' if 'train' == 'train' else '.'))
    parser.add_argument('--seed', type=int, default=2026, help='Training metadata construction seed. The formal CASE benchmark uses 2026.')
    return parser.parse_args()

def _train_main():
    global _train_ANN_FILE, _train_OUTPUT_DB_DIR
    args = _train_parse_args()
    _train_ANN_FILE = str(args.annotation_file.expanduser().resolve())
    _train_OUTPUT_DB_DIR = str(args.output_root.expanduser().resolve() / 'vc_cat' / 'Union' / 'train')
    os.makedirs(_train_OUTPUT_DB_DIR, exist_ok=True)
    random.seed(args.seed)
    if not os.path.exists(_train_ANN_FILE):
        print(f'錯誤：找不到 {_train_ANN_FILE}')
        return
    print(f'載入 Union Train 標註檔: {_train_ANN_FILE}')
    coco = COCO(_train_ANN_FILE)
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {cat['name']: cat['id'] for cat in cats}
    id_to_name = {cat['id']: cat['name'] for cat in cats}
    CatToImgs = {cat['id']: set() for cat in cats}
    for ann in coco.dataset['annotations']:
        if ann.get('iscrowd', 0):
            continue
        if 'bbox' not in ann or len(ann['bbox']) != 4:
            continue
        CatToImgs[ann['category_id']].add(ann['image_id'])
    total_episodes = 0
    print('\n[Step 2] 開始生成 Union 訓練集 (Target 放寬 + 隨機採樣池 + 嚴格 2Pos/1Neg 防守)...')
    for intent in _train_UNION_INTENTS:
        try:
            pos_cids = [name_to_id[n] for n in intent['pos_categories']]
            neg_cids = [name_to_id[n] for n in intent['neg_categories']]
        except KeyError:
            continue
        intent_episodes = []
        concept_target_count = 0
        potential_target_ids = set()
        for cid in pos_cids:
            potential_target_ids.update(CatToImgs[cid])
        for target_img_id in potential_target_ids:
            if concept_target_count >= _train_MAX_TARGETS_PER_CONCEPT:
                break
            fg_pixels = _train_calculate_union_gt_area(coco, target_img_id, pos_cids)
            if fg_pixels < _train_MIN_FG_PIXELS:
                continue
            pos_tgt_bboxes = []
            neg_tgt_bboxes = []
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                if ann['category_id'] in neg_cids and len(ann['bbox']) == 4:
                    bbox = ann['bbox']
                    if not _train_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.7):
                        neg_tgt_bboxes.append(bbox)
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=target_img_id, iscrowd=False)):
                if ann['category_id'] in pos_cids and len(ann['bbox']) == 4:
                    bbox = ann['bbox']
                    if not _train_is_duplicate(bbox, neg_tgt_bboxes, iou_thresh=0.5) and (not _train_is_duplicate(bbox, pos_tgt_bboxes, iou_thresh=0.7)):
                        pos_tgt_bboxes.append(bbox)
            if not pos_tgt_bboxes:
                continue
            candidate_supp_ids = set()
            for cid in pos_cids + neg_cids:
                candidate_supp_ids.update(CatToImgs[cid])
            candidate_supp_ids -= {target_img_id}
            qualified_supports = []
            for supp_id in candidate_supp_ids:
                pos_bboxes = []
                neg_bboxes = []
                pos_cats = set()
                neg_cats = set()
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                    if ann['category_id'] in pos_cids and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _train_is_duplicate(bbox, pos_bboxes, iou_thresh=0.7):
                            pos_bboxes.append(bbox)
                            pos_cats.add(ann['category_id'])
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=supp_id, iscrowd=False)):
                    if ann['category_id'] in neg_cids and len(ann['bbox']) == 4:
                        bbox = ann['bbox']
                        if not _train_is_duplicate(bbox, pos_bboxes, iou_thresh=0.5) and (not _train_is_duplicate(bbox, neg_bboxes, iou_thresh=0.7)):
                            neg_bboxes.append(bbox)
                            neg_cats.add(ann['category_id'])
                if len(pos_cats) >= 2 and len(neg_bboxes) >= 1:
                    diversity_score = len(pos_cats) + len(neg_cats)
                    qualified_supports.append({'image_id': supp_id, 'image_path': f"train2017/{coco.loadImgs(supp_id)[0]['file_name']}", 'diversity_score': diversity_score, 'pos_bboxes': pos_bboxes, 'neg_bboxes': neg_bboxes, 'present_pos_categories': [id_to_name[pid] for pid in pos_cats], 'present_neg_categories': [id_to_name[nid] for nid in neg_cats]})
            if not qualified_supports:
                continue
            qualified_supports.sort(key=lambda x: x['diversity_score'], reverse=True)
            top_candidates = qualified_supports[:_train_CANDIDATE_POOL_SIZE]
            if len(top_candidates) > _train_MAX_SUPPORT_POOL_SIZE:
                final_supports = random.sample(top_candidates, _train_MAX_SUPPORT_POOL_SIZE)
            else:
                final_supports = top_candidates
            random.shuffle(final_supports)
            episode_id = f"Train_Union_{_train_safe_filename(intent['intent_name'])}_{total_episodes:06d}"
            intent_episodes.append({'episode_id': episode_id, 'task': 'Union', 'supercategory': 'Mixed', 'intent': intent['intent_name'], 'target': {'image_id': target_img_id, 'image_path': f"train2017/{coco.loadImgs(target_img_id)[0]['file_name']}", 'gt_foreground_pixels': int(fg_pixels), 'pos_bboxes': pos_tgt_bboxes, 'neg_bboxes': neg_tgt_bboxes}, 'dynamic_gt_config': {'positive_category_ids': pos_cids}, 'support_pool': final_supports})
            total_episodes += 1
            concept_target_count += 1
        if intent_episodes:
            out_json = os.path.join(_train_OUTPUT_DB_DIR, f"{_train_safe_filename(intent['intent_name'])}.json")
            with open(out_json, 'w') as f:
                json.dump(intent_episodes, f, indent=4)
            print(f'  - 生成 {len(intent_episodes)} 筆 -> {out_json}')
    print(f'\n Union 訓練集構建完成，共生成 {total_episodes} 筆。')
# END CASE LEGACY TRAIN IMPLEMENTATION


# ======================================================================
# Unified public CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build intermediate CASE Union train or validation metadata."
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
            "Outputs are written under vc_cat/Union."
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
