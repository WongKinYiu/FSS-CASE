import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from data.coco20i_dataset import COCOFSSDataset
except ImportError as exc:
    raise ImportError(
        "Failed to import SAM 3 or COCO-20i dataset modules. "
        "Run this script from the repository environment after installing dependencies."
    ) from exc


CANVAS_SIZE = 1008

# Formal COCO-20i evaluation geometry used by the submitted CPA experiments.
ONE_SHOT_SUPPORT_RATIO = 0.60
ONE_SHOT_SWAP_ORDER = True

FIVE_SHOT_SUPPORT_RATIO = 0.30
FIVE_SHOT_LAYOUT = "l_shape"


class ClassWiseIoUMetric:
    """Pixel-aggregate IoU per class, followed by macro averaging over classes."""

    def __init__(self):
        self.intersections = {}
        self.unions = {}

    def update(self, pred, gt, class_id):
        pred = np.asarray(pred) > 0
        gt = np.asarray(gt) > 0

        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        self.intersections[class_id] = (
            self.intersections.get(class_id, 0.0) + float(intersection)
        )
        self.unions[class_id] = (
            self.unions.get(class_id, 0.0) + float(union)
        )

    def per_class(self):
        scores = {}
        for class_id in sorted(self.intersections):
            intersection = self.intersections[class_id]
            union = self.unions[class_id]
            scores[class_id] = intersection / union if union > 0 else 0.0
        return scores

    def compute(self):
        scores = list(self.per_class().values())
        return float(np.mean(scores)) if scores else 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint, bpe_path, device):
    print("Loading SAM 3...")
    model = build_sam3_image_model(bpe_path=str(bpe_path))

    if checkpoint is not None:
        print(f"Loading CPA checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(model, str(checkpoint))
        model = model.merge_and_unload()
        model_name = "CPA"
    else:
        print("No checkpoint provided; evaluating frozen SAM 3.")
        model_name = "Frozen SAM 3"

    model.to(device)
    model.eval()

    processor = Sam3Processor(model, device=str(device))
    return model, processor, model_name


class OneShotEvaluator:
    """Formal 1-shot COCO-20i shared-canvas evaluator."""

    def __init__(
        self,
        processor,
        support_ratio=ONE_SHOT_SUPPORT_RATIO,
        swap_order=ONE_SHOT_SWAP_ORDER,
        canvas_size=CANVAS_SIZE,
    ):
        self.processor = processor
        self.device = processor.device
        self.support_ratio = support_ratio
        self.swap_order = swap_order
        self.canvas_size = int(canvas_size)
        self.metric = ClassWiseIoUMetric()

    def create_canvas(self, support_data, query_data):
        canvas = Image.new(
            "RGB",
            (self.canvas_size, self.canvas_size),
            (0, 0, 0),
        )

        support_height = int(self.canvas_size * self.support_ratio)
        query_height = self.canvas_size - support_height

        support_rect = (
            0,
            0,
            self.canvas_size,
            support_height,
        )
        query_rect = (
            0,
            support_height,
            self.canvas_size,
            query_height,
        )

        if self.swap_order:
            support_rect = (
                0,
                query_height,
                self.canvas_size,
                support_height,
            )
            query_rect = (
                0,
                0,
                self.canvas_size,
                query_height,
            )

        placements = {
            "canvas_size": (self.canvas_size, self.canvas_size),
        }

        for key, data, rect in (
            ("support", support_data, support_rect),
            ("query", query_data, query_rect),
        ):
            x, y, width, height = rect
            resized = data["image"].resize(
                (int(width), int(height)),
                Image.Resampling.BILINEAR,
            )
            canvas.paste(resized, (x, y))

            placements[key] = {
                "offset": (x, y),
                "curr_size": (int(width), int(height)),
                "orig_size": data["image"].size,
                "orig_box": data.get("box"),
                "orig_mask": data.get("mask"),
            }

        return canvas, placements

    def get_support_box(self, placements):
        support = placements["support"]
        canvas_width, canvas_height = placements["canvas_size"]

        box = support["orig_box"]
        if box is None:
            raise ValueError("1-shot support sample is missing its prompt box.")

        bx, by, bw, bh = box
        offset_x, offset_y = support["offset"]
        current_w, current_h = support["curr_size"]
        original_w, original_h = support["orig_size"]

        scale_x = current_w / original_w
        scale_y = current_h / original_h

        x = bx * scale_x + offset_x
        y = by * scale_y + offset_y
        width = bw * scale_x
        height = bh * scale_y

        return [
            (x + width / 2.0) / canvas_width,
            (y + height / 2.0) / canvas_height,
            width / canvas_width,
            height / canvas_height,
        ]

    @torch.inference_mode()
    def predict(self, support_data, query_data):
        canvas, placements = self.create_canvas(
            support_data,
            query_data,
        )
        support_box = self.get_support_box(placements)

        state = self.processor.set_image(canvas)
        self.processor.reset_all_prompts(state)
        state = self.processor.add_geometric_prompt(
            state=state,
            box=support_box,
            label=True,
        )

        pred_masks = state.get("masks")
        if pred_masks is not None and pred_masks.shape[0] > 0:
            canvas_mask = torch.max(pred_masks[:, 0], dim=0)[0].int()
        else:
            canvas_mask = torch.zeros(
                (self.canvas_size, self.canvas_size),
                dtype=torch.int32,
                device=self.device,
            )

        query = placements["query"]
        qx, qy = query["offset"]
        qw, qh = query["curr_size"]

        query_mask = (
            canvas_mask[qy : qy + qh, qx : qx + qw]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        pred = np.array(
            Image.fromarray(query_mask).resize(
                query["orig_size"],
                Image.Resampling.NEAREST,
            )
        )
        return pred

    def run_episode(self, support_data, query_data_list, class_id):
        if isinstance(support_data, list):
            if len(support_data) != 1:
                raise ValueError(
                    f"Expected one support image for 1-shot evaluation, "
                    f"got {len(support_data)}."
                )
            support_data = support_data[0]

        for query_data in query_data_list:
            pred = self.predict(support_data, query_data)

            gt = query_data.get("mask")
            if gt is None:
                raise ValueError("Query sample is missing its ground-truth mask.")

            self.metric.update(pred, gt, class_id)


class FiveShotEvaluator:
    """Formal 5-shot COCO-20i L-shaped shared-canvas evaluator."""

    def __init__(
        self,
        processor,
        support_ratio=FIVE_SHOT_SUPPORT_RATIO,
        canvas_size=CANVAS_SIZE,
    ):
        self.processor = processor
        self.device = processor.device
        self.support_ratio = support_ratio
        self.canvas_size = int(canvas_size)
        self.metric = ClassWiseIoUMetric()

    def create_canvas(self, support_data, query_data):
        if len(support_data) != 5:
            raise ValueError(
                f"Expected five support images for 5-shot evaluation, "
                f"got {len(support_data)}."
            )

        canvas = Image.new(
            "RGB",
            (self.canvas_size, self.canvas_size),
            (0, 0, 0),
        )

        support_thickness = int(
            round(self.canvas_size * self.support_ratio)
        )
        query_size = self.canvas_size - support_thickness

        top_width = self.canvas_size // 3
        left_height = query_size // 2

        layouts = [
            (0, 0, top_width, support_thickness),
            (
                top_width,
                0,
                top_width,
                support_thickness,
            ),
            (
                top_width * 2,
                0,
                self.canvas_size - top_width * 2,
                support_thickness,
            ),
            (
                0,
                support_thickness,
                support_thickness,
                left_height,
            ),
            (
                0,
                support_thickness + left_height,
                support_thickness,
                query_size - left_height,
            ),
            (
                support_thickness,
                support_thickness,
                query_size,
                query_size,
            ),
        ]

        placements = []
        all_data = list(support_data) + [query_data]

        for index, (data, layout) in enumerate(zip(all_data, layouts)):
            is_query = index == 5
            x, y, width, height = layout

            image = data["image"]
            original_w, original_h = image.size

            resized = image.resize(
                (int(width), int(height)),
                Image.Resampling.BILINEAR,
            )
            canvas.paste(resized, (x, y))

            original_box = data.get("box")
            scaled_box = None

            if original_box is not None:
                bx, by, bw, bh = original_box
                scale_x = width / float(original_w)
                scale_y = height / float(original_h)
                scaled_box = [
                    x + bx * scale_x,
                    y + by * scale_y,
                    bw * scale_x,
                    bh * scale_y,
                ]

            placements.append(
                {
                    "type": "query" if is_query else "support",
                    "offset": (x, y),
                    "orig_size": (original_w, original_h),
                    "curr_size": (int(width), int(height)),
                    "scaled_box": scaled_box,
                    "orig_mask": data.get("mask"),
                }
            )

        return canvas, placements

    @torch.inference_mode()
    def predict(self, support_data, query_data):
        canvas, placements = self.create_canvas(
            support_data,
            query_data,
        )

        state = self.processor.set_image(canvas)
        self.processor.reset_all_prompts(state)

        for placement in placements:
            if placement["type"] != "support":
                continue

            scaled_box = placement["scaled_box"]
            if scaled_box is None:
                raise ValueError(
                    "5-shot support sample is missing its prompt box."
                )

            bx, by, bw, bh = scaled_box
            norm_box = [
                (bx + bw / 2.0) / self.canvas_size,
                (by + bh / 2.0) / self.canvas_size,
                bw / self.canvas_size,
                bh / self.canvas_size,
            ]

            state = self.processor.add_geometric_prompt(
                state=state,
                box=norm_box,
                label=True,
            )

        pred_masks = state.get("masks")
        if pred_masks is not None and pred_masks.shape[0] > 0:
            canvas_mask = torch.max(pred_masks[:, 0], dim=0)[0].int()
        else:
            canvas_mask = torch.zeros(
                (self.canvas_size, self.canvas_size),
                dtype=torch.int32,
                device=self.device,
            )

        query = next(
            placement
            for placement in placements
            if placement["type"] == "query"
        )

        qx, qy = query["offset"]
        qw, qh = query["curr_size"]

        query_mask = (
            canvas_mask[qy : qy + qh, qx : qx + qw]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        pred = np.array(
            Image.fromarray(query_mask).resize(
                query["orig_size"],
                Image.Resampling.NEAREST,
            )
        )
        return pred

    def run_episode(self, support_data, query_data_list, class_id):
        if not isinstance(support_data, list):
            raise ValueError(
                "5-shot evaluation expects support_data to be a list."
            )

        if len(support_data) != 5:
            raise ValueError(
                f"Expected five support images, got {len(support_data)}."
            )

        for query_data in query_data_list:
            pred = self.predict(support_data, query_data)

            gt = query_data.get("mask")
            if gt is None:
                raise ValueError("Query sample is missing its ground-truth mask.")

            self.metric.update(pred, gt, class_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CPA or frozen SAM 3 on COCO-20i using the "
            "shared-canvas protocol from the CASE paper."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a CPA PEFT checkpoint directory. "
            "If omitted, evaluates frozen SAM 3."
        ),
    )
    parser.add_argument(
        "--shot",
        type=int,
        choices=[1, 5],
        required=True,
        help="COCO-20i shot setting.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "COCO root containing val2014/ and "
            "annotations/instances_val2014.json."
        ),
    )
    parser.add_argument(
        "--bpe-path",
        type=Path,
        default=(
            ROOT_DIR
            / "model"
            / "sam3"
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        ),
        help="Path to the SAM 3 BPE vocabulary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="COCO-20i episode sampling seed used by the formal evaluation.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        choices=[0, 1, 2, 3],
        default=[0, 1, 2, 3],
        help="Folds to evaluate. Default: all four folds.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for saving fold-wise and mean mIoU.",
    )
    return parser.parse_args()


def validate_paths(args):
    if not args.data_root.exists():
        raise FileNotFoundError(
            f"COCO data root not found: {args.data_root}"
        )

    val_dir = args.data_root / "val2014"
    ann_file = (
        args.data_root
        / "annotations"
        / "instances_val2014.json"
    )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"COCO val2014 directory not found: {val_dir}"
        )
    if not ann_file.exists():
        raise FileNotFoundError(
            f"COCO val2014 annotations not found: {ann_file}"
        )
    if not args.bpe_path.exists():
        raise FileNotFoundError(
            f"SAM 3 BPE vocabulary not found: {args.bpe_path}"
        )
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise FileNotFoundError(
            f"CPA checkpoint not found: {args.checkpoint}"
        )


def main():
    args = parse_args()
    validate_paths(args)
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, processor, model_name = load_model(
        checkpoint=args.checkpoint,
        bpe_path=args.bpe_path,
        device=device,
    )

    fold_results = {}

    print("=" * 72)
    print("COCO-20i Evaluation")
    print("=" * 72)
    print(f"Model       : {model_name}")
    print(f"Shot        : {args.shot}")
    print(f"Seed        : {args.seed}")

    if args.shot == 1:
        print(
            "Protocol    : vertical shared canvas, "
            "support ratio=0.60, query-first ordering"
        )
    else:
        print(
            "Protocol    : L-shaped shared canvas, "
            "support ratio=0.30"
        )

    print(f"Folds       : {args.folds}")
    print("=" * 72)

    for fold in args.folds:
        dataset = COCOFSSDataset(
            lists_root=None,
            data_root=str(args.data_root),
            fold=fold,
            k_shot=args.shot,
            mode="val",
            seed=args.seed,
        )

        if args.shot == 1:
            evaluator = OneShotEvaluator(processor)
        else:
            evaluator = FiveShotEvaluator(processor)

        pbar = tqdm(
            range(len(dataset)),
            desc=f"COCO-20i Fold-{fold}",
        )

        processed_items = 0
        processed_queries = 0

        for index in pbar:
            try:
                support_data, query_data_list, class_id, _ = dataset[index]

                if not isinstance(query_data_list, list):
                    query_data_list = [query_data_list]

                evaluator.run_episode(
                    support_data,
                    query_data_list,
                    class_id,
                )

                processed_items += 1
                processed_queries += len(query_data_list)

            except Exception as exc:
                raise RuntimeError(
                    f"COCO-20i evaluation failed at "
                    f"fold={fold}, dataset_index={index}."
                ) from exc

            pbar.set_postfix(
                {
                    "mIoU": f"{evaluator.metric.compute() * 100.0:.2f}",
                }
            )

        fold_miou = evaluator.metric.compute()
        fold_results[str(fold)] = {
            "miou": fold_miou,
            "miou_percent": fold_miou * 100.0,
            "num_dataset_items": processed_items,
            "num_queries": processed_queries,
            "num_classes": len(evaluator.metric.per_class()),
            "per_class_iou": {
                str(class_id): score
                for class_id, score in evaluator.metric.per_class().items()
            },
        }

        print(
            f"Fold-{fold}: "
            f"mIoU={fold_miou * 100.0:.2f}% "
            f"({processed_items} items, "
            f"{processed_queries} queries)"
        )

    fold_mious = [
        fold_results[str(fold)]["miou"]
        for fold in args.folds
    ]
    mean_miou = float(np.mean(fold_mious))

    result = {
        "model": model_name,
        "checkpoint": (
            str(args.checkpoint)
            if args.checkpoint is not None
            else None
        ),
        "shot": args.shot,
        "seed": args.seed,
        "canvas_size": CANVAS_SIZE,
        "folds": args.folds,
        "protocol": {
            "one_shot_support_ratio": ONE_SHOT_SUPPORT_RATIO,
            "one_shot_swap_order": ONE_SHOT_SWAP_ORDER,
            "five_shot_support_ratio": FIVE_SHOT_SUPPORT_RATIO,
            "five_shot_layout": FIVE_SHOT_LAYOUT,
        },
        "fold_results": fold_results,
        "mean_miou": mean_miou,
        "mean_miou_percent": mean_miou * 100.0,
    }

    print("\n" + "=" * 72)
    print("COCO-20i Results")
    print("=" * 72)
    for fold in args.folds:
        score = fold_results[str(fold)]["miou_percent"]
        print(f"Fold-{fold:<2}     : {score:.2f}%")
    print("-" * 72)
    print(f"Mean mIoU   : {mean_miou * 100.0:.2f}%")
    print("=" * 72)

    if args.output_json is not None:
        args.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with args.output_json.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
