import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from data.case_dataset import CASEDataset
except ImportError as exc:
    raise ImportError(
        "Failed to import SAM 3 or CASE dataset modules. "
        "Run this script from the repository environment after installing dependencies."
    ) from exc


CANVAS_SIZE = 1008
SPLIT_RATIO = 0.5

TASKS_BY_CHAPTER = {
    "vc_cat": {"SuperEx", "FineIso", "Union"},
    "vc_attr": {"AttrEx", "AttrIso", "AttrUnion"},
}


class ClassWiseIoUMetric:
    """Pixel-aggregate IoU per concept, followed by macro averaging over concepts."""

    def __init__(self):
        self.tp = {}
        self.fp = {}
        self.fn = {}

    def update(self, pred, gt, concept_key):
        pred = np.asarray(pred).astype(bool)
        gt = np.asarray(gt).astype(bool)

        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pred), gt).sum()

        self.tp[concept_key] = self.tp.get(concept_key, 0.0) + float(tp)
        self.fp[concept_key] = self.fp.get(concept_key, 0.0) + float(fp)
        self.fn[concept_key] = self.fn.get(concept_key, 0.0) + float(fn)

    def per_concept(self):
        result = {}
        keys = sorted(set(self.tp) | set(self.fp) | set(self.fn))
        for key in keys:
            tp = self.tp.get(key, 0.0)
            fp = self.fp.get(key, 0.0)
            fn = self.fn.get(key, 0.0)
            denom = tp + fp + fn
            result[key] = tp / denom if denom > 0 else 0.0
        return result

    def compute(self):
        values = list(self.per_concept().values())
        return float(np.mean(values)) if values else 0.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resize_with_padding(image, target_size, resample, use_padding=False):
    target_w, target_h = target_size
    if not use_padding:
        resized = image.resize((target_w, target_h), resample)
        return resized, (0, 0), (target_w, target_h)

    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    resized = image.resize((new_w, new_h), resample)
    fill_color = 0 if image.mode in ("1", "L", "I", "F") else (0, 0, 0)
    canvas = Image.new(image.mode, (target_w, target_h), fill_color)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, (pad_x, pad_y), (new_w, new_h)


def create_kshot_canvas(
    ref_data,
    target_data,
    k_shot,
    canvas_size=CANVAS_SIZE,
    use_padding=False,
):
    """Create the paper's fixed 1-shot vertical or 5-shot L-shaped canvas."""

    if len(ref_data) != k_shot:
        raise ValueError(
            f"Expected exactly {k_shot} support images, got {len(ref_data)}."
        )

    if k_shot == 1:
        support_height = int(canvas_size * SPLIT_RATIO)
        target_height = canvas_size - support_height

        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

        ref_resized, ref_pad, ref_content_size = _resize_with_padding(
            ref_data[0]["image"],
            (canvas_size, support_height),
            Image.Resampling.LANCZOS,
            use_padding=use_padding,
        )
        target_resized, target_pad, target_content_size = _resize_with_padding(
            target_data["image"],
            (canvas_size, target_height),
            Image.Resampling.BILINEAR,
            use_padding=use_padding,
        )

        canvas.paste(ref_resized, (0, 0))
        canvas.paste(target_resized, (0, support_height))

        placements = {
            "ref_positions": [{
                "offset": (0, 0),
                "size": (canvas_size, support_height),
                "orig_size": ref_data[0]["image"].size,
                "content_offset": ref_pad,
                "content_size": ref_content_size,
            }],
            "tgt_position": {
                "offset": (0, support_height),
                "size": (canvas_size, target_height),
                "content_offset": (
                    target_pad[0],
                    support_height + target_pad[1],
                ),
                "content_size": target_content_size,
            },
        }
        return canvas, placements

    if k_shot == 5:
        support_thickness = int(round(canvas_size * 0.33))
        target_size = canvas_size - support_thickness

        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        ref_positions = []

        top_width = canvas_size // 3
        for idx in range(3):
            x = idx * top_width
            width = top_width if idx < 2 else canvas_size - x
            resized, pad_xy, content_size = _resize_with_padding(
                ref_data[idx]["image"],
                (width, support_thickness),
                Image.Resampling.LANCZOS,
                use_padding=use_padding,
            )
            canvas.paste(resized, (x, 0))
            ref_positions.append({
                "offset": (x, 0),
                "size": (width, support_thickness),
                "orig_size": ref_data[idx]["image"].size,
                "content_offset": pad_xy,
                "content_size": content_size,
            })

        left_height = target_size // 2
        for idx in range(2):
            y = support_thickness + idx * left_height
            height = left_height if idx == 0 else target_size - left_height
            resized, pad_xy, content_size = _resize_with_padding(
                ref_data[3 + idx]["image"],
                (support_thickness, height),
                Image.Resampling.LANCZOS,
                use_padding=use_padding,
            )
            canvas.paste(resized, (0, y))
            ref_positions.append({
                "offset": (0, y),
                "size": (support_thickness, height),
                "orig_size": ref_data[3 + idx]["image"].size,
                "content_offset": pad_xy,
                "content_size": content_size,
            })

        target_resized, target_pad, target_content_size = _resize_with_padding(
            target_data["image"],
            (target_size, target_size),
            Image.Resampling.BILINEAR,
            use_padding=use_padding,
        )
        canvas.paste(target_resized, (support_thickness, support_thickness))

        placements = {
            "ref_positions": ref_positions,
            "tgt_position": {
                "offset": (support_thickness, support_thickness),
                "size": (target_size, target_size),
                "content_offset": (
                    support_thickness + target_pad[0],
                    support_thickness + target_pad[1],
                ),
                "content_size": target_content_size,
            },
        }
        return canvas, placements

    raise ValueError(f"Unsupported shot setting: {k_shot}. Expected 1 or 5.")


def get_norm_box_for_region(
    box,
    orig_size,
    region,
    content_region=None,
    canvas_size=CANVAS_SIZE,
):
    rx, ry, rw, rh = region
    orig_w, orig_h = orig_size
    bx, by, bw, bh = box

    if content_region is None:
        scale_x = rw / orig_w
        scale_y = rh / orig_h
        new_x = bx * scale_x + rx
        new_y = by * scale_y + ry
        new_w = bw * scale_x
        new_h = bh * scale_y
    else:
        cx0, cy0, _, _ = content_region
        scale = min(rw / orig_w, rh / orig_h)
        new_x = bx * scale + cx0
        new_y = by * scale + cy0
        new_w = bw * scale
        new_h = bh * scale

    cx = (new_x + new_w / 2.0) / canvas_size
    cy = (new_y + new_h / 2.0) / canvas_size
    nw = new_w / canvas_size
    nh = new_h / canvas_size
    return [cx, cy, nw, nh]


def collect_boxes(ref_data, placements, prompt_mode, use_padding=False):
    """Transform support-image boxes into shared-canvas coordinates."""

    boxes = []
    labels = []
    include_negative = prompt_mode == "posneg"

    for idx, support in enumerate(ref_data):
        ref_pos = placements["ref_positions"][idx]
        offset = ref_pos["offset"]
        size = ref_pos["size"]
        orig_size = ref_pos["orig_size"]

        content_region = None
        if use_padding:
            content_offset = ref_pos["content_offset"]
            content_size = ref_pos["content_size"]
            content_region = (
                offset[0] + content_offset[0],
                offset[1] + content_offset[1],
                content_size[0],
                content_size[1],
            )

        for box, label in zip(
            support.get("boxes", []),
            support.get("labels", []),
        ):
            is_positive = int(label) == 1
            if not include_negative and not is_positive:
                continue

            boxes.append(
                get_norm_box_for_region(
                    box,
                    orig_size,
                    (offset[0], offset[1], size[0], size[1]),
                    content_region=content_region,
                )
            )
            labels.append(is_positive)

    return boxes, labels


@torch.inference_mode()
def encode_canvas(processor, image):
    state = {}
    image_tensor = v2.functional.to_image(image).to(processor.device)
    image_tensor = processor.transform(image_tensor).unsqueeze(0)

    state["original_height"] = image.height
    state["original_width"] = image.width
    state["backbone_out"] = processor.model.backbone.forward_image(image_tensor)
    return state


@torch.inference_mode()
def forward_grounding_logits(
    processor,
    state,
    boxes,
    labels,
    target_size=(CANVAS_SIZE, CANVAS_SIZE),
):
    """Run SAM 3 grounding and return full-resolution raw mask logits."""

    if "backbone_out" not in state:
        raise ValueError("Canvas backbone features are missing.")

    if "language_features" not in state["backbone_out"]:
        text_outputs = processor.model.backbone.forward_text(
            ["visual"],
            device=processor.device,
        )
        state["backbone_out"].update(text_outputs)

    geometric_prompt = processor.model._get_dummy_prompt()

    for box, label in zip(boxes, labels):
        box_tensor = torch.tensor(
            box,
            device=processor.device,
            dtype=torch.float32,
        ).view(1, 1, 4)
        label_tensor = torch.tensor(
            [1 if label else 0],
            device=processor.device,
            dtype=torch.long,
        ).view(1, 1)
        geometric_prompt.append_boxes(box_tensor, label_tensor)

    outputs = processor.model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=processor.find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )

    mask_logits = outputs["pred_masks"]

    if mask_logits.shape[0] == 1 and mask_logits.ndim == 4:
        mask_logits = mask_logits.squeeze(0)

    if mask_logits.ndim == 3:
        mask_logits = mask_logits.unsqueeze(1)
    elif mask_logits.ndim == 4 and mask_logits.shape[1] != 1:
        mask_logits = mask_logits[:, :1]

    return F.interpolate(
        mask_logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )


def aggregate_mask_candidates(mask_logits):
    """Element-wise maximum over SAM 3 mask candidates."""

    if mask_logits is None or mask_logits.numel() == 0:
        raise RuntimeError("SAM 3 returned no mask candidates.")

    if mask_logits.ndim != 4 or mask_logits.shape[1] != 1:
        raise ValueError(
            f"Unexpected mask-logit shape: {tuple(mask_logits.shape)}."
        )

    return mask_logits.max(dim=0)[0].squeeze(0)


def crop_query_logit_to_original_size(
    canvas_logit,
    placements,
    target_data,
    use_padding=False,
):
    target_pos = placements["tgt_position"]
    tx, ty = target_pos["offset"]
    tw, th = target_pos["size"]

    if use_padding:
        cx, cy = target_pos["content_offset"]
        cw, ch = target_pos["content_size"]
        query_logit = canvas_logit[cy:cy + ch, cx:cx + cw]
    else:
        query_logit = canvas_logit[ty:ty + th, tx:tx + tw]

    orig_w, orig_h = target_data["image"].size

    return F.interpolate(
        query_logit.unsqueeze(0).unsqueeze(0).float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def patch_sam3_for_prompt_only_inference(model):
    """Compatibility patch for SAM 3 calls without training targets."""

    if hasattr(model, "back_convert"):
        original_back_convert = model.back_convert

        def safe_back_convert(targets):
            return None if targets is None else original_back_convert(targets)

        model.back_convert = safe_back_convert

    if hasattr(model, "_compute_matching"):
        original_compute_matching = model._compute_matching

        def safe_compute_matching(outputs, targets):
            return None if targets is None else original_compute_matching(
                outputs,
                targets,
            )

        model._compute_matching = safe_compute_matching

    for module in model.modules():
        if hasattr(module, "use_act_ckpt"):
            module.use_act_ckpt = False
        if hasattr(module, "act_ckpt_enable"):
            module.act_ckpt_enable = False
        if hasattr(module, "checkpoint"):
            module.checkpoint = False


def load_cpa(checkpoint, bpe_path, device):
    print("Loading SAM 3...")
    model = build_sam3_image_model(bpe_path=str(bpe_path))

    print(f"Loading CPA checkpoint: {checkpoint}")
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model = model.merge_and_unload()

    patch_sam3_for_prompt_only_inference(model)

    model.to(device)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Concept Prompt Adaptation (CPA) on CASE."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a CPA PEFT checkpoint directory.",
    )
    parser.add_argument(
        "--chapter",
        choices=sorted(TASKS_BY_CHAPTER),
        required=True,
        help="CASE chapter: category-level (vc_cat) or attribute-level (vc_attr).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="CASE task, e.g. SuperEx, FineIso, Union, AttrEx, AttrIso, AttrUnion.",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        choices=[1, 5],
        default=1,
        help="Number of support images. Uses the paper's fixed canvas layout.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["pos", "posneg"],
        default="posneg",
        help="Use positive-only or positive-negative support prompts.",
    )
    parser.add_argument(
        "--case-data-root",
        type=Path,
        required=True,
        help="Root of the released CASE_DATA benchmark.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Root directory containing the COCO images referenced by CASE.",
    )
    parser.add_argument(
        "--bpe-path",
        type=Path,
        default=ROOT_DIR / "model" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        help="Path to SAM 3 BPE vocabulary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Mask probability threshold.",
    )
    parser.add_argument(
        "--padding",
        action="store_true",
        help="Preserve aspect ratio with padding inside each canvas cell.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for saving evaluation results as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task not in TASKS_BY_CHAPTER[args.chapter]:
        valid = ", ".join(sorted(TASKS_BY_CHAPTER[args.chapter]))
        raise ValueError(
            f"Task '{args.task}' does not belong to chapter '{args.chapter}'. "
            f"Expected one of: {valid}."
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.case_data_root.exists():
        raise FileNotFoundError(f"CASE data root not found: {args.case_data_root}")
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root not found: {args.image_root}")
    if not args.bpe_path.exists():
        raise FileNotFoundError(f"BPE vocabulary not found: {args.bpe_path}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_cpa(args.checkpoint, args.bpe_path, device)
    processor = Sam3Processor(model, device=str(device))

    dataset = CASEDataset(
        benchmark_root=args.case_data_root,
        image_root=args.image_root,
        split="eval",
        k_shot=args.k_shot,
        task=args.task,
    )

    metric = ClassWiseIoUMetric()
    evaluated = 0

    pbar = tqdm(
        range(len(dataset)),
        desc=f"CASE {args.task} | {args.k_shot}-shot | {args.prompt_mode}",
    )

    for index in pbar:
        ref_data, target_data, meta_info = dataset[index]

        if not isinstance(ref_data, list):
            ref_data = [ref_data]
        if isinstance(target_data, list):
            target_data = target_data[0]

        if "mask" not in target_data:
            raise KeyError(f"Episode {index} has no query mask.")

        if len(ref_data) != args.k_shot:
            raise ValueError(
                f"Episode {index}: expected {args.k_shot} supports, "
                f"got {len(ref_data)}."
            )

        canvas, placements = create_kshot_canvas(
            ref_data,
            target_data,
            k_shot=args.k_shot,
            canvas_size=CANVAS_SIZE,
            use_padding=args.padding,
        )

        boxes, labels = collect_boxes(
            ref_data,
            placements,
            prompt_mode=args.prompt_mode,
            use_padding=args.padding,
        )
        if not boxes:
            raise ValueError(
                f"Episode {index} has no prompts under mode '{args.prompt_mode}'."
            )

        state = encode_canvas(processor, canvas)
        mask_logits = forward_grounding_logits(
            processor,
            state,
            boxes,
            labels,
            target_size=(CANVAS_SIZE, CANVAS_SIZE),
        )

        canvas_logit = aggregate_mask_candidates(mask_logits)
        query_logit = crop_query_logit_to_original_size(
            canvas_logit,
            placements,
            target_data,
            use_padding=args.padding,
        )

        pred_mask = (
            torch.sigmoid(query_logit) > args.threshold
        ).cpu().numpy()

        gt_mask = target_data["mask"]
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.detach().cpu().numpy()
        gt_mask = np.asarray(gt_mask) > 0

        concept_key = meta_info.get("intent")
        if concept_key is None:
            raise KeyError(f"Episode {index} is missing meta_info['intent'].")

        metric.update(pred_mask, gt_mask, concept_key)
        evaluated += 1

        pbar.set_postfix({
            "mIoU": f"{metric.compute() * 100.0:.2f}",
            "episodes": evaluated,
        })

    final_miou = metric.compute()
    per_concept = metric.per_concept()

    result = {
        "chapter": args.chapter,
        "task": args.task,
        "k_shot": args.k_shot,
        "prompt_mode": args.prompt_mode,
        "threshold": args.threshold,
        "padding": args.padding,
        "checkpoint": str(args.checkpoint),
        "num_episodes": evaluated,
        "num_concepts": len(per_concept),
        "miou": final_miou,
        "miou_percent": final_miou * 100.0,
        "per_concept_iou": per_concept,
    }

    print("\n" + "=" * 72)
    print("CASE Evaluation")
    print("=" * 72)
    print(f"Task        : {args.task}")
    print(f"Shot        : {args.k_shot}")
    print(f"Prompt mode : {args.prompt_mode}")
    print(f"Episodes    : {evaluated}")
    print(f"Concepts    : {len(per_concept)}")
    print(f"mIoU        : {final_miou * 100.0:.2f}%")
    print("=" * 72)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.output_json}")


if __name__ == "__main__":
    main()
