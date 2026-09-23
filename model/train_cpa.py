"""Train Concept Prompt Adaptation (CPA) on CASE.

This script implements the training protocol described in the CASE paper:
  * frozen SAM 3 backbone and pretrained weights;
  * LoRA on transformer encoder/decoder q/k/v/out projections and FFN layers;
  * CASE supervision under positive-negative prompting;
  * positive-only preservation against frozen SAM 3;
  * 1-shot vertical and 5-shot L-shaped shared-canvas layouts.

The script expects the released CASE training metadata and the source
images referenced by that metadata. Target masks are provided as frozen
COCO-style RLE in the CASE release, so runtime training does not require
COCO/PACO annotation files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms import v2
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from data.case_dataset import CASEDataset
except ImportError as exc:
    raise ImportError(
        "Failed to import SAM 3 or CASE dataset modules. "
        "Run this script from the repository environment and verify that "
        "model/sam3 and data/case_dataset.py are available."
    ) from exc


CANVAS_SIZE = 1008
ONE_SHOT_SUPPORT_RATIO = 0.5
FIVE_SHOT_SUPPORT_RATIO = 0.33
LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "linear1",
    "linear2",
)


def resize_with_padding(
    image: Image.Image,
    target_size: Tuple[int, int],
    resample: Image.Resampling,
    use_padding: bool = False,
):
    """Resize an image to ``target_size``, optionally preserving aspect ratio."""
    target_w, target_h = target_size
    if not use_padding:
        return image.resize((target_w, target_h), resample), (0, 0), (target_w, target_h)

    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    resized = image.resize((new_w, new_h), resample)
    fill = 0 if image.mode in ("1", "L", "I", "F") else (0, 0, 0)
    output = Image.new(image.mode, (target_w, target_h), fill)
    output.paste(resized, (pad_x, pad_y))
    return output, (pad_x, pad_y), (new_w, new_h)


def resize_mask_with_padding(
    mask: np.ndarray,
    target_size: Tuple[int, int],
    use_padding: bool = False,
) -> np.ndarray:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized, _, _ = resize_with_padding(
        mask_image,
        target_size,
        Image.Resampling.NEAREST,
        use_padding=use_padding,
    )
    return np.asarray(resized) > 128


def create_shared_canvas(
    supports: Sequence[Dict],
    query: Dict,
    k_shot: int,
    canvas_size: int = CANVAS_SIZE,
    use_padding: bool = False,
):
    """Compose support/query images using the paper's fixed shared-canvas layouts."""
    if k_shot == 1:
        if len(supports) < 1:
            raise ValueError("1-shot training requires at least one support image.")

        support_h = int(canvas_size * ONE_SHOT_SUPPORT_RATIO)
        query_h = canvas_size - support_h
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))

        support_image, support_pad, support_content_size = resize_with_padding(
            supports[0]["image"],
            (canvas_size, support_h),
            Image.Resampling.LANCZOS,
            use_padding=use_padding,
        )
        query_image, query_pad, query_content_size = resize_with_padding(
            query["image"],
            (canvas_size, query_h),
            Image.Resampling.BILINEAR,
            use_padding=use_padding,
        )

        canvas.paste(support_image, (0, 0))
        canvas.paste(query_image, (0, support_h))

        placements = {
            "ref_positions": [
                {
                    "offset": (0, 0),
                    "size": (canvas_size, support_h),
                    "orig_size": supports[0]["image"].size,
                    "content_offset": support_pad,
                    "content_size": support_content_size,
                }
            ],
            "tgt_position": {
                "offset": (0, support_h),
                "size": (canvas_size, query_h),
                "content_offset": (query_pad[0], support_h + query_pad[1]),
                "content_size": query_content_size,
            },
        }
        return canvas, placements

    if k_shot == 5:
        if len(supports) < 5:
            raise ValueError(f"5-shot training requires five support images, got {len(supports)}.")

        support_thickness = int(round(canvas_size * FIVE_SHOT_SUPPORT_RATIO))
        query_size = canvas_size - support_thickness
        canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
        ref_positions = []

        # Three support images along the top row.
        top_w = canvas_size // 3
        for index in range(3):
            x = index * top_w
            width = top_w if index < 2 else canvas_size - x
            support_image, pad_xy, content_size = resize_with_padding(
                supports[index]["image"],
                (width, support_thickness),
                Image.Resampling.LANCZOS,
                use_padding=use_padding,
            )
            canvas.paste(support_image, (x, 0))
            ref_positions.append(
                {
                    "offset": (x, 0),
                    "size": (width, support_thickness),
                    "orig_size": supports[index]["image"].size,
                    "content_offset": pad_xy,
                    "content_size": content_size,
                }
            )

        # Two support images along the left column.
        left_h = query_size // 2
        for index in range(2):
            y = support_thickness + index * left_h
            height = left_h if index == 0 else query_size - left_h
            support_image, pad_xy, content_size = resize_with_padding(
                supports[3 + index]["image"],
                (support_thickness, height),
                Image.Resampling.LANCZOS,
                use_padding=use_padding,
            )
            canvas.paste(support_image, (0, y))
            ref_positions.append(
                {
                    "offset": (0, y),
                    "size": (support_thickness, height),
                    "orig_size": supports[3 + index]["image"].size,
                    "content_offset": pad_xy,
                    "content_size": content_size,
                }
            )

        query_image, query_pad, query_content_size = resize_with_padding(
            query["image"],
            (query_size, query_size),
            Image.Resampling.BILINEAR,
            use_padding=use_padding,
        )
        canvas.paste(query_image, (support_thickness, support_thickness))

        placements = {
            "ref_positions": ref_positions,
            "tgt_position": {
                "offset": (support_thickness, support_thickness),
                "size": (query_size, query_size),
                "content_offset": (
                    support_thickness + query_pad[0],
                    support_thickness + query_pad[1],
                ),
                "content_size": query_content_size,
            },
        }
        return canvas, placements

    raise ValueError(f"Unsupported shot setting: {k_shot}. Expected 1 or 5.")


def transform_box_to_canvas(
    box: Sequence[float],
    original_size: Tuple[int, int],
    region: Tuple[int, int, int, int],
    content_region: Tuple[int, int, int, int] | None = None,
) -> List[float]:
    """Map an absolute xywh support box to normalized canvas cxcywh coordinates."""
    region_x, region_y, region_w, region_h = region
    original_w, original_h = original_size
    box_x, box_y, box_w, box_h = box

    if content_region is None:
        scale_x = region_w / original_w
        scale_y = region_h / original_h
        new_x = box_x * scale_x + region_x
        new_y = box_y * scale_y + region_y
        new_w = box_w * scale_x
        new_h = box_h * scale_y
    else:
        content_x, content_y, _, _ = content_region
        scale = min(region_w / original_w, region_h / original_h)
        new_x = box_x * scale + content_x
        new_y = box_y * scale + content_y
        new_w = box_w * scale
        new_h = box_h * scale

    center_x = (new_x + new_w / 2.0) / CANVAS_SIZE
    center_y = (new_y + new_h / 2.0) / CANVAS_SIZE
    norm_w = new_w / CANVAS_SIZE
    norm_h = new_h / CANVAS_SIZE
    return [center_x, center_y, norm_w, norm_h]


def collect_prompt_boxes(
    supports: Sequence[Dict],
    placements: Dict,
    include_negative: bool,
    use_padding: bool = False,
):
    """Collect positive-only or positive-negative support prompts."""
    boxes: List[List[float]] = []
    labels: List[bool] = []

    for index, support in enumerate(supports):
        placement = placements["ref_positions"][index]
        offset = placement["offset"]
        size = placement["size"]
        original_size = placement["orig_size"]

        content_region = None
        if use_padding:
            content_offset = placement["content_offset"]
            content_size = placement["content_size"]
            content_region = (
                content_offset[0] + offset[0],
                content_offset[1] + offset[1],
                content_size[0],
                content_size[1],
            )

        for box, label in zip(support.get("boxes", []), support.get("labels", [])):
            is_positive = int(label) == 1
            if not include_negative and not is_positive:
                continue

            boxes.append(
                transform_box_to_canvas(
                    box,
                    original_size,
                    (offset[0], offset[1], size[0], size[1]),
                    content_region=content_region,
                )
            )
            labels.append(is_positive)

    return boxes, labels


@torch.no_grad()
def encode_canvas(processor: Sam3Processor, image: Image.Image) -> Dict:
    """Compute frozen image-backbone features shared by the three CPA forwards."""
    state: Dict = {}
    image_tensor = v2.functional.to_image(image).to(processor.device)
    image_tensor = processor.transform(image_tensor).unsqueeze(0)
    state["original_height"] = image.height
    state["original_width"] = image.width
    state["backbone_out"] = processor.model.backbone.forward_image(image_tensor)
    return state


def forward_grounding_logits(
    processor: Sam3Processor,
    state: Dict,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[bool],
    target_size: Tuple[int, int] = (CANVAS_SIZE, CANVAS_SIZE),
) -> torch.Tensor:
    """Run SAM 3 grounding and return full-resolution raw mask logits."""
    if "backbone_out" not in state:
        raise ValueError("encode_canvas() must be called before grounding.")

    if "language_features" not in state["backbone_out"]:
        text_outputs = processor.model.backbone.forward_text(
            ["visual"],
            device=processor.device,
        )
        state["backbone_out"].update(text_outputs)

    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

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
        state["geometric_prompt"].append_boxes(box_tensor, label_tensor)

    outputs = processor.model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=processor.find_stage,
        geometric_prompt=state["geometric_prompt"],
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


def aggregate_mask_candidates(mask_logits: torch.Tensor | None) -> torch.Tensor | None:
    """Aggregate SAM 3 mask candidates by element-wise maximum, as in the paper."""
    if mask_logits is None or mask_logits.numel() == 0:
        return None
    if mask_logits.ndim >= 3 and mask_logits.shape[0] > 1:
        mask_logits = mask_logits.max(dim=0, keepdim=True).values
    return mask_logits


def crop_region(tensor: torch.Tensor, region: Tuple[int, int, int, int]) -> torch.Tensor:
    x, y, width, height = map(int, region)
    return tensor[..., y : y + height, x : x + width]


def mask_loss(
    mask_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    target_region: Tuple[int, int, int, int],
) -> torch.Tensor:
    """20 * sigmoid focal loss + Dice loss on the query region."""
    mask_logits = aggregate_mask_candidates(mask_logits)
    if mask_logits is None:
        raise ValueError("SAM 3 returned no mask candidates.")

    if mask_logits.ndim == 3:
        mask_logits = mask_logits.unsqueeze(0)

    gt_mask = gt_mask.to(device=mask_logits.device, dtype=mask_logits.dtype)
    if gt_mask.ndim == 2:
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
    elif gt_mask.ndim == 3:
        gt_mask = gt_mask.unsqueeze(1)

    if mask_logits.shape[-2:] != gt_mask.shape[-2:]:
        mask_logits = F.interpolate(
            mask_logits,
            size=gt_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    mask_logits = crop_region(mask_logits, target_region)
    gt_mask = crop_region(gt_mask, target_region)

    focal = sigmoid_focal_loss(
        mask_logits,
        gt_mask,
        alpha=0.25,
        gamma=2.0,
        reduction="mean",
    )
    probabilities = torch.sigmoid(mask_logits)
    intersection = (probabilities * gt_mask).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return 20.0 * focal + dice.mean()


def prepare_cpa_model(
    bpe_path: str,
    rank: int,
    alpha: int,
    dropout: float,
):
    """Load frozen SAM 3 and attach CPA LoRA adapters."""
    print("[model] Loading frozen SAM 3...")
    model = build_sam3_image_model(bpe_path=bpe_path)

    for parameter in model.parameters():
        parameter.requires_grad = False

    target_modules = [
        name
        for name, _ in model.named_modules()
        if ("transformer.encoder" in name or "transformer.decoder" in name)
        and name.endswith(LORA_TARGET_SUFFIXES)
    ]
    if not target_modules:
        raise RuntimeError("No CPA LoRA target modules were found in the SAM 3 model.")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )
    cpa_model = get_peft_model(model, lora_config)

    # SAM 3 compatibility for inference-style grounding during CPA training.
    base_sam3 = (
        cpa_model.base_model.model
        if hasattr(cpa_model, "base_model")
        else cpa_model
    )
    original_back_convert = base_sam3.back_convert
    original_compute_matching = base_sam3._compute_matching

    def back_convert_without_targets(targets):
        return None if targets is None else original_back_convert(targets)

    def compute_matching_without_targets(output, targets):
        return None if targets is None else original_compute_matching(output, targets)

    base_sam3.back_convert = back_convert_without_targets
    base_sam3._compute_matching = compute_matching_without_targets

    # Activation checkpointing is disabled because CPA trains only LoRA adapters.
    for module in cpa_model.modules():
        if hasattr(module, "use_act_ckpt"):
            module.use_act_ckpt = False
        if hasattr(module, "act_ckpt_enable"):
            module.act_ckpt_enable = False
        if hasattr(module, "checkpoint"):
            module.checkpoint = False

    print(f"[model] LoRA target modules: {len(target_modules)}")
    cpa_model.print_trainable_parameters()
    return cpa_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_data_worker(worker_id: int, base_seed: int) -> None:
    """Seed Python/NumPy in a DataLoader worker using the submitted protocol."""
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def collate_single(batch):
    """Unwrap the single episode used by batch_size=1 training."""
    return batch[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Concept Prompt Adaptation (CPA) on CASE."
    )

    # Data and output paths. Defaults match the repository layout but the large
    # datasets/metadata are expected to be downloaded separately.
    parser.add_argument(
        "--case-data-root",
        type=str,
        required=True,
        help="Root of the released CASE_DATA benchmark.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=str(REPO_ROOT / "data" / "coco"),
        help="Image root used by CASE metadata image_path entries.",
    )
    parser.add_argument(
        "--bpe-path",
        type=str,
        default=str(SCRIPT_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        help="SAM 3 BPE vocabulary path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for CPA LoRA checkpoints and the resolved training config.",
    )

    # Paper defaults.
    parser.add_argument("--k-shot", type=int, default=1, choices=(1, 5))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--epoch-size", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-pres", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--padding",
        action="store_true",
        help="Preserve image aspect ratio with zero padding. Disabled in the default paper configuration.",
    )
    return parser


def validate_paths(args: argparse.Namespace) -> None:
    required = {
        "CASE data root": args.case_data_root,
        "image root": args.image_root,
        "SAM 3 BPE vocabulary": args.bpe_path,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required training data/files:\n  - " + "\n  - ".join(missing)
        )


def train(args: argparse.Namespace) -> None:
    validate_paths(args)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = vars(args).copy()
    resolved_config["canvas_size"] = CANVAS_SIZE
    resolved_config["layout"] = "vertical" if args.k_shot == 1 else "l_shape"
    resolved_config["device"] = device
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2)

    model = prepare_cpa_model(
        args.bpe_path,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.to(device)
    model.train()

    sam3_model = model.base_model.model if hasattr(model, "base_model") else model
    processor = Sam3Processor(sam3_model, device=device)

    dataset = CASEDataset(
        benchmark_root=args.case_data_root,
        image_root=args.image_root,
        split="train",
        k_shot=args.k_shot,
        epoch_size=args.epoch_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_single,
        worker_init_fn=partial(seed_data_worker, base_seed=args.seed),
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=(device == "cuda"))

    print(
        f"[train] k={args.k_shot}, layout={resolved_config['layout']}, "
        f"epochs={args.num_epochs}, steps/epoch={len(dataloader)}, device={device}"
    )

    for epoch in range(args.num_epochs):
        model.train()
        running_total: List[float] = []
        running_svps: List[float] = []
        running_pres: List[float] = []

        progress = tqdm(dataloader, desc=f"CPA {epoch + 1}/{args.num_epochs}")
        for supports, query, meta_info in progress:
            if not isinstance(supports, list):
                supports = [supports]
            if isinstance(query, list):
                query = query[0]
            if "mask" not in query:
                continue

            optimizer.zero_grad(set_to_none=True)

            canvas, placements = create_shared_canvas(
                supports,
                query,
                k_shot=args.k_shot,
                canvas_size=CANVAS_SIZE,
                use_padding=args.padding,
            )

            query_offset = placements["tgt_position"]["offset"]
            query_size = placements["tgt_position"]["size"]
            query_region = (
                query_offset[0],
                query_offset[1],
                query_size[0],
                query_size[1],
            )

            query_mask = query["mask"]
            if isinstance(query_mask, torch.Tensor):
                query_mask = query_mask.detach().cpu().numpy()
            resized_query_mask = resize_mask_with_padding(
                query_mask,
                query_size,
                use_padding=args.padding,
            )

            canvas_gt_mask = torch.zeros(
                (CANVAS_SIZE, CANVAS_SIZE),
                dtype=torch.float32,
                device=device,
            )
            x, y = query_offset
            width, height = query_size
            canvas_gt_mask[y : y + height, x : x + width] = torch.from_numpy(
                resized_query_mask.astype(np.float32)
            ).to(device)

            backbone_state = encode_canvas(processor, canvas)
            posneg_boxes, posneg_labels = collect_prompt_boxes(
                supports,
                placements,
                include_negative=True,
                use_padding=args.padding,
            )
            pos_boxes, pos_labels = collect_prompt_boxes(
                supports,
                placements,
                include_negative=False,
                use_padding=args.padding,
            )

            with autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=(device == "cuda"),
            ):
                # Frozen SAM 3 under positive-only prompting.
                prob_frozen_pos = None
                if pos_boxes:
                    with torch.no_grad(), model.disable_adapter():
                        frozen_state = deepcopy(backbone_state)
                        processor.reset_all_prompts(frozen_state)
                        logits_frozen_pos = aggregate_mask_candidates(
                            forward_grounding_logits(
                                processor,
                                frozen_state,
                                pos_boxes,
                                pos_labels,
                            )
                        )
                        if logits_frozen_pos is not None:
                            prob_frozen_pos = torch.sigmoid(
                                crop_region(logits_frozen_pos, query_region)
                            ).detach()

                # CPA under positive-only prompting.
                prob_cpa_pos = None
                if pos_boxes:
                    cpa_pos_state = deepcopy(backbone_state)
                    processor.reset_all_prompts(cpa_pos_state)
                    logits_cpa_pos = aggregate_mask_candidates(
                        forward_grounding_logits(
                            processor,
                            cpa_pos_state,
                            pos_boxes,
                            pos_labels,
                        )
                    )
                    if logits_cpa_pos is not None:
                        prob_cpa_pos = torch.sigmoid(
                            crop_region(logits_cpa_pos, query_region)
                        )

                # CPA under positive-negative prompting.
                cpa_posneg_state = deepcopy(backbone_state)
                processor.reset_all_prompts(cpa_posneg_state)
                logits_cpa_posneg = aggregate_mask_candidates(
                    forward_grounding_logits(
                        processor,
                        cpa_posneg_state,
                        posneg_boxes,
                        posneg_labels,
                    )
                )
                if logits_cpa_posneg is None:
                    continue

                loss_svps = mask_loss(
                    logits_cpa_posneg,
                    canvas_gt_mask,
                    target_region=query_region,
                )
                loss_pres = torch.zeros((), device=device)
                if prob_frozen_pos is not None and prob_cpa_pos is not None:
                    loss_pres = F.mse_loss(prob_cpa_pos, prob_frozen_pos)

                loss = loss_svps + args.lambda_pres * loss_pres

            if loss.detach().item() == 0.0:
                del backbone_state
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_total.append(float(loss.detach().cpu()))
            running_svps.append(float(loss_svps.detach().cpu()))
            running_pres.append(float(loss_pres.detach().cpu()))

            progress.set_postfix(
                {
                    "loss": f"{np.mean(running_total):.4f}",
                    "svps": f"{np.mean(running_svps):.4f}",
                    "pres": f"{np.mean(running_pres):.4f}",
                    "task": meta_info.get("task", "N/A"),
                }
            )

            del backbone_state
            if device == "cuda":
                torch.cuda.empty_cache()

        epoch_dir = output_dir / f"epoch_{epoch + 1:02d}"
        model.save_pretrained(epoch_dir)
        print(f"[checkpoint] Saved CPA LoRA weights to {epoch_dir}")


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
