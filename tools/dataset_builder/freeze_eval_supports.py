#!/usr/bin/env python3
"""Freeze the official nested 1-shot / 5-shot CASE evaluation supports.

This utility operates on the intermediate validation metadata produced by the
CASE task builders. For every task it samples five support images once and
defines the 1-shot support as the first item of that same 5-shot set.

This preserves the evaluation protocol used by the CASE paper:

    global seed = 42
    1-shot support ⊂ 5-shot supports

The output of this script is internal fixed-split metadata. Use
`compile_case_data.py` afterwards to produce the public frozen-RLE CASE_DATA
release.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence


DEFAULT_SEED = 42

TASKS = (
    ("vc_cat", "SuperEx"),
    ("vc_cat", "FineIso"),
    ("vc_cat", "Union"),
    ("vc_attr", "AttrIso"),
    ("vc_attr", "AttrUnion"),
    ("vc_attr", "AttrEx"),
)


def freeze_task(
    metadata_root: Path,
    output_root: Path,
    chapter: str,
    task: str,
    seed: int,
) -> dict:
    input_path = (
        metadata_root
        / chapter
        / task
        / "val.json"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Validation metadata not found for {task}: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        episodes = json.load(handle)

    if not isinstance(episodes, list):
        raise ValueError(
            f"Expected a JSON episode list: {input_path}"
        )

    # Exact submitted protocol: reset the RNG for every CASE task.
    rng = random.Random(seed)

    frozen = {
        1: [],
        5: [],
    }

    skipped = 0

    for episode in episodes:
        pool = episode.get("support_pool", [])

        if not pool:
            skipped += 1
            continue

        if len(pool) >= 5:
            sampled_5 = rng.sample(
                pool,
                5,
            )
        else:
            # Preserve the original protocol for concepts with fewer than
            # five candidate support images.
            sampled_5 = rng.choices(
                pool,
                k=5,
            )

        base_episode = {
            key: value
            for key, value in episode.items()
            if key != "support_pool"
        }

        one_shot = dict(base_episode)
        one_shot["support"] = sampled_5[:1]

        five_shot = dict(base_episode)
        five_shot["support"] = sampled_5[:5]

        frozen[1].append(one_shot)
        frozen[5].append(five_shot)

    output_dir = (
        output_root
        / chapter
        / task
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for shot in (1, 5):
        output_path = (
            output_dir
            / f"{task.lower()}_{shot}shot_seed{seed}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                frozen[shot],
                handle,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[{task}] {shot}-shot: "
            f"{len(frozen[shot]):,} episodes -> {output_path}"
        )

    if skipped:
        print(
            f"[{task}] skipped {skipped:,} episodes "
            "with an empty support pool"
        )

    return {
        "input": len(episodes),
        "output": len(frozen[5]),
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the official nested CASE 1-shot/5-shot "
            "evaluation support sets."
        )
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help=(
            "Intermediate CASE metadata root containing "
            "vc_cat/<task>/val.json and vc_attr/<task>/val.json."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination for the fixed official evaluation splits.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Support sampling seed. The paper protocol uses 42.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.metadata_root.exists():
        raise FileNotFoundError(
            f"Metadata root not found: {args.metadata_root}"
        )

    total = 0

    for chapter, task in TASKS:
        result = freeze_task(
            metadata_root=args.metadata_root,
            output_root=args.output_root,
            chapter=chapter,
            task=task,
            seed=args.seed,
        )

        total += result["output"]

    print()
    print(
        f"Frozen CASE evaluation queries per shot: "
        f"{total:,}"
    )


if __name__ == "__main__":
    main()
