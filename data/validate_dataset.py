from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import config as cfg


@dataclass
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


def _token_from_stem(stem: str) -> str:
    marker = "_CAM_"
    idx = stem.find(marker)
    if idx == -1:
        return stem
    return stem[:idx]


def _scan_pairs(images_dir: Path, masks_dir: Path) -> Dict[str, object]:
    image_stems = {p.stem for p in images_dir.glob("*.jpg")}
    mask_stems = {p.stem for p in masks_dir.glob("*.png")}

    paired = sorted(image_stems & mask_stems)
    missing_images = sorted(mask_stems - image_stems)
    missing_masks = sorted(image_stems - mask_stems)

    return {
        "paired": paired,
        "missing_images": missing_images,
        "missing_masks": missing_masks,
    }


def _group_by_token(stems: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for stem in stems:
        grouped[_token_from_stem(stem)].append(stem)
    for token in grouped:
        grouped[token].sort()
    return grouped


def _token_split(tokens: List[str], ratios: SplitRatios, seed: int) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    tokens = list(tokens)
    rng.shuffle(tokens)

    n = len(tokens)
    n_train = max(1, int(n * ratios.train))
    n_val = max(1, int(n * ratios.val))
    n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train : n_train + n_val]
    test_tokens = tokens[n_train + n_val :]

    return {"train": sorted(train_tokens), "val": sorted(val_tokens), "test": sorted(test_tokens)}


def _write_split_files(split_dir: Path, grouped: Dict[str, List[str]], token_splits: Dict[str, List[str]]) -> Dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    pair_counts: Dict[str, int] = {}

    for split_name, tokens in token_splits.items():
        stems: List[str] = []
        for token in tokens:
            stems.extend(grouped[token])
        stems = sorted(stems)

        out_path = split_dir / f"{split_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(stems))
        pair_counts[split_name] = len(stems)

    return pair_counts


def _read_split_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _validate_samples(stems: List[str], images_dir: Path, masks_dir: Path, empty_thresh: float) -> Dict[str, object]:
    unreadable_images = 0
    unreadable_masks = 0
    shape_mismatch = 0
    empty_masks = 0

    for stem in stems:
        img_path = images_dir / f"{stem}.jpg"
        mask_path = masks_dir / f"{stem}.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            unreadable_images += 1
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            unreadable_masks += 1
            continue

        if img.shape[:2] != mask.shape[:2]:
            shape_mismatch += 1

        fg_ratio = float((mask > 0).mean())
        if fg_ratio < empty_thresh:
            empty_masks += 1

    total = max(1, len(stems))
    return {
        "unreadable_images": unreadable_images,
        "unreadable_masks": unreadable_masks,
        "shape_mismatch": shape_mismatch,
        "empty_masks": empty_masks,
        "empty_mask_ratio": empty_masks / total,
    }


def _split_leakage_report(token_splits: Dict[str, List[str]]) -> Dict[str, object]:
    seen: Dict[str, str] = {}
    collisions: List[Tuple[str, str, str]] = []

    for split, tokens in token_splits.items():
        for t in tokens:
            if t in seen and seen[t] != split:
                collisions.append((t, seen[t], split))
            seen[t] = split

    return {
        "has_leakage": len(collisions) > 0,
        "leakage_count": len(collisions),
        "examples": collisions[:20],
    }


def run_validation(write_splits: bool, seed: int, empty_thresh: float) -> Dict[str, object]:
    cfg.ensure_paths()

    scanned = _scan_pairs(cfg.IMAGES_DIR, cfg.MASKS_DIR)
    paired_stems: List[str] = scanned["paired"]

    grouped = _group_by_token(paired_stems)
    all_tokens = sorted(grouped.keys())

    token_splits = _token_split(all_tokens, SplitRatios(), seed)
    if write_splits:
        pair_counts = _write_split_files(cfg.SPLITS_DIR, grouped, token_splits)
    else:
        pair_counts = {}

    # Validate from written split files if available, otherwise use generated split map.
    split_stems: Dict[str, List[str]] = {}
    for split in ["train", "val", "test"]:
        p = cfg.SPLITS_DIR / f"{split}.txt"
        stems = _read_split_file(p)
        if stems:
            split_stems[split] = stems
        else:
            tmp: List[str] = []
            for t in token_splits[split]:
                tmp.extend(grouped[t])
            split_stems[split] = sorted(tmp)

    split_tokens = {
        split: sorted({_token_from_stem(s) for s in stems})
        for split, stems in split_stems.items()
    }

    leakage = _split_leakage_report(split_tokens)

    quality = {
        split: _validate_samples(stems, cfg.IMAGES_DIR, cfg.MASKS_DIR, empty_thresh)
        for split, stems in split_stems.items()
    }

    report = {
        "summary": {
            "total_images": len(list(cfg.IMAGES_DIR.glob("*.jpg"))),
            "total_masks": len(list(cfg.MASKS_DIR.glob("*.png"))),
            "paired_samples": len(paired_stems),
            "missing_images_for_masks": len(scanned["missing_images"]),
            "missing_masks_for_images": len(scanned["missing_masks"]),
            "total_token_groups": len(all_tokens),
        },
        "split_counts": {
            "tokens": {k: len(v) for k, v in split_tokens.items()},
            "pairs": {k: len(v) for k, v in split_stems.items()},
            "pairs_written": pair_counts,
        },
        "leakage": leakage,
        "quality": quality,
        "examples": {
            "missing_images_for_masks": scanned["missing_images"][:20],
            "missing_masks_for_images": scanned["missing_masks"][:20],
        },
    }

    report_path = cfg.LOG_DIR / "data_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Data validation complete")
    print(f"Report: {report_path}")
    print(
        "Summary:",
        f"paired={report['summary']['paired_samples']}",
        f"tokens={report['summary']['total_token_groups']}",
        f"missing_img={report['summary']['missing_images_for_masks']}",
        f"missing_mask={report['summary']['missing_masks_for_images']}",
    )
    print(
        "Splits:",
        f"train={report['split_counts']['pairs']['train']}",
        f"val={report['split_counts']['pairs']['val']}",
        f"test={report['split_counts']['pairs']['test']}",
    )
    print(
        "Leakage:",
        "YES" if report["leakage"]["has_leakage"] else "NO",
        f"count={report['leakage']['leakage_count']}",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe splits and validate dataset integrity")
    parser.add_argument("--write-splits", action="store_true", help="Write train/val/test split files")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--empty-mask-thresh", type=float, default=0.005)
    args = parser.parse_args()

    run_validation(write_splits=args.write_splits, seed=args.seed, empty_thresh=args.empty_mask_thresh)


if __name__ == "__main__":
    main()
