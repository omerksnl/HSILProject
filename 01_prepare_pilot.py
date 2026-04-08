import argparse
import csv
import random
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(dataset_root: Path):
    files = []
    for p in dataset_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return files


def image_id_key(path: Path):
    stem = path.stem
    return int(stem) if stem.isdigit() else 10**9


def parse_image_id(name: str):
    stem = Path(str(name)).stem
    return int(stem) if stem.isdigit() else None


def rows_from_images(images, dataset_root: Path):
    rows = []
    for img in images:
        rel = img.relative_to(dataset_root).as_posix()
        rows.append(
            {
                "image_path": rel,
                "image_name": img.name,
                "label": "",
                "confidence": "",
                "notes": "",
            }
        )
    return rows


def write_template_rows(out_csv: Path, rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "image_name", "label", "confidence", "notes"])
        for r in rows:
            writer.writerow(
                [
                    r.get("image_path", ""),
                    r.get("image_name", ""),
                    r.get("label", ""),
                    r.get("confidence", ""),
                    r.get("notes", ""),
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Sample pilot images and create manual labeling template."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="spinal-ai2024",
        help="Root directory that contains subset folders and images.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of images for pilot labeling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "first_ids"],
        help="random: sample randomly, first_ids: take smallest numeric image IDs.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="labels/pilot_labels_template.csv",
        help="Output CSV path for annotation template.",
    )
    parser.add_argument("--start-id", type=int, default=0, help="Inclusive start image ID.")
    parser.add_argument("--end-id", type=int, default=0, help="Inclusive end image ID.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new range rows to existing CSV (dedupe by image_name).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_csv = Path(args.out_csv).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    images = list_images(dataset_root)
    if len(images) == 0:
        raise RuntimeError(
            f"No images found under {dataset_root}. "
            "Check your subset folder structure."
        )
    if args.sample_size > len(images):
        raise ValueError(
            f"sample-size ({args.sample_size}) > total images ({len(images)})."
        )

    if args.start_id > 0 and args.end_id > 0:
        if args.start_id > args.end_id:
            raise ValueError("start-id must be <= end-id")
        ranged = []
        for p in images:
            pid = parse_image_id(p.name)
            if pid is not None and args.start_id <= pid <= args.end_id:
                ranged.append(p)
        sample = sorted(ranged, key=image_id_key)
        if len(sample) == 0:
            raise RuntimeError("No images found in requested ID range.")
    elif args.mode == "random":
        random.seed(args.seed)
        sample = random.sample(images, args.sample_size)
        sample.sort(key=lambda p: p.name)
    else:
        ordered = sorted(images, key=image_id_key)
        sample = ordered[: args.sample_size]

    new_rows = rows_from_images(sample, dataset_root)
    if args.append and out_csv.exists():
        existing = []
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.append(
                    {
                        "image_path": r.get("image_path", ""),
                        "image_name": r.get("image_name", ""),
                        "label": r.get("label", ""),
                        "confidence": r.get("confidence", ""),
                        "notes": r.get("notes", ""),
                    }
                )
        by_name = {r["image_name"]: r for r in existing}
        for r in new_rows:
            if r["image_name"] not in by_name:
                by_name[r["image_name"]] = r
        merged = list(by_name.values())
        merged.sort(key=lambda r: (parse_image_id(r["image_name"]) or 10**9))
        write_template_rows(out_csv, merged)
        print(f"Appended rows. Total CSV rows now: {len(merged)}")
    else:
        write_template_rows(out_csv, new_rows)

    print(f"Total images found: {len(images)}")
    print(f"Pilot sample size: {len(sample)}")
    print(f"Template written to: {out_csv}")
    print("Label values must be one of: N, C, S, Unknown")


if __name__ == "__main__":
    main()
