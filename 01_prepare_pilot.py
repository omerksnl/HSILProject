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


def write_template(out_csv: Path, images, dataset_root: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "image_name", "label", "confidence", "notes"])
        for img in images:
            rel = img.relative_to(dataset_root).as_posix()
            writer.writerow([rel, img.name, "", "", ""])


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
        "--out-csv",
        type=str,
        default="labels/pilot_labels_template.csv",
        help="Output CSV path for annotation template.",
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

    random.seed(args.seed)
    sample = random.sample(images, args.sample_size)
    sample.sort(key=lambda p: p.name)

    write_template(out_csv, sample, dataset_root)
    print(f"Total images found: {len(images)}")
    print(f"Pilot sample size: {len(sample)}")
    print(f"Template written to: {out_csv}")
    print("Label values must be one of: C, S, Unknown")


if __name__ == "__main__":
    main()
