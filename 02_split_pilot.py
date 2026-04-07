import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ALLOWED_LABELS = {"N", "C", "S", "Unknown"}


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].astype(str).str.strip()
    mapping = {"n": "N", "c": "C", "s": "S", "unknown": "Unknown"}
    df["label"] = df["label"].str.lower().map(mapping).fillna(df["label"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Split manually labeled pilot CSV into train/val."
    )
    parser.add_argument(
        "--in-csv",
        type=str,
        default="labels/pilot_labels_template.csv",
        help="CSV filled manually with labels.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=240,
        help="Train set size.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=60,
        help="Validation set size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="labels")
    args = parser.parse_args()

    in_csv = Path(args.in_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    required_cols = {"image_path", "image_name", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = normalize_labels(df)
    df = df[df["label"].isin(ALLOWED_LABELS)].copy()
    total_needed = args.train_size + args.val_size
    if len(df) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} valid labeled rows, found {len(df)}."
        )

    # If more than needed is present, keep only the first N for a strict pilot split.
    df = df.iloc[:total_needed].copy()

    train_df, val_df = train_test_split(
        df,
        train_size=args.train_size,
        random_state=args.seed,
        stratify=df["label"],
    )

    train_path = out_dir / "pilot_train.csv"
    val_path = out_dir / "pilot_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train saved: {train_path} ({len(train_df)} rows)")
    print(f"Val saved: {val_path} ({len(val_df)} rows)")
    print("Class counts (train):")
    print(train_df["label"].value_counts())
    print("Class counts (val):")
    print(val_df["label"].value_counts())


if __name__ == "__main__":
    main()
