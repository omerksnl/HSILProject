import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_NAMES = ["N", "C", "S", "Unknown"]


def load_cobb_txt(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            rows.append(
                {
                    "image_name": parts[0].strip(),
                    "cobb_1": float(parts[1]),
                    "cobb_2": float(parts[2]),
                    "cobb_3": float(parts[3]),
                }
            )
    return pd.DataFrame(rows)


def build_image_index(dataset_root: Path):
    index = {}
    for p in dataset_root.rglob("*.jpg"):
        index[p.name] = p.relative_to(dataset_root).as_posix()
    return index


def compute_cobb_norm(df: pd.DataFrame):
    vals = df[["cobb_1", "cobb_2", "cobb_3"]].to_numpy(dtype=np.float32)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_cobb_norm(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
    out = df.copy()
    vals = out[["cobb_1", "cobb_2", "cobb_3"]].to_numpy(dtype=np.float32)
    out[["cobb_1", "cobb_2", "cobb_3"]] = (vals - mean) / std
    return out


def make_dataset(df: pd.DataFrame, dataset_root: Path, image_size, batch_size, training):
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augment",
    )

    img_paths = [str((dataset_root / p).resolve()) for p in df["image_path"].tolist()]
    # Dummy class labels for pretraining phase. Class loss is zero-weighted.
    class_y = np.zeros((len(df),), dtype=np.int32)
    cobb_y = df[["cobb_1", "cobb_2", "cobb_3"]].to_numpy(dtype=np.float32)

    def _load(path, cls, cobb):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = aug(img, training=True)
        targets = {"class_output": cls, "cobb_output": cobb}
        return img, targets

    ds = tf.data.Dataset.from_tensor_slices((img_paths, class_y, cobb_y))
    if training:
        ds = ds.shuffle(min(len(img_paths), 4096), seed=42)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(image_size=(320, 320)):
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    class_output = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="class_output"
    )(x)
    cobb_output = tf.keras.layers.Dense(3, activation="linear", name="cobb_output")(x)
    model = tf.keras.Model(inputs, {"class_output": class_output, "cobb_output": cobb_output})
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage-1 pretrain on Cobb only (16k).")
    parser.add_argument("--dataset-root", type=str, default="spinal-ai2024")
    parser.add_argument(
        "--cobb-train-txt", type=str, default="spinal-ai2024/Cobb_spinal-AI2024-train_gt.txt"
    )
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default="runs/pretrain_cobb_resnet50")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve()

    cobb_df = load_cobb_txt(Path(args.cobb_train_txt).resolve())
    index = build_image_index(dataset_root)
    cobb_df["image_path"] = cobb_df["image_name"].map(index)
    cobb_df = cobb_df.dropna(subset=["image_path"]).copy()
    if len(cobb_df) < 1000:
        raise RuntimeError("Too few Cobb samples found. Check dataset path/indexing.")

    # Reproducible split for pretraining.
    cobb_df = cobb_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = max(1, int(len(cobb_df) * args.val_ratio))
    val_df = cobb_df.iloc[:n_val].copy()
    train_df = cobb_df.iloc[n_val:].copy()

    cobb_mean, cobb_std = compute_cobb_norm(train_df)
    train_df = apply_cobb_norm(train_df, cobb_mean, cobb_std)
    val_df = apply_cobb_norm(val_df, cobb_mean, cobb_std)

    image_size = (args.img_size, args.img_size)
    train_ds = make_dataset(train_df, dataset_root, image_size, args.batch_size, True)
    val_ds = make_dataset(val_df, dataset_root, image_size, args.batch_size, False)

    model = build_model(image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss={
            "class_output": "sparse_categorical_crossentropy",
            "cobb_output": tf.keras.losses.Huber(delta=1.0),
        },
        loss_weights={"class_output": 0.0, "cobb_output": 1.0},
        metrics={"cobb_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")]},
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_cobb_output_mae", mode="min", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_cobb_output_mae",
            mode="min",
            save_best_only=True,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    eval_res = model.evaluate(val_ds, return_dict=True, verbose=0)
    print(f"Pretrain val cobb mae(norm): {eval_res.get('cobb_output_mae', 0.0):.4f}")

    with (out_dir / "cobb_norm.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": cobb_mean.tolist(), "std": cobb_std.tolist()}, f, indent=2)

    with (out_dir / "pretrain_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"train_samples={len(train_df)}\n")
        f.write(f"val_samples={len(val_df)}\n")
        f.write(f"val_cobb_mae_norm={eval_res.get('cobb_output_mae', 0.0):.4f}\n")


if __name__ == "__main__":
    main()
