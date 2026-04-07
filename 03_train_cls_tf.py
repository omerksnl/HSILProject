import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_NAMES = ["N", "C", "S", "Unknown"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}


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


def merge_with_cobb(df: pd.DataFrame, cobb_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].astype(str).str.strip()
    out = out.merge(cobb_df, on="image_name", how="left")
    if out[["cobb_1", "cobb_2", "cobb_3"]].isna().any().any():
        missing = out[out["cobb_1"].isna()]["image_name"].tolist()[:5]
        raise ValueError(f"Missing Cobb labels for samples like: {missing}")
    return out


def compute_class_weights(train_df: pd.DataFrame):
    counts = train_df["label"].value_counts()
    total = len(train_df)
    num_classes = len(CLASS_NAMES)
    weights = {}
    for name in CLASS_NAMES:
        c = int(counts.get(name, 0))
        weights[name] = (total / (num_classes * c)) if c > 0 else 1.0
    return weights


def compute_cobb_norm(train_df: pd.DataFrame):
    vals = train_df[["cobb_1", "cobb_2", "cobb_3"]].to_numpy(dtype=np.float32)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_cobb_norm(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
    out = df.copy()
    vals = out[["cobb_1", "cobb_2", "cobb_3"]].to_numpy(dtype=np.float32)
    norm = (vals - mean) / std
    out[["cobb_1", "cobb_2", "cobb_3"]] = norm
    return out


def make_dataset(
    df: pd.DataFrame,
    dataset_root: Path,
    image_size,
    batch_size,
    training,
    class_weights,
    use_sample_weights,
):
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augment",
    )

    img_paths = [str((dataset_root / p).resolve()) for p in df["image_path"].tolist()]
    class_y = np.array([CLASS_TO_ID[l] for l in df["label"].tolist()], dtype=np.int32)
    cobb_y = df[["cobb_1", "cobb_2", "cobb_3"]].to_numpy(dtype=np.float32)
    class_sw = np.array([class_weights[l] for l in df["label"].tolist()], dtype=np.float32)
    cobb_sw = np.ones((len(df),), dtype=np.float32)

    def _load(path, cls, cobb, sw_cls, sw_cobb):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = aug(img, training=True)
        targets = {"class_output": cls, "cobb_output": cobb}
        if use_sample_weights:
            sample_weights = {"class_output": sw_cls, "cobb_output": sw_cobb}
            return img, targets, sample_weights
        return img, targets

    ds = tf.data.Dataset.from_tensor_slices((img_paths, class_y, cobb_y, class_sw, cobb_sw))
    if training:
        ds = ds.shuffle(min(len(img_paths), 1024), seed=42)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(image_size=(320, 320)):
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)

    class_output = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="class_output"
    )(x)
    # Linear head prevents dead-ReLU collapse on one Cobb channel.
    cobb_output = tf.keras.layers.Dense(3, activation="linear", name="cobb_output")(x)

    model = tf.keras.Model(inputs, {"class_output": class_output, "cobb_output": cobb_output})
    return model, base


def compile_model(model, lr, class_loss_w, cobb_loss_w):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
            "class_output": "sparse_categorical_crossentropy",
            "cobb_output": tf.keras.losses.Huber(delta=1.0),
        },
        loss_weights={"class_output": class_loss_w, "cobb_output": cobb_loss_w},
        metrics={
            "class_output": ["accuracy"],
            "cobb_output": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-task model: class (N/C/S/Unknown) + 3 Cobb angles."
    )
    parser.add_argument("--dataset-root", type=str, default="spinal-ai2024")
    parser.add_argument("--train-csv", type=str, default="labels/pilot_train.csv")
    parser.add_argument("--val-csv", type=str, default="labels/pilot_val.csv")
    parser.add_argument(
        "--cobb-train-txt", type=str, default="spinal-ai2024/Cobb_spinal-AI2024-train_gt.txt"
    )
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-ft", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-ft", type=float, default=1e-5)
    parser.add_argument("--class-loss-w", type=float, default=1.0)
    parser.add_argument("--cobb-loss-w", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="runs/pilot_resnet50_multitask")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve()

    train_df = pd.read_csv(Path(args.train_csv).resolve())
    val_df = pd.read_csv(Path(args.val_csv).resolve())
    cobb_df = load_cobb_txt(Path(args.cobb_train_txt).resolve())

    train_df = merge_with_cobb(train_df, cobb_df)
    val_df = merge_with_cobb(val_df, cobb_df)

    if not set(train_df["label"]).issubset(set(CLASS_NAMES)):
        raise ValueError("Train CSV has invalid labels. Allowed: N, C, S, Unknown")
    if not set(val_df["label"]).issubset(set(CLASS_NAMES)):
        raise ValueError("Val CSV has invalid labels. Allowed: N, C, S, Unknown")

    class_weights = compute_class_weights(train_df)
    cobb_mean, cobb_std = compute_cobb_norm(train_df)
    train_df = apply_cobb_norm(train_df, cobb_mean, cobb_std)
    val_df = apply_cobb_norm(val_df, cobb_mean, cobb_std)

    image_size = (args.img_size, args.img_size)
    train_ds = make_dataset(
        train_df, dataset_root, image_size, args.batch_size, True, class_weights, True
    )
    val_ds = make_dataset(
        val_df, dataset_root, image_size, args.batch_size, False, class_weights, False
    )

    model, base = build_model(image_size=image_size)
    compile_model(model, args.lr_head, args.class_loss_w, args.cobb_loss_w)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_class_output_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_class_output_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
    )

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    compile_model(model, args.lr_ft, args.class_loss_w, args.cobb_loss_w)
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_ft,
        callbacks=callbacks,
    )

    eval_res = model.evaluate(val_ds, verbose=0, return_dict=True)
    print(f"Final val class acc: {eval_res.get('class_output_accuracy', 0.0):.4f}")
    print(f"Final val cobb mae: {eval_res.get('cobb_output_mae', 0.0):.4f}")

    with (out_dir / "label_map.txt").open("w", encoding="utf-8") as f:
        for idx, name in enumerate(CLASS_NAMES):
            f.write(f"{idx},{name}\n")

    with (out_dir / "cobb_norm.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": [float(x) for x in cobb_mean],
                "std": [float(x) for x in cobb_std],
                "class_weights": {k: float(v) for k, v in class_weights.items()},
            },
            f,
            indent=2,
        )

    with (out_dir / "train_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Head training epochs completed: ")
        f.write(str(len(history_head.history.get("loss", []))))
        f.write("\nFine-tune epochs completed: ")
        f.write(str(len(history_ft.history.get("loss", []))))
        f.write(f"\nFinal val class acc: {eval_res.get('class_output_accuracy', 0.0):.4f}")
        f.write(f"\nFinal val cobb mae: {eval_res.get('cobb_output_mae', 0.0):.4f}\n")


if __name__ == "__main__":
    main()
