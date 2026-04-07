import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_NAMES = ["C", "S", "Unknown"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}


def build_paths(df: pd.DataFrame, dataset_root: Path):
    img_paths = [str((dataset_root / p).resolve()) for p in df["image_path"].tolist()]
    y = [CLASS_TO_ID[lbl] for lbl in df["label"].tolist()]
    return img_paths, np.array(y, dtype=np.int32)


def make_dataset(img_paths, labels, image_size, batch_size, training):
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augment",
    )

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = aug(img, training=True)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
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
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base


def main():
    parser = argparse.ArgumentParser(description="Train C/S/Unknown classifier.")
    parser.add_argument("--dataset-root", type=str, default="spinal-ai2024")
    parser.add_argument("--train-csv", type=str, default="labels/pilot_train.csv")
    parser.add_argument("--val-csv", type=str, default="labels/pilot_val.csv")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-ft", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-ft", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=str, default="runs/pilot_resnet50")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root).resolve()

    train_df = pd.read_csv(Path(args.train_csv).resolve())
    val_df = pd.read_csv(Path(args.val_csv).resolve())

    train_df["label"] = train_df["label"].astype(str).str.strip()
    val_df["label"] = val_df["label"].astype(str).str.strip()

    if not set(train_df["label"]).issubset(set(CLASS_NAMES)):
        raise ValueError("Train CSV has invalid labels. Allowed: C, S, Unknown")
    if not set(val_df["label"]).issubset(set(CLASS_NAMES)):
        raise ValueError("Val CSV has invalid labels. Allowed: C, S, Unknown")

    train_paths, train_y = build_paths(train_df, dataset_root)
    val_paths, val_y = build_paths(val_df, dataset_root)

    image_size = (args.img_size, args.img_size)
    train_ds = make_dataset(train_paths, train_y, image_size, args.batch_size, True)
    val_ds = make_dataset(val_paths, val_y, image_size, args.batch_size, False)

    model, base = build_model(image_size=image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
    )

    # Fine-tune the last part of the backbone.
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_ft),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_ft,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Final val accuracy: {val_acc:.4f}")
    print(f"Final val loss: {val_loss:.4f}")

    # Save label map for inference scripts.
    with (out_dir / "label_map.txt").open("w", encoding="utf-8") as f:
        for idx, name in enumerate(CLASS_NAMES):
            f.write(f"{idx},{name}\n")

    # Save concise training summary.
    with (out_dir / "train_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Head training epochs completed: ")
        f.write(str(len(history_head.history.get("loss", []))))
        f.write("\nFine-tune epochs completed: ")
        f.write(str(len(history_ft.history.get("loss", []))))
        f.write(f"\nFinal val accuracy: {val_acc:.4f}\n")


if __name__ == "__main__":
    main()
