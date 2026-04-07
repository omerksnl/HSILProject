import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_NAMES = ["N", "C", "S", "Unknown"]


def reported_angles_by_class(pred_class: str, a1: float, a2: float, a3: float):
    vals = sorted([float(a1), float(a2), float(a3)], reverse=True)
    if pred_class == "C":
        return [vals[0]]
    if pred_class == "S":
        return [vals[0], vals[1]]
    if pred_class == "N":
        return []
    return [vals[0], vals[1], vals[2]]


def main():
    parser = argparse.ArgumentParser(
        description="Predict class + Cobb and apply C/S reporting rule."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="spinal-ai2024")
    parser.add_argument("--in-csv", type=str, default="labels/pilot_labels_template.csv")
    parser.add_argument("--out-csv", type=str, default="runs/predictions_multitask.csv")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument(
        "--norm-json",
        type=str,
        default="",
        help="Path to cobb_norm.json saved during training. "
        "Default: same folder as model-path.",
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.model_path).resolve())
    if args.norm_json:
        norm_path = Path(args.norm_json).resolve()
    else:
        norm_path = Path(args.model_path).resolve().parent / "cobb_norm.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    cobb_mean = np.array(norm["mean"], dtype=np.float32)
    cobb_std = np.array(norm["std"], dtype=np.float32)

    df = pd.read_csv(Path(args.in_csv).resolve())
    dataset_root = Path(args.dataset_root).resolve()

    paths = [str((dataset_root / p).resolve()) for p in df["image_path"].tolist()]
    out_rows = []
    for row, p in zip(df.to_dict(orient="records"), paths):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (args.img_size, args.img_size))
        img = tf.cast(img, tf.float32) / 255.0
        pred = model.predict(tf.expand_dims(img, axis=0), verbose=0)

        class_probs = pred["class_output"][0]
        cobb_norm = pred["cobb_output"][0]
        cobb = (cobb_norm * cobb_std) + cobb_mean
        cobb = np.maximum(cobb, 0.0)
        class_idx = int(np.argmax(class_probs))
        pred_class = CLASS_NAMES[class_idx]
        top_angles = reported_angles_by_class(pred_class, cobb[0], cobb[1], cobb[2])

        out_rows.append(
            {
                "image_path": row["image_path"],
                "image_name": row["image_name"],
                "pred_class": pred_class,
                "pred_class_conf": float(class_probs[class_idx]),
                "pred_cobb_1": float(cobb[0]),
                "pred_cobb_2": float(cobb[1]),
                "pred_cobb_3": float(cobb[2]),
                "reported_major_angles": ";".join([f"{x:.2f}" for x in top_angles]),
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()
