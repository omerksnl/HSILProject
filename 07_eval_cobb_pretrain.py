import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def load_cobb_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=["image_name", "cobb_1", "cobb_2", "cobb_3"],
        dtype={"image_name": str, "cobb_1": float, "cobb_2": float, "cobb_3": float},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cobb pretrain model on a subset of Spinal-AI2024."
    )
    parser.add_argument(
        "--dataset-root", type=str, default="spinal-ai2024", help="Dataset root."
    )
    parser.add_argument(
        "--gt-txt",
        type=str,
        default="spinal-ai2024/Cobb_spinal-AI2024-test_gt.txt",
        help="GT txt (train or test).",
    )
    parser.add_argument(
        "--subset-dir",
        type=str,
        default="Spinal-AI2024-subset5",
        help="Relative subset dir containing the images.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="runs/pretrain_cobb_resnet50/best_model.keras",
        help="Pretrained Cobb model path.",
    )
    parser.add_argument(
        "--norm-json",
        type=str,
        default="runs/pretrain_cobb_resnet50/cobb_norm.json",
        help="Normalization stats JSON from pretrain.",
    )
    parser.add_argument(
        "--img-size", type=int, default=320, help="Image size used at train time."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max number of images to evaluate (0 = all).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    model = tf.keras.models.load_model(Path(args.model_path).resolve())

    norm = pd.read_json(Path(args.norm_json).resolve(), typ="series")
    cobb_mean = np.array(norm["mean"], dtype=np.float32)
    cobb_std = np.array(norm["std"], dtype=np.float32)

    gt_df = load_cobb_txt(Path(args.gt_txt).resolve())
    gt_df["image_path"] = args.subset_dir.rstrip("/") + "/" + gt_df["image_name"]

    if args.max_samples > 0:
        gt_df = gt_df.iloc[: args.max_samples].copy()

    paths = [str((dataset_root / p).resolve()) for p in gt_df["image_path"]]

    pred_list = []
    gt_list = []
    per_sample_mae = []

    for (p, row) in zip(paths, gt_df.itertuples(index=False)):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (args.img_size, args.img_size))
        img = tf.cast(img, tf.float32) / 255.0
        pred = model.predict(tf.expand_dims(img, 0), verbose=0)
        cobb_norm = pred["cobb_output"][0]
        cobb = (cobb_norm * cobb_std) + cobb_mean
        cobb = np.maximum(cobb, 0.0)

        gt_vec = np.array(
            [row.cobb_1, row.cobb_2, row.cobb_3], dtype=np.float32
        )
        err = np.abs(cobb - gt_vec)

        pred_list.append(cobb)
        gt_list.append(gt_vec)
        per_sample_mae.append(err.mean())

    pred_arr = np.stack(pred_list)
    gt_arr = np.stack(gt_list)
    err_arr = np.array(per_sample_mae, dtype=np.float32)

    overall_mae = float(err_arr.mean())
    mae_angles = np.mean(np.abs(pred_arr - gt_arr), axis=0)
    rmse_angles = np.sqrt(np.mean((pred_arr - gt_arr) ** 2, axis=0))

    print(f"Evaluated {len(err_arr)} samples.")
    print(f"Overall Cobb MAE (degrees): {overall_mae:.3f}")
    print(
        "MAE per angle [cobb_1, cobb_2, cobb_3]:",
        [round(float(x), 3) for x in mae_angles],
    )
    print(
        "RMSE per angle [cobb_1, cobb_2, cobb_3]:",
        [round(float(x), 3) for x in rmse_angles],
    )

    print("\nError distribution (per-sample mean abs error):")
    print("  median:", round(float(np.median(err_arr)), 3))
    print("  p90:", round(float(np.quantile(err_arr, 0.90)), 3))
    print("  p95:", round(float(np.quantile(err_arr, 0.95)), 3))
    print("  within 5 deg:", float((err_arr <= 5.0).mean()))
    print("  within 10 deg:", float((err_arr <= 10.0).mean()))

    worst_idx = np.argsort(-err_arr)[:20]
    print("\nTop 20 worst samples:")
    for idx in worst_idx:
        row = gt_df.iloc[idx]
        gt_vec = gt_arr[idx]
        cobb = pred_arr[idx]
        e = err_arr[idx]
        print(
            f"{row.image_name}: overall_err={e:.3f} | "
            f"gt={[round(float(x),2) for x in gt_vec]} | "
            f"pred={[round(float(x),2) for x in cobb]}"
        )


if __name__ == "__main__":
    main()
