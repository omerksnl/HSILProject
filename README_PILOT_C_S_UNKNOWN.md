# Pilot Pipeline (C / S / Unknown)

This pipeline lives in the main project directory and keeps the dataset folder as raw data storage.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Create pilot labeling template (300 images)

```bash
python 01_prepare_pilot.py --dataset-root spinal-ai2024 --sample-size 300 --out-csv labels/pilot_labels_template.csv
```

Fill `labels/pilot_labels_template.csv` manually:
- `label` must be one of `C`, `S`, `Unknown`
- leave no unlabeled rows for split/training

## 3) Split into train/val (240/60)

```bash
python 02_split_pilot.py --in-csv labels/pilot_labels_template.csv --train-size 240 --val-size 60
```

Outputs:
- `labels/pilot_train.csv`
- `labels/pilot_val.csv`

## 4) Train TensorFlow ResNet50 classifier

```bash
python 03_train_cls_tf.py --dataset-root spinal-ai2024 --train-csv labels/pilot_train.csv --val-csv labels/pilot_val.csv --img-size 320
```

Outputs in `runs/pilot_resnet50`:
- `best_model.keras`
- `label_map.txt`
- `train_summary.txt`

---

## Implementation choices you selected

- Framework: TensorFlow/Keras
- Backbone: ResNet50
- Input size: 320x320
- Labeling flow: manual CSV
- API pre-label: not included in this first version

## Pros / Cons of this selected setup

Pros:
- Quick to start and easy to modify.
- Strong baseline with transfer learning.
- Manual CSV workflow gives full label control.

Cons:
- Manual labeling is slower without UI/API assist.
- Pilot set (300) can be noisy and class-imbalanced.
- Classification-only model does not directly predict Cobb angles yet.
