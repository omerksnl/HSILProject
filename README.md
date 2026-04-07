# HSILProject

Pilot workflow for scoliosis X-ray curve type classification (`C`, `S`, `Unknown` for now) using TensorFlow + ResNet50.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset setup (for collaborators)

1. Each teammate downloads the Spinal-AI2024 dataset locally.
2. Put it in the project root as `spinal-ai2024/`.
3. Keep this folder local-only (it is ignored by git).

Expected local structure:

```text
HSILProject/
  spinal-ai2024/
    subset1/
    subset2/
    subset3/
    subset4/
    subset5/
```

## Pilot pipeline

1. Prepare 300-image labeling template:

```bash
python 01_prepare_pilot.py --dataset-root spinal-ai2024 --sample-size 300
```

2. Fill labels in `labels/pilot_labels_template.csv` (`C`, `S`, `Unknown`).

3. Split into train/val (240/60):

```bash
python 02_split_pilot.py --in-csv labels/pilot_labels_template.csv --train-size 240 --val-size 60
```

4. Train classifier:

```bash
python 03_train_cls_tf.py --dataset-root spinal-ai2024 --train-csv labels/pilot_train.csv --val-csv labels/pilot_val.csv --img-size 320
```

## Notes

- Keep raw dataset under `spinal-ai2024/` (ignored by git).
- Model artifacts are saved under `runs/` (ignored by git).
- Share/commit only code and label CSV files, not raw images.
