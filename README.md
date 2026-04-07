# HSILProject

Pilot workflow for scoliosis X-ray curve type classification (`N`, `C`, `S`, `Unknown`) using TensorFlow + ResNet50.

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

2. Fill labels in `labels/pilot_labels_template.csv` (`N`, `C`, `S`, `Unknown`).

### Labeling notes guideline

- If the case is clear (`N`, `C`, or `S`) and confidence is high, keep `notes` empty.
- If label is `Unknown` or borderline, add a short reason (2-6 words).
- Keep wording consistent across teammates.

Recommended note values:
- `low image quality`
- `rotation artifact`
- `curve not clear`
- `possible double curve`
- `posture ambiguous`
- `hardware overlap`
- `borderline C/S`

3. Split into train/val (240/60):

```bash
python 02_split_pilot.py --in-csv labels/pilot_labels_template.csv --train-size 240 --val-size 60
```

4. Train multi-task model (class + 3 Cobb angles):

```bash
python 03_train_cls_tf.py --dataset-root spinal-ai2024 --train-csv labels/pilot_train.csv --val-csv labels/pilot_val.csv --cobb-train-txt spinal-ai2024/Cobb_spinal-AI2024-train_gt.txt --img-size 320
```

5. Predict and apply reporting rule:
- `C` => report highest 1 Cobb angle
- `S` => report highest 2 Cobb angles

```bash
python 06_predict_multitask_tf.py --model-path runs/pilot_resnet50_multitask/best_model.keras --dataset-root spinal-ai2024 --in-csv labels/pilot_labels_template.csv --out-csv runs/predictions_multitask.csv
```

## Two-stage training 

Stage-1: Pretrain on 16k with Cobb regression only.

```bash
python 03a_pretrain_cobb_tf.py --dataset-root spinal-ai2024 --cobb-train-txt spinal-ai2024/Cobb_spinal-AI2024-train_gt.txt --out-dir runs/pretrain_cobb_resnet50
```

Stage-2: Finetune multitask on labeled first-300 (`N/C/S/Unknown` + Cobb), starting from stage-1 model.

```bash
python 03b_finetune_multitask_tf.py --dataset-root spinal-ai2024 --train-csv labels/pilot_train.csv --val-csv labels/pilot_val.csv --cobb-train-txt spinal-ai2024/Cobb_spinal-AI2024-train_gt.txt --pretrain-model runs/pretrain_cobb_resnet50/best_model.keras --pretrain-norm-json runs/pretrain_cobb_resnet50/cobb_norm.json --out-dir runs/finetune_multitask_from_pretrain
```

Predict from stage-2 model:

```bash
python 06_predict_multitask_tf.py --model-path runs/finetune_multitask_from_pretrain/best_model.keras --norm-json runs/finetune_multitask_from_pretrain/cobb_norm.json --dataset-root spinal-ai2024 --in-csv labels/pilot_labels_template.csv --out-csv runs/predictions_multitask_stage2.csv
```

## Optional: OpenAI Batch API pre-label 

If you want lower cost than synchronous API calls, use Batch API.

1. Set key (PowerShell):

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Or set it once in `.env`:

```text
OPENAI_API_KEY=your_api_key_here
```

2. Submit batch job for empty labels only:

```bash
python 05_prelable_openai_batch.py submit --csv labels/pilot_labels_template.csv --dataset-root spinal-ai2024 --model gpt-4o-mini
```

3. Later, apply results after batch finishes:

```bash
python 05_prelable_openai_batch.py apply
```

Behavior:
- Fills only empty `label` cells.
- Writes `label`, `confidence`, and note reason.
- Adds `low_conf_from_ai` note for low confidence rows.
- Creates backup: `labels/pilot_labels_template.before_openai_batch_backup.csv`.

Tip:
- Start with `--max-rows 20` on submit to test first.

## Notes

- Keep raw dataset under `spinal-ai2024/` (ignored by git).
- Model artifacts are saved under `runs/` (ignored by git).
- Share/commit only code and label CSV files, not raw images.
