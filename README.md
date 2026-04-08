# HSILProject

Pilot workflow for scoliosis X-ray curve type classification (`N`, `C`, `S`, `Unknown`) using TensorFlow + ResNet50.

## Setup

```bash
pip install -r requirements.txt
```

## GPU usage (Windows / WSL2) and macOS notes

### Windows

- TensorFlow on native Windows is often **CPU-only** for recent versions.
- If you want to use an NVIDIA GPU, run training inside **WSL2 (Ubuntu)**.

### WSL2 (Ubuntu) + NVIDIA GPU (recommended)

1. Confirm WSL2 distros:

```powershell
wsl -l -v
```

2. Open Ubuntu:

```powershell
wsl -d Ubuntu-22.04
```

3. Inside Ubuntu, verify GPU:

```bash
nvidia-smi
```

4. Create venv + install TensorFlow with CUDA:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

python3 -m venv ~/hsi
source ~/hsi/bin/activate

pip install -U pip
pip install "tensorflow[and-cuda]==2.15.1"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

5. Install project deps in the same venv:

```bash
cd /mnt/c/Users/EXCALIBUR/OneDrive/Masaüstü/HSILProject
pip install pandas numpy scikit-learn pillow tqdm openai
```

Note:
- Do not install any DirectML / `tfdml` plugins inside WSL. They are Windows-specific and can break TensorFlow in Linux.

Then run scripts from WSL as usual (recommended for training speed).

### macOS

- Apple Silicon can use Metal acceleration (MPS) with `tensorflow-macos` + `tensorflow-metal`.
- Otherwise, training runs on CPU.

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

## Daily startup (Windows + WSL)

Open Ubuntu:

```powershell
wsl -d Ubuntu-22.04
```

Inside Ubuntu:

```bash
source ~/hsi/bin/activate
cd /mnt/c/Users/EXCALIBUR/OneDrive/Masaüstü/HSILProject
```

Optional one-liner from Windows:

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/c/Users/EXCALIBUR/OneDrive/Masaüstü/HSILProject
```

## Pilot labeling flow

1. Prepare a block of images.
2. Fill labels in `labels/pilot_labels_template.csv` (`N`, `C`, `S`, `Unknown`).
3. Split labeled data to train/val.

Labeling note rule:
- Clear (`N`, `C`, `S`) -> keep `notes` empty.
- Borderline/uncertain -> short note (2-6 words).

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

## Range-based labeling workflow

Use ID ranges so you can grow labels in blocks (e.g., 1-300, then 301-600).

Create first block:

```bash
python 01_prepare_pilot.py --dataset-root spinal-ai2024 --start-id 1 --end-id 300 --out-csv labels/pilot_labels_template.csv
```

Append next block to same CSV:

```bash
python 01_prepare_pilot.py --dataset-root spinal-ai2024 --start-id 301 --end-id 600 --out-csv labels/pilot_labels_template.csv --append
```

Pre-label only a specific range:

```bash
python 05_prelable_openai_batch.py submit --csv labels/pilot_labels_template.csv --dataset-root spinal-ai2024 --model gpt-4o-mini --start-id 301 --end-id 600 --out-dir batch_jobs/openai_prelabel_301_600
```

Split only a specific range (optional):

```bash
python 02_split_pilot.py --in-csv labels/pilot_labels_template.csv --start-id 301 --end-id 600 --train-size 240 --val-size 60
```

## Notes

- Keep raw dataset under `spinal-ai2024/` (ignored by git).
- Model artifacts are saved under `runs/` (ignored by git).
- Share/commit only code and label CSV files, not raw images.
