# Mitral-AIM

4D TEE classification of clinically significant mitral regurgitation. One sample is one cardiac cycle, resampled to 64×64×64.

## Installation

Python 3.10 and PyTorch 2.7.1+cu118 (CUDA 11.8).

```bash
conda env create -f environment.yml
conda activate tee_motionr
```

Without conda:

```bash
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

The CSV needs `patient_id`, `file_name`, `fold`, and `binary`. `fold` is 0, 1, or 2. Each clip is a directory of `frame_*.mha`.

```bash
python scripts/preprocess.py \
  --manifest folds.csv \
  --root /path/to/volumes \
  --out cache_64
```

Point `data.manifest` and `data.cache` in the yaml at those paths. Volumes, the cache, and checkpoints are not part of this repository.

## Training

```bash
python scripts/train.py --config configs/fold0.yaml --fold 0 --out runs/fold0
python scripts/train.py --config configs/fold1.yaml --fold 1 --out runs/fold1
python scripts/train.py --config configs/fold2.yaml --fold 2 --out runs/fold2
```

Run directories must sit inside this repo, or inside `MOTIONR_OUTPUT_ROOT` if that variable is set.

## Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint runs/fold0/best.pt \
  --out runs/fold0/eval
```

```bash
python -m unittest discover -s tests
```
