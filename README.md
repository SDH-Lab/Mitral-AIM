# MotionR_S

4D TEE 序列上的二尖瓣反流二分类。一个 clip 是一个心动周期，体积为 64³。

## 环境

Python 3.10，PyTorch 2.7.1+cu118，CUDA 11.8。其余依赖见 `requirements.txt`。

```bash
conda create -n tee_motionr python=3.10.20 -y
conda activate tee_motionr
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 数据

清单 CSV 列：`patient_id,file_name,fold,binary`。`fold` 为 0/1/2，`binary` 为 0/1。每个 clip 一个目录，里面是 `frame_*.mha`。

把配置里的 `manifest` 和 `cache` 改成自己的路径。

```bash
python scripts/preprocess.py --manifest folds.csv --root /path/to/volumes --out cache_64
python scripts/train.py --config configs/fold0.yaml --fold 0 --out runs/fold0
python scripts/evaluate.py --checkpoint runs/fold0/best.pt --out runs/fold0/eval
```

Fold 0 的学习率是 3e-5 / 9e-5，fold 1 和 fold 2 是 1e-4 / 3e-4，对应 `configs/fold*.yaml`。输出写在本目录下；要写到别处时设置 `MOTIONR_OUTPUT_ROOT`。
