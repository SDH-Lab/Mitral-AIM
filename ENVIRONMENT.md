# 环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| SimpleITK | 2.5.6 |
| PyYAML | 6.0.3 |
| matplotlib | 3.10.9 |

```bash
conda create -n tee_motionr python=3.10.20 -y
conda activate tee_motionr
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

训练用 24 GB GPU。没有 CUDA 11.8 时，先装与本机 CUDA 匹配的 PyTorch 2.7，再安装 `requirements.txt`。
