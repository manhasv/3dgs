# 3D Gaussian Splatting

This project is a small experiment on 3DGS using Colmap output and trained with gsplat library.

### 1. Prerequisites

- **Package Manager**: Conda / Miniconda
- **Hardware**: NVIDIA GPU with CUDA support

---

### 2. Environment Setup

### A. Create and Activate Conda Environment

```bash
conda create -n 3dgs python=3.11 -y
conda activate 3dgs
```
### B. Install python packages
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install pycolmap gsplat opencv-python plyfile
```

### There might be some problem with gcc like with Fedora 44, so if you need to config a separate gcc for this env, use:
```bash
conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```


## 3. Prepare training images
### Recommended: Use ffmpeg to turn a video into sequences of images
```bash
ffmpeg -i file.MOV -qscale:v 1 -qmin 1 -vf fps=2 images/%04d.jpg
```

## 4. Run
### Digest images with Colmap first
```bash
python reconstruction.py
```
### Run 3dgs training on Colmap output
```bash
python train.py
```