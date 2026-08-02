# Setup

This page installs the environment used for both LIBERO training and LIBERO evaluation.

## Requirements

The pinned dependency stack targets Python 3.10, PyTorch 2.2.0, and CUDA 12.1. Training a 7B backbone with LoRA needs at least one 40GB GPU, and the default recipe assumes two GPUs.

## Conda environment

```bash
conda create -n actionmap python=3.10 -y
conda activate actionmap
```

Install PyTorch first, so that the remaining packages resolve against it:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

## ActionMap

```bash
git clone https://github.com/showlab/ActionMap.git
cd ActionMap/actionmap_oft
pip install -e .
```

## LIBERO

LIBERO is installed separately, following the convention of the OpenVLA-OFT codebase this implementation builds on:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO --config-settings editable_mode=compat
pip install -r experiments/robot/libero/libero_requirements.txt
```

The `editable_mode=compat` setting is needed with recent versions of setuptools. Without it the install reports success but leaves `libero` unimportable, which later surfaces as `ModuleNotFoundError: No module named 'libero'`.

## Flash Attention 2

Flash Attention is compiled from source, so it needs the CUDA toolkit and it must be installed after the editable install. Training and evaluation both run without it, so this section can be skipped if the build gives trouble:

```bash
export CUDA_HOME=/usr/local/cuda    # must contain bin/nvcc
pip install packaging ninja
ninja --version; echo $?            # should print 0
pip install "flash-attn==2.5.5" --no-build-isolation
```

A build that fails with `No such file or directory: .../nvcc` means `CUDA_HOME` points at a directory without the compiler. If the build fails for other reasons, run `pip cache remove flash_attn` and retry.

## Verify

```bash
python -c "from actionmap import HeatmapActionHead; print(HeatmapActionHead(input_dim=64, hidden_dim=64))"
```

Next, see [LIBERO.md](LIBERO.md) for downloading the datasets and running training and evaluation.
