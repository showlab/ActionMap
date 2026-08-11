<div align="center">
<h1>ActionMap: Robot Policy Learning via Voxel Action Heatmap</h1>
</div>

<div align="center">
    <a href="https://yangpei-comp.github.io/">Pei Yang</a><sup>1,&#42;</sup>&nbsp;, <a href="https://scholar.google.com/citations?user=GMrjppAAAAAJ&hl=en">Hai Ci</a><sup>1,&#42;</sup>&nbsp;, <a href="https://chenanno.github.io/">Yanzhe Chen</a><sup>1,&#42;</sup>&nbsp;, <a href="https://aopolin-lv.github.io/">Qi Lv</a><sup>1</sup>&nbsp;, <a href="https://han-cai.github.io/">Han Cai</a><sup>2</sup>&nbsp;, and <a href="https://sites.google.com/view/showlab">Mike Zheng Shou</a><sup>1,&#x2709;</sup>
</div>

<div align="center">
    <sup>1</sup> <a href='https://sites.google.com/view/showlab/home?authuser=0' target='_blank'>Show Lab</a>, National University of Singapore
    &nbsp;&nbsp;
    <sup>2</sup> NVIDIA
</div>

<!--
<div align="center">
    <sup>&#42;</sup> Equal contribution &nbsp;&nbsp; <sup>&#x2709;</sup> Corresponding author
</div>
-->


<br/>

<div align="center">
    <a href='https://showlab.github.io/ActionMap/'>https://showlab.github.io/ActionMap/</a>
</div>

<br/>

<div align="center">
    <a href="https://arxiv.org/abs/2606.06904">
        <img src="https://img.shields.io/badge/arXiv-2606.06904-b31b1b.svg?logo=arXiv">
    </a>
</div>

<br/>

<div align="center">
    <img src="assets/teaser.png" width="1024">
</div>

<br/>

<div align="center">
    <b>ActionMap replaces the single-point action decoder of vision-language-action models with a voxel action heatmap, improving success rate, data efficiency, and convergence across LIBERO simulation and real-world Franka manipulation.</b>
</div>

<br/>

## 🧩 Overview

ActionMap replaces the single-point action decoder of a vision-language-action model with a voxel action heatmap. The head predicts a probability distribution over a discretized action space, and then decodes that distribution back into a continuous action.

This repository holds one folder per base model, so that each implementation stays self-contained. The OpenVLA-OFT-based implementation is released here, and the Pi-0.5-based implementation will follow in its own folder.

```
ActionMap/
├── actionmap_oft/          ActionMap on OpenVLA-OFT: LIBERO training, evaluation, real-robot serving
│   └── actionmap/          The voxel action heatmap head
└── actionmap_pi/           ActionMap on Pi-0.5 (planned)
```

## ⚙️ Installation

Installation covers a conda environment, the pinned dependency stack, and the LIBERO benchmark:

```bash
git clone https://github.com/showlab/ActionMap.git
cd ActionMap/actionmap_oft
```

Follow [actionmap_oft/SETUP.md](actionmap_oft/SETUP.md) for the full sequence. Compiling Flash Attention from source dominates the install time, so expect the last step to take a while.

## 🚀 Quick Start

Download the LIBERO RLDS datasets, which are the ones released with OpenVLA:

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

Finetune OpenVLA-7B on LIBERO-Spatial with the ActionMap head, using the configuration behind our reported results:

```bash
bash scripts/train_libero.sh libero_spatial /path/to/modified_libero_rlds 2
```

Evaluate the resulting policy over 50 trials on each of the 10 tasks in the suite:

```bash
bash scripts/eval_libero.sh libero_spatial ./runs/<run_name>--10000_chkpt
```

Both scripts wrap the underlying Python entry points, and [actionmap_oft/LIBERO.md](actionmap_oft/LIBERO.md) shows those commands in full.

## 📚 Documentation

- [SETUP.md](actionmap_oft/SETUP.md) installs the environment, the dependency stack, and the LIBERO benchmark.
- [LIBERO.md](actionmap_oft/LIBERO.md) walks through LIBERO training and evaluation, including how the two must be kept consistent.
- [ARGUMENTS.md](actionmap_oft/ARGUMENTS.md) documents every training and evaluation argument.
- [REAL_ROBOT.md](actionmap_oft/REAL_ROBOT.md) covers the data format, dataset registration, and policy serving for a real robot arm.

## 🔧 Using the Head on Its Own

The head is a self-contained module, so it can replace the action decoder of another vision-language-action model. The example below plugs it into a backbone directly.

```python
import torch
from actionmap import HeatmapActionHead

head = HeatmapActionHead(
    input_dim=4096,           # VLA backbone hidden size
    num_actions_chunk=8,      # action tokens per chunk
    action_dim=7,             # [x, y, z, r, p, w, grip]
    trans_grid=(48, 48, 24),  # translation voxel grid
    rot_grid=(24, 24, 24),    # rotation voxel grid
)

# Run your VLA backbone and keep the last hidden layer.
outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
hidden = outputs.hidden_states[-1]                       # (B, seq_len, llm_dim)

# Gather the hidden states at the action-token positions:
#   (B, num_actions_chunk * action_dim, llm_dim)
actions_hidden = hidden[:, action_token_indices]

# Training (ground-truth actions are normalized to [-1, 1]):
pred_actions, loss = head.predict_action_with_loss(actions_hidden, gt_actions)
loss.backward()

# Inference:
pred_actions = head.predict_action(actions_hidden)       # (B, num_actions_chunk, 7)
```

## 📌 TODO

- [x] **Stage 1**: Core implementation of the voxel action heatmap head
- [x] **Stage 2**: OpenVLA-OFT-based training and inference code
- [x] **Stage 3**: Pi-0.5-based training and inference code

## 📄 Citation

```bibtex
@article{actionmap,
    title={ActionMap: Robot Policy Learning via Voxel Action Heatmap}, 
    author={Pei Yang and Hai Ci and Yanzhe Chen and Qi Lv and Han Cai and Mike Zheng Shou},
    year={2026},
    eprint={2606.06904},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2606.06904}, 
}
```
