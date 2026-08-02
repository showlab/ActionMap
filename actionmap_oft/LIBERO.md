# LIBERO Training and Evaluation

This page covers the four LIBERO task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. Complete [SETUP.md](SETUP.md) first.

## Datasets

Training uses the modified LIBERO RLDS datasets released with OpenVLA, which remove no-op actions:

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

The download is roughly 10GB and contains one dataset per task suite. Evaluation reads its initial states from the LIBERO benchmark itself, so the datasets are needed only for training.

To rebuild these datasets from the raw LIBERO demonstrations instead, `experiments/robot/libero/regenerate_libero_dataset.py` replays them at 256x256 resolution and drops the no-op transitions. Its docstring documents the arguments.

## Training

The command below finetunes OpenVLA-7B on LIBERO-Spatial with the ActionMap head, using the configuration that produced our reported results:

```bash
bash scripts/train_libero.sh libero_spatial /path/to/modified_libero_rlds 2
```

The third argument is the number of GPUs, and the script keeps the effective batch size at 64 regardless of that number. Checkpoints are written to `./runs`, and training runs for 10K steps.

To train on another suite, replace `libero_spatial` with one of the following:

- `libero_object` for LIBERO-Object.
- `libero_goal` for LIBERO-Goal.
- `libero_10` for LIBERO-Long, which the benchmark names `libero_10`.

The script appends the `_no_noops` suffix for you. Every argument is documented in [ARGUMENTS.md](ARGUMENTS.md), including the voxel grid resolutions and the Gaussian target widths.

## Evaluation

```bash
bash scripts/eval_libero.sh libero_spatial ./runs/<run_name>--10000_chkpt
```

This runs 50 trials for each of the 10 tasks in the suite, at seed 7. The script sets the EGL environment variables that MuJoCo needs for headless rendering, so remove them if you have a display attached.

Success rates are printed per task and in aggregate, logs are written to `./experiments/logs`, and rollout videos are written to `./rollouts`.

### Matching evaluation to training

The heatmap arguments define the model architecture, so `--heatmap_trans_grid` and `--heatmap_rot_grid` must repeat the values used during training. A mismatch produces a checkpoint loading error rather than silently wrong actions.

Decoding arguments are the exception, since they only affect how a predicted heatmap is reduced to an action. The released scripts decode with `--heatmap_decode_mode topk --heatmap_decode_top_k 10`, which averages the 10 highest-probability voxels.

## Checkpoint format

Training saves LoRA adapters rather than merged weights, which keeps each checkpoint near 1GB instead of 15GB. The evaluation script detects this format and pulls the `openvla/openvla-7b` base model from the Hugging Face Hub automatically.

Pass `--merge_lora_during_training True` if you would rather write self-contained merged checkpoints, or use `vla-scripts/merge_lora_weights_and_save.py` to merge one after the fact.
