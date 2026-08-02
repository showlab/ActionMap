# Argument Reference

This page documents the arguments of `vla-scripts/finetune.py` and `experiments/robot/libero/run_libero_eval.py`. Both scripts are configured through [draccus](https://github.com/dlwh/draccus), so every field below is passed as `--field value`.

## ActionMap head

These arguments are shared by training and evaluation, and they define the architecture of the head.

| Argument | Default | Description |
| --- | --- | --- |
| `--use_heatmap` | `False` | Enables the ActionMap voxel heatmap head. Mutually exclusive with `--use_l1_regression` and `--use_diffusion`. |
| `--heatmap_trans_grid` | `32,32,16` | Translation voxel grid resolution, as `X,Y,Z`. |
| `--heatmap_rot_grid` | `16,16,16` | Rotation voxel grid resolution, as `R,P,W`. |
| `--heatmap_trans_sigma` | `0.1` | Width of the Gaussian target placed on the translation grid, in normalized action units. |
| `--heatmap_rot_sigma` | `0.2` | Width of the Gaussian target placed on the rotation grid. |
| `--heatmap_grid_pad` | `0.0` | Extends the translation grid beyond the normalized range, so that targets near the boundary stay fully inside the grid. |
| `--periodic_rot` | `False` | Treats the rotation axes as circular, which matters when the rotation representation wraps around. |

The two grid arguments change the size of the output layers, so evaluation must repeat the values used during training. Our reported LIBERO results use `--heatmap_trans_grid 48,48,24 --heatmap_rot_grid 24,24,24` with both sigmas at `0.10`.

## Training only

| Argument | Default | Description |
| --- | --- | --- |
| `--vla_path` | `openvla/openvla-7b` | Base VLA checkpoint, either a Hugging Face Hub identifier or a local directory. |
| `--data_root_dir` | `datasets/rlds` | Directory containing the RLDS datasets. |
| `--dataset_name` | | Dataset to train on, for example `libero_spatial_no_noops`. |
| `--run_root_dir` | `runs` | Destination for checkpoints and logs. |
| `--heatmap_loss_type` | `ce` | Training objective, either `ce` for cross-entropy against the Gaussian target or `l2` for mean squared error. |
| `--batch_size` | `8` | Per-device batch size. The effective batch size is this value times the number of GPUs times `--grad_accumulation_steps`. |
| `--grad_accumulation_steps` | `1` | Gradient accumulation steps. |
| `--learning_rate` | `5e-4` | Peak learning rate. |
| `--lr_warmup_steps` | `0` | Steps spent warming the learning rate up from 10 percent to 100 percent. |
| `--num_steps_before_decay` | `100000` | Step at which the learning rate drops by a factor of 10. |
| `--max_steps` | `200000` | Total training steps. |
| `--image_aug` | `True` | Random resized cropping during training. Evaluation must then use `--center_crop True`. |
| `--num_images_in_input` | `1` | Number of camera views. LIBERO uses 2, a third-person view and a wrist view. |
| `--use_proprio` | `False` | Appends the proprioceptive state to the input. LIBERO uses `True`. |
| `--use_lora` | `True` | LoRA finetuning rather than full finetuning. |
| `--lora_rank` | `32` | LoRA rank. Evaluation must repeat this value. |
| `--merge_lora_during_training` | `True` | Writes merged weights alongside the adapter. Setting this to `False` keeps checkpoints near 1GB instead of 15GB. |
| `--save_freq` | `10000` | Checkpoint interval in steps. |
| `--max_checkpoints` | `None` | Retains only this many checkpoints, deleting the oldest first. |
| `--pinned_checkpoints` | `None` | Comma-separated steps that are exempt from the rotation above. |
| `--resume`, `--resume_step` | `False`, `None` | Resumes training from an existing checkpoint at the given step. |
| `--shuffle_buffer_size` | `10000` | Dataloader shuffle buffer. Reduce this if the host runs out of memory. |
| `--use_val_set` | `False` | Computes validation metrics every `--val_freq` steps. |
| `--use_wandb` | `False` | Logs training metrics to Weights and Biases. Left off, training needs no Weights and Biases account. |
| `--wandb_entity`, `--wandb_project` | placeholders | Weights and Biases destination, used when `--use_wandb True`. |
| `--run_id_note` | `None` | Suffix appended to the generated run directory name. |

## Evaluation only

| Argument | Default | Description |
| --- | --- | --- |
| `--pretrained_checkpoint` | | Checkpoint directory to evaluate, in either the LoRA or the merged format. |
| `--task_suite_name` | `libero_spatial` | One of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`. |
| `--num_trials_per_task` | `50` | Rollouts per task. |
| `--seed` | `7` | Seed for the environment and the policy. |
| `--center_crop` | `True` | Center crops the input images, matching `--image_aug` during training. |
| `--num_open_loop_steps` | `8` | Actions executed per policy query, which should equal the trained chunk length. |
| `--num_steps_wait` | `10` | Simulation steps waited at episode start, letting objects settle. |
| `--env_img_res` | `256` | Rendering resolution of the environment, which is independent of the policy input resolution. |
| `--unnorm_key` | | Dataset statistics key used to un-normalize actions. It is derived from the task suite when left empty. |
| `--local_log_dir` | `./experiments/logs` | Destination for evaluation logs. |
| `--load_in_8bit`, `--load_in_4bit` | `False` | Quantized loading, which lowers memory use at some cost in accuracy. |

### Decoding

Decoding arguments reduce a predicted heatmap to a single action, and they can be changed without retraining.

| Argument | Default | Description |
| --- | --- | --- |
| `--heatmap_decode_mode` | `soft_argmax` | One of `soft_argmax`, `topk`, `hard_argmax`, `mean`. |
| `--heatmap_decode_top_k` | `0` | Number of highest-probability voxels kept in `topk` and `mean` modes. |
| `--heatmap_decode_temperature` | `1.0` | Softmax temperature, where values below 1 sharpen the distribution. |

`soft_argmax` averages over the whole grid, which is smooth but sensitive to probability mass far from the peak. `topk` restricts that average to the highest-probability voxels, and we use it with `--heatmap_decode_top_k 10` for our reported results.
