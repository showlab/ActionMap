# Adapting to a Real Robot

ActionMap is not tied to LIBERO, and the LIBERO pipeline is the template for a real arm. This page covers the two pieces that differ: preparing your data, and serving the trained policy to your robot.

## Data format

Training reads RLDS datasets, the same format the LIBERO pipeline uses. Each step of a trajectory holds:

| Field | Shape | Description |
| --- | --- | --- |
| `observation/image` | `(H, W, 3)` | Third-person camera view. |
| `observation/wrist_image` | `(H, W, 3)` | Wrist camera view, used when `--num_images_in_input 2`. |
| `observation/state` | `(D,)` | Proprioceptive state, used when `--use_proprio True`. |
| `action` | `(7,)` | Delta end-effector pose plus a gripper command. |
| `language_instruction` | string | Task description given to the policy. |

Actions are delta end-effector commands, laid out as three translation values, three rotation values, and one gripper value. The gripper value is binary, where 1 opens and 0 closes.

Normalization is handled for you. The training pipeline computes the 1st and 99th percentile of your action and proprioceptive data, then maps that range onto the interval that the voxel grid covers.

## Converting your data

Use the standard [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) to turn your recorded episodes into RLDS, which is the same tool the LIBERO and Open X-Embodiment datasets were built with. Its example builder is a good starting point, since you only need to emit the fields in the table above.

Once the dataset is built, register it in three places so the pipeline can find it:

- `prismatic/vla/datasets/rlds/oxe/transforms.py`, adding a transform that reshapes your raw fields into the layout above. The LIBERO transform splits `state` into an `EEF_state` and a `gripper_state` field, and yours can be a passthrough if your data already matches.
- `prismatic/vla/datasets/rlds/oxe/configs.py`, declaring your image keys and the state keys the transform produces. Follow the `libero_spatial_no_noops` entry, whose `state_obs_keys` names the two fields its transform creates rather than anything in the raw dataset.
- `prismatic/vla/datasets/rlds/oxe/mixtures.py`, adding a single-entry mixture with your dataset name.

Training then runs exactly as in [LIBERO.md](LIBERO.md), with `--dataset_name` set to your dataset:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /path/to/your/rlds \
    --dataset_name your_dataset_name \
    --use_l1_regression False --use_heatmap True \
    --heatmap_trans_grid "48,48,24" --heatmap_rot_grid "24,24,24" \
    --heatmap_trans_sigma 0.10 --heatmap_rot_sigma 0.10 \
    --num_images_in_input 2 --use_proprio True \
    --batch_size 8 --grad_accumulation_steps 4 --learning_rate 5e-4 \
    --lora_rank 32 --merge_lora_during_training False
```

### Choosing a grid

The voxel grid discretizes the normalized action range, so its resolution sets the finest action the policy can express. Dividing your action range by the grid resolution gives the physical size of one voxel, which is worth checking against the precision your task needs.

Set `--periodic_rot True` if your rotation representation wraps around, so that values at opposite ends of the range are treated as neighbors rather than opposites.

## Serving the policy

`vla-scripts/deploy.py` runs the trained policy behind an HTTP endpoint, which keeps your robot control code independent of this repository:

```bash
python vla-scripts/deploy.py \
    --pretrained_checkpoint /path/to/checkpoint \
    --unnorm_key your_dataset_name \
    --use_l1_regression False --use_heatmap True \
    --heatmap_trans_grid "48,48,24" --heatmap_rot_grid "24,24,24" \
    --heatmap_trans_sigma 0.10 --heatmap_rot_sigma 0.10 \
    --heatmap_decode_mode topk --heatmap_decode_top_k 10 \
    --num_images_in_input 2 --use_proprio True --lora_rank 32 \
    --port 8777
```

Your control loop then posts an observation and receives an action chunk:

```python
import json_numpy, requests
json_numpy.patch()

action_chunk = requests.post(
    "http://<server-host>:8777/act",
    json={
        "full_image": third_person_rgb,   # (H, W, 3) uint8
        "wrist_image": wrist_rgb,         # (H, W, 3) uint8
        "state": proprio_state,           # (D,) float
        "instruction": "pick up the red block",
    },
).json()
```

The server returns a chunk of consecutive actions rather than one action. Executing the whole chunk before querying again is what makes inference fast, and the chunk length is fixed at training time by `NUM_ACTIONS_CHUNK` in `prismatic/vla/constants.py`.

## Constants to check

`prismatic/vla/constants.py` sets the action dimension, the chunk length, and the proprioceptive dimension. The defaults match a 7-dimensional end-effector action with 8-dimensional proprioception, so edit them if your robot differs.
