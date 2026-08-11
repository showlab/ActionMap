import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_real_robot_example() -> dict:
    """Creates a random input example for the real robot (Panda) policy."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 HWC format.

    LeRobot stores images as float32 (C, H, W) in [0, 1]. During inference
    images come in as uint8 (H, W, C). This function handles both.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RealRobotInputs(transforms.DataTransformFn):
    """Input transform for the real Panda robot dataset.

    Dataset cameras:
      - ``image``       : third-person / base camera  → base_0_rgb
      - ``wrist_image`` : right wrist camera           → right_wrist_0_rgb
      - ``left_image``  : left wrist camera            → left_wrist_0_rgb

    State: 8-D  [x, y, z, roll, pitch, yaw, gripper_open, gripper_close]
    Actions: 7-D delta  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        left_image = _parse_image(data["observation/left_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # For pi0 (not pi0-FAST), the third slot is masked if unused.
                # All three cameras are present here, so always True.
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealRobotOutputs(transforms.DataTransformFn):
    """Output transform for the real Panda robot.

    Truncates the model output (padded to action_dim=32) back to the 7
    action dimensions used by the robot.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
