"""Main script for running pi05_droid policy on Franka robot.

This script runs on your workstation with the 5090 GPU and:
1. Loads the pi05_droid policy locally (on GPU)
2. Captures images from RealSense cameras
3. Connects to the Franka NUC via ZeroRPC
4. Runs policy inference and executes actions

"""

import contextlib
import datetime
import signal
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import tyro
import json
from moviepy import ImageSequenceClip  # type: ignore
from datetime import datetime

# Import openpi modules
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from PIL import Image

# Import local modules
import camera_utils
import config as cfg
from franka_interface import FrankaInterface, MockRobot
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

GRIPPER_CLOSE_THRESHOLD = 0.07  # meters (70mm)


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until after policy inference completes."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: [qx, qy, qz, qw] (Franka format)

    Returns:
        [roll, pitch, yaw] in radians
    """
    rot = R.from_quat(quat)  # scipy expects [qx, qy, qz, qw]
    euler = rot.as_euler("xyz", degrees=False)
    return euler


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to quaternion.

    Args:
        euler: [roll, pitch, yaw] in radians

    Returns:
        [qx, qy, qz, qw] quaternion (Franka format)
    """
    rot = R.from_euler("xyz", euler, degrees=False)
    quat = rot.as_quat()  # Returns [qx, qy, qz, qw]
    return quat


def apply_euler_delta(current_euler: np.ndarray, delta_euler: np.ndarray, mode="add") -> np.ndarray:
    """
    Apply Euler angle delta using rotation composition.
    This is more robust than simple addition for large rotations.

    Args:
        current_euler: [roll, pitch, yaw] current orientation
        delta_euler: [Δroll, Δpitch, Δyaw] delta orientation
        mode: "add" to add delta, "multiply" to compose rotations

    Returns:
        [roll, pitch, yaw] new orientation
    """

    if mode == "add":
        # Simple addition (not recommended for large angles)
        new_euler = current_euler + delta_euler
    elif mode == "multiply":
        # Convert to rotation matrices
        R_current = R.from_euler("xyz", current_euler, degrees=False)
        R_delta = R.from_euler("xyz", delta_euler, degrees=False)
        # Compose: R_new = R_delta * R_current
        R_new = R_delta * R_current
        new_euler = R_new.as_euler("xyz", degrees=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Convert back to Euler
    return new_euler


class FrankaPolicyRunner:
    """Manages policy inference and robot control."""

    def __init__(self, config: cfg.Config):
        self.config = config
        self.robot: Optional[FrankaInterface] = None
        self.external_cam = None
        self.wrist_cam = None
        self.left_cam = None
        self.policy = None

        # Safety limits
        self.max_position_delta = 0.05  # 5cm per step
        self.max_rotation_delta = 0.2  # ~11 degrees per step

        # use normal
        self.use_normalize = False

        # Gripper state management
        self.last_gripper_state = None  # Track last gripper command (True=closed, False=open)
        self.gripper_cooldown_steps = 10  # Number of steps to wait before allowing another gripper command
        self.steps_since_last_gripper_cmd = 0  # Counter for cooldown

    def setup(self):
        """Initialize all components."""
        print("=" * 70)
        print("Setting up Franka Pi05 Policy Runner")
        print("=" * 70)

        # 1. Setup cameras
        self._setup_cameras()

        # 2. Connect to robot
        self._connect_robot()

        # 3. Load policy model (locally on this GPU workstation)
        self._load_policy()

        print("\n" + "=" * 70)
        print("Setup complete! Ready to run.")
        print("=" * 70 + "\n")

    def _setup_cameras(self):
        """Initialize RealSense cameras."""
        print("\n[1/3] Setting up cameras...")

        if self.config.camera.use_mock_cameras:
            print("  Using MOCK cameras (no real hardware)")
            self.external_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            self.wrist_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            self.left_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )

        else:
            # List available cameras
            devices = camera_utils.list_realsense_devices()
            print(f"  Found {len(devices)} RealSense device(s)")

            if not devices:
                raise RuntimeError(
                    "No RealSense cameras found! "
                    "Run 'python camera_utils.py' to list devices, "
                    "or use --use-mock-cameras for testing."
                )

            for i, dev in enumerate(devices):
                print(f"    [{i}] {dev['name']} (Serial: {dev['serial_number']})")

            # Initialize cameras
            print(f"  Initializing external camera (serial: {self.config.camera.external_camera_serial})...")
            self.external_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.external_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
            print(f"  Initializing wrist camera (serial: {self.config.camera.wrist_camera_serial})...")
            self.wrist_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.wrist_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
            print(f"  Initializing left camera (serial: {self.config.camera.left_camera_serial})...")
            self.left_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.left_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )

        print("  ✓ Cameras ready")

    def _connect_robot(self):
        """Connect to Franka robot via ZeroRPC."""
        if self.config.robot.use_mock_robot:
            print(f"\n[2/3] Using MOCK robot (no real hardware)...")
            self.robot = MockRobot(ip=self.config.robot.nuc_ip, port=self.config.robot.nuc_port)
        else:
            print(f"\n[2/3] Connecting to Franka NUC at {self.config.robot.nuc_ip}:{self.config.robot.nuc_port}...")

            try:
                self.robot = FrankaInterface(ip=self.config.robot.nuc_ip, port=self.config.robot.nuc_port)
            except Exception as e:
                raise RuntimeError(f"Failed to connect to robot: {e}")

        # Test connection by getting robot state
        joint_pos = self.robot.get_joint_positions()
        gripper_pos = self.robot.get_gripper_position()

        print(f"  ✓ Connected to robot")
        print(f"    Joint positions: {joint_pos}")
        print(f"    Gripper position: {gripper_pos}")

        # Initialize robot controller based on control mode
        if self.config.robot.control_mode == "joint":
            print("  Starting joint impedance controller...")
            self.robot.start_joint_impedance(Kq=None, Kqd=None)
        elif self.config.robot.control_mode == "eef":
            print("  Starting Cartesian impedance controller...")
            # Stiffness and damping for position (x,y,z) and orientation (rx,ry,rz)
            Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
            Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
            self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
            # Note: End-effector poses use quaternion format: [x, y, z, qw, qx, qy, qz]

        print("  ✓ Robot controller started")

    def _load_policy(self):
        """Load pi05_droid policy model on local GPU."""
        print(f"\n[3/3] Loading policy model '{self.config.policy.checkpoint_name}'...")
        print(f"  Checkpoint: {self.config.policy.checkpoint_path}")

        # Get policy config
        policy_cfg = _config.get_config(self.config.policy.checkpoint_name)

        # Download checkpoint if needed
        print("  Downloading checkpoint (if not cached)...")
        checkpoint_dir = download.maybe_download(self.config.policy.checkpoint_path)
        print(f"  Using checkpoint from: {checkpoint_dir}")

        # Create policy (loads model weights onto GPU)
        print("  Loading model weights onto GPU...")
        self.policy = policy_config.create_trained_policy(policy_cfg, checkpoint_dir)

        print("  ✓ Policy loaded and ready")

    def _get_robot_state(self, target_pose) -> np.ndarray:
        """
        Get current robot state in model format.

        Returns:
            state (8D): [x, y, z, roll, pitch, yaw, gripper_1, gripper_2]
        """
        # TODO: Get end-effector pose: [x, y, z, qx, qy, qz, qw]
        if target_pose is not None:
            ee_pose = target_pose
        else:
            ee_pose = self.robot.get_ee_pose()

        # Extract position
        position = ee_pose[:3]

        # Convert quaternion to Euler
        quat = ee_pose[3:]  # [qx, qy, qz, qw] - Franka format
        euler = quaternion_to_euler(quat)  # [roll, pitch, yaw]

        # Get gripper width
        gripper_width = self.robot.get_gripper_position()[0]

        # Convert to symmetric format (matches training data)
        gripper_1 = gripper_width / 2.0
        gripper_2 = -gripper_width / 2.0

        # [x, y, z]  # [roll, pitch, yaw]  # symmetric gripper
        state = np.concatenate([position, euler, [gripper_1, gripper_2]]).astype(np.float32)

        return state

    def _execute_action(self, pred_action: np.ndarray, current_state: np.ndarray, dt: float):
        """
        Execute action on robot.

        Args:
            raw_action (7D): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_binary]
            current_state (8D): Current robot state (raw)
            dt: Control timestep
        """
        # Parse action
        delta_pos = pred_action[:3]
        delta_rot = pred_action[3:6]
        gripper_cmd = pred_action[6]

        # Safety clipping
        delta_pos_clipped = np.clip(delta_pos, -self.max_position_delta, self.max_position_delta)
        delta_rot_clipped = np.clip(delta_rot, -self.max_rotation_delta, self.max_rotation_delta)

        if not np.allclose(delta_pos, delta_pos_clipped) or not np.allclose(delta_rot, delta_rot_clipped):
            print(f"  [WARNING] Action clipped for safety!")
            print(f"    Δpos: {delta_pos} → {delta_pos_clipped}")
            print(f"    Δrot: {delta_rot} → {delta_rot_clipped}")

        # Current state
        current_pos = current_state[:3]
        current_euler = current_state[3:6]

        # Compute target
        target_pos = current_pos + delta_pos_clipped
        target_euler = apply_euler_delta(current_euler, delta_rot_clipped, mode="add")
        target_quat = euler_to_quaternion(target_euler)

        # Ensure z-axis is vertical by zeroing out roll and pitch
        # target_euler_vertical = np.array([0.0, 0.0, target_euler[2]])
        # target_quat = euler_to_quaternion(target_euler_vertical)

        # Build target pose
        target_ee_pose = np.concatenate([target_pos, target_quat]).astype(np.float32)

        # Send to robot
        self.robot.update_desired_ee_pose(target_ee_pose)

        # Control gripper with state tracking and cooldown
        gripper_open = bool(gripper_cmd > 0.5)  # 1.0 means open
        gripper_close = not gripper_open  # Invert for robot API (True=close, False=open)

        # Increment cooldown counter
        self.steps_since_last_gripper_cmd += 1

        # Initialize gripper state on first call
        if self.last_gripper_state is None:
            # Get current gripper state from observation
            current_gripper_width = current_state[6:8].mean() * 2  # Convert symmetric format back to width
            self.last_gripper_state = current_gripper_width < GRIPPER_CLOSE_THRESHOLD
            print(
                f"  [GRIPPER] Initialized state: {'CLOSED' if self.last_gripper_state else 'OPEN'} (width: {current_gripper_width*1000:.1f}mm)"
            )

        # Only send control command if:
        # 1. State has changed from last commanded state
        # 2. Cooldown period has passed
        # 3. gripper prev cmd success
        if (
            gripper_close != self.last_gripper_state
            and self.steps_since_last_gripper_cmd >= self.gripper_cooldown_steps
            and self.robot.get_gripper_prev_cmd_success()
        ):
            print(
                f"  [GRIPPER] State change detected: {'CLOSED' if self.last_gripper_state else 'OPEN'} → {'CLOSED' if gripper_close else 'OPEN'}"
            )
            self.robot.control_gripper(gripper_close)
            self.last_gripper_state = gripper_close
            self.steps_since_last_gripper_cmd = 0
        elif gripper_close != self.last_gripper_state:
            # State change requested but cooldown not elapsed
            print(
                f"  [GRIPPER] State change requested but in cooldown ({self.steps_since_last_gripper_cmd}/{self.gripper_cooldown_steps} steps)"
            )

        return target_ee_pose

    def run_rollout(self, instruction: str, max_timesteps: Optional[int] = None):
        """Execute rollout with proper normalization."""
        if max_timesteps is None:
            max_timesteps = self.config.max_timesteps

        print(f"\n{'=' * 70}")
        print(f"TASK: {instruction}")
        print(f"Max timesteps: {max_timesteps}")
        print(f"{'=' * 70}")

        # Reset gripper state tracking for new rollout
        self.last_gripper_state = None
        self.steps_since_last_gripper_cmd = 0

        actions_from_chunk_completed = 0
        pred_action_chunk = None

        video_frames = []
        dt = 1.0 / self.config.robot.control_frequency
        inference_times = []

        print("Running rollout... (Press Ctrl+C to stop)")

        target_pose = None
        try:
            for t_step in range(max_timesteps):
                step_start_time = time.time()

                # 1. Get raw robot state
                raw_state = self._get_robot_state(target_pose)

                # 2. Capture images
                ret_ext, external_img, _ = self.external_cam.read()
                ret_wrist, wrist_img, _ = self.wrist_cam.read()
                ret_left, left_img, _ = self.left_cam.read()

                if not ret_ext or not ret_wrist or not ret_left:
                    print(f"  [Step {t_step}] Failed to capture images!")
                    break

                external_img_rgb = cv2.cvtColor(external_img, cv2.COLOR_BGR2RGB)
                wrist_img_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

                if self.config.save_video:
                    video_frames.append(external_img_rgb.copy())

                # 3. Check against actual chunk length, not config
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.config.policy.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    obs = {
                        "observation/image": external_img_rgb,
                        "observation/wrist_image": wrist_img_rgb,
                        "observation/left_image": left_img_rgb,
                        "observation/state": raw_state,
                        "prompt": instruction,
                    }

                    if actions_from_chunk_completed == 0:
                        Image.fromarray(external_img_rgb).save("./debug_front.png")
                        Image.fromarray(left_img_rgb).save("./debug_left.png")

                    # Run inference
                    inference_start = time.time()

                    result = self.policy.infer(obs)
                    inference_time = (time.time() - inference_start) * 1000
                    inference_times.append(inference_time)

                    pred_action_chunk = result["actions"]

                    print(f"\n  [Step {t_step:3d}] NEW ACTION CHUNK")
                    print(f"    Inference: {inference_time:.1f}ms")
                    print(f"    Chunk size: {len(pred_action_chunk)} actions")
                    print(f"    First action: {pred_action_chunk[0]}")
                    print(f"      Δpos: {pred_action_chunk[0, :3]}")
                    print(f"      Δrot (deg): {np.rad2deg(pred_action_chunk[0, 3:6])}")
                    print(f"      Gripper: {pred_action_chunk[0, 6]:.3f}")

                # 4. Execute denormalized action
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1  # Small delay to ensure timing accuracy

                # Print periodic updates
                if t_step % 10 == 0 or actions_from_chunk_completed == 1:
                    print(f"  [Step {t_step:3d}] Action {actions_from_chunk_completed}/{len(pred_action_chunk)}")
                    print(
                        f"      Gripper cmd: {pred_action_chunk[0, 6]:.3f} -> {'OPEN' if pred_action_chunk[0, 6] > 0.5 else 'CLOSE'}"
                    )

                target_pose = self._execute_action(action, raw_state, dt)

                # 5. Regulate timing
                elapsed = time.time() - step_start_time
                # print(f"    Step time: {elapsed*1000:.1f}ms (target: {dt*1000:.1f}ms)")
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                elif t_step % 10 == 0:
                    print(f"  [WARNING] Step {t_step} overran: {elapsed*1000:.1f}ms (target: {dt*1000:.1f}ms)")

        except KeyboardInterrupt:
            print("\n\n  Rollout interrupted by user")

        finally:
            print(f"\n{'=' * 70}")
            print("ROLLOUT COMPLETE")
            print(f"{'=' * 70}")
            print(f"  Steps: {len(video_frames)}")
            if inference_times:
                print(f"  Inference: {np.mean(inference_times):.1f}ms ± {np.std(inference_times):.1f}ms")
                print(f"    Min: {np.min(inference_times):.1f}ms, Max: {np.max(inference_times):.1f}ms")

            if self.config.save_video and video_frames:
                self._save_video(video_frames, instruction)

            if self.config.show_cameras:
                try:
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print(f"  Note: Could not destroy windows (headless mode): {e}")

    def _save_video(self, frames: list[np.ndarray], instruction: str):
        """Save rollout video."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_instruction = "".join(c if c.isalnum() else "_" for c in instruction[:30])
        filename = f"rollout_{timestamp}_{safe_instruction}.mp4"

        print(f"\nSaving video to {filename}...")
        try:
            clip = ImageSequenceClip(frames, fps=self.config.video_fps)
            clip.write_videofile(filename, codec="libx264", verbose=False, logger=None)
            print(f"  ✓ Video saved: {filename}")
        except Exception as e:
            print(f"  ✗ Failed to save video: {e}")

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        if self.robot is not None:
            try:
                self.robot.terminate_current_policy()
                self.robot.close()
                print("  ✓ Robot disconnected")
            except Exception as e:
                print(f"  ✗ Error disconnecting robot: {e}")

        if self.external_cam is not None:
            self.external_cam.release()

        if self.wrist_cam is not None:
            self.wrist_cam.release()

        if self.left_cam is not None:
            self.left_cam.release()

        # cv2.destroyAllWindows()
        print("  ✓ Cleanup complete")


def main(
    # Camera settings
    external_camera: Optional[str] = None,
    wrist_camera: Optional[str] = None,
    left_camera: Optional[str] = None,
    use_mock_cameras: bool = False,
    # Robot settings
    nuc_ip: str = "192.168.1.112",
    nuc_port: int = 4242,
    control_mode: str = "eef",
    use_mock_robot: bool = False,
    # Policy settings
    checkpoint_name: str = "pi05_flow_matching_build_block",
    checkpoint_path: str = "./checkpoints/pi05_flow_matching_build_block",
    # Rollout settings
    instruction: str = "Place the orange block on top of the gray block",
    max_timesteps: int = 600,
    show_cameras: bool = True,
    save_video: bool = True,
):
    """
    Run pi05 real-robot policy on Franka.

    Args:
        external_camera: Serial number of external (base) RealSense camera
        wrist_camera: Serial number of right-wrist RealSense camera
        left_camera: Serial number of left-wrist RealSense camera
        use_mock_cameras: Use mock cameras for testing (no real hardware)
        nuc_ip: IP address of Franka NUC running Polymetis
        nuc_port: Port of ZeroRPC server on NUC
        control_mode: Control mode ('eef' for Cartesian, 'joint' for joint impedance)
        use_mock_robot: Use mock robot for testing (no real robot hardware)
        instruction: Natural-language task instruction passed to the policy
        checkpoint_name: Name of policy checkpoint
        checkpoint_path: Path to policy checkpoint
        max_timesteps: Maximum steps per rollout
        show_cameras: Display camera feeds during rollout
        save_video: Save rollout video
    """

    # Build configuration
    config = cfg.Config(
        camera=cfg.CameraConfig(
            external_camera_serial=external_camera,
            wrist_camera_serial=wrist_camera,
            left_camera_serial=left_camera,
            use_mock_cameras=use_mock_cameras,
        ),
        robot=cfg.RobotConfig(
            nuc_ip=nuc_ip,
            nuc_port=nuc_port,
            control_mode=control_mode,
            use_mock_robot=use_mock_robot,
        ),
        policy=cfg.PolicyConfig(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
        ),
        max_timesteps=max_timesteps,
        show_cameras=show_cameras,
        save_video=save_video,
    )

    # Create runner
    runner = FrankaPolicyRunner(config)

    try:
        # Setup all components
        runner.setup()

        # Main loop: ask for instructions and run rollouts
        while True:
            print("\n" + "=" * 70)
            print(f"Instruction: {instruction}")

            # Run rollout
            runner.run_rollout(instruction)

            # Ask if user wants to continue
            continue_prompt = input("\nRun another task? (y/n): ").strip().lower()
            if continue_prompt != "y":
                break

    except KeyboardInterrupt:
        print("\n\nShutdown requested (Ctrl+C)")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        runner.cleanup()
        print("\nGoodbye!")


if __name__ == "__main__":
    tyro.cli(main)
