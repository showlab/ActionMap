"""Heatmap (voxel map) action head for JAX/Flax NNX.

Ported from OpenVLA-OFT (PyTorch) to JAX. Predicts 3D probability distributions
over discretized action grids, decoded via soft-argmax.

Architecture:
    Shared trunk (LayerNorm + Linear + ReLU + 2 ResBlocks)
    -> Translation branch: Linear -> ReLU -> Linear -> logits over trans grid
    -> Rotation branch:    Linear -> ReLU -> Linear -> logits over rot grid
    -> Gripper branch:     Linear -> scalar logit
"""

import dataclasses
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class MLPResNetBlock(nnx.Module):
    """One MLP ResNet block with pre-norm residual connection."""

    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.linear = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):
        identity = x
        x = self.layer_norm(x)
        x = self.linear(x)
        x = jax.nn.relu(x)
        return x + identity


def make_grid(grid_shape: tuple[int, int, int], pad: float = 0.0) -> jnp.ndarray:
    """Create uniformly-spaced grid centers in [-(1+pad), 1+pad]^3.

    Returns:
        centers: (N, 3) where N = gx * gy * gz
    """
    gx, gy, gz = grid_shape
    lo, hi = -1.0 - pad, 1.0 + pad
    xs = jnp.linspace(lo, hi, gx)
    ys = jnp.linspace(lo, hi, gy)
    zs = jnp.linspace(lo, hi, gz)
    grid_x, grid_y, grid_z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    centers = jnp.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    return centers


def gaussian_target(
    true_action: jnp.ndarray,
    grid_centers: jnp.ndarray,
    sigma: float,
    periodic: bool = False,
) -> jnp.ndarray:
    """Compute Gaussian soft targets over grid centers.

    Args:
        true_action: (B, T, 3) ground-truth 3D coordinates in [-1, 1]
        grid_centers: (N, 3) grid center positions
        sigma: standard deviation of Gaussian
        periodic: if True, use circular distance on [-1, 1] with period 2

    Returns:
        targets: (B, T, N) normalized probability distributions
    """
    diff = true_action[:, :, None, :] - grid_centers[None, None, :, :]  # (B, T, N, 3)
    if periodic:
        abs_diff = jnp.abs(diff)
        diff = jnp.minimum(abs_diff, 2.0 - abs_diff)
    sq_dist = jnp.sum(diff**2, axis=-1)  # (B, T, N)
    log_targets = -0.5 * sq_dist / (sigma**2)
    targets = jax.nn.softmax(log_targets, axis=-1)  # (B, T, N)
    return targets


def soft_argmax(
    probs: jnp.ndarray,
    grid_centers: jnp.ndarray,
    periodic: bool = False,
) -> jnp.ndarray:
    """Compute soft-argmax: expected coordinate under probability distribution.

    Args:
        probs: (B, T, N) probability distribution over grid
        grid_centers: (N, 3) grid center coordinates
        periodic: if True, use circular mean on [-1, 1] with period 2

    Returns:
        coords: (B, T, 3)
    """
    if periodic:
        return _circular_mean(probs, grid_centers)
    return jnp.matmul(probs, grid_centers)


def _circular_mean(probs: jnp.ndarray, grid_centers: jnp.ndarray) -> jnp.ndarray:
    """Circular weighted mean on [-1, 1] with period 2.

    Maps grid coordinates to angles on the unit circle, computes the
    weighted mean direction via atan2, then maps back.
    """
    angles = grid_centers * jnp.pi  # (N, 3)
    weighted_sin = jnp.matmul(probs, jnp.sin(angles))  # (B, T, 3)
    weighted_cos = jnp.matmul(probs, jnp.cos(angles))  # (B, T, 3)
    mean_angle = jnp.arctan2(weighted_sin, weighted_cos)
    return mean_angle / jnp.pi


def _sigmoid_bce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable sigmoid binary cross-entropy."""
    return jnp.maximum(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


@dataclasses.dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for the heatmap action head."""

    input_dim: int = 1024
    hidden_dim: int = 1024
    action_dim: int = 7  # Always 7: xyz(3) + rpy(3) + grip(1)
    trans_grid: tuple[int, int, int] = (48, 48, 24)
    rot_grid: tuple[int, int, int] = (24, 24, 24)
    trans_sigma: float = 0.20
    rot_sigma: float = 0.20
    loss_type: str = "ce"  # "ce" or "l2"
    decode_mode: str = "soft_argmax"
    decode_temperature: float = 1.0
    periodic_rot: bool = True
    grid_pad: float = 0.0


class HeatmapHead(nnx.Module):
    """Heatmap action head that predicts 3D probability distributions over discretized action space."""

    def __init__(self, config: HeatmapConfig, rngs: nnx.Rngs):
        self.config = config

        trans_num_bins = config.trans_grid[0] * config.trans_grid[1] * config.trans_grid[2]
        rot_num_bins = config.rot_grid[0] * config.rot_grid[1] * config.rot_grid[2]

        # Shared trunk: LayerNorm + Linear + ReLU + 2 ResBlocks
        self.trunk_ln = nnx.LayerNorm(config.input_dim, rngs=rngs)
        self.trunk_fc = nnx.Linear(config.input_dim, config.hidden_dim, rngs=rngs)
        self.resblock1 = MLPResNetBlock(config.hidden_dim, rngs)
        self.resblock2 = MLPResNetBlock(config.hidden_dim, rngs)

        # Translation branch: 2-layer MLP -> logits
        self.trans_fc1 = nnx.Linear(config.hidden_dim, 1024, rngs=rngs)
        self.trans_fc2 = nnx.Linear(1024, trans_num_bins, rngs=rngs)

        # Rotation branch: 2-layer MLP -> logits
        self.rot_fc1 = nnx.Linear(config.hidden_dim, 512, rngs=rngs)
        self.rot_fc2 = nnx.Linear(512, rot_num_bins, rngs=rngs)

        # Gripper branch: single linear -> scalar logit
        self.grip_fc = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

        # Pre-compute grid centers (non-trainable constants)
        self.trans_centers = nnx.Variable(make_grid(config.trans_grid, pad=config.grid_pad))
        self.rot_centers = nnx.Variable(make_grid(config.rot_grid))

    def _trunk(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through shared trunk. Input/output: (B, T, dim)."""
        x = self.trunk_ln(x)
        x = self.trunk_fc(x)
        x = jax.nn.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x

    def predict_action(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Inference-time action prediction via soft-argmax decoding.

        Args:
            hidden_states: (B, T, hidden_dim) from action expert

        Returns:
            actions: (B, T, 7) predicted actions [x,y,z, r,p,w, grip]
        """
        h = self._trunk(hidden_states)

        # Translation (always non-periodic)
        trans_logits = self.trans_fc2(jax.nn.relu(self.trans_fc1(h)))
        trans_coords = self._decode_logits(trans_logits, self.trans_centers.value, periodic=False)

        # Rotation (periodic if enabled)
        rot_logits = self.rot_fc2(jax.nn.relu(self.rot_fc1(h)))
        rot_coords = self._decode_logits(rot_logits, self.rot_centers.value, periodic=self.config.periodic_rot)

        # Gripper: threshold at 0
        grip_logits = self.grip_fc(h)[..., 0]  # (B, T)
        grip_action = (grip_logits > 0).astype(jnp.float32) * 2 - 1  # {-1, +1}

        actions = jnp.concatenate([trans_coords, rot_coords, grip_action[..., None]], axis=-1)
        return actions

    def _decode_logits(self, logits: jnp.ndarray, grid_centers: jnp.ndarray, *, periodic: bool) -> jnp.ndarray:
        """Decode logits to coordinates using the configured decode_mode."""
        mode = self.config.decode_mode
        temp = self.config.decode_temperature
        if mode == "hard_argmax":
            idx = jnp.argmax(logits, axis=-1)  # (B, T)
            return grid_centers[idx]  # (B, T, 3)
        elif mode == "soft_argmax":
            probs = jax.nn.softmax(logits / temp, axis=-1)
            return soft_argmax(probs, grid_centers, periodic=periodic)
        else:
            raise ValueError(f"Unknown decode_mode: {mode}")

    def compute_loss(self, hidden_states: jnp.ndarray, ground_truth_actions: jnp.ndarray) -> jnp.ndarray:
        """Compute per-timestep heatmap loss.

        Args:
            hidden_states: (B, T, hidden_dim) from action expert
            ground_truth_actions: (B, T, 7) [x,y,z, r,p,w, grip] in [-1, 1]

        Returns:
            loss: (B, T) per-timestep loss
        """
        h = self._trunk(hidden_states)

        # Clamp GT to grid range so out-of-range quantile-normalized values get
        # valid Gaussian targets instead of falling outside the grid support.
        pad = self.config.grid_pad
        clamped_actions = jnp.clip(ground_truth_actions[..., :6], -1.0 - pad, 1.0 + pad)

        gt_trans = clamped_actions[..., :3]  # (B, T, 3)
        gt_rot = clamped_actions[..., 3:6]  # (B, T, 3)
        gt_grip = ground_truth_actions[..., 6]  # (B, T)

        # --- Translation loss (always non-periodic) ---
        trans_logits = self.trans_fc2(jax.nn.relu(self.trans_fc1(h)))
        trans_targets = gaussian_target(gt_trans, self.trans_centers.value, self.config.trans_sigma, periodic=False)
        if self.config.loss_type == "ce":
            trans_log_probs = jax.nn.log_softmax(trans_logits, axis=-1)
            loss_trans = -jnp.sum(trans_targets * trans_log_probs, axis=-1)  # (B, T)
        else:
            trans_probs = jax.nn.softmax(trans_logits, axis=-1)
            loss_trans = jnp.mean((trans_probs - trans_targets) ** 2, axis=-1)  # (B, T)

        # --- Rotation loss (periodic if enabled) ---
        rot_logits = self.rot_fc2(jax.nn.relu(self.rot_fc1(h)))
        rot_targets = gaussian_target(gt_rot, self.rot_centers.value, self.config.rot_sigma, periodic=self.config.periodic_rot)
        if self.config.loss_type == "ce":
            rot_log_probs = jax.nn.log_softmax(rot_logits, axis=-1)
            loss_rot = -jnp.sum(rot_targets * rot_log_probs, axis=-1)  # (B, T)
        else:
            rot_probs = jax.nn.softmax(rot_logits, axis=-1)
            loss_rot = jnp.mean((rot_probs - rot_targets) ** 2, axis=-1)  # (B, T)

        # --- Gripper loss (binary cross-entropy) ---
        grip_logits = self.grip_fc(h)[..., 0]  # (B, T)
        grip_binary_target = (gt_grip > 0).astype(jnp.float32)
        loss_grip = _sigmoid_bce(grip_logits, grip_binary_target)  # (B, T)

        return loss_trans + loss_rot + loss_grip
