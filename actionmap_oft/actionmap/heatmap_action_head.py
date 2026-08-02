"""
ActionMap voxel action heatmap head.

This implementation can be plugged into a VLA in place of its native action decoder,
predicts a heatmap over the action space instead of a single point, and computes its
own training loss.
"""

import math

import torch
import torch.nn as nn


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.ffn(x)


class HeatmapActionHead(nn.Module):
    """
    Predicts a probability heatmap over a discretized action space and decodes it
    back to a continuous action by soft-argmax (or top-k / argmax).

    A shared trunk feeds a translation voxel branch, a rotation voxel branch, and a
    binary gripper branch. Actions are expected to be normalized to [-1, 1].
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_actions_chunk=8,
        trans_grid=(32, 32, 16),
        rot_grid=(16, 16, 16),
        trans_sigma=0.1,
        rot_sigma=0.2,
        loss_type="ce",
        decode_mode="soft_argmax",
        decode_temperature=1.0,
        decode_top_k=0,
        periodic_rot=False,
        grid_pad=0.0,
    ):
        super().__init__()
        assert loss_type in ("ce", "l2"), f"loss_type must be 'ce' or 'l2', got '{loss_type}'"
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.trans_grid = trans_grid
        self.rot_grid = rot_grid
        self.trans_sigma = trans_sigma
        self.rot_sigma = rot_sigma
        self.loss_type = loss_type
        self.decode_mode = decode_mode
        self.decode_temperature = decode_temperature
        self.decode_top_k = decode_top_k
        self.periodic_rot = periodic_rot
        self.grid_pad = grid_pad
        self.trans_num_bins = trans_grid[0] * trans_grid[1] * trans_grid[2]
        self.rot_num_bins = rot_grid[0] * rot_grid[1] * rot_grid[2]

        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim * action_dim),
            nn.Linear(input_dim * action_dim, hidden_dim),
            nn.ReLU(),
            MLPResNetBlock(hidden_dim),
            MLPResNetBlock(hidden_dim),
        )

        # one voxel-grid classifier per branch
        self.trans_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.trans_num_bins),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.rot_num_bins),
        )
        self.grip_head = nn.Linear(hidden_dim, 1)

        # translation grid is padded, rotation grid is not (it wraps around)
        self.register_buffer("trans_centers", self._make_grid(trans_grid, pad=grid_pad))
        self.register_buffer("rot_centers", self._make_grid(rot_grid))

    @staticmethod
    def _make_grid(grid_shape, pad=0.0):
        """Uniformly-spaced grid centers in [-(1+pad), 1+pad]^3, shape (N, 3)."""
        gx, gy, gz = grid_shape
        lo, hi = -1.0 - pad, 1.0 + pad
        xs = torch.linspace(lo, hi, gx)
        ys = torch.linspace(lo, hi, gy)
        zs = torch.linspace(lo, hi, gz)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    @staticmethod
    def _gaussian_target(true_action, grid_centers, sigma, periodic=False):
        """Softmax-normalized Gaussian blob over the grid, centered at true_action. Returns (B, T, N)."""
        diff = true_action.unsqueeze(-2) - grid_centers.unsqueeze(0).unsqueeze(0)  # (B, T, N, 3)
        if periodic:
            # wrap onto the [-1, 1] circle (period 2)
            abs_diff = diff.abs()
            diff = torch.min(abs_diff, 2.0 - abs_diff)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, T, N)
        return torch.softmax(-0.5 * sq_dist / (sigma ** 2), dim=-1)

    @staticmethod
    def _soft_argmax(probs, grid_centers, periodic=False):
        """Expected grid coordinate under probs. Returns (B, T, 3)."""
        if periodic:
            return HeatmapActionHead._circular_mean(probs, grid_centers)
        return torch.matmul(probs, grid_centers)

    @staticmethod
    def _circular_mean(probs, grid_centers):
        """Weighted mean on the periodic [-1, 1] axis, via the mean angle on the unit circle."""
        angles = grid_centers * math.pi  # [-1, 1] -> [-pi, pi]
        weighted_sin = torch.matmul(probs, torch.sin(angles))
        weighted_cos = torch.matmul(probs, torch.cos(angles))
        return torch.atan2(weighted_sin, weighted_cos) / math.pi

    @staticmethod
    def _circular_weighted_mean(weights, coords):
        """Circular mean for a subset of grid centers (used by topk / mean decoding)."""
        angles = coords * math.pi  # (..., K, 3)
        weighted_sin = (weights.unsqueeze(-1) * torch.sin(angles)).sum(dim=-2)
        weighted_cos = (weights.unsqueeze(-1) * torch.cos(angles)).sum(dim=-2)
        return torch.atan2(weighted_sin, weighted_cos) / math.pi

    def _forward_trunk(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(batch_size, self.num_actions_chunk, -1)  # (B, T, action_dim * llm_dim)
        return self.trunk(x)  # (B, T, hidden_dim)

    def _decode_logits(self, logits, grid_centers, periodic=False):
        """Turn heatmap logits (B, T, N) into coordinates (B, T, 3) under the configured decode_mode."""
        if self.decode_mode == "hard_argmax":
            idx = logits.argmax(dim=-1)
            coords = grid_centers[idx]
        elif self.decode_mode == "topk":
            k = min(self.decode_top_k, logits.shape[-1])
            topk_logits, topk_idx = logits.topk(k, dim=-1)
            topk_probs = torch.softmax(topk_logits / self.decode_temperature, dim=-1)
            topk_centers = grid_centers[topk_idx]  # (B, T, k, 3)
            if periodic:
                coords = HeatmapActionHead._circular_weighted_mean(topk_probs, topk_centers)
            else:
                coords = (topk_probs.unsqueeze(-1) * topk_centers).sum(dim=-2)
        elif self.decode_mode == "mean":
            k = min(self.decode_top_k, logits.shape[-1])
            _, topk_idx = logits.topk(k, dim=-1)
            topk_centers = grid_centers[topk_idx]  # (B, T, k, 3)
            if periodic:
                uniform_probs = torch.ones_like(topk_idx, dtype=topk_centers.dtype) / k
                coords = HeatmapActionHead._circular_weighted_mean(uniform_probs, topk_centers)
            else:
                coords = topk_centers.mean(dim=-2)
        else:  # soft_argmax
            probs = torch.softmax(logits / self.decode_temperature, dim=-1)
            if periodic:
                coords = HeatmapActionHead._circular_mean(probs, grid_centers)
            else:
                coords = torch.matmul(probs, grid_centers)
        return coords

    def predict_action(self, actions_hidden_states):
        """
        Inference. Decode an action chunk from the predicted heatmaps.

        actions_hidden_states: (B, num_actions_chunk * action_dim, llm_dim)
        returns: (B, num_actions_chunk, 7)  ->  [x, y, z, r, p, w, grip]
        """
        h = self._forward_trunk(actions_hidden_states)

        trans_coords = self._decode_logits(self.trans_head(h), self.trans_centers, periodic=False)
        rot_coords = self._decode_logits(self.rot_head(h), self.rot_centers, periodic=self.periodic_rot)

        grip_logits = self.grip_head(h).squeeze(-1)
        grip_action = (grip_logits > 0).float() * 2 - 1  # {-1, +1}

        return torch.cat([trans_coords, rot_coords, grip_action.unsqueeze(-1)], dim=-1)

    def predict_action_with_loss(self, actions_hidden_states, ground_truth_actions):
        """
        Training. Predict an action chunk and compute the loss in one pass.

        actions_hidden_states: (B, num_actions_chunk * action_dim, llm_dim)
        ground_truth_actions:  (B, num_actions_chunk, 7), normalized to [-1, 1]
        returns: (predicted_actions (B, num_actions_chunk, 7), loss)
        """
        h = self._forward_trunk(actions_hidden_states)

        gt_trans = ground_truth_actions[..., :3]
        gt_rot = ground_truth_actions[..., 3:6]
        gt_grip = ground_truth_actions[..., 6]

        # translation: cross-entropy (or l2) against the Gaussian-blob heatmap
        trans_logits = self.trans_head(h)
        trans_targets = self._gaussian_target(gt_trans, self.trans_centers, self.trans_sigma, periodic=False)
        trans_probs = torch.softmax(trans_logits, dim=-1)
        if self.loss_type == "ce":
            loss_trans = -(trans_targets * torch.log_softmax(trans_logits, dim=-1)).sum(dim=-1).mean()
        else:
            loss_trans = nn.functional.mse_loss(trans_probs, trans_targets)
        trans_coords = self._soft_argmax(trans_probs, self.trans_centers, periodic=False)

        # rotation: same recipe, optionally periodic
        rot_logits = self.rot_head(h)
        rot_targets = self._gaussian_target(gt_rot, self.rot_centers, self.rot_sigma, periodic=self.periodic_rot)
        rot_probs = torch.softmax(rot_logits, dim=-1)
        if self.loss_type == "ce":
            loss_rot = -(rot_targets * torch.log_softmax(rot_logits, dim=-1)).sum(dim=-1).mean()
        else:
            loss_rot = nn.functional.mse_loss(rot_probs, rot_targets)
        rot_coords = self._soft_argmax(rot_probs, self.rot_centers, periodic=self.periodic_rot)

        # gripper: binary cross-entropy
        grip_logits = self.grip_head(h).squeeze(-1)
        loss_grip = nn.functional.binary_cross_entropy_with_logits(grip_logits, (gt_grip > 0).float())
        grip_action = (grip_logits > 0).float() * 2 - 1

        loss = loss_trans + loss_rot + loss_grip
        predicted_actions = torch.cat([trans_coords, rot_coords, grip_action.unsqueeze(-1)], dim=-1)
        return predicted_actions, loss
