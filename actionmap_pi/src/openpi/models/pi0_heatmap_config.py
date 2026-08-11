"""Configuration for the Pi0.5 model with heatmap action head."""

import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_heatmap import Pi0Heatmap


# Action head modes:
#   "single_pass"  - Learned query tokens, single forward pass, heatmap CE loss.
#   "flow_heatmap" - Flow matching input pathway (noisy actions), heatmap decodes x0 at each step.
#   "flow_matching" - Vanilla Pi0.5 flow matching (no heatmap head at all).
ActionHeadMode = Literal["single_pass", "flow_heatmap", "flow_matching"]


@dataclasses.dataclass(frozen=True)
class Pi0HeatmapConfig(_model.BaseModelConfig):
    """Config for Pi0.5 with configurable action head mode.

    Supports three modes via ``action_head_mode``:
        - ``single_pass``:  Heatmap head with learned action queries (no denoising).
        - ``flow_heatmap``: Flow matching input + heatmap x0 prediction at each ODE step.
        - ``flow_matching``: Vanilla Pi0.5 flow matching (baseline).
    """

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Model dimensions (keep 32 for pipeline compatibility with Pi0.5)
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200  # Same as Pi0.5

    # Used by ModelTransformFactory to handle state encoding
    discrete_state_input: bool = False

    # ---- Action head mode ----
    action_head_mode: ActionHeadMode = "single_pass"

    # ---- Heatmap head parameters (ignored when action_head_mode="flow_matching") ----
    heatmap_hidden_dim: int = 1024
    heatmap_trans_grid: tuple[int, int, int] = (48, 48, 24)
    heatmap_rot_grid: tuple[int, int, int] = (24, 24, 24)
    heatmap_trans_sigma: float = 0.20
    heatmap_rot_sigma: float = 0.20
    heatmap_loss_type: str = "ce"
    heatmap_decode_mode: str = "soft_argmax"
    heatmap_decode_temperature: float = 1.0
    heatmap_periodic_rot: bool = False
    heatmap_grid_pad: float = 0.0

    pytorch_compile_mode: str | None = "max-autotune"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Reuse PI05 for transform compatibility (same image/text/action pipeline)
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Heatmap":
        from openpi.models.pi0_heatmap import Pi0Heatmap

        return Pi0Heatmap(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        if "lora" in self.action_expert_variant:
            if not has_lora:
                filters.append(action_expert_params_filter)
            has_lora = True
        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
            return nnx.All(*filters)
        return nnx.Nothing
