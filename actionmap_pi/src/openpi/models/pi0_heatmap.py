"""Pi0.5 model with configurable action head: single-pass heatmap, flow+heatmap, or vanilla flow matching.

Three modes controlled by ``action_head_mode``:
  - ``single_pass``:   Learned query tokens, single forward pass, heatmap CE loss.
  - ``flow_heatmap``:  Flow matching input pathway (noisy actions x_t), heatmap predicts x0
                        at training and at each denoising ODE step.
  - ``flow_matching``:  Vanilla Pi0.5 flow matching (no heatmap head, pure MSE velocity loss).
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models.heatmap_head import HeatmapConfig, HeatmapHead
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class Pi0Heatmap(_model.BaseModel):
    """Pi0.5 backbone with configurable action head.

    Shares the PaliGemma backbone + adaRMS time conditioning across all modes.
    The ``action_head_mode`` config field selects which output head and training
    objective are used.
    """

    def __init__(self, config, rngs: nnx.Rngs):
        from openpi.models import pi0_heatmap_config

        assert isinstance(config, pi0_heatmap_config.Pi0HeatmapConfig)

        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self._mode = config.action_head_mode

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # PaliGemma backbone (with adaRMS in action expert, same as Pi0.5)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Shared across all modes: action input projection + time MLPs for adaRMS
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # flow_matching mode needs the velocity output projection
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # single_pass mode needs learned action queries
        self.action_queries = nnx.Param(
            jax.random.normal(rngs.params(), (config.action_horizon, action_expert_config.width)) * 0.02
        )

        # Heatmap head (used by single_pass and flow_heatmap modes)
        heatmap_cfg = HeatmapConfig(
            input_dim=action_expert_config.width,
            hidden_dim=config.heatmap_hidden_dim,
            action_dim=7,
            trans_grid=config.heatmap_trans_grid,
            rot_grid=config.heatmap_rot_grid,
            trans_sigma=config.heatmap_trans_sigma,
            rot_sigma=config.heatmap_rot_sigma,
            loss_type=config.heatmap_loss_type,
            decode_mode=config.heatmap_decode_mode,
            decode_temperature=config.heatmap_decode_temperature,
            periodic_rot=config.heatmap_periodic_rot,
            grid_pad=config.heatmap_grid_pad,
        )
        self.heatmap_head = HeatmapHead(heatmap_cfg, rngs)

        self._action_expert_width = action_expert_config.width
        self.deterministic = True

    # ------------------------------------------------------------------
    # Prefix / suffix embedding (shared across modes)
    # ------------------------------------------------------------------

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Encode images and language into prefix tokens (same as Pi0.embed_prefix)."""
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _time_cond(self, timestep: jnp.ndarray) -> jnp.ndarray:
        """Compute adaRMS conditioning from a timestep vector (B,)."""
        time_emb = _pi0.posemb_sincos(timestep, self._action_expert_width, min_period=4e-3, max_period=4.0)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        return time_emb

    @at.typecheck
    def embed_suffix_flow(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        """Suffix for flow matching modes: project noisy actions + time conditioning."""
        action_tokens = self.action_in_proj(noisy_actions)
        adarms_cond = self._time_cond(timestep)
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_suffix_query(self, obs: _model.Observation) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"],
    ]:
        """Suffix for single_pass mode: learned queries + fixed-time conditioning."""
        batch_size = obs.state.shape[0]
        action_tokens = einops.repeat(self.action_queries.value, "s e -> b s e", b=batch_size)
        adarms_cond = self._time_cond(jnp.full((batch_size,), 0.5))
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, input_mask, ar_mask, adarms_cond

    def _backbone_forward(
        self,
        prefix_tokens: jnp.ndarray,
        prefix_mask: jnp.ndarray,
        prefix_ar_mask: jnp.ndarray,
        suffix_tokens: jnp.ndarray,
        suffix_mask: jnp.ndarray,
        suffix_ar_mask: jnp.ndarray,
        adarms_cond: jnp.ndarray,
    ) -> jnp.ndarray:
        """Run the full prefix+suffix through PaliGemma and return action-position hidden states."""
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out[:, -self.action_horizon :]

    def _pad_actions(self, actions_7: jnp.ndarray) -> jnp.ndarray:
        """Pad 7-dim actions to action_dim for pipeline compatibility."""
        if self.action_dim > 7:
            padding = jnp.zeros((*actions_7.shape[:-1], self.action_dim - 7))
            return jnp.concatenate([actions_7, padding], axis=-1)
        return actions_7

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        if self._mode == "single_pass":
            return self._loss_single_pass(rng, observation, actions, train=train)
        elif self._mode == "flow_heatmap":
            return self._loss_flow_heatmap(rng, observation, actions, train=train)
        else:  # flow_matching
            return self._loss_flow_matching(rng, observation, actions, train=train)

    def _loss_single_pass(self, rng, observation, actions, *, train):
        """Learned queries → single forward pass → heatmap CE loss on clean actions."""
        observation = _model.preprocess_observation(rng if train else None, observation, train=train)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_query(observation)
        hidden = self._backbone_forward(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
        )
        return self.heatmap_head.compute_loss(hidden, actions[..., :7])

    def _loss_flow_heatmap(self, rng, observation, actions, *, train):
        """Flow matching input (noisy x_t) → backbone → heatmap CE loss on clean actions (x0-prediction)."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        # Only add noise to the 7 supervised action dims; zero-pad the rest to avoid
        # unsupervised noisy channels polluting the hidden representations.
        noise_7 = jax.random.normal(noise_rng, (*batch_shape, self.action_horizon, 7))
        clean_7 = actions[..., :7]
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t_7 = time_expanded * noise_7 + (1 - time_expanded) * clean_7
        x_t = self._pad_actions(x_t_7)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_flow(observation, x_t, time)
        hidden = self._backbone_forward(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
        )
        # Heatmap predicts clean action x0 (not velocity), supervised with CE on ground truth
        return self.heatmap_head.compute_loss(hidden, actions[..., :7])

    def _loss_flow_matching(self, rng, observation, actions, *, train):
        """Vanilla Pi0.5 flow matching: predict velocity v_t, MSE loss."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_flow(observation, x_t, time)
        hidden = self._backbone_forward(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
        )
        v_t = self.action_out_proj(hidden)
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # ------------------------------------------------------------------
    # sample_actions
    # ------------------------------------------------------------------

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        **kwargs,
    ) -> _model.Actions:
        if self._mode == "single_pass":
            return self._sample_single_pass(rng, observation)
        elif self._mode == "flow_heatmap":
            return self._sample_flow_heatmap(rng, observation, num_steps=num_steps, noise=noise)
        else:  # flow_matching
            return self._sample_flow_matching(rng, observation, num_steps=num_steps, noise=noise)

    def _sample_single_pass(self, rng, observation):
        """Single forward pass with learned queries → heatmap decode."""
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_query(observation)
        hidden = self._backbone_forward(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
        )
        return self._pad_actions(self.heatmap_head.predict_action(hidden))

    def _sample_flow_heatmap(self, rng, observation, *, num_steps, noise):
        """Iterative denoising: at each step, heatmap predicts x̂₀, then update x_t toward x̂₀."""
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            # Only noise the 7 supervised dims, zero-pad the rest to match action_dim.
            noise_7 = jax.random.normal(rng, (batch_size, self.action_horizon, 7))
            noise = self._pad_actions(noise_7)

        # Fill KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_flow(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            full_prefix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([full_prefix_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            hidden = suffix_out[:, -self.action_horizon :]

            # Heatmap predicts x̂₀ (clean action) from hidden states
            x0_pred_7 = self.heatmap_head.predict_action(hidden)  # (B, T, 7)
            x0_pred = self._pad_actions(x0_pred_7)

            # WARNING: ODE update: x_t = t*noise + (1-t)*x₀  =>  v_t = dx/dt = noise - x₀
            # From x₀ prediction: noise = (x_t - (1-t)*x₀)/t  =>  v_t = (x_t - x₀)/t
            # Then x_{t+dt} = x_t + dt * v_t
            v_t = (x_t - x0_pred) / jnp.maximum(time, 1e-6)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _sample_flow_matching(self, rng, observation, *, num_steps, noise):
        """Vanilla Pi0.5 iterative denoising with velocity output (no heatmap)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Fill KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_flow(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            full_prefix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([full_prefix_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
