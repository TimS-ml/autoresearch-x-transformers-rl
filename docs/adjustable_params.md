# x-transformers-rl Adjustable Parameters

Complete reference of all tunable parameters for the autoresearch RL agent.
Organized by where they appear in `train.py`.

---

## 1. Top-Level Training Config (`train.py` constants)

These are the constants at the top of `train.py` that the agent directly edits.

### Time Budget

| Constant | Default | Description |
|---|---|---|
| `TIME_BUDGET` | `300` | Wall-clock training time in seconds (5 minutes) |

### Reproducibility

| Constant | Default | Description |
|---|---|---|
| `SEED` | `42` | Global seed for Python random, NumPy, PyTorch, CUDA, and gymnasium eval envs |

### Environment

| Constant | Default | Description |
|---|---|---|
| `ENV_NAME` | `'LunarLander-v3'` | Gymnasium environment name |
| `MAX_TIMESTEPS` | `1000` | Max steps per episode (also positional encoding limit) |

### Critic Value Range

| Constant | Default | Description |
|---|---|---|
| `REWARD_RANGE` | `(-5., 5.)` | Min/max for the HL-Gauss distributional critic bins. Following upstream `train_lander.py`, rewards are clipped to [-5, 5] |

**Critical**: Setting this wrong breaks the critic. Too narrow clips value estimates; too wide wastes bins on unused ranges.

### World Model Architecture

| Constant | Default | Description |
|---|---|---|
| `HIDDEN_DIM` | `64` | Transformer residual-stream dimension (the `dim` passed to Decoder) |
| `WORLD_MODEL_DEPTH` | `4` | Number of transformer layers |
| `WORLD_MODEL_HEADS` | `4` | Number of attention heads |
| `WORLD_MODEL_DIM_HEAD` | `16` | Per-head dimension for Q/K/V projections |

**Note**: `HIDDEN_DIM` and `HEADS * DIM_HEAD` are independent. The attention projects from `HIDDEN_DIM` → `HEADS * DIM_HEAD` → back to `HIDDEN_DIM`. Default: 64 → 64 → 64 (no mismatch).

### Training Hyperparameters

| Constant | Default | Description |
|---|---|---|
| `NUM_EPISODES_PER_UPDATE` | `25` | Episodes collected before each PPO update. Must divide cleanly |
| `BATCH_SIZE` | `5` | PPO mini-batch size. Must divide `NUM_EPISODES_PER_UPDATE` |
| `PPO_EPOCHS` | `3` | Number of PPO gradient-step epochs per update |
| `LEARNING_RATE` | `8e-4` | Learning rate for AdoptAtan2 optimizer |
| `BETAS` | `(0.9, 0.99)` | Adam-style beta coefficients |
| `GAMMA` | `0.99` | Discount factor for future rewards |
| `LAM` | `0.95` | GAE lambda (1.0 = Monte Carlo, 0.0 = one-step TD) |
| `ENTROPY_WEIGHT` | `0.01` | Entropy bonus coefficient (higher = more exploration) |
| `EPS_CLIP` | `0.2` | PPO clipping epsilon for the surrogate objective |
| `VALUE_CLIP` | `0.4` | Clipping radius for value function updates |
| `EMA_DECAY` | `0.9` | EMA decay for the behavior policy (higher = slower update) |
| `REGEN_REG_RATE` | `1e-4` | Regenerative regularization for AdoptAtan2 |
| `CAUTIOUS_FACTOR` | `0.1` | Cautious update factor for AdoptAtan2 |

### World Model Schedule

| Constant | Default | Description |
|---|---|---|
| `WORLD_MODEL_EMBED_SCHEDULE` | `(5., 20.)` | Linear ramp: world model embedding contribution goes from 0 → 1 over training steps 5-20. `None` = always 1.0 |

---

## 2. Decoder (World Model) Dict Parameters

These go into `WORLD_MODEL = dict(...)` and are passed to `x_transformers.Decoder`.

### Currently Used

| Parameter | Default | Description |
|---|---|---|
| `depth` | `4` | Number of transformer layers |
| `attn_gate_values` | `True` | Gate attention values with a learned scalar |
| `add_value_residual` | `True` | ResFormer value residual connections |
| `ff_relu_squared` | `True` | ReLU² activation (Primer paper) |
| `learned_value_residual_mix` | `True` | Learn mixing coefficient for value residuals |
| `attn_flash` | `True` | Use Flash Attention (PyTorch SDP) |

### Worth Trying

| Parameter | Type | Description |
|---|---|---|
| `ff_glu` | `bool` | Gated Linear Unit in FFN (usually helps) |
| `ff_swish` | `bool` | SwiGLU activation (combine with `ff_glu`) |
| `ff_mult` | `int` | FFN expansion factor (default 4) |
| `use_rmsnorm` | `bool` | RMSNorm instead of LayerNorm |
| `attn_qk_norm` | `bool` | QK normalization for stability |
| `attn_kv_heads` | `int` | Grouped-query attention (fewer KV heads = less memory) |
| `attn_num_mem_kv` | `int` | Persistent memory key-values |
| `macaron` | `bool` | Macaron (sandwich FFN) structure |
| `gate_residual` | `bool` | Gated residual connections |
| `scale_residual` | `bool` | Scaled residual connections |
| `sandwich_norm` | `bool` | Sandwich normalization |

### Hardcoded by Agent (not overridable via dict)

| Parameter | Value | Description |
|---|---|---|
| `dim` | `HIDDEN_DIM` | Set to Agent's `hidden_dim` parameter |
| `rotary_pos_emb` | `True` | Always enabled |
| `attn_dropout` | `0.0` | From Agent's `dropout` param |
| `ff_dropout` | `0.0` | From Agent's `dropout` param |
| `attn_dim_head` | `WORLD_MODEL_DIM_HEAD` | Per-head dimension |
| `heads` | `WORLD_MODEL_HEADS` | Number of attention heads |

---

## 3. Agent Extra Kwargs (`AGENT_KWARGS`)

These are passed through `agent_kwargs=dict(...)` in `train.py`.

### Model Architecture

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | `48` (Agent default) | Transformer hidden dimension. **Must be set explicitly to match HIDDEN_DIM** |
| `world_model_attn_dim_head` | `16` | Per-head attention dimension |
| `world_model_heads` | `4` | Number of attention heads |
| `world_model_attn_hybrid_gru` | `True` | Use hybrid GRU + attention mechanism |
| `dropout` | `0.0` | Dropout rate for attention and feedforward |

### Loss Weights

| Parameter | Default | Description |
|---|---|---|
| `actor_loss_weight` | `1.0` | Weight for actor (policy) loss |
| `critic_loss_weight` | `1.0` | Weight for critic (value) loss |
| `autoregressive_loss_weight` | `1.0` | Weight for world model + done prediction losses |

### Training Infrastructure

| Parameter | Default | Description |
|---|---|---|
| `max_grad_norm` | `0.5` | Maximum gradient norm for clipping |
| `critic_pred_num_bins` | `100` | Number of HL-Gauss bins for distributional critic |
| `world_model_embed_linear_schedule` | `None` | Schedule for world model embedding contribution |

### EMA Settings (via `ema_kwargs`)

| Parameter | Default | Description |
|---|---|---|
| `update_model_with_ema_every` | `1250` | Copy EMA weights back to online model every N steps |

---

## 4. Actor-Critic Head Parameters (`actor_critic_world_model`)

These go inside `AGENT_KWARGS['actor_critic_world_model'] = dict(...)` and control the `WorldModelActorCritic` module.

### Gradient Flow

| Parameter | Default | Description |
|---|---|---|
| `frac_actor_head_gradient` | `0.05` | Fraction [0,1] of actor gradient flowing into transformer backbone |
| `frac_critic_head_gradient` | `0.05` | Fraction [0,1] of critic gradient flowing into transformer backbone |

### Policy Objective

| Parameter | Default | Description |
|---|---|---|
| `use_simple_policy_optimization` | `False` | Use SPO (Xie et al.) instead of PPO. SPO uses quadratic penalty instead of clipping |
| `entropy_weight` | `0.02` | Entropy bonus in actor loss (note: separate from top-level ENTROPY_WEIGHT) |
| `eps_clip` | `0.2` | PPO clipping epsilon (also used as SPO trust-region radius) |
| `value_clip` | `0.4` | Critic value clipping |

### Advantage Normalization

| Parameter | Default | Description |
|---|---|---|
| `normalize_advantages` | `True` | Z-score normalize advantages |
| `distributed_normalize` | `True` | Sync advantage stats across processes |
| `norm_advantages_stats_momentum` | `0.25` | EMA momentum for running advantage stats (1.0 = batch only) |

### Entropy-Augmented Advantages (Cheng et al. 2025)

| Parameter | Default | Description |
|---|---|---|
| `add_entropy_to_advantage` | `False` | Add entropy bonus to advantages |
| `entropy_to_advantage_kappa` | `2.0` | Max entropy bonus relative to |A| |
| `entropy_to_advantage_scale` | `0.1` | Scale factor for entropy bonus |

### Head Architecture

| Parameter | Default | Description |
|---|---|---|
| `actor_ff_depth` | `1` | Depth of actor feedforward head |
| `critic_ff_depth` | `2` | Depth of critic feedforward head |
| `actor_use_norm_ff` | `False` | Use magnitude-preserving nFeedforwards for actor |
| `critic_use_norm_ff` | `False` | Use magnitude-preserving nFeedforwards for critic |

### World Model Prediction

| Parameter | Default | Description |
|---|---|---|
| `state_pred_num_bins` | `50` | Number of HL-Gauss bins for next-state prediction |
| `reward_dropout` | `0.5` | Probability of zeroing reward conditioning (robustness) |

---

## 5. Learner-Level Parameters

These are set directly in the `Learner()` constructor in `train.py`.

### Continuous Action Space (not used for CartPole)

| Parameter | Default | Description |
|---|---|---|
| `continuous_actions` | `False` | Whether action space is continuous |
| `discretize_continuous` | `True` | If continuous, discretize into bins |
| `continuous_discretized_bins` | `50` | Number of bins |
| `squash_continuous` | `True` | Apply tanh squashing |
| `continuous_actions_clamp` | `None` | (min, max) clamp range |

### Dropout (World Model Robustness)

| Parameter | Default | Description |
|---|---|---|
| `reward_dropout_prob` | `0.5` | Probability of dropping reward input per episode |
| `action_dropout_prob` | `0.5` | Probability of dropping previous-action input per episode |

### Curiosity

| Parameter | Default | Description |
|---|---|---|
| `curiosity_reward_weight` | `0.0` | Weight for world model uncertainty as exploration bonus (0 = disabled) |

### Evolutionary Training

| Parameter | Default | Description |
|---|---|---|
| `evolutionary` | `False` | Enable evolutionary gene pool |
| `evolve_every` | `10` | Evolve every N training steps |
| `evolve_after_step` | `20` | Start evolution after this many steps |

---

## 6. LatentGenePool Parameters (evolutionary mode only)

Passed via `latent_gene_pool=dict(...)` in `Learner`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim` | `int` | *required* | Gene vector dimensionality |
| `num_genes_per_island` | `int` | *required* | Sub-population size per island |
| `num_selected` | `int` | *required* | Genes surviving selection (2 ≤ n < genes_per_island) |
| `tournament_size` | `int` | *required* | Tournament selection group size |
| `num_elites` | `int` | `1` | Top genes exempt from mutation |
| `mutation_prob` | `float` | `0.25` | Probability of mutation for non-elites |
| `mutation_std_dev` | `float` | `0.1` | Gaussian mutation noise std |
| `num_islands` | `int` | `1` | Number of independent sub-populations |
| `migrate_genes_every` | `int` | `10` | Migration interval (in evolution steps) |
| `num_frac_migrate` | `float` | `0.1` | Fraction of genes that migrate |
| `fitness_var_threshold` | `float` | `0.0` | Skip evolution when fitness variance ≤ threshold |

---

## 7. Parameter Constraints

- `BATCH_SIZE` must divide `NUM_EPISODES_PER_UPDATE`
- `REWARD_RANGE` must cover the clipped reward range (LunarLander: -5 to 5)
- `HIDDEN_DIM` must be passed explicitly via `agent_kwargs` (Agent default is only 48)
- When using `ff_glu=True`, the effective FFN width is halved (compensate with `ff_mult=8`)
- `attn_flash=True` requires compatible attention configuration (no custom attention patterns)

## 8. Experiment Priority (for autoresearch agent)

### High Impact (try first)

1. **`WORLD_MODEL_DEPTH`**: 4 → 6 or 8 (more capacity)
2. **`NUM_EPISODES_PER_UPDATE`**: 25 → 50 (lower variance gradients)
3. **`HIDDEN_DIM`**: 64 → 96 or 128 (more model capacity)
4. **`PPO_EPOCHS`**: 3 → 5 (more gradient steps per update)
5. **`LEARNING_RATE`**: 8e-4 → 3e-4 or 1.5e-3

### Medium Impact

6. **`ff_glu=True, ff_swish=True`** in WORLD_MODEL (SwiGLU)
7. **`GAMMA`**: 0.99 → 0.995 or 0.98
8. **`ENTROPY_WEIGHT`**: 0.01 → 0.001 or 0.05
9. **`world_model_attn_hybrid_gru=False`** (disable GRU, pure attention — it's on by default now)
10. **`add_entropy_to_advantage=True`** (Cheng et al.)

### Lower Impact / Experimental

11. **`use_simple_policy_optimization=True`** (SPO instead of PPO)
12. **`curiosity_reward_weight=0.1`** (exploration bonus)
13. **`WORLD_MODEL_EMBED_SCHEDULE`**: try `None` or `(10., 50.)`
14. **`frac_actor_head_gradient`**: 0.05 → 0.1 or 0.01
15. **`actor_ff_depth`**: 1 → 2
