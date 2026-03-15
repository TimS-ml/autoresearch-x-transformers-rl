# autoresearch-x-transformers-rl — Design Document

This document records the architectural and implementation decisions for
autonomous RL experimentation using Phil Wang's
[x-transformers-rl](https://github.com/lucidrains/x-transformers-rl) library,
following the [autoresearch](https://github.com/karpathy/autoresearch) pattern.

---

## 1. Scope and Goals

An autonomous experiment loop for **reinforcement learning** on classic control
tasks. An AI agent modifies `train.py`, trains for 5 minutes, checks if the
result improved, keeps or discards, and repeats.

Key goals:

- One editable file (`train.py`), fixed 5-minute time budget, TSV logging,
  git-branch-per-run.
- `x-transformers-rl` as a local submodule (read-only).
- Benchmark environment that is fast, non-trivial, and GPU-friendly.
- Structured output block (`---`) for automated result extraction.

---

## 2. Benchmark Environment Choice

**Decision: `CartPole-v1` (gymnasium)**

Rationale:

| Option | Pros | Cons |
|---|---|---|
| CartPole-v1 | 4-dim obs, 2-action, well-known baseline (500 = solved), fast | Very easy, less room to demonstrate RL improvements |
| LunarLander-v3 | 8-dim obs, 4-action, non-trivial, established baselines | Slow Box2D physics, ~30s/episode at rollout depth 500 |
| MountainCar-v0 | Sparse reward, classic exploration challenge | Sparse reward makes early training signal near-zero |
| Pendulum-v1 | Continuous action, good gradient signal | Continuous action space adds complexity to first baseline |

CartPole-v1 was chosen as the **primary benchmark** because:
1. It is instant to install (no Box2D), runs at thousands of steps/second on CPU.
2. The solved score of 500 is well-defined, making "better" unambiguous.
3. 4-dimensional observations are trivial for the world model — experiments
   can isolate the RL algorithm differences rather than model capacity.

LunarLander-v3 is available as a secondary target for experiments after a solid
CartPole baseline exists.

**Metric: `mean_episode_reward`** over the last N episodes in each training run
(higher is better).  Reported alongside `peak_vram_mb` and `total_seconds`.

---

## 3. Training Script Design (`train.py`)

- Single file, all configuration at the top as named constants.
- Fixed **5-minute wall clock** training time (`TIME_BUDGET = 300`).
  The script runs as many `Learner` updates as fit in the time budget,
  calling `learner(env, 1)` in a loop.
- Intermediate evaluations are printed every `EVAL_INTERVAL` updates.
- Output block at the end:
  ```
  ---
  mean_reward:       139.8333
  best_reward:       500.0000
  num_updates:       32
  total_episodes:    800
  training_seconds:  304.5
  total_seconds:     305.2
  peak_vram_mb:      96.6
  num_params_M:      0.2647
  hidden_dim:        64
  world_model_depth: 2
  ```
- Logs to `results.tsv`.

The number of updates varies with model size and episode length (CartPole
episodes can be 10–500 steps), which means more-capable agents actually
collect fewer episodes but each episode is longer.  ~30–40 updates in
5 minutes is typical.

---

## 4. World Model Architecture Defaults

Starting configuration for `train.py`:

```python
HIDDEN_DIM         = 64
WORLD_MODEL_DEPTH  = 2
WORLD_MODEL_HEADS  = 4
WORLD_MODEL_DIM_HEAD = 16
REWARD_RANGE       = (0., 250.)

WORLD_MODEL = dict(
    depth = 2,
    attn_gate_values = True,
    add_value_residual = True,
    ff_relu_squared = True,
    learned_value_residual_mix = True,
    attn_flash = True,
)
AGENT_KWARGS = dict(
    hidden_dim = 64,                # must be set explicitly (Agent default is 48!)
    world_model_attn_dim_head = 16,
    world_model_heads = 4,
    world_model_embed_linear_schedule = (5., 20.),
)
```

**Key fix**: `hidden_dim` must be set explicitly via `agent_kwargs`.  The
`Agent` default is only 48, but we want 64 (= 4 heads × 16 dim_head).

**Key fix**: `REWARD_RANGE` must cover the expected discounted return range.
CartPole gives +1/step with gamma=0.99 and max 500 steps, so returns are
in roughly `(0, 200)`.  We use `(0, 250)`.  The previous default of `(-1, 1)`
broke the HL-Gauss distributional critic and prevented learning.

Architecture: ~265K parameters, enough to learn CartPole but small enough
for fast training (~2.5s per update with 25 episodes).

---

## 5. Bugs Found and Fixed in x-transformers-rl

### 5.1 `compute_actor_loss` — SPO/PPO condition inversion

**File:** `x-transformers-rl/x_transformers_rl/x_transformers_rl.py`, line ~1212

**Bug:** The `if not self.use_spo:` branch contained the SPO formula and the
`else:` branch contained PPO.  This is backwards — when `use_spo=True`, PPO
was executed, and when `use_spo=False`, SPO was executed.

**Fix:** Changed `if not self.use_spo:` to `if self.use_spo:`.

```python
# Before (wrong):
if not self.use_spo:
    # SPO formula ...
else:
    # PPO formula ...

# After (correct):
if self.use_spo:
    # SPO formula ...
else:
    # PPO formula ...
```

### 5.2 `Learner.forward` — reward not robust to non-scalar numpy arrays

**File:** `x-transformers-rl/x_transformers_rl/x_transformers_rl.py`, line ~3035

**Bug:** `float(reward)` fails when the environment returns a 1-element numpy
array (e.g. `np.random.randn(1)`).  This is a valid return type per the
gymnasium API.

**Fix:** `float(np.asarray(reward).flat[0])` — works for 0-d arrays, scalars,
and 1-element arrays.

### 5.3 `Learner.forward` — truncation bootstrap memory appended to wrong list

**File:** `x-transformers-rl/x_transformers_rl/x_transformers_rl.py`, line ~3090

**Bug:** When an episode is **truncated** (time limit hit, not terminated), the
code creates a bootstrap Memory for GAE value estimation.  This memory was
appended to the outer `memories` list (which is a list of lists) instead of
to `one_episode_memories` (the inner list for the current episode).  This
caused `learn()` to crash with `TypeError: iteration over a 0-d tensor`
when trying to unpack the bare Memory tuple as if it were a list of Memories.

The bug only triggers when the agent gets good enough that episodes reach
the maximum timestep (500 for CartPole), causing truncation.

**Fix:** Changed `memories.append(bootstrap_value_memory)` to
`one_episode_memories.append(bootstrap_value_memory)`.

---

## 6. Autoresearch Protocol

See `AGENTS.md` for the full machine-specific protocol.

| | This repo |
|---|---|
| Metric | mean_reward (higher = better) |
| Budget | 5-minute wall clock |
| Environment | CartPole-v1 (stochastic env) |
| Editable file | `train.py` |
| Results log | `results.tsv` |

**Variance handling:** RL experiments have high variance.
A result should only be kept if it improves by more than ~10 reward units over
the current best, to avoid chasing noise.

**Baseline results** (5-minute runs on RTX 4090):
- Baseline config: mean_reward ≈ 120–140 (high variance)
- CartPole-v1 is "solved" at mean_reward ≥ 475
- Plenty of room for improvement via architecture, hyperparameters, and training config

---

## 7. Parameter Search Space for RL Experiments

The `WorldModelActorCritic` / `Decoder` configuration is the primary search
space.  Most-impactful parameters (ordered by expected impact):

### World Model Architecture

| Parameter | Description | Default | Try |
|---|---|---|---|
| `depth` | Transformer layers | 2 | 1, 3, 4 |
| `attn_gate_values` | Gate attention values | True | False |
| `add_value_residual` | ResFormer value residual | True | False |
| `ff_relu_squared` | ReLU² activation | True | `ff_glu=True, ff_swish=True` |
| `learned_value_residual_mix` | Learned mix coefficient | True | False |
| `attn_flash` | Flash attention | True | — |
| `world_model_heads` | Attention heads | 4 | 2, 8 |
| `world_model_attn_dim_head` | Head dimension | 16 | 32, 64 |

### Agent / Training

| Parameter | Description | Default | Try |
|---|---|---|---|
| `lr` | Learning rate | 8e-4 | 3e-4, 1e-3, 3e-3 |
| `epochs` | PPO epochs per update | 3 | 1, 5, 10 |
| `batch_size` | Mini-batch size | 5 | 10, 25 |
| `gamma` | Discount factor | 0.99 | 0.95, 0.999 |
| `lam` | GAE lambda | 0.95 | 0.9, 0.98 |
| `entropy_weight` | Exploration bonus | 0.01 | 0.001, 0.1 |
| `use_spo` | SPO vs PPO | False | True |
| `add_entropy_to_advantage` | Entropy-augmented advantages | False | True |
| `world_model_embed_linear_schedule` | WM warmup schedule | (5, 20) | None, (10, 50) |

### Advanced

| Parameter | Description |
|---|---|
| `world_model_attn_hybrid_gru` | Hybrid GRU+attention for recurrence |
| `curiosity_reward_weight` | Curiosity bonus from world model uncertainty |
| `evolutionary` | Evolutionary gene pool (population diversity) |
| `num_episodes_per_update` | Rollouts per PPO update (affects variance) |

---

## 8. File Layout

```
autoresearch-x-transformers-rl/
├── train.py             — RL autoresearch (agent edits this)
├── results.tsv          — experiment log (untracked)
├── AGENTS.md            — machine-specific protocol (not committed)
├── docs/
│   ├── design.md        — this file
│   └── adjustable_params.md — x-transformers-rl parameter reference
├── x-transformers/      — x-transformers library submodule (read-only)
└── x-transformers-rl/   — RL library submodule (read-only)
    ├── x_transformers_rl/
    │   ├── x_transformers_rl.py   — core (fixed 3 bugs, see §5)
    │   ├── evolution.py           — evolutionary gene pool
    │   └── distributed.py        — DDP utilities
    └── train_lander.py            — reference LunarLander example
```
