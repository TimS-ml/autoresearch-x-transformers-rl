# autoresearch-x-transformers-rl — Design Document

This document records the architectural and implementation decisions made when
extending the base `autoresearch-x-transformers` repository into the RL edition.

---

## 1. Scope and Goals

The base repo is an autonomous experiment loop for **character-level language
modelling** (enwik8, BPC metric, 5-minute training budget).  This extension adds
a parallel loop for **reinforcement learning** using Phil Wang's
[x-transformers-rl](https://github.com/lucidrains/x-transformers-rl) library.

Key goals:

- Keep the same autoresearch pattern: one editable file (`train_rl.py`), fixed
  time budget, TSV logging, git-branch-per-run.
- Use `x-transformers-rl` as a local submodule (read-only, like `x-transformers`).
- Choose a benchmark environment that is:
  - Fast enough to collect many episodes inside the time budget.
  - Hard enough to show meaningful improvement across experiments.
  - GPU-friendly (runs in Python, no display needed).
- Expose the same kind of structured output block (`---`) so results can be
  grepped and logged automatically.

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

## 3. Training Script Design (`train_rl.py`)

The script mirrors `train.py` in structure:

- Single file, all configuration at the top as named constants.
- Fixed **episode budget** (not time budget) for comparability: 500 episodes
  per run (~30 seconds on CartPole), configurable via `NUM_EPISODES`.
- Output block at the end:
  ```
  ---
  mean_reward:      123.45
  best_reward:      456.78
  total_episodes:   500
  total_seconds:    32.1
  peak_vram_mb:     1234.5
  num_params_M:     0.12
  world_model_dim:  48
  world_model_depth: 1
  ```
- Logs to `results_rl.tsv` (separate from the LM results).

**Why episode budget instead of time budget?**

For RL the training time varies with:
- Episode length (CartPole can end at step 1 or step 500).
- Number of PPO epochs.
- Model size.

A fixed episode count gives more stable comparisons.  We use 500 episodes
which completes in ~20-60s on CartPole with a small model on the 4090.

---

## 4. World Model Architecture Defaults

Starting configuration for `train_rl.py`:

```python
world_model = dict(
    depth = 2,
    attn_gate_values = True,
    add_value_residual = True,
    ff_relu_squared = True,
    learned_value_residual_mix = True,
    attn_flash = True,
)
agent_kwargs = dict(
    world_model_attn_dim_head = 16,
    world_model_heads = 4,          # hidden_dim = 4 * 16 = 64
    world_model_embed_linear_schedule = (5., 20.),
)
```

This is taken directly from `train_lander.py` with `depth` reduced from 4 to 2
for speed on a trivial environment.  The hidden dim of 64 gives ~100K parameters
total — small enough to train quickly, large enough to learn CartPole.

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

---

## 6. Autoresearch Protocol for RL

See `AGENTS_RL.md` for the full machine-specific protocol.

Key differences from the LM autoresearch:

| | LM (`train.py`) | RL (`train_rl.py`) |
|---|---|---|
| Metric | val_bpc (lower = better) | mean_reward (higher = better) |
| Budget | 5-minute wall clock | 500 episodes |
| Environment | enwik8 (fixed dataset) | CartPole-v1 (stochastic env) |
| Variance | Low (deterministic data) | High (stochastic rollouts) |
| Editable file | `train.py` | `train_rl.py` |
| Results log | `results.tsv` | `results_rl.tsv` |

**Variance handling:** RL experiments have higher variance than LM experiments.
A result should only be kept if it improves by more than ~5 reward units over
the baseline, to avoid chasing noise.

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
| `batch_size` | Mini-batch size | 8 | 4, 16, 32 |
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
├── train.py             — LM autoresearch (existing, unchanged)
├── train_rl.py          — RL autoresearch (new, agent edits this)
├── results.tsv          — LM experiment log (untracked)
├── results_rl.tsv       — RL experiment log (untracked)
├── AGENTS.md            — LM machine-specific protocol
├── AGENTS_RL.md         — RL machine-specific protocol (new)
├── docs/
│   ├── design.md        — this file
│   ├── adjustable_params_basemodel.md
│   └── adjustable_params_basemodel_v2.md
├── x-transformers/      — LM library submodule (read-only)
└── x-transformers-rl/   — RL library submodule (read-only)
    ├── x_transformers_rl/
    │   ├── x_transformers_rl.py   — core (fixed 2 bugs, see §5)
    │   ├── evolution.py           — evolutionary gene pool
    │   └── distributed.py        — DDP utilities
    └── train_lander.py            — reference LunarLander example
```
