# autoresearch (x-transformers-rl edition) — Machine-Specific Config

**This file is not committed.** It contains settings specific to this machine.
The agent should read this file first and follow these overrides.

## Hardware & Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 4090 eGPU (24 GB VRAM, SM89 Ada Lovelace) |
| GPU (internal) | NVIDIA GTX 1650 Max-Q (4 GB, display only — do NOT use for training) |
| Python | `/home/tim/miniforge3/envs/torch/bin/python` (`conda activate torch`) |
| PyTorch | 2.10.0+cu130 |
| x-transformers-rl | Local submodule at `./x-transformers-rl` |
| x-transformers | Local submodule at `./x-transformers` |
| Environment | CartPole-v1 (gymnasium) |
| Metric | **mean_reward** — higher is better. Solved = 475+ |

## Machine-Specific Overrides

- **Python path**: Always use `/home/tim/miniforge3/envs/torch/bin/python`.
- **GPU selection**: Set `CUDA_VISIBLE_DEVICES=0` to target the RTX 4090.
- **LD_LIBRARY_PATH**: Prepend the conda env lib to avoid GLIBCXX errors:
  ```
  LD_LIBRARY_PATH="/home/tim/miniforge3/envs/torch/lib:$LD_LIBRARY_PATH"
  ```

### Launch command template

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH="/home/tim/miniforge3/envs/torch/lib:$LD_LIBRARY_PATH" \
/home/tim/miniforge3/envs/torch/bin/python train_rl.py > run_rl.log 2>&1
```

## Setup

To set up a new RL experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15-rl`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `AGENTS_RL.md` — this file.
   - `train_rl.py` — the file you modify.
   - `docs/design.md` — design decisions and parameter search space reference.
   - `x-transformers-rl/x_transformers_rl/x_transformers_rl.py` — core RL library (reference, do not modify).
   - `x-transformers-rl/train_lander.py` — reference training script with annotations.
4. **Verify dependencies**:
   ```bash
   python -c "from x_transformers_rl import Learner; print('OK')"
   python -c "import gymnasium; print('OK')"
   ```
   If missing, install:
   ```bash
   pip install gymnasium hl-gauss-pytorch assoc-scan x-mlps-pytorch accelerate adam-atan2-pytorch ema-pytorch einx einops
   ```
5. **Initialize results_rl.tsv**: Create with just the header row.
6. **Confirm and go**.

## Running Experiments

Each experiment runs for a **fixed episode budget of 500 episodes** (wall clock
varies: ~30–120s depending on the architecture).

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH="/home/tim/miniforge3/envs/torch/lib:$LD_LIBRARY_PATH" \
/home/tim/miniforge3/envs/torch/bin/python train_rl.py > run_rl.log 2>&1
```

**What you CAN do:**
- Modify `train_rl.py` — this is the only file you edit. Everything is fair game:
  world-model architecture, PPO hyperparameters, agent config, etc.

**What you CANNOT do:**
- Modify files inside `x-transformers-rl/`. The library is read-only reference.
- Break the output format (the `---` summary block at the end must remain parseable).

**The goal: get the highest mean_reward on CartPole-v1 in 500 episodes.**
CartPole-v1 is "solved" at 475+ mean reward. A well-trained agent should reach
100–200+ within 500 episodes. 

**Variance**: RL training is noisier than LM training. A result should only be
kept if it improves by **more than ~5 reward units** over the baseline, to avoid
chasing noise. Consider running each config twice and averaging if marginal.

**Simplicity criterion**: Same as the LM autoresearch — simpler is better if
results are equal or better. Adding complexity for a 1-point reward improvement
is not worth it.

**The first run**: Always run `train_rl.py` as-is to establish the baseline.

## Output Format

```
---
mean_reward:       123.4500
best_reward:       456.0000
total_episodes:    500
training_seconds:  45.2
total_seconds:     46.1
peak_vram_mb:      1234.5
num_params_M:      0.1618
world_model_dim:   64
world_model_depth: 2
```

Extract key metrics:
```bash
grep "^mean_reward:\|^peak_vram_mb:" run_rl.log
```

## Logging Results

Log to `results_rl.tsv` (tab-separated, NOT comma-separated).

Header + 5 columns:

```
commit  mean_reward     memory_gb       status  description
```

1. git commit hash (short, 7 chars)
2. mean_reward (e.g. 123.4500) — use 0.0000 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description

Example:
```
commit	mean_reward	memory_gb	status	description
a1b2c3d	13.8500	0.0	keep	baseline (depth=2 heads=4 dim_head=16)
b2c3d4e	45.2000	0.0	keep	depth=4 epochs=5
c3d4e5f	12.1000	0.0	discard	lr=1e-3 (worse than baseline)
d4e5f6g	0.0000	0.0	crash	num_episodes_per_update not divisible by batch_size
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15-rl`).

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Tune `train_rl.py` with an experimental idea.
3. git commit
4. Run: `python train_rl.py > run_rl.log 2>&1`
5. Read results: `grep "^mean_reward:\|^peak_vram_mb:" run_rl.log`
6. If grep is empty — crashed. Run `tail -n 50 run_rl.log` for stack trace.
7. Record in `results_rl.tsv` (not committed).
8. If mean_reward improved by >5 units: keep the commit.
9. If not: `git reset --hard HEAD~1`.

**Timeout**: Each experiment should complete in ≤5 minutes. If it exceeds
10 minutes, kill and treat as failure (discard and revert).

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run
continuously until manually stopped.

## Experiment Ideas (Prioritized)

### Quick Wins
1. **More episodes**: Try `NUM_EPISODES=1000` (doubles training time but gives
   more learning signal)
2. **More PPO epochs**: `PPO_EPOCHS=5` or `PPO_EPOCHS=10`
3. **Deeper world model**: `WORLD_MODEL_DEPTH=4`
4. **Larger hidden dim**: `WORLD_MODEL_HEADS=8` (dim=8*16=128)
5. **More updates**: `NUM_EPISODES_PER_UPDATE=10` (more frequent PPO steps)
6. **Higher learning rate**: Try `LEARNING_RATE=1e-3` or `3e-3`

### Architecture Changes
7. Try `world_model_attn_hybrid_gru=True` (GRU + attention hybrid)
8. Enable `add_entropy_to_advantage=True` (Cheng et al. entropy-augmented)
9. Enable `use_simple_policy_optimization=True` (SPO instead of PPO)
10. Tune `world_model_embed_linear_schedule` — try `None` (full WM from step 0)

### Exploration / Curiosity
11. Enable `curiosity_reward_weight=0.1` (world model uncertainty as bonus)
12. Tune `entropy_weight` (higher = more exploration)
13. Try `gamma=0.95` (less discounting)

### Evolutionary
14. Enable `evolutionary=True` for population-based policy diversity
    (adds overhead; only try after baseline is solid)

## Key Constraints

- `NUM_EPISODES_PER_UPDATE` must divide `NUM_EPISODES`
- `BATCH_SIZE` must divide `NUM_EPISODES_PER_UPDATE`
- `WORLD_MODEL_DIM = WORLD_MODEL_HEADS * WORLD_MODEL_DIM_HEAD`
  (this is a comment convention; the library computes dim as `heads * dim_head`)
