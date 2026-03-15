# autoresearch (x-transformers-rl edition)

This is an experiment to have the LLM do its own research on reinforcement
learning using [x-transformers-rl](https://github.com/lucidrains/x-transformers-rl).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15-rl`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `AGENTS.md` — **machine-specific overrides** (Python path, GPU, etc.). Always read this first and follow its settings. This file is not committed — it is customized per machine.
   - `train.py` — the file you modify. World-model config, PPO hyperparameters, training loop.
   - `docs/adjustable_params.md` — x-transformers-rl parameter reference.
   - `docs/design.md` — design decisions and parameter search space.
   - `x-transformers-rl/x_transformers_rl/x_transformers_rl.py` — the core RL library (reference, do not modify).
   - `x-transformers-rl/train_lander.py` — reference LunarLander training script with annotations.
4. **Verify dependencies**: Run `python -c "from x_transformers_rl import Learner; print('OK')"` and `python -c "import gymnasium; print('OK')"`. If missing deps, install:
   ```bash
   pip install gymnasium hl-gauss-pytorch assoc-scan x-mlps-pytorch accelerate adam-atan2-pytorch ema-pytorch einx einops
   ```
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (300 seconds wall clock training time). The script runs as many learning updates as fit in the time budget, then does a final greedy evaluation.

```bash
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: world-model architecture, PPO hyperparameters, agent config, training loop, etc.

**What you CANNOT do:**
- Modify files inside `x-transformers-rl/` or `x-transformers/`. The libraries are read-only reference.
- Break the output format (the `---` summary block at the end must remain parseable).

**The goal: get the highest mean_reward on CartPole-v1 in 5 minutes.** CartPole-v1 is "solved" at 475+ mean reward over 100 consecutive episodes. The baseline gets ~120-140 mean_reward; there is substantial room for improvement.

**Variance**: RL training is noisier than supervised learning. A result should only be kept if it improves by **more than ~10 reward units** over the current best, to avoid chasing noise.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
mean_reward:       139.8333
best_reward:       500.0000
num_updates:       33
total_episodes:    825
training_seconds:  304.7
total_seconds:     310.2
peak_vram_mb:      118.4
num_params_M:      0.2647
hidden_dim:        64
world_model_depth: 2
```

You can extract key metrics from the log file:

```
grep "^mean_reward:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	mean_reward	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. mean_reward (e.g. 139.8333) — use 0.0000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	mean_reward	memory_gb	status	description
a1b2c3d	139.8333	0.1	keep	baseline (dim=64 depth=2 heads=4 5min)
b2c3d4e	200.5000	0.1	keep	depth=3 epochs=5
c3d4e5f	120.1000	0.1	discard	lr=1e-3 (worse than baseline)
d4e5f6g	0.0000	0.0	crash	batch_size doesn't divide num_episodes_per_update
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15-rl`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^mean_reward:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If mean_reward improved by >10 units, you "advance" the branch, keeping the git commit
9. If mean_reward is not sufficiently better, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment takes ~5.5 minutes total (5 min training + eval). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read `docs/adjustable_params.md` and the x-transformers-rl source for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## x-transformers-rl Parameter Space

The x-transformers-rl library wraps a causal Transformer world model with PPO actor-critic heads. The main tunable components are:

### World Model (Decoder kwargs in `WORLD_MODEL` dict)

| Parameter | Type | Description | Default |
|---|---|---|---|
| `depth` | int | Number of transformer layers | 2 |
| `attn_gate_values` | bool | Gate attention values with a learned scalar | True |
| `add_value_residual` | bool | ResFormer value residual connections | True |
| `ff_relu_squared` | bool | ReLU^2 activation (Primer) | True |
| `learned_value_residual_mix` | bool | Learn mixing coefficient for value residuals | True |
| `attn_flash` | bool | Flash Attention (PyTorch SDP) | True |

### Worth Trying (ordered by expected impact)

| Parameter | Type | Description |
|---|---|---|
| `ff_glu` | bool | Gated Linear Unit in FFN (usually helps) |
| `ff_swish` | bool | SwiGLU activation (combine with ff_glu) |
| `ff_mult` | int | FFN expansion factor (default 4) |
| `use_rmsnorm` | bool | RMSNorm instead of LayerNorm |
| `attn_qk_norm` | bool | QK normalization for stability |
| `attn_kv_heads` | int | Grouped-query attention |
| `macaron` | bool | Macaron (sandwich FFN) structure |
| `gate_residual` | bool | Gated residual connections |
| `scale_residual` | bool | Scaled residual connections |

### Model Architecture (via `AGENT_KWARGS`)

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | 64 | Transformer residual-stream dimension. **Must be set explicitly** (Agent default is only 48) |
| `world_model_attn_dim_head` | 16 | Per-head Q/K/V dimension |
| `world_model_heads` | 4 | Number of attention heads |
| `world_model_attn_hybrid_gru` | False | Hybrid GRU + attention mechanism |

### PPO / Training Hyperparameters

| Constant | Default | Description |
|---|---|---|
| `LEARNING_RATE` | 8e-4 | Learning rate for AdoptAtan2 optimizer |
| `PPO_EPOCHS` | 3 | PPO gradient-step epochs per update |
| `BATCH_SIZE` | 5 | PPO mini-batch size (must divide `NUM_EPISODES_PER_UPDATE`) |
| `NUM_EPISODES_PER_UPDATE` | 25 | Episodes collected before each PPO update |
| `GAMMA` | 0.99 | Discount factor |
| `LAM` | 0.95 | GAE lambda (1.0=MC, 0.0=TD(0)) |
| `ENTROPY_WEIGHT` | 0.01 | Entropy bonus coefficient |
| `EPS_CLIP` | 0.2 | PPO clipping epsilon |

### Actor-Critic Head Config (via `actor_critic_world_model` dict)

| Parameter | Default | Description |
|---|---|---|
| `use_simple_policy_optimization` | False | SPO instead of PPO |
| `add_entropy_to_advantage` | False | Entropy-augmented advantages (Cheng et al.) |
| `frac_actor_head_gradient` | 0.05 | Fraction of actor gradient flowing into transformer backbone |
| `frac_critic_head_gradient` | 0.05 | Fraction of critic gradient flowing into transformer backbone |

### Key Constraints

- `BATCH_SIZE` must divide `NUM_EPISODES_PER_UPDATE`
- `REWARD_RANGE` must cover the expected discounted return range (CartPole: ~0 to 200)
- `HIDDEN_DIM` must be passed explicitly via `agent_kwargs` (Agent default is only 48)

## Experiment Ideas (Prioritized)

Start with baseline, then try roughly in this order:

### Quick Wins
1. **More PPO epochs**: `PPO_EPOCHS=5` (more gradient steps per update)
2. **Deeper world model**: `WORLD_MODEL_DEPTH=3` (more capacity)
3. **Larger hidden dim**: `HIDDEN_DIM=96` with `WORLD_MODEL_HEADS=6`
4. **More episodes per update**: `NUM_EPISODES_PER_UPDATE=50, BATCH_SIZE=10`
5. **Tune learning rate**: Try `LEARNING_RATE=3e-4` or `1.5e-3`
6. **Wider reward range**: `REWARD_RANGE=(0., 500.)` (covers full episode return)

### Architecture Changes
7. Add `ff_glu=True, ff_swish=True` in WORLD_MODEL (SwiGLU)
8. Try `world_model_attn_hybrid_gru=True` (GRU + attention hybrid)
9. Enable `add_entropy_to_advantage=True` (Cheng et al. entropy-augmented)
10. Enable `use_simple_policy_optimization=True` (SPO instead of PPO)
11. Tune `world_model_embed_linear_schedule` — try `None` (full WM from step 0)

### Exploration / Curiosity
12. Enable `curiosity_reward_weight=0.1` (world model uncertainty as bonus)
13. Tune `entropy_weight` (higher = more exploration)
14. Try `gamma=0.95` or `gamma=0.995`

### Advanced
15. Tune `frac_actor_head_gradient` / `frac_critic_head_gradient`
16. Try `actor_ff_depth=2` or `critic_ff_depth=3`
17. Enable `evolutionary=True` for population-based policy diversity
18. Try different model size trade-offs (wider+shallower vs narrower+deeper)
