Autonomous LLM-driven research on reinforcement learning using my fork of [x-transformers-rl](https://github.com/TimS-ml/x-transformers-rl). A transformer world model learns to play LunarLander-v3 via PPO, and an AI agent iterates on the architecture and hyperparameters overnight using population-based multi-branch search.

# autoresearch (x-transformers-rl edition)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real RL training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better agent. This edition uses Phil Wang's [x-transformers-rl](https://github.com/lucidrains/x-transformers-rl) library — a causal Transformer world model with PPO actor-critic heads — and the same autonomous loop from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## How it works

The repo has a few key files:

- **`train.py`** — the single file the agent edits. Contains the x-transformers-rl Learner config (world model architecture, PPO hyperparameters, training loop). Everything is fair game. **This file is edited and iterated on by the agent**.
- **`program.md`** — instructions for the agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`AGENTS.md`** — machine-specific overrides (GPU, Python path, etc.). Not committed — customized per machine.
- **`docs/adjustable_params.md`** — comprehensive reference of all adjustable x-transformers-rl parameters.
- **`docs/design.md`** — design decisions, bug fixes, and parameter search space.
- **`x-transformers-rl/`** — the RL library (git submodule, read-only reference).
- **`x-transformers/`** — the base transformer library (git submodule, read-only reference).

Environment: **LunarLander-v3** (gymnasium). Metric: **mean_reward** — higher is better. Solved at 200+.

By design, training runs for a **fixed 5-minute time budget** (wall clock, hard-killed by `timeout 300`). This makes experiments directly comparable. The agent uses **population-based multi-branch search** (3 parallel git branches) to escape local optima.

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+. Any package manager works (uv / conda / mamba / pip).

```bash
# 1. Clone the repo
git clone --recursive https://github.com/TimS-ml/autoresearch-x-transformers-rl
cd autoresearch-x-transformers-rl

# 2. Install dependencies
pip install "gymnasium[box2d]" hl-gauss-pytorch assoc-scan x-mlps-pytorch \
    accelerate adam-atan2-pytorch ema-pytorch einx einops

# 3. Verify imports work
python -c "from x_transformers_rl import Learner; print('OK')"
python -c "import gymnasium; print('OK')"

# 4. Run a single training experiment (~5 min, hard-killed at 300s)
CUDA_VISIBLE_DEVICES=0 timeout 300 python train.py
```

## Running the agent

Spin up your Claude/Codex or whatever you want in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The `program.md` file is the "skill" that drives the autonomous agent. `AGENTS.md` provides machine-specific overrides.

## Project structure

```
train.py                    — Learner config, PPO hyperparameters, training loop (agent modifies this)
program.md                  — agent instructions (public)
AGENTS.md                   — machine-specific overrides (private, not committed)
analysis.py                 — experiment analysis script (jupytext percent format)
docs/
  adjustable_params.md      — x-transformers-rl parameter reference
  design.md                 — design decisions and bug log
x-transformers-rl/          — RL library (git submodule, read-only)
x-transformers/             — base transformer library (git submodule, read-only)
```

## Design choices

- **x-transformers-rl as the backbone.** Phil Wang's library wraps a causal Transformer world model with PPO actor-critic heads, distributional critic (HL-Gauss), curiosity rewards, evolutionary gene pools, and more — a huge search space for the agent.
- **LunarLander-v3.** 8-dim observations, 4 discrete actions, shaped rewards. Non-trivial enough to expose real architecture differences, fast enough for rapid iteration.
- **Fixed 5-minute time budget.** Training hard-killed by `timeout 300`. ~12 experiments/hour, ~100 overnight.
- **Population-based multi-branch search.** 3 parallel git branches explore different regions of the search space. Periodic migration (cherry-pick) of winning changes prevents getting stuck in local optima.
- **Seeded reproducibility.** Global seed (torch, numpy, random, gymnasium) for cross-run comparability.
- **Self-contained.** One GPU, one file, one metric. The submodules provide the model; everything else is standard PyTorch + gymnasium.

## Bug fixes in x-transformers-rl

Three bugs were found and fixed in the local fork (see `docs/design.md` §5 for details):

1. **SPO/PPO condition inversion** — `if not self.use_spo` had the SPO and PPO code blocks swapped.
2. **Non-scalar reward crash** — `float(reward)` fails on 1-element numpy arrays from some environments.
3. **Truncation bootstrap crash** — bootstrap memory for GAE was appended to the wrong list, crashing when episodes hit the time limit.

## Credits

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the original concept
- [Phil Wang's x-transformers-rl](https://github.com/lucidrains/x-transformers-rl) here is [my fork](https://github.com/TimS-ml/x-transformers-rl) — the RL library
- [Phil Wang's x-transformers](https://github.com/lucidrains/x-transformers) here is [my fork](https://github.com/TimS-ml/x-transformers) — the base transformer library

## License

MIT
