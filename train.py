"""
train_rl.py — autoresearch training script for x-transformers-rl.

This is the RL counterpart of train.py.  The agent modifies this file to
experiment with different world-model architectures, PPO hyperparameters, and
training configurations.

Environment: LunarLander-v3 (gymnasium)
  - Observation: 8-dim continuous state vector (x, y, vx, vy, angle, angular
    velocity, left leg contact, right leg contact).
  - Action: 4 discrete actions (do nothing, fire left, fire main, fire right).
  - Reward: shaped — hover/land near pad = positive, crash/fuel = negative.
  - Solved: mean episode reward >= 200 over 100 consecutive episodes.
  - Max episode length: 1000 steps.

Metric: mean_reward (higher is better), evaluated greedily after training.

Budget: Fixed **5-minute wall clock** training time (TIME_BUDGET = 300 seconds),
matching the LM autoresearch.  The script runs as many learning updates as fit
in the time budget, then does a final evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=... python train_rl.py

Output block (parseable, grepped by the experiment loop):
    ---
    mean_reward:       123.45
    best_reward:       456.78
    num_updates:       150
    total_episodes:    3750
    training_seconds:  300.1
    total_seconds:     305.2
    peak_vram_mb:      123.5
    num_params_M:      0.12
    hidden_dim:        64
    world_model_depth: 2
"""

import sys
import os
import time
import gc
import random

# ---------------------------------------------------------------------------
# Path setup — pick up the local submodules first
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "x-transformers-rl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "x-transformers"))

# ---------------------------------------------------------------------------
# Hyperparameters — everything the agent may tune
# ---------------------------------------------------------------------------

# --- Reproducibility ---
SEED = 42  # Global seed for torch, numpy, random, gymnasium

# --- Time budget ---
TIME_BUDGET = 300  # 5-minute wall clock training time (seconds)

# --- Environment ---
ENV_NAME = "LunarLander-v3"
MAX_TIMESTEPS = 1000  # Max steps per episode (LunarLander default)

# --- Critic value range ---
# LunarLander rewards: landing ~100-140, crash ~-100, fuel cost ~-50.
# With gamma=0.99, discounted returns roughly in (-200, 300).
# We clip observed rewards to [-5, 5] (following upstream train_lander.py)
# so the critic range covers the clipped return distribution.
REWARD_RANGE = (-5.0, 5.0)

# --- World model architecture ---
HIDDEN_DIM = 64  # Transformer residual-stream dimension
WORLD_MODEL_DEPTH = 4  # Number of transformer layers (deeper for LunarLander)
WORLD_MODEL_HEADS = 4  # Number of attention heads
WORLD_MODEL_DIM_HEAD = 16  # Per-head dimension

WORLD_MODEL = dict(
    depth=WORLD_MODEL_DEPTH,
    attn_gate_values=True,
    add_value_residual=True,
    ff_relu_squared=True,
    learned_value_residual_mix=True,
    attn_flash=True,
)

# --- Training ---
NUM_EPISODES_PER_UPDATE = 25  # Episodes collected before each PPO update
BATCH_SIZE = 5  # PPO mini-batch size (must divide NUM_EPISODES_PER_UPDATE)
PPO_EPOCHS = 3  # PPO epochs per update
LEARNING_RATE = 8e-4
BETAS = (0.9, 0.99)
GAMMA = 0.99  # Discount factor
LAM = 0.95  # GAE lambda
ENTROPY_WEIGHT = 0.01  # Entropy bonus coefficient
EPS_CLIP = 0.2  # PPO clipping epsilon
VALUE_CLIP = 0.4  # Critic value clipping
EMA_DECAY = 0.9  # EMA decay for behavior policy
REGEN_REG_RATE = 1e-4  # Regenerative regularization rate
CAUTIOUS_FACTOR = 0.1  # Cautious update factor

# --- World model schedule ---
# Ramp the world-model embedding contribution from 0 to 1 over steps 5-20.
WORLD_MODEL_EMBED_SCHEDULE = (5.0, 20.0)

# --- Agent extra kwargs ---
AGENT_KWARGS = dict(
    hidden_dim=HIDDEN_DIM,
    world_model_attn_dim_head=WORLD_MODEL_DIM_HEAD,
    world_model_heads=WORLD_MODEL_HEADS,
    world_model_attn_hybrid_gru=True,
    world_model_embed_linear_schedule=WORLD_MODEL_EMBED_SCHEDULE,
    actor_critic_world_model=dict(
        frac_critic_head_gradient=5e-2,
        frac_actor_head_gradient=5e-2,
        use_simple_policy_optimization=False,
        add_entropy_to_advantage=False,
    ),
)

# --- Evaluation ---
NUM_EVAL_EPISODES = 30  # Greedy eval episodes after training
EVAL_INTERVAL = 10  # Eval every N learning updates (0 = no intermediate eval)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

from x_transformers_rl import Learner

# ---------------------------------------------------------------------------
# Reproducibility: seed everything
# ---------------------------------------------------------------------------


def seed_everything(seed: int):
    """Seed Python, NumPy, PyTorch, and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slight perf cost, better reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helper: count model parameters
# ---------------------------------------------------------------------------


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Evaluation: run greedy rollouts, return mean episode reward
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(agent, env_name, num_eval_episodes=30, max_timesteps=1000, seed=None):
    """Run the agent greedily and return (mean_reward, best_reward)."""
    agent.eval()
    rewards = []
    eval_env = gym.make(env_name)

    for ep in range(num_eval_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        obs, _ = eval_env.reset(seed=ep_seed)
        hiddens = None
        episode_reward = 0.0
        for _ in range(max_timesteps):
            raw_actions, hiddens = agent(obs, hiddens=hiddens)
            # Greedy action selection (works for any discrete action count)
            action = raw_actions.argmax().item()
            obs, reward, terminated, truncated, *_ = eval_env.step(action)
            episode_reward += float(reward)
            if terminated or truncated:
                break
        rewards.append(episode_reward)

    eval_env.close()
    return float(np.mean(rewards)), float(np.max(rewards))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    t_start_total = time.time()

    # -- Seed everything for reproducibility --
    seed_everything(SEED)

    # -- Environment --
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]  # 8 for LunarLander
    num_actions = env.action_space.n  # 4 for LunarLander

    # -- Learner (world model + actor + critic + PPO) --
    learner = Learner(
        state_dim=state_dim,
        num_actions=num_actions,
        reward_range=REWARD_RANGE,
        max_timesteps=MAX_TIMESTEPS,
        batch_size=BATCH_SIZE,
        num_episodes_per_update=NUM_EPISODES_PER_UPDATE,
        world_model=WORLD_MODEL,
        lr=LEARNING_RATE,
        betas=BETAS,
        gamma=GAMMA,
        lam=LAM,
        entropy_weight=ENTROPY_WEIGHT,
        eps_clip=EPS_CLIP,
        value_clip=VALUE_CLIP,
        ema_decay=EMA_DECAY,
        regen_reg_rate=REGEN_REG_RATE,
        cautious_factor=CAUTIOUS_FACTOR,
        epochs=PPO_EPOCHS,
        agent_kwargs=AGENT_KWARGS,
    )

    agent = learner.agent
    num_params = count_params(agent.model)
    print(f"Model parameters: {num_params / 1e6:.4f}M")

    # -- Training with time budget --
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Freeze GC for speed after first update
    gc_frozen = False

    t_start_train = time.time()
    num_updates = 0
    total_episodes = 0
    best_eval_reward = -float("inf")

    while True:
        elapsed = time.time() - t_start_train
        if elapsed >= TIME_BUDGET:
            break

        # Run one learning update (collects NUM_EPISODES_PER_UPDATE episodes,
        # then does PPO_EPOCHS of gradient steps).
        learner(env, 1)
        num_updates += 1
        total_episodes += NUM_EPISODES_PER_UPDATE

        # Freeze GC after first update for speed
        if not gc_frozen:
            gc.disable()
            gc_frozen = True

        # Intermediate evaluation
        if EVAL_INTERVAL > 0 and num_updates % EVAL_INTERVAL == 0:
            mean_r, best_r = evaluate(
                agent,
                ENV_NAME,
                num_eval_episodes=10,
                max_timesteps=MAX_TIMESTEPS,
                seed=SEED + 10000 + num_updates,
            )
            elapsed = time.time() - t_start_train
            print(
                f"[update {num_updates:4d} | {elapsed:5.0f}s] mean_reward={mean_r:.1f}  best={best_r:.0f}  episodes={total_episodes}"
            )
            if mean_r > best_eval_reward:
                best_eval_reward = mean_r

    t_end_train = time.time()
    env.close()

    # Re-enable GC
    if gc_frozen:
        gc.enable()

    training_seconds = t_end_train - t_start_train

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # -- Final evaluation (greedy policy, no exploration noise) --
    mean_reward, best_reward = evaluate(
        agent,
        ENV_NAME,
        num_eval_episodes=NUM_EVAL_EPISODES,
        max_timesteps=MAX_TIMESTEPS,
        seed=SEED + 20000,
    )

    total_seconds = time.time() - t_start_total

    # -- Summary block (parseable by grep) --
    print()
    print("training complete")
    print()
    print("---")
    print(f"mean_reward:       {mean_reward:.4f}")
    print(f"best_reward:       {best_reward:.4f}")
    print(f"num_updates:       {num_updates}")
    print(f"total_episodes:    {total_episodes}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_params_M:      {num_params / 1e6:.4f}")
    print(f"hidden_dim:        {HIDDEN_DIM}")
    print(f"world_model_depth: {WORLD_MODEL_DEPTH}")


if __name__ == "__main__":
    main()
