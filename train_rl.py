"""
train_rl.py — autoresearch training script for x-transformers-rl.

This is the RL counterpart of train.py.  The agent modifies this file to
experiment with different world-model architectures, PPO hyperparameters, and
training configurations.

Environment: CartPole-v1 (gymnasium)
  - Observation: 4-dim continuous state vector.
  - Action: 2 discrete actions (push left / push right).
  - Reward: +1 per timestep the pole stays upright.
  - Solved: mean episode reward >= 475 over 100 consecutive episodes.

Metric: mean_reward (higher is better).

Budget: NUM_EPISODES episodes total.  Each episode can last up to 500 steps
(gymnasium default).  At a small model size this takes ~20-120 seconds on a
4090, depending on NUM_EPISODES and the architecture.

Usage:
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=... python train_rl.py

Output block (parseable, grepped by the experiment loop):
    ---
    mean_reward:       123.45
    best_reward:       456.78
    total_episodes:    500
    total_seconds:     32.1
    peak_vram_mb:      1234.5
    num_params_M:      0.12
    world_model_dim:   64
    world_model_depth: 2
"""

import sys
import os
import time
import math

# ---------------------------------------------------------------------------
# Path setup — pick up the local x-transformers-rl submodule first
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "x-transformers-rl"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "x-transformers"))

# ---------------------------------------------------------------------------
# Hyperparameters — everything the agent may tune
# ---------------------------------------------------------------------------

# --- Environment ---
ENV_NAME = "CartPole-v1"
NUM_EPISODES = 500  # Total episodes in one run
MAX_TIMESTEPS = 500  # Max steps per episode (gymnasium default)
REWARD_RANGE = (
    -1.0,
    1.0,
)  # HL-Gauss critic value range; CartPole rewards are +1 per step
# but we clip tightly so the critic focuses on episode structure

# --- World model architecture ---
WORLD_MODEL_DIM = 64  # hidden_dim = attn_dim_head * heads
WORLD_MODEL_DEPTH = 2  # number of transformer layers
WORLD_MODEL_HEADS = 4  # number of attention heads
WORLD_MODEL_DIM_HEAD = 16  # per-head dimension (WORLD_MODEL_DIM = HEADS * DIM_HEAD)

WORLD_MODEL = dict(
    depth=WORLD_MODEL_DEPTH,
    attn_gate_values=True,  # Gate attention values with a learned scalar
    add_value_residual=True,  # ResFormer value residual connections
    ff_relu_squared=True,  # ReLU² activation (Primer paper)
    learned_value_residual_mix=True,  # Learn the mixing coefficient for value residuals
    attn_flash=True,  # Flash Attention for memory efficiency
)

# --- Training ---
NUM_EPISODES_PER_UPDATE = (
    25  # Episodes collected before each PPO update (must divide NUM_EPISODES)
)
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
# Before step 5 the actor/critic use only the direct state embedding.
WORLD_MODEL_EMBED_SCHEDULE = (5.0, 20.0)

# --- Agent extra kwargs ---
AGENT_KWARGS = dict(
    world_model_attn_dim_head=WORLD_MODEL_DIM_HEAD,
    world_model_heads=WORLD_MODEL_HEADS,
    world_model_attn_hybrid_gru=False,  # Hybrid GRU+attention
    world_model_embed_linear_schedule=WORLD_MODEL_EMBED_SCHEDULE,
    actor_critic_world_model=dict(
        frac_critic_head_gradient=5e-2,
        frac_actor_head_gradient=5e-2,
        use_simple_policy_optimization=False,  # SPO vs PPO
        add_entropy_to_advantage=False,  # Cheng et al. entropy-augmented advantages
    ),
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

from x_transformers_rl import Learner

# ---------------------------------------------------------------------------
# Helper: count model parameters
# ---------------------------------------------------------------------------


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Evaluation: run greedy rollouts, return mean episode reward
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(agent, env_name, num_eval_episodes=20, max_timesteps=500):
    """Run the agent's EMA policy greedily and return mean episode reward."""
    agent.eval()
    rewards = []
    eval_env = gym.make(env_name)

    for _ in range(num_eval_episodes):
        obs, _ = eval_env.reset()
        hiddens = None
        episode_reward = 0.0
        for _ in range(max_timesteps):
            raw_actions, hiddens = agent(obs, hiddens=hiddens)
            # Greedy action selection
            if hasattr(raw_actions, "argmax"):
                action = raw_actions.argmax().item()
            else:
                action = raw_actions.softmax(dim=-1).argmax().item()
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

    # -- Environment --
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    num_actions = env.action_space.n  # 2 for CartPole

    assert NUM_EPISODES_PER_UPDATE <= NUM_EPISODES, (
        "NUM_EPISODES_PER_UPDATE must be <= NUM_EPISODES"
    )
    assert NUM_EPISODES % NUM_EPISODES_PER_UPDATE == 0, (
        "NUM_EPISODES must be divisible by NUM_EPISODES_PER_UPDATE"
    )

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

    # -- Training --
    num_updates = NUM_EPISODES // NUM_EPISODES_PER_UPDATE

    t_start_train = time.time()

    # Track VRAM before/during training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Run training: the Learner handles rollout + PPO internally.
    # We run all num_updates in one call.
    learner(env, num_updates)

    t_end_train = time.time()
    env.close()

    training_seconds = t_end_train - t_start_train
    total_seconds = t_end_train - t_start_total

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # -- Final evaluation (greedy policy, no exploration noise) --
    mean_reward, best_reward = evaluate(
        agent,
        ENV_NAME,
        num_eval_episodes=20,
        max_timesteps=MAX_TIMESTEPS,
    )

    # -- Summary block (parseable by grep) --
    print()
    print("---")
    print(f"mean_reward:       {mean_reward:.4f}")
    print(f"best_reward:       {best_reward:.4f}")
    print(f"total_episodes:    {NUM_EPISODES}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_params_M:      {num_params / 1e6:.4f}")
    print(f"world_model_dim:   {WORLD_MODEL_DIM}")
    print(f"world_model_depth: {WORLD_MODEL_DEPTH}")


if __name__ == "__main__":
    main()
