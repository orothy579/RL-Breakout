"""PPO builder — uses SB3's PPO unmodified.

================================ ATTRIBUTION ================================
External library: Stable-Baselines3 (SB3). PPO (Schulman et al., 2017) is
used AS-IS; we make no algorithmic modification to it.

MY CONTRIBUTION here is intentionally small: a thin factory that maps our
YAML ``algo`` block onto SB3's ``PPO(...)`` so PPO plugs into the exact same
training/eval/logging pipeline as the (modified) DQN family, keeping the
on-policy vs. off-policy comparison fair. The interesting design choices for
PPO live in the *hyper-parameters* (configs/ppo.yaml), which are justified
there; the most relevant ones for Atari Breakout are:
    learning_rate=2.5e-4, n_steps=128 (x8 envs = 1024 steps/update),
    batch_size=256, n_epochs=4, clip_range=0.1, ent_coef=0.01, vf_coef=0.5
    — the standard SB3/OpenAI-baselines Atari PPO recipe.
============================================================================
"""

from __future__ import annotations

import copy
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def build_ppo(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> PPO:
    algo_cfg = cfg["algo"]
    # deepcopy: never mutate the caller's config (it gets snapshotted to disk).
    kwargs = copy.deepcopy(algo_cfg.get("kwargs", {}) or {})
    policy = kwargs.pop("policy", "CnnPolicy")  # Nature-CNN actor-critic for pixels

    return PPO(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,          # forwarded for reproducibility (see utils/seeding.py)
        device=device,
        verbose=0,
        **kwargs,
    )
