"""A2C builder — uses SB3's A2C unmodified.

================================ ATTRIBUTION ================================
External library: Stable-Baselines3 (SB3). A2C — the synchronous, batched
variant of A3C (Mnih et al., 2016) — is used AS-IS; no algorithmic change.

MY CONTRIBUTION here is the same thin-factory pattern as ppo.py: it lets A2C
share the identical preprocessing, seeding and evaluation protocol as the
other algorithms so it serves as the "simplest actor-critic" baseline in the
comparison. A2C's design choices are expressed through hyper-parameters
(configs/a2c.yaml), justified there; the notable ones vs. PPO are:
    use_rms_prop=true (original A3C optimizer), n_steps=5 (short n-step
    rollout), gae_lambda=1.0 (pure n-step return), vf_coef=0.25 and
    normalize_advantage=false — i.e. the "vanilla" actor-critic setting,
    deliberately kept simple so the PPO improvements stand out by contrast.
============================================================================
"""

from __future__ import annotations

import copy
from typing import Any

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv


def build_a2c(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> A2C:
    algo_cfg = cfg["algo"]
    # deepcopy: never mutate the caller's config (it gets snapshotted to disk).
    kwargs = copy.deepcopy(algo_cfg.get("kwargs", {}) or {})
    policy = kwargs.pop("policy", "CnnPolicy")  # Nature-CNN actor-critic for pixels

    return A2C(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,          # forwarded for reproducibility (see utils/seeding.py)
        device=device,
        verbose=0,
        **kwargs,
    )
