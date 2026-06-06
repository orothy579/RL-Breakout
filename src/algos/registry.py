"""Algorithm-name -> builder mapping.

MY ORIGINAL CONTRIBUTION: a tiny registry/dispatch layer so the training and
evaluation scripts stay algorithm-agnostic. ``cfg["algo"]["name"]`` (a string
from YAML) is the single switch that selects which builder runs; adding a new
algorithm means adding one entry here, nothing else changes downstream.
``load_model`` mirrors the same dispatch for reloading saved checkpoints,
including the subtlety that our custom ``DoubleDQN`` must be reloaded with its
own class (not SB3's ``DQN``) so the overridden ``train`` is preserved.
"""

from __future__ import annotations

from typing import Any, Callable

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from src.algos.a2c import build_a2c
from src.algos.dqn import build_dqn
from src.algos.ppo import build_ppo

BuilderFn = Callable[..., BaseAlgorithm]

REGISTRY: dict[str, BuilderFn] = {
    "dqn": build_dqn,   # also covers Double / Dueling DQN via algo.features flags
    "ppo": build_ppo,
    "a2c": build_a2c,
}


def list_algorithms() -> list[str]:
    return sorted(REGISTRY.keys())


def build_model(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> BaseAlgorithm:
    name = str(cfg["algo"]["name"]).lower()
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {list_algorithms()}"
        )
    return REGISTRY[name](
        cfg, env=env, tensorboard_log=tensorboard_log, seed=seed, device=device
    )


def load_model(
    cfg: dict[str, Any],
    model_path: str,
    *,
    env: VecEnv | None = None,
    device: str = "auto",
) -> BaseAlgorithm:
    from stable_baselines3 import A2C, DQN, PPO

    from src.algos.dqn import DoubleDQN

    name = str(cfg["algo"]["name"]).lower()
    features = cfg["algo"].get("features", {}) or {}

    if name == "ppo":
        return PPO.load(model_path, env=env, device=device)
    if name == "a2c":
        return A2C.load(model_path, env=env, device=device)
    if name == "dqn":
        # IMPORTANT: a Double-DQN checkpoint must be reloaded with our
        # DoubleDQN subclass so the overridden Double-Q ``train`` survives a
        # save/load round-trip (e.g. when resuming training). Dueling is
        # carried by the saved policy weights, so no special-casing needed.
        cls = DoubleDQN if features.get("double_q", False) else DQN
        return cls.load(model_path, env=env, device=device)
    raise KeyError(f"Unknown algorithm '{name}'.")
