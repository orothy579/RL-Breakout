"""Breakout v5 environment factory.

================================ ATTRIBUTION ================================
External library: Stable-Baselines3 (SB3). We rely on SB3's
``make_atari_env`` (which applies the standard DeepMind ``AtariWrapper``
stack) plus the vectorised wrappers ``VecFrameStack`` / ``VecTransposeImage``.

MY ORIGINAL CONTRIBUTION: defining the environment ONCE here so every
algorithm (DQN/DDQN/Dueling/A2C/PPO) trains and is evaluated on a byte-for-byte
identical pipeline — this is what makes the cross-algorithm comparison fair.
The non-trivial design choice is the *deliberate train/eval mismatch*:

  * Training env  : terminal_on_life_loss=True, clip_reward=True
        -> denser learning signal (each life is an episode; rewards in
           {-1,0,+1}) — the DeepMind 2015 training protocol.
  * Eval / play   : terminal_on_life_loss=False, clip_reward=False
        -> measures the TRUE game score over a full 5-life episode, which is
           the number we actually report.

Baseline preprocessing fixed by the assignment manual (§1.2 / Guide §B.4):

    env_kwargs = {
        "frameskip": 1,                      # AtariWrapper does its own skip
        "repeat_action_probability": 0.0,    # no sticky actions in baseline
        "full_action_space": False,
    }
    AtariWrapper = NoopReset + MaxAndSkip(4) + EpisodicLife + FireReset
                   + WarpFrame(84x84 gray) + ClipReward
    + VecFrameStack(n_stack=4)               # 4 frames -> motion/velocity cues
============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecTransposeImage,
)

ENV_ID_DEFAULT = "ALE/Breakout-v5"


def _ensure_registered() -> None:
    # ale-py >= 0.11 no longer auto-registers; do it explicitly so
    # ``gym.make("ALE/Breakout-v5")`` works regardless of import order.
    gym.register_envs(ale_py)


def _coerce_env_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalise types coming from YAML (string 'true'/'false' -> bool)."""
    out: dict[str, Any] = {}
    for k, v in env_kwargs.items():
        if isinstance(v, str) and v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
        else:
            out[k] = v
    return out


def build_train_env(
    env_cfg: dict[str, Any],
    *,
    seed: int,
    monitor_dir: str | Path | None = None,
) -> VecEnv:
    """Build the vectorised TRAINING env (n_envs parallel Atari games)."""
    _ensure_registered()

    env_id = env_cfg.get("env_id", ENV_ID_DEFAULT)
    n_envs = int(env_cfg.get("n_envs", 8))          # 8 parallel envs (config default)
    frame_stack = int(env_cfg.get("frame_stack", 4))
    env_kwargs = _coerce_env_kwargs(env_cfg.get("env_kwargs", {}))
    wrapper_kwargs = dict(env_cfg.get("wrapper_kwargs", {}))

    monitor_dir_str = str(monitor_dir) if monitor_dir is not None else None
    if monitor_dir_str is not None:
        # Monitor CSVs capture raw episode returns/lengths for later plotting.
        Path(monitor_dir_str).mkdir(parents=True, exist_ok=True)

    # make_atari_env applies the full DeepMind AtariWrapper stack to each env.
    venv = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=monitor_dir_str,
        env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs or None,
    )
    if frame_stack and frame_stack > 1:
        # Stack the last 4 frames so the agent can perceive ball velocity
        # (a single 84x84 frame is not enough to infer motion direction).
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return venv


def build_eval_env(
    env_cfg: dict[str, Any],
    eval_env_cfg: dict[str, Any],
    *,
    seed: int,
) -> VecEnv:
    """Build the EVALUATION env (true full-episode score, no reward clipping)."""
    _ensure_registered()

    env_id = env_cfg.get("env_id", ENV_ID_DEFAULT)
    n_envs = int(eval_env_cfg.get("n_envs", 1))
    frame_stack = int(env_cfg.get("frame_stack", 4))
    env_kwargs = _coerce_env_kwargs(env_cfg.get("env_kwargs", {}))

    # Start from the training wrappers, then force the eval-specific overrides
    # so the reported score reflects the real game, not the training proxy.
    eval_wrapper = dict(env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.update(eval_env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.setdefault("terminal_on_life_loss", False)  # full 5-life episode
    eval_wrapper.setdefault("clip_reward", False)            # real (unclipped) score

    venv: VecEnv = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_kwargs=eval_wrapper,
    )
    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    # VecTransposeImage: SB3's predict() expects channel-first images; the eval
    # path is built manually here, so we add the transpose that the training
    # path gets automatically.
    return VecTransposeImage(venv)


def build_play_env(
    env_cfg: dict[str, Any],
    eval_env_cfg: dict[str, Any],
    *,
    seed: int,
    render_mode: str = "human",
) -> VecEnv:
    """Build a single renderable env for interactive watching (scripts/play.py)."""
    _ensure_registered()
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    from stable_baselines3.common.monitor import Monitor

    env_id = env_cfg.get("env_id", ENV_ID_DEFAULT)
    frame_stack = int(env_cfg.get("frame_stack", 4))
    env_kwargs = _coerce_env_kwargs(env_cfg.get("env_kwargs", {}))

    # Same eval-style wrappers (no life-loss termination, no reward clipping).
    eval_wrapper = dict(env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.update(eval_env_cfg.get("wrapper_kwargs", {}))
    eval_wrapper.setdefault("terminal_on_life_loss", False)
    eval_wrapper.setdefault("clip_reward", False)

    # Built by hand (not make_atari_env) because we need render_mode on the
    # underlying gym env so a window can be shown.
    def _make() -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
        env = Monitor(env)
        env = AtariWrapper(env, **eval_wrapper)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    venv: VecEnv = DummyVecEnv([_make])
    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return venv
