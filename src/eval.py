"""Evaluation utilities — run n episodes and compute quantitative metrics.

External library: we use SB3's ``evaluate_policy`` only to roll out episodes
and collect raw per-episode returns/lengths.

MY ORIGINAL CONTRIBUTION: the statistics layer on top. Rather than reporting a
bare mean, I summarise each evaluation with mean/std/median/min/max AND a
non-parametric 95% bootstrap confidence interval of the mean (``_bootstrap_ci``).
Atari return distributions are skewed and heavy-tailed, so a bootstrap CI is a
more honest uncertainty estimate than a Gaussian ±std, and it is what the
plotting/report code consumes to compare algorithms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv


@dataclass
class EvalSummary:
    """Statistical summary of a single evaluation run."""

    n_episodes: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    ci95_low: float
    ci95_high: float
    rewards: list[float]
    lengths: list[int]

    def to_dict(self, *, drop_lists: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if drop_lists:
            data.pop("rewards", None)
            data.pop("lengths", None)
        return data


def _bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    """Non-parametric 95% bootstrap CI of the mean (my contribution).

    Resample the episode returns with replacement ``n_boot`` times, take the
    mean of each resample, then read off the 2.5th / 97.5th percentiles. This
    makes no normality assumption, which matters for skewed Atari scores. The
    RNG is seeded by the caller so the CI is reproducible.
    """
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = rng if rng is not None else np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate_model(
    model: BaseAlgorithm,
    env: VecEnv,
    *,
    n_eval_episodes: int = 50,
    deterministic: bool = True,
    seed: int | None = None,
) -> EvalSummary:
    """Roll out episodes via SB3's ``evaluate_policy`` then return statistics.

    ``deterministic=True`` by default: we report the greedy/argmax policy so
    the score is reproducible and not inflated/deflated by exploration noise.
    """
    if seed is not None:
        env.seed(seed)

    rewards_list, lengths_list = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=False,
        return_episode_rewards=True,
        warn=True,
    )
    rewards = np.asarray(rewards_list, dtype=np.float64)
    lengths = np.asarray(lengths_list, dtype=np.int64)
    if len(rewards) == 0:
        raise RuntimeError("No evaluation episodes were completed.")

    ci_low, ci_high = _bootstrap_ci(rewards, rng=np.random.default_rng(seed or 0))

    return EvalSummary(
        n_episodes=int(len(rewards)),
        mean=float(rewards.mean()),
        std=float(rewards.std(ddof=0)),
        median=float(np.median(rewards)),
        min=float(rewards.min()),
        max=float(rewards.max()),
        ci95_low=ci_low,
        ci95_high=ci_high,
        rewards=rewards.tolist(),
        lengths=lengths.tolist(),
    )


def format_summary(summary: EvalSummary, *, name: str = "") -> str:
    head = f"[{name}] " if name else ""
    return (
        f"{head}episodes={summary.n_episodes}  "
        f"mean={summary.mean:.2f} ± {summary.std:.2f}  "
        f"median={summary.median:.1f}  "
        f"[min,max]=[{summary.min:.1f}, {summary.max:.1f}]  "
        f"95% CI=[{summary.ci95_low:.2f}, {summary.ci95_high:.2f}]"
    )
