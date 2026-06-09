"""Multi-faceted analysis of Breakout experiment runs.

This module is the reusable, side-effect-free core behind both
``scripts/plot_results.py`` and ``reports/analysis.ipynb``. It reads ONLY the
artifacts that training/evaluation already produced (no re-training, no
re-evaluation) and turns them into the metrics needed to compare runs along
several axes:

    seed · algorithm · hyperparameter · timestep-budget          (the obvious axes)
    + sample efficiency · training stability · compute cost       (the "why" axes)
    + score-distribution shape · behavior · statistical significance

Data sources per run directory (see src/envs.py, scripts/train.py, scripts/evaluate.py):
    monitor_train/*.monitor.csv      raw TRAINING episodes (reward r, length l, time t)
    eval/evaluations.npz             periodic in-training eval (timesteps, results)
    eval_runs/seed<E>/summary_*.json final 100-episode eval aggregate + meta
    eval_runs/seed<E>/episodes_*.csv final 100-episode eval, PER-EPISODE reward+length
    config.yaml                      frozen run config

Design notes
------------
* IQM and the stratified bootstrap CI are implemented here directly, so the
  cross-seed aggregate (rliable-style) works WITHOUT the optional ``rliable``
  dependency. If ``rliable`` is installed, ``rliable_metrics`` exposes its
  richer estimators too (soft import).
* The small label/seed helpers mirror the conventions used by
  scripts/plot_results.py so labels are consistent across both surfaces.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

try:  # numpy>=2.0 renamed trapz -> trapezoid
    from numpy import trapezoid as _trapz
except ImportError:  # numpy<2.0
    from numpy import trapz as _trapz

ROOT = Path(__file__).resolve().parent.parent

# Atari time-limit for ALE/Breakout-v5 (max env steps per eval episode). Eval
# episodes whose recorded length reaches this were truncated by the time cap
# rather than by losing all lives -> a "survived but stopped scoring" signal.
EPISODE_TIME_CAP = 108_000


# ---------------------------------------------------------------------------
# Small label / parsing helpers (kept consistent with scripts/plot_results.py)
# ---------------------------------------------------------------------------
def extract_seed(run_name: str) -> int | None:
    """Parse ``seed<N>`` from a run directory name -> int (or None)."""
    for token in run_name.split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError:
                return None
    return None


def extract_config_variant(run_name: str, algo: str) -> str:
    """Recover the config 'variant' tag from ``<date>_<time>_<stem>_seed<N>``.

        dqn_baseline / dqn  -> "baseline"
        dqn_buffer_100k      -> "buffer_100k"
        ppo_clip_0.2         -> "clip_0.2"
    """
    parts = run_name.split("_")
    if len(parts) < 4 or not parts[-1].startswith("seed"):
        return ""
    config_stem = "_".join(parts[2:-1])
    if config_stem in (algo, f"{algo}_baseline"):
        return "baseline"
    if config_stem.startswith(f"{algo}_"):
        return config_stem[len(algo) + 1 :]
    return config_stem


def group_label(algo: str, features: Mapping[str, Any] | None, variant: str = "") -> str:
    """Human-readable algorithm label, e.g. ``dqn[double+dueling]`` or ``ppo[clip_0.2]``.

    This is the canonical comparison key: runs that share a ``group_label``
    differ only by seed/budget and can be aggregated together.
    """
    feats = features or {}
    label = algo
    if algo == "dqn":
        tags = []
        if feats.get("double_q"):
            tags.append("double")
        if feats.get("dueling"):
            tags.append("dueling")
        if tags:
            label = "dqn[" + "+".join(tags) + "]"
    # Variants whose meaning is already captured by the feature flags above
    # (ddqn -> double, dueling_dqn -> dueling) are NOT shown as a separate tag.
    feature_variants = {"ddqn", "dueling_dqn", "dueling", "double", "double_dueling"}
    if variant and variant != "baseline" and variant not in feature_variants:
        return f"{label}[{variant}]"
    return label


def format_timesteps(n: int) -> str:
    """1_000_000 -> '1M', 250_000 -> '250k'."""
    if n >= 1_000_000:
        val = n / 1_000_000
        return f"{val:g}M" if val == int(val) else f"{val:.1f}M"
    if n >= 1_000:
        val = n / 1_000
        return f"{val:g}k" if val == int(val) else f"{val:.1f}k"
    return str(n)


def experiment_group(path: Path) -> str | None:
    """Return the ``experiments/<group>`` folder name (e.g. '10m', 'ablation_0603')."""
    parts = Path(path).resolve().parts
    if "experiments" not in parts:
        return None
    idx = parts.index("experiments")
    return parts[idx + 1] if idx + 1 < len(parts) else None


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    """Lightweight handle to one experiment run (lazy: holds paths, not data)."""

    run_dir: Path
    name: str
    algo: str
    features: dict[str, Any]
    variant: str
    seed: int | None
    budget: int                      # train.total_timesteps
    bucket: str | None               # experiments/<bucket> folder (budget/ablation group)
    cfg: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def label(self) -> str:
        return group_label(self.algo, self.features, self.variant)


def _read_record(run_dir: Path) -> RunRecord | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    algo = str(cfg.get("algo", {}).get("name", "?"))
    return RunRecord(
        run_dir=run_dir,
        name=run_dir.name,
        algo=algo,
        features=dict(cfg.get("algo", {}).get("features", {}) or {}),
        variant=extract_config_variant(run_dir.name, algo),
        seed=extract_seed(run_dir.name),
        budget=int((cfg.get("train", {}) or {}).get("total_timesteps", 0)),
        bucket=experiment_group(run_dir),
        cfg=cfg,
    )


def discover_runs(paths: Sequence[str | Path]) -> list[RunRecord]:
    """Collect runs under one or more experiment paths.

    Accepts a group folder (``experiments/10m``), a single run dir, or a parent
    that contains groups; scans up to two levels deep, like
    scripts/plot_results.py ``_collect_runs``. De-duplicates by resolved path.
    """
    out: list[RunRecord] = []
    seen: set[Path] = set()

    def _add(d: Path) -> None:
        key = d.resolve()
        if key in seen:
            return
        rec = _read_record(d)
        if rec is not None:
            out.append(rec)
            seen.add(key)

    for p in paths:
        root = Path(p).resolve()
        if not root.exists():
            continue
        if (root / "config.yaml").exists():
            _add(root)
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / "config.yaml").exists():
                _add(child)
                continue
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and (sub / "config.yaml").exists():
                    _add(sub)
    return out


def dedup_latest(runs: Sequence[RunRecord]) -> list[RunRecord]:
    """Keep one run per (label, budget, seed) — the latest by timestamped name.

    The ablation folder has a few accidental duplicates (e.g. dqn_target_10000,
    dqn_expl_frac_0.2 appear twice); this removes them deterministically.
    """
    best: dict[tuple[str, int, int | None], RunRecord] = {}
    for r in runs:
        key = (r.label, r.budget, r.seed)
        if key not in best or r.name > best[key].name:  # names start with sortable date
            best[key] = r
    return list(best.values())


# ---------------------------------------------------------------------------
# Per-run data loaders
# ---------------------------------------------------------------------------
def load_summary(run: RunRecord, eval_seed: int | None = None) -> dict[str, Any] | None:
    """Final-eval ``summary*.json`` (mean/std/median/min/max/ci95 + meta).

    Picks ``eval_runs/seed<eval_seed>`` if given, else the first available.
    """
    base = run.run_dir / "eval_runs"
    if not base.exists():
        return None
    seed_dirs = sorted(base.glob(f"seed{eval_seed}")) if eval_seed is not None else sorted(base.glob("seed*"))
    for sd in seed_dirs:
        files = [sd / "summary.json"] if (sd / "summary.json").exists() else sorted(sd.glob("summary_*.json"))
        for f in files:
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def load_episodes(run: RunRecord) -> pd.DataFrame:
    """Per-episode final-eval rewards+lengths across all eval seeds.

    Columns: reward, length, eval_seed, plus run identity (label, seed, budget).
    Empty DataFrame if the run was never finally-evaluated.
    """
    base = run.run_dir / "eval_runs"
    frames: list[pd.DataFrame] = []
    if base.exists():
        for csv in sorted(base.glob("seed*/episodes*.csv")):
            df = pd.read_csv(csv)
            if "reward" not in df.columns:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["reward", "length", "label", "seed", "budget"])
    out = pd.concat(frames, ignore_index=True)
    out["label"] = run.label
    out["seed"] = run.seed
    out["budget"] = run.budget
    out["bucket"] = run.bucket
    return out


def load_monitor(run: RunRecord) -> pd.DataFrame:
    """Concatenated raw TRAINING episodes from ``monitor_train/*.monitor.csv``.

    SB3 Monitor files have a ``#{json}`` header line then columns r,l,t where
    ``t`` is seconds since training start. Columns out: r, l, t, env.
    """
    mdir = run.run_dir / "monitor_train"
    frames: list[pd.DataFrame] = []
    if mdir.exists():
        for csv in sorted(mdir.glob("*.monitor.csv")):
            try:
                df = pd.read_csv(csv, skiprows=1)  # skip the #{json} header
            except Exception:
                continue
            if not {"r", "l", "t"}.issubset(df.columns):
                continue
            df["env"] = csv.stem.split(".")[0]
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["r", "l", "t", "env"])
    return pd.concat(frames, ignore_index=True)


def load_evaluations(run: RunRecord) -> dict[str, np.ndarray] | None:
    """Periodic in-training eval from ``eval/evaluations.npz``.

    Returns timesteps, mean/std of eval reward, and mean episode length per point.
    """
    npz = run.run_dir / "eval" / "evaluations.npz"
    if not npz.exists():
        return None
    data = np.load(npz)
    results = np.asarray(data["results"], dtype=np.float64)  # (n_points, n_eval)
    out = {
        "timesteps": np.asarray(data["timesteps"], dtype=np.int64),
        "mean": results.mean(axis=1),
        "std": results.std(axis=1),
    }
    if "ep_lengths" in data.files:
        out["ep_len_mean"] = np.asarray(data["ep_lengths"], dtype=np.float64).mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Statistics primitives
# ---------------------------------------------------------------------------
def _skewness(values: Sequence[float]) -> float:
    """Fisher-Pearson skewness, computed locally so scipy is NOT required."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    mu, sd = x.mean(), x.std()
    if sd == 0:
        return 0.0
    return float((((x - mu) / sd) ** 3).mean())


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> tuple[float, float]:
    """Non-parametric bootstrap CI of the mean (same idea as src/eval.py)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo = (100.0 - ci) / 2.0
    return float(np.percentile(boots, lo)), float(np.percentile(boots, 100.0 - lo))


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% (robust central tendency,
    the rliable-recommended aggregate for RL because it ignores the lucky/unlucky
    tails that a plain mean is sensitive to with few seeds)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.size == 0:
        return float("nan")
    if v.size < 4:
        return float(v.mean())
    lo, hi = int(np.floor(v.size * 0.25)), int(np.ceil(v.size * 0.75))
    return float(v[lo:hi].mean())


def stratified_bootstrap_iqm_ci(
    per_seed_scores: Sequence[float], *, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap CI of the IQM by resampling seeds with replacement."""
    v = np.asarray(per_seed_scores, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [iqm(rng.choice(v, size=v.size, replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# Per-run metrics (the comparison criteria)
# ---------------------------------------------------------------------------
def sample_efficiency(run: RunRecord, *, threshold: float | None = None) -> dict[str, float]:
    """Learning-speed metrics from the in-training eval curve.

    auc_mean_reward : average eval reward over training (area under the eval
                      curve / total steps) — higher = learned more, sooner.
    steps_to_threshold : first env-step at which mean eval reward >= threshold
                      (NaN if never reached). Lets you compare *speed* to a fixed bar.
    final / best : last and best eval-curve points.
    """
    ev = load_evaluations(run)
    if ev is None or len(ev["timesteps"]) == 0:
        return {"auc_mean_reward": float("nan"), "steps_to_threshold": float("nan"),
                "final": float("nan"), "best": float("nan")}
    ts, mean = ev["timesteps"].astype(np.float64), ev["mean"]
    span = ts[-1] - ts[0]
    auc = float(_trapz(mean, ts) / span) if span > 0 else float(mean.mean())
    steps_to = float("nan")
    if threshold is not None:
        hit = np.where(mean >= threshold)[0]
        if hit.size:
            steps_to = float(ts[hit[0]])
    return {
        "auc_mean_reward": auc,
        "steps_to_threshold": steps_to,
        "final": float(mean[-1]),
        "best": float(mean.max()),
    }


def training_stability(run: RunRecord, *, last_k: int = 3) -> dict[str, float]:
    """How steady the in-training eval curve is near the end.

    tail_std / tail_cv : std (and coeff. of variation) of the last ``last_k``
                         eval points — large = oscillating/unstable late training.
    drawdown : best - final (how much it fell back from its peak).
    """
    ev = load_evaluations(run)
    if ev is None or len(ev["mean"]) == 0:
        return {"tail_std": float("nan"), "tail_cv": float("nan"), "drawdown": float("nan")}
    mean = ev["mean"]
    tail = mean[-last_k:]
    tail_std = float(tail.std(ddof=0))
    tail_cv = float(tail_std / abs(tail.mean())) if tail.mean() != 0 else float("nan")
    return {"tail_std": tail_std, "tail_cv": tail_cv, "drawdown": float(mean.max() - mean[-1])}


def compute_stats(run: RunRecord) -> dict[str, float]:
    """Wall-clock / throughput from the training Monitor logs.

    fps : env steps per second (sum of episode lengths / wall-clock span).
    wall_clock_hours : training duration.
    Useful for the off-policy(DQN) vs on-policy(PPO/A2C) compute trade-off and
    'reward per GPU-hour' (combine with a final score downstream).
    """
    mon = load_monitor(run)
    if mon.empty:
        return {"fps": float("nan"), "wall_clock_hours": float("nan"), "train_steps": float("nan")}
    total_steps = float(mon["l"].sum())
    wall = float(mon["t"].max())  # seconds since training start (shared t_start across envs)
    fps = total_steps / wall if wall > 0 else float("nan")
    return {"fps": fps, "wall_clock_hours": wall / 3600.0, "train_steps": total_steps}


def behavior_stats(run: RunRecord) -> dict[str, float]:
    """Behavioral / distribution-shape metrics from the final per-episode eval.

    mean_length / cap_rate : average eval episode length and the fraction of
        episodes that hit the time cap (108000) — distinguishes 'actively
        clearing bricks' from 'survives but stalls'.
    skew : skewness of the return distribution (a box plot hides this).
    """
    ep = load_episodes(run)
    if ep.empty:
        return {"mean_length": float("nan"), "cap_rate": float("nan"), "skew": float("nan"),
                "reward_mean": float("nan"), "reward_median": float("nan"), "n_episodes": 0}
    lengths = ep["length"].to_numpy() if "length" in ep.columns else np.array([])
    rewards = ep["reward"].to_numpy(dtype=np.float64)
    cap_rate = float((lengths >= EPISODE_TIME_CAP).mean()) if lengths.size else float("nan")
    return {
        "mean_length": float(lengths.mean()) if lengths.size else float("nan"),
        "cap_rate": cap_rate,
        "skew": _skewness(rewards),
        "reward_mean": float(rewards.mean()),
        "reward_median": float(np.median(rewards)),
        "n_episodes": int(rewards.size),
    }


def final_performance(run: RunRecord) -> dict[str, float]:
    """Final 100-episode score: prefer summary.json, fall back to episodes.csv."""
    s = load_summary(run)
    if s is not None and "mean" in s:
        return {k: float(s[k]) for k in
                ("mean", "std", "median", "min", "max", "ci95_low", "ci95_high") if k in s}
    ep = load_episodes(run)
    if ep.empty:
        return {"mean": float("nan")}
    r = ep["reward"].to_numpy(dtype=np.float64)
    lo, hi = bootstrap_ci(r)
    return {"mean": float(r.mean()), "std": float(r.std()), "median": float(np.median(r)),
            "min": float(r.min()), "max": float(r.max()), "ci95_low": lo, "ci95_high": hi}


# ---------------------------------------------------------------------------
# Cross-run aggregation tables
# ---------------------------------------------------------------------------
def runs_table(runs: Sequence[RunRecord], *, threshold: float | None = None) -> pd.DataFrame:
    """One row per run with every per-run metric — the master analysis table."""
    rows: list[dict[str, Any]] = []
    for r in dedup_latest(runs):
        row: dict[str, Any] = {
            "run": r.name, "label": r.label, "algo": r.algo, "variant": r.variant,
            "seed": r.seed, "budget": r.budget, "bucket": r.bucket,
        }
        row.update({f"final_{k}": v for k, v in final_performance(r).items()})
        row.update(sample_efficiency(r, threshold=threshold))
        row.update(training_stability(r))
        row.update(compute_stats(r))
        row.update(behavior_stats(r))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_config(
    runs: Sequence[RunRecord], *, metric: str = "final_mean"
) -> pd.DataFrame:
    """Cross-SEED aggregate per (label, budget): mean, IQM, and bootstrap CIs.

    This is where multi-seed groups (1m: seeds 7/77/777) become rigorous —
    columns ``iqm`` / ``iqm_ci_low|high`` are meaningful only when ``n_seeds``>=3;
    single-seed rows fall back to the per-run value and are flagged by ``n_seeds==1``.
    """
    tbl = runs_table(runs)
    if tbl.empty or metric not in tbl.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (label, budget), grp in tbl.groupby(["label", "budget"]):
        vals = grp[metric].dropna().to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        mean_lo, mean_hi = bootstrap_ci(vals) if vals.size > 1 else (float("nan"), float("nan"))
        iqm_lo, iqm_hi = stratified_bootstrap_iqm_ci(vals) if vals.size >= 3 else (float("nan"), float("nan"))
        rows.append({
            "label": label, "budget": budget, "budget_str": format_timesteps(int(budget)),
            "n_seeds": int(vals.size), "seeds": sorted(grp["seed"].dropna().astype(int).tolist()),
            "mean": float(vals.mean()), "mean_ci_low": mean_lo, "mean_ci_high": mean_hi,
            "iqm": iqm(vals), "iqm_ci_low": iqm_lo, "iqm_ci_high": iqm_hi,
            "std_across_seeds": float(vals.std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values(["budget", "mean"], ascending=[True, False])


def hyperparameter_response(
    runs: Sequence[RunRecord], algo: str, knob: str, *, metric: str = "final_mean"
) -> pd.DataFrame:
    """Build a response curve 'metric vs knob value' for one swept hyperparameter.

    ``knob`` matches the variant prefix, e.g. algo='ppo', knob='clip' picks up
    variants clip_0.05/clip_0.2/clip_0.3 plus the baseline. The numeric value is
    parsed from the variant tag; the baseline is read from the run's config so it
    is placed correctly on the x-axis.
    """
    rows: list[dict[str, Any]] = []
    for r in dedup_latest(runs):
        if r.algo != algo:
            continue
        v = r.variant
        value: float | None = None
        if v == "baseline":
            value = _baseline_knob_value(r, knob)
        elif v.startswith(f"{knob}_"):
            value = _parse_trailing_number(v[len(knob) + 1 :])
        if value is None:
            continue
        rows.append({"variant": v, "value": value, "seed": r.seed, "budget": r.budget,
                     metric: final_performance(r).get("mean", float("nan"))})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # A response curve must compare runs at the SAME training budget, otherwise a
    # baseline trained at a different budget would distort the curve. Keep only
    # the budget the swept (non-baseline) runs actually used.
    swept = df[df["variant"] != "baseline"]
    if not swept.empty:
        target_budget = int(swept["budget"].mode().iloc[0])
        df = df[df["budget"] == target_budget]
    return df.sort_values("value")


# Map a knob name to its config.yaml location so the baseline point is anchored.
_KNOB_TO_CFG = {
    "clip": ("algo", "kwargs", "clip_range"),
    "ent": ("algo", "kwargs", "ent_coef"),
    "gae": ("algo", "kwargs", "gae_lambda"),
    "gamma": ("algo", "kwargs", "gamma"),
    "lr": ("algo", "kwargs", "learning_rate"),
    "ne": ("algo", "kwargs", "n_epochs"),
    "nstep": ("algo", "kwargs", "n_steps"),
    "vf": ("algo", "kwargs", "vf_coef"),
    "buffer": ("algo", "kwargs", "buffer_size"),
    "batch": ("algo", "kwargs", "batch_size"),
    "target": ("algo", "kwargs", "target_update_interval"),
    "expl": ("algo", "kwargs", "exploration_fraction"),
    "fs": ("env", "frame_stack"),
}


def _baseline_knob_value(run: RunRecord, knob: str) -> float | None:
    path = _KNOB_TO_CFG.get(knob)
    if path is None:
        return None
    node: Any = run.cfg
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def _parse_trailing_number(s: str) -> float | None:
    """'0.2' -> 0.2 ; '100k' -> 100000 ; '5000' -> 5000 ; '1.5e-4' -> 0.00015."""
    s = s.strip()
    m = re.match(r"^([0-9]*\.?[0-9]+(?:e-?[0-9]+)?)([kKmM]?)$", s)
    if not m:
        return None
    val = float(m.group(1))
    return val * {"k": 1e3, "m": 1e6, "": 1.0}[m.group(2).lower()]


# ---------------------------------------------------------------------------
# Statistical significance between two runs (eval-level)
# ---------------------------------------------------------------------------
def pairwise_significance(run_a: RunRecord, run_b: RunRecord, *, seed: int = 0) -> dict[str, Any]:
    """Compare two runs' final 100-episode return distributions.

    Reports a Mann-Whitney U test (distribution-free: does A stochastically
    dominate B?), the probability of improvement P(A>B) estimated over episode
    pairs, and a bootstrap CI for the mean difference. NOTE: this is *evaluation*
    variance over episodes of single models — it is NOT a seed-level test, so it
    cannot by itself prove one algorithm beats another in general.
    """
    a = load_episodes(run_a)["reward"].to_numpy(dtype=np.float64)
    b = load_episodes(run_b)["reward"].to_numpy(dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {"error": "missing episodes.csv for one of the runs"}
    # Mann-Whitney U needs scipy; it's optional — degrade to NaN p-value if absent
    # (the bootstrap diff + P(A>B) below are scipy-free and still reported).
    try:
        from scipy.stats import mannwhitneyu

        _u, _p = mannwhitneyu(a, b, alternative="two-sided")
        u, p = float(_u), float(_p)
    except Exception:
        u, p = float("nan"), float("nan")
    prob_improve = float((a[:, None] > b[None, :]).mean())  # P(random A-ep > random B-ep)
    rng = np.random.default_rng(seed)
    diffs = (rng.choice(a, (4000, a.size)).mean(1) - rng.choice(b, (4000, b.size)).mean(1))
    return {
        "label_a": run_a.label, "label_b": run_b.label,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(a.mean() - b.mean()),
        "diff_ci_low": float(np.percentile(diffs, 2.5)), "diff_ci_high": float(np.percentile(diffs, 97.5)),
        "mannwhitney_u": float(u), "p_value": float(p),
        "prob_a_gt_b": prob_improve,
    }


# ---------------------------------------------------------------------------
# Optional rliable backend (soft import)
# ---------------------------------------------------------------------------
def rliable_available() -> bool:
    try:
        import rliable  # noqa: F401
        return True
    except Exception:
        return False
