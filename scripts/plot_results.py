#!/usr/bin/env python3
"""experiments 디렉토리를 훑어 학습/평가 결과를 figures & tables 로 정리한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Reusable analysis core (loaders + metrics). The richer multi-axis plots in the
# `--analysis` suite below consume this; the legacy plots keep their own loaders.
from src import analysis as A


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "runs"


def _group_name_under_experiments(path: Path) -> str | None:
    """``experiments/<group>/...`` 에서 ``group`` 이름을 반환."""
    parts = path.resolve().parts
    if "experiments" not in parts:
        return None
    idx = parts.index("experiments")
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return None


def _is_run_dir(path: Path) -> bool:
    return (path / "config.yaml").exists()


def _is_group_dir(path: Path) -> bool:
    """``experiments/<group>/`` 처럼 하위 run 을 담는 폴더인지 판별."""
    if not path.is_dir() or path.name.startswith("."):
        return False
    if _is_run_dir(path):
        return False
    return any(_is_run_dir(child) for child in path.iterdir() if child.is_dir())


def _collect_group_labels(experiment_paths: list[Path]) -> list[str]:
    """출력 slug 용 그룹 라벨 수집 (개별 run 폴더명은 제외)."""
    labels: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen:
            labels.append(name)
            seen.add(name)

    for root in experiment_paths:
        root = root.resolve()
        group = _group_name_under_experiments(root)
        if group and _is_group_dir(root):
            _add(group)
            continue
        if group and _is_run_dir(root):
            parent = root.parent
            if parent.name != "experiments":
                _add(parent.name)
            else:
                _add("root")
            continue
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if _is_group_dir(child):
                _add(child.name)
            elif _is_run_dir(child):
                _add("root")
    return labels


def _truncate_slug(slug: str, *, max_len: int = 80) -> str:
    """파일명 길이 제한 (Linux 255 byte) — 초과 시 짧게 자르고 해시 접미사."""
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha256(slug.encode()).hexdigest()[:8]
    return f"{slug[: max_len - 9]}_{digest}"


def _output_slug(experiment_paths: list[Path], runs: list[dict[str, Any]]) -> str:
    """출력 파일명 접미사: ``experiments`` 바로 아래 그룹 디렉터리명(들)."""
    labels = _collect_group_labels(experiment_paths)
    if not labels:
        for r in runs:
            parent = r["run_dir"].resolve().parent
            if parent.name == "experiments":
                labels.append("root")
            elif parent.name not in labels:
                labels.append(parent.name)
    if not labels:
        return "runs"
    unique = sorted(set(labels))
    if len(unique) > 8:
        digest = hashlib.sha256("_".join(unique).encode()).hexdigest()[:8]
        head = "_".join(unique[:4])
        slug = f"{head}_plus{len(unique) - 4}_{digest}"
    else:
        slug = "_".join(unique)
    return _truncate_slug(_slugify(slug))


def _slugged_path(path: Path, slug: str) -> Path:
    return path.with_name(f"{path.stem}_{slug}{path.suffix}")


def _read_run(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    eval_npz = run_dir / "eval" / "evaluations.npz"
    timesteps: np.ndarray | None = None
    results_mean: np.ndarray | None = None
    results_std: np.ndarray | None = None
    if eval_npz.exists():
        data = np.load(eval_npz)
        timesteps = np.asarray(data["timesteps"])
        results = np.asarray(data["results"])
        results_mean = results.mean(axis=1)
        results_std = results.std(axis=1)

    summary_files: list[Path] = []
    if (run_dir / "eval_runs").exists():
        for seed_dir in sorted((run_dir / "eval_runs").glob("seed*/")):
            canonical = seed_dir / "summary.json"
            if canonical.exists():
                summary_files.append(canonical)
            else:
                summary_files.extend(sorted(seed_dir.glob("summary_*.json")))

    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in summary_files]

    algo_name = cfg.get("algo", {}).get("name", "?")
    train_cfg = cfg.get("train", {}) or {}
    return {
        "run_dir": run_dir,
        "name": run_dir.name,
        "algo": algo_name,
        "features": cfg.get("algo", {}).get("features", {}) or {},
        "seed": _extract_seed(run_dir.name),
        "variant": _extract_config_variant(run_dir.name, algo_name),
        "total_timesteps": int(train_cfg.get("total_timesteps", 0)),
        "timesteps": timesteps,
        "ep_rew_mean": results_mean,
        "ep_rew_std": results_std,
        "summaries": summaries,
    }


def _extract_seed(name: str) -> int | None:
    for token in name.split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError:
                return None
    return None


def _extract_config_variant(run_name: str, algo: str) -> str:
    """run 폴더명 ``<date>_<time>_<config_stem>_seed<N>`` 에서 variant 태그 추출.

    ``dqn_baseline`` / ``dqn`` → "baseline"
    ``dqn_buffer_500k``        → "buffer_500k"
    ``ppo_clip_0.2``           → "clip_0.2"
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


def _parse_eval_summary(raw: dict[str, Any]) -> dict[str, float | int]:
    """``summary.json`` / ``EvalSummary`` 필드를 플롯용으로 정규화."""
    keys = ("mean", "std", "median", "min", "max", "ci95_low", "ci95_high")
    missing = [k for k in keys if k not in raw]
    if missing:
        raise KeyError(f"summary missing fields: {missing}")
    return {
        "n_episodes": int(raw.get("n_episodes", 0)),
        **{k: float(raw[k]) for k in keys},
    }


def _group_label(run: dict[str, Any]) -> str:
    algo = run["algo"]
    feats = run["features"]
    if algo == "dqn":
        tags = []
        if feats.get("double_q"):
            tags.append("double")
        if feats.get("dueling"):
            tags.append("dueling")
        if tags:
            algo = "dqn[" + "+".join(tags) + "]"
    variant = run.get("variant", "")
    if variant and variant != "baseline":
        return f"{algo}[{variant}]"
    return algo


def _format_timesteps(n: int) -> str:
    if n >= 1_000_000:
        val = n / 1_000_000
        return f"{val:g}M" if val == int(val) else f"{val:.1f}M"
    if n >= 1_000:
        val = n / 1_000
        return f"{val:g}k" if val == int(val) else f"{val:.1f}k"
    return str(n)


def _run_training_seed(run: dict[str, Any], raw: dict[str, Any] | None = None) -> int | str:
    if run.get("seed") is not None:
        return run["seed"]
    meta = (raw or {}).get("meta", {}) or {}
    training_seed = meta.get("training_seed")
    if training_seed is not None and str(training_seed) != "unknown":
        try:
            return int(training_seed)
        except ValueError:
            return str(training_seed)
    return "?"


def _run_total_timesteps(run: dict[str, Any], raw: dict[str, Any] | None = None) -> int:
    meta = (raw or {}).get("meta", {}) or {}
    if meta.get("total_timesteps") is not None:
        return int(meta["total_timesteps"])
    return int(run.get("total_timesteps") or 0)


def _eval_box_label(run: dict[str, Any], raw: dict[str, Any]) -> str:
    algo = _group_label(run)
    seed = _run_training_seed(run, raw)
    steps = _run_total_timesteps(run, raw)
    steps_str = _format_timesteps(steps) if steps > 0 else "?"
    seed_line = f"seed={seed}"
    if len(run["summaries"]) > 1:
        eval_seed = (raw.get("meta", {}) or {}).get("eval_seed", "?")
        seed_line = f"seed={seed} (eval={eval_seed})"
    return f"{algo}\n{seed_line}\n{steps_str} steps"


def _eval_box_sort_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple:
    run, raw = item
    seed = _run_training_seed(run, raw)
    seed_key: int | str
    if isinstance(seed, int):
        seed_key = seed
    else:
        try:
            seed_key = int(seed)
        except ValueError:
            seed_key = str(seed)
    return (_group_label(run), _run_total_timesteps(run, raw), seed_key)


def _collect_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """실험 루트(들)에서 run 디렉토리를 수집.

    - ``experiments/2m`` 처럼 그룹 폴더면 하위 run 을 스캔
    - ``experiments/2m/2026-..._dqn_seed7`` 처럼 run 하나만 지정해도 동작
    """
    runs: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def _add(run_dir: Path) -> None:
        key = run_dir.resolve()
        if key in seen:
            return
        rec = _read_run(run_dir)
        if rec is not None:
            runs.append(rec)
            seen.add(key)

    for root in paths:
        root = root.resolve()
        if not root.exists():
            print(f"[plot] skip (not found): {root}")
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

    return runs


def plot_learning_curves(runs: list[dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        if r["timesteps"] is None:
            continue
        groups.setdefault(_group_label(r), []).append(r)

    if not groups:
        print("[plot] eval/evaluations.npz 가 어느 런에도 없습니다.")
        return

    for label, members in sorted(groups.items()):
        all_steps = sorted(
            {
                int(t)
                for r in members
                for t in np.asarray(r["timesteps"]).ravel().tolist()
            }
        )
        grid = np.array(all_steps)
        if len(grid) == 0:
            continue
        stacked = []
        for r in members:
            xs = np.asarray(r["timesteps"], dtype=np.float64)
            ys = np.asarray(r["ep_rew_mean"], dtype=np.float64)
            stacked.append(np.interp(grid, xs, ys))
        stacked_arr = np.stack(stacked, axis=0)
        mean = stacked_arr.mean(axis=0)
        std = stacked_arr.std(axis=0)
        plt.plot(grid, mean, label=f"{label} (n={len(members)})")
        plt.fill_between(grid, mean - std, mean + std, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Eval episode reward (mean ± std over seeds)")
    plt.title("Breakout v5 — Learning Curves")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_eval_distribution(runs: list[dict[str, Any]], out_path: Path) -> None:
    """``summary.json`` 의 에피소드 통계(mean/std/median/min/max/ci95)를 시각화."""
    raw_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for r in runs:
        for raw in r["summaries"]:
            raw_entries.append((r, raw))

    entries: list[tuple[str, dict[str, float | int]]] = []
    for run, raw in sorted(raw_entries, key=_eval_box_sort_key):
        try:
            stats = _parse_eval_summary(raw)
        except KeyError as exc:
            print(f"[plot] skip summary in {run['name']}: {exc}")
            continue
        entries.append((_eval_box_label(run, raw), stats))

    if not entries:
        print("[plot] eval_runs/*/summary*.json 이 없으면 분포 plot 을 건너뜁니다.")
        return

    labels = [label for label, _ in entries]
    x_centers = np.arange(len(labels), dtype=float)
    bar_half = 0.14

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.4), 6))

    for xi, (_, stats) in zip(x_centers, entries, strict=True):
        xi = float(xi)
        # min–max (episode rewards)
        ax.vlines(
            xi,
            stats["min"],
            stats["max"],
            colors="#333333",
            linewidth=1.8,
            zorder=2,
        )
        # 95% bootstrap CI of mean (from evaluate.py)
        ax.add_patch(
            Rectangle(
                (xi - bar_half, stats["ci95_low"]),
                2 * bar_half,
                stats["ci95_high"] - stats["ci95_low"],
                facecolor="#A8D4F0",
                edgecolor="#1F77B4",
                linewidth=1,
                alpha=0.55,
                zorder=3,
            )
        )
        # ±1 std around mean (episode reward spread)
        ax.errorbar(
            xi,
            stats["mean"],
            yerr=stats["std"],
            fmt="none",
            ecolor="#666666",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4,
        )
        # median
        ax.hlines(
            stats["median"],
            xi - bar_half * 1.1,
            xi + bar_half * 1.1,
            colors="#FF7F0E",
            linewidth=2.5,
            zorder=5,
        )
        # mean
        ax.scatter(
            [xi],
            [stats["mean"]],
            marker="^",
            s=72,
            c="#2CA02C",
            edgecolors="#2CA02C",
            zorder=6,
        )

    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#A8D4F0",
            edgecolor="#1F77B4",
            label="Blue band: 95% CI of mean (bootstrap)",
        ),
        Line2D([0], [0], color="#666666", linewidth=1.2, label="Gray caps: mean ± std"),
        Line2D([0], [0], color="#333333", linewidth=1.8, label="Vertical line: min–max"),
        Line2D([0], [0], color="#FF7F0E", linewidth=2.5, label="Orange line: median"),
        Line2D(
            [0],
            [0],
            marker="^",
            color="#2CA02C",
            linestyle="None",
            markersize=8,
            label="Green triangle: mean",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=7,
        frameon=True,
        framealpha=0.9,
    )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("Episode reward")
    ax.set_title(
        "Evaluation (100 episodes)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def write_runs_summary(runs: list[dict[str, Any]], out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for r in runs:
        timesteps = r["timesteps"]
        last_eval_mean = (
            float(r["ep_rew_mean"][-1]) if r["ep_rew_mean"] is not None else float("nan")
        )
        last_eval_std = (
            float(r["ep_rew_std"][-1]) if r["ep_rew_std"] is not None else float("nan")
        )
        last_step = (
            int(timesteps[-1]) if timesteps is not None and len(timesteps) > 0 else None
        )
        rows.append(
            {
                "run": r["name"],
                "algo": _group_label(r),
                "seed": r["seed"],
                "last_step": last_step,
                "last_eval_mean": last_eval_mean,
                "last_eval_std": last_eval_std,
                "n_final_eval_runs": len(r["summaries"]),
                "final_eval_mean_avg": (
                    float(np.mean([s["mean"] for s in r["summaries"]]))
                    if r["summaries"]
                    else float("nan")
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values(by=["algo", "seed", "run"], na_position="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[plot] saved {out_path}")


# ===========================================================================
# Extended multi-axis analysis suite (--analysis)
# ---------------------------------------------------------------------------
# These plots consume src/analysis.py (RunRecord + metrics) and cover the axes
# the legacy plots don't: score-distribution SHAPE, sample EFFICIENCY,
# training STABILITY, COMPUTE cost, hyperparameter RESPONSE curves, cross-SEED
# aggregates (IQM + bootstrap CI), and eval-level statistical SIGNIFICANCE.
# ===========================================================================

# Stable colour per base algorithm so the same algo looks the same everywhere.
_ALGO_COLORS = {
    "dqn": "#1F77B4", "dqn[double]": "#17BECF", "dqn[dueling]": "#9467BD",
    "dqn[double+dueling]": "#8C564B", "ppo": "#2CA02C", "a2c": "#FF7F0E",
}


def _algo_color(label: str) -> str:
    base = label.split("[")[0]
    for key in (label, base):
        if key in _ALGO_COLORS:
            return _ALGO_COLORS[key]
    return "#%06x" % (abs(hash(base)) % 0xFFFFFF)


def _rec_label(r: "A.RunRecord") -> str:
    return f"{r.label}\nseed={r.seed} · {A.format_timesteps(r.budget)}"


def _sorted_recs(records: list["A.RunRecord"]) -> list["A.RunRecord"]:
    return sorted(records, key=lambda r: (r.label, r.budget, r.seed if r.seed is not None else -1))


def plot_score_distributions(
    records: list["A.RunRecord"], out_path: Path, *, kind: str = "violin"
) -> None:
    """Per-config distribution of the 100 final-eval episode returns.

    A violin/box reveals SHAPE (skew, multimodality, the time-cap pile-up) that
    a single mean ± CI hides — directly answers "compare the box plots across
    configs", with full per-episode data instead of just summary stats.
    """
    data: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    for r in _sorted_recs(records):
        ep = A.load_episodes(r)
        if ep.empty:
            continue
        data.append(ep["reward"].to_numpy(dtype=np.float64))
        labels.append(_rec_label(r))
        colors.append(_algo_color(r.label))
    if not data:
        print("[plot] no episodes_*.csv found — skipping score distribution")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 1.5), 6))
    pos = np.arange(1, len(data) + 1)
    if kind == "violin":
        parts = ax.violinplot(data, positions=pos, showmeans=True, showmedians=True, widths=0.8)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.5)
    else:
        ax.boxplot(data, positions=pos, showmeans=True, widths=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Final-eval episode reward (100 episodes)")
    ax.set_title(f"Score distribution per config ({kind})")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_ecdf(records: list["A.RunRecord"], out_path: Path) -> None:
    """Empirical CDF of final-eval returns — read off P(return ≤ x) per config.

    ECDFs that cross indicate no clear dominance; an ECDF fully to the right of
    another means stochastic dominance (that config is better at every quantile).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    n = 0
    for r in _sorted_recs(records):
        ep = A.load_episodes(r)
        if ep.empty:
            continue
        x = np.sort(ep["reward"].to_numpy(dtype=np.float64))
        y = np.arange(1, x.size + 1) / x.size
        ax.step(x, y, where="post", label=f"{r.label} (s{r.seed}, {A.format_timesteps(r.budget)})",
                color=_algo_color(r.label), alpha=0.85)
        n += 1
    if n == 0:
        print("[plot] no episodes for ECDF")
        plt.close(fig)
        return
    ax.set_xlabel("Episode reward")
    ax.set_ylabel("Empirical CDF  P(reward ≤ x)")
    ax.set_title("Final-eval return ECDF")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_sample_efficiency(
    records: list["A.RunRecord"], out_path: Path, *, threshold: float = 50.0
) -> None:
    """Learning SPEED: area-under-eval-curve and steps-to-threshold.

    AUC = average eval reward over training (higher ⇒ learned more, sooner).
    steps-to-threshold = first env-step reaching mean eval ≥ ``threshold``.
    """
    rows = []
    for r in _sorted_recs(records):
        se = A.sample_efficiency(r, threshold=threshold)
        if np.isnan(se["auc_mean_reward"]):
            continue
        rows.append((f"{r.label}\n{A.format_timesteps(r.budget)} s{r.seed}", se, _algo_color(r.label)))
    if not rows:
        print("[plot] no evaluations.npz for sample-efficiency")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11, len(rows) * 1.4), 5))
    xs = np.arange(len(rows))
    ax1.bar(xs, [r[1]["auc_mean_reward"] for r in rows], color=[r[2] for r in rows])
    ax1.set_xticks(xs); ax1.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax1.set_ylabel("AUC (avg eval reward over training)")
    ax1.set_title("Sample efficiency — area under eval curve")
    ax1.grid(True, axis="y", alpha=0.3)

    s2t = [r[1]["steps_to_threshold"] for r in rows]
    ax2.bar(xs, [v / 1e6 if not np.isnan(v) else 0 for v in s2t], color=[r[2] for r in rows])
    for i, v in enumerate(s2t):
        if np.isnan(v):
            ax2.text(i, 0, "never", ha="center", va="bottom", fontsize=7, rotation=90)
    ax2.set_xticks(xs); ax2.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax2.set_ylabel("Million steps to reach threshold")
    ax2.set_title(f"Steps to mean eval ≥ {threshold:g}")
    ax2.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_compute_tradeoff(records: list["A.RunRecord"], out_path: Path) -> None:
    """Final score vs wall-clock training time — the compute/performance frontier.

    Surfaces the off-policy(DQN) vs on-policy(PPO/A2C) cost trade-off: who gets
    the most reward per training hour at a given budget.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0
    for r in _sorted_recs(records):
        comp = A.compute_stats(r)
        perf = A.final_performance(r).get("mean", float("nan"))
        if np.isnan(comp["wall_clock_hours"]) or np.isnan(perf):
            continue
        ax.scatter(comp["wall_clock_hours"], perf, s=80, color=_algo_color(r.label),
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(f"{r.label}\n{A.format_timesteps(r.budget)} s{r.seed}",
                    (comp["wall_clock_hours"], perf), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")
        plotted += 1
    if plotted == 0:
        print("[plot] no monitor logs for compute trade-off")
        plt.close(fig)
        return
    ax.set_xlabel("Wall-clock training time (hours)")
    ax.set_ylabel("Final-eval mean reward")
    ax.set_title("Compute vs performance")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_seed_aggregate(
    records: list["A.RunRecord"], out_path: Path, *, metric: str = "final_mean"
) -> None:
    """Cross-seed aggregate: mean + bootstrap CI, IQM where ≥3 seeds, seed dots.

    This is the statistically honest comparison: a bar is only trustworthy when
    ``n_seeds`` is large. Single-seed bars are drawn but flagged (n=1), so the
    reader sees that 10m/50m results lack training-seed replication.
    """
    agg = A.aggregate_by_config(records, metric=metric)
    if agg.empty:
        print("[plot] nothing to aggregate")
        return
    tbl = A.runs_table(records)
    agg = agg.sort_values(["budget", "mean"], ascending=[True, False]).reset_index(drop=True)
    xs = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(max(9, len(agg) * 1.3), 6))
    for x, (_, row) in zip(xs, agg.iterrows()):
        c = _algo_color(row["label"])
        err = [[row["mean"] - row["mean_ci_low"]], [row["mean_ci_high"] - row["mean"]]] \
            if not np.isnan(row["mean_ci_low"]) else None
        ax.bar(x, row["mean"], color=c, alpha=0.55, yerr=err, capsize=4)
        if not np.isnan(row["iqm"]):
            ax.scatter(x, row["iqm"], marker="D", s=45, color="k", zorder=5,
                       label="IQM" if x == 0 else None)
        # overlay individual seed scores
        sub = tbl[(tbl["label"] == row["label"]) & (tbl["budget"] == row["budget"])]
        ax.scatter(np.full(len(sub), x), sub[metric], color="k", s=18, alpha=0.7, zorder=6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['label']}\n{r['budget_str']} (n={r['n_seeds']})"
                        for _, r in agg.iterrows()], fontsize=7)
    ax.set_ylabel(f"{metric} (bar=mean±95%CI, ◆=IQM, dots=seeds)")
    ax.set_title("Cross-seed aggregate per config")
    ax.grid(True, axis="y", alpha=0.3)
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_, fontsize=8)
    _save(fig, out_path)


def plot_response_curves(records: list["A.RunRecord"], out_path: Path) -> None:
    """One panel per swept hyperparameter: final reward vs knob value.

    Auto-detects sweeps (≥2 distinct values for an (algo, knob) pair) and draws
    the response curve, with the baseline point included when present. Reveals
    'too low → baseline → too high' shapes that justify the chosen value.
    """
    found: list[tuple[str, str, pd.DataFrame]] = []
    algos = sorted({r.algo for r in records})
    for algo in algos:
        for knob in A._KNOB_TO_CFG:
            df = A.hyperparameter_response(records, algo, knob)
            if not df.empty and df["value"].nunique() >= 2:
                found.append((algo, knob, df))
    if not found:
        print("[plot] no hyperparameter sweeps detected")
        return
    ncol = min(3, len(found))
    nrow = int(np.ceil(len(found) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for ax, (algo, knob, df) in zip(axes.ravel(), found):
        g = df.groupby("value")["final_mean"].mean().reset_index()
        ax.plot(g["value"], g["final_mean"], "o-", color=_algo_color(algo))
        ax.scatter(df["value"], df["final_mean"], color=_algo_color(algo), alpha=0.5, s=25)
        ax.set_title(f"{algo}: {knob}")
        ax.set_xlabel(knob)
        ax.set_ylabel("final-eval mean")
        ax.grid(True, alpha=0.3)
    for ax in axes.ravel()[len(found):]:
        ax.set_visible(False)
    fig.suptitle("Hyperparameter response curves", y=1.0)
    _save(fig, out_path)


def plot_significance_heatmap(records: list["A.RunRecord"], out_path: Path) -> None:
    """P(A > B) over eval episodes, for distinct configs at the largest shared budget.

    Caveat (drawn in the title): this is EVALUATION-level variance over episodes
    of single models, NOT a training-seed test — read it as 'how separated are
    these two trained policies', not 'algorithm A beats B in general'.
    """
    recs = A.dedup_latest(records)
    by_budget: dict[int, list[A.RunRecord]] = {}
    for r in recs:
        by_budget.setdefault(r.budget, []).append(r)
    # pick the budget with the most distinct labels (most informative matrix)
    budget = max(by_budget, key=lambda b: len({r.label for r in by_budget[b]}), default=None)
    if budget is None:
        print("[plot] nothing for significance heatmap")
        return
    chosen: dict[str, A.RunRecord] = {}
    for r in by_budget[budget]:
        if r.label not in chosen:  # one representative run per label
            chosen[r.label] = r
    labels = sorted(chosen)
    if len(labels) < 2:
        print("[plot] need ≥2 configs at a shared budget for significance heatmap")
        return
    mat = np.full((len(labels), len(labels)), np.nan)
    for i, la in enumerate(labels):
        for j, lb in enumerate(labels):
            if i == j:
                mat[i, j] = 0.5
                continue
            res = A.pairwise_significance(chosen[la], chosen[lb])
            if "prob_a_gt_b" in res:
                mat[i, j] = res["prob_a_gt_b"]
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.4 * len(labels) + 1))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(row > col) per episode")
    ax.set_title(f"Eval-level dominance @ {A.format_timesteps(int(budget))}\n(episode variance, NOT seed variance)")
    _save(fig, out_path)


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def generate_analysis_suite(
    experiment_paths: list[Path],
    figures_dir: Path,
    tables_dir: Path,
    slug: str,
    *,
    training_seed: int | None = None,
    threshold: float = 50.0,
) -> None:
    """Run the full extended analysis suite and write all figures + the master table."""
    records = A.dedup_latest(A.discover_runs(experiment_paths))
    if training_seed is not None:
        records = [r for r in records if r.seed == training_seed]
    if not records:
        print("[plot] analysis: no runs found")
        return
    print(f"[plot] analysis: {len(records)} runs")

    def _p(name: str) -> Path:
        return _slugged_path(figures_dir / f"analysis_{name}.png", slug)

    plot_score_distributions(records, _p("score_violin"), kind="violin")
    plot_ecdf(records, _p("score_ecdf"))
    plot_sample_efficiency(records, _p("sample_efficiency"), threshold=threshold)
    plot_compute_tradeoff(records, _p("compute_tradeoff"))
    plot_seed_aggregate(records, _p("seed_aggregate"))
    plot_response_curves(records, _p("response_curves"))
    plot_significance_heatmap(records, _p("significance"))

    # Master per-run metrics table (every criterion in one CSV).
    table = A.runs_table(records, threshold=threshold)
    out_csv = _slugged_path(tables_dir / "analysis_metrics.csv", slug)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"[plot] saved {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate experiment runs into figures & tables "
            "(filename suffix = experiments/<group> name)."
        )
    )
    p.add_argument(
        "--experiments",
        nargs="+",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "집계할 실험 디렉토리(복수 가능). "
            "예: experiments/2m experiments/1m 또는 단일 run 경로. "
            "미지정 시 experiments/ 전체를 스캔"
        ),
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports",
        help="출력 루트 (figures/, tables/ 하위에 저장)",
    )
    p.add_argument(
        "--slug",
        type=str,
        default=None,
        help=(
            "출력 파일명 접미사 (기본: --experiments 경로의 experiments/ 하위 "
            "디렉터리명, 복수면 정렬 후 '_' 로 연결)"
        ),
    )
    p.add_argument(
        "--training-seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "학습 시드가 N 인 run 만 포함 (run 폴더명의 seedN). "
            "지정 시 slug 에 _seedN 이 자동 추가 (--slug 가 없을 때)"
        ),
    )
    p.add_argument(
        "--analysis",
        action="store_true",
        help=(
            "추가 다축 분석 figure/table 생성 (violin/ECDF, sample-efficiency, "
            "compute trade-off, cross-seed aggregate, hyperparameter response "
            "curves, significance heatmap). src/analysis.py 사용."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        metavar="R",
        help="sample-efficiency 의 steps-to-threshold 기준 보상값 (기본 50)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.experiments:
        experiment_paths = [p.resolve() for p in args.experiments]
    else:
        experiment_paths = [ROOT / "experiments"]

    reports = args.reports_dir.resolve()
    figures_dir = reports / "figures"
    tables_dir = reports / "tables"

    runs = _collect_runs(experiment_paths)
    if not runs:
        print(f"[plot] no runs under {experiment_paths}")
        return

    if args.training_seed is not None:
        before = len(runs)
        runs = [r for r in runs if r["seed"] == args.training_seed]
        print(
            f"[plot] training-seed={args.training_seed}: "
            f"{len(runs)}/{before} runs kept"
        )
        if not runs:
            print("[plot] no runs left after --training-seed filter")
            return

    slug = _slugify(args.slug) if args.slug else _output_slug(experiment_paths, runs)
    if args.training_seed is not None and args.slug is None:
        slug = f"{slug}_seed{args.training_seed}"

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = f"{slug}_{ts}"

    print(f"[plot] slug        = {slug}")
    print(f"[plot] experiments = {', '.join(str(p) for p in experiment_paths)}")
    print(f"[plot] runs found  = {len(runs)}")

    plot_learning_curves(
        runs, _slugged_path(figures_dir / "learning_curves.png", slug)
    )
    plot_eval_distribution(
        runs, _slugged_path(figures_dir / "eval_distribution.png", slug)
    )
    write_runs_summary(runs, _slugged_path(tables_dir / "runs_summary.csv", slug))

    if args.analysis:
        generate_analysis_suite(
            experiment_paths,
            figures_dir,
            tables_dir,
            slug,
            training_seed=args.training_seed,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
