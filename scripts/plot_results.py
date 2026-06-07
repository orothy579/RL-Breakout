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


if __name__ == "__main__":
    main()
