#!/usr/bin/env python3
"""Breakout v5 — 학습된 정책 평가."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algos.registry import load_model
from src.envs import build_eval_env
from src.eval import EvalSummary, evaluate_model, format_summary
from src.utils.config import load_config
from src.utils.logging import write_episode_csv

SUMMARY_JSON = "summary.json"
EPISODES_CSV = "episodes.csv"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained agent on ALE/Breakout-v5.")
    p.add_argument("--run", type=Path, default=None, help="실험 run 디렉토리 (단일) 또는 하위 run 을 담은 그룹 폴더")
    p.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "여러 실험 디렉토리를 한 번에 평가. 디렉토리 경로나 글로브 패턴을 "
            "공백으로 나열 (예: --runs experiments/*ppo* experiments/*dqn*). "
            "각 run 별 결과 저장 + 평균보상 기준 leaderboard 와 비교 CSV 출력."
        ),
    )
    p.add_argument("--model", type=Path, default=None, help="모델 .zip")
    p.add_argument("--config", type=Path, default=None, help="config yaml")
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0, help="평가 env 시드")
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="결과 저장 디렉토리 (기본: <run>/eval_runs/seed{eval_seed})",
    )
    return p.parse_args()


def _algo_slug(cfg: dict) -> str:
    name = str(cfg["algo"]["name"]).lower()
    feats = cfg["algo"].get("features", {}) or {}
    if name == "dqn":
        if feats.get("dueling"):
            return "dueling_dqn"
        if feats.get("double_q"):
            return "ddqn"
        return "dqn"
    return name


def _training_seed(run_dir: Path | None) -> int | str:
    if run_dir is None:
        return "unknown"
    for token in run_dir.name.split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError:
                return token
    return "unknown"


def _eval_meta(
    cfg: dict,
    *,
    run_dir: Path | None,
    model_path: Path,
    eval_seed: int,
    n_eval_episodes: int,
    deterministic: bool,
) -> dict:
    return {
        "algo": _algo_slug(cfg),
        "training_seed": str(_training_seed(run_dir)),
        "total_timesteps": int(cfg.get("train", {}).get("total_timesteps", 0)),
        "eval_seed": eval_seed,
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
        "model_path": str(model_path),
    }


def _save_eval_artifacts(
    out_dir: Path,
    summary,
    cfg: dict,
    *,
    run_dir: Path | None,
    model_path: Path,
    eval_seed: int,
    n_eval_episodes: int,
    deterministic: bool,
) -> tuple[Path, Path]:
    json_path = out_dir / SUMMARY_JSON
    csv_path = out_dir / EPISODES_CSV
    meta = _eval_meta(
        cfg,
        run_dir=run_dir,
        model_path=model_path,
        eval_seed=eval_seed,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )
    payload = summary.to_dict(drop_lists=True)
    payload["meta"] = meta
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_episode_csv(
        csv_path,
        rewards=summary.rewards,
        lengths=summary.lengths,
        meta={k: str(v) for k, v in meta.items()},
    )
    return json_path, csv_path


def _resolve_run_artifacts(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None]:
    if args.run is not None:
        run = args.run.resolve()
        cfg_path = run / "config.yaml"
        cand = [run / "best_model" / "best_model.zip", run / "final_model.zip"]
        model_path = next((p for p in cand if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError(f"No best/final model in {run}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.yaml in {run}")
        return model_path, cfg_path, run
    if args.model is None or args.config is None:
        raise SystemExit("--run 또는 --model + --config 둘 중 하나는 지정해야 합니다.")
    return args.model.resolve(), args.config.resolve(), None


def evaluate_run(
    run_dir: Path,
    *,
    n_eval_episodes: int = 50,
    eval_seed: int = 0,
    deterministic: bool = True,
    device: str = "auto",
) -> EvalSummary:
    """완료된 학습 run 을 평가하고 ``eval_runs/seed<N>/`` 에 summary/CSV 저장.

    ``train.py`` 의 ``--evaluate-after-train`` 경로와 ``evaluate.py`` 의
    ``--run`` / ``--runs`` 경로가 공유한다. 산출된 ``EvalSummary`` 를 반환해
    다중 run 비교(leaderboard)에서 재사용한다.
    """
    run_dir = run_dir.resolve()
    cfg_path = run_dir / "config.yaml"
    cand = [run_dir / "best_model" / "best_model.zip", run_dir / "final_model.zip"]
    model_path = next((p for p in cand if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"No best/final model in {run_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml in {run_dir}")

    cfg = load_config(cfg_path)
    out_dir = run_dir / "eval_runs" / f"seed{eval_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model  = {model_path}")
    print(f"[eval] config = {cfg_path}")

    eval_env = build_eval_env(
        cfg["env"], cfg.get("eval_env", {}), seed=eval_seed + 10_000
    )
    model = load_model(cfg, str(model_path), env=eval_env, device=device)

    summary = evaluate_model(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        seed=eval_seed,
    )
    eval_env.close()

    print(format_summary(summary, name=cfg["algo"]["name"]))

    json_path, csv_path = _save_eval_artifacts(
        out_dir,
        summary,
        cfg,
        run_dir=run_dir,
        model_path=model_path,
        eval_seed=eval_seed,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )
    print(f"[eval] saved  = {json_path}")
    print(f"[eval] saved  = {csv_path}")
    return summary


def _is_evaluable_run(path: Path) -> bool:
    return (path / "config.yaml").exists() and (
        (path / "best_model" / "best_model.zip").exists()
        or (path / "final_model.zip").exists()
    )


def _scan_run_dirs(root: Path) -> list[Path]:
    """``root`` 자체 또는 하위(최대 2 depth)에서 평가 가능한 run 을 수집."""
    root = root.resolve()
    if _is_evaluable_run(root):
        return [root]

    found: list[Path] = []
    if not root.is_dir():
        return found

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if _is_evaluable_run(child):
            found.append(child.resolve())
            continue
        if (child / "config.yaml").exists():
            print(f"[eval] skip {child.name}: config.yaml/model 누락")
            continue
        for sub in sorted(child.iterdir()):
            if not sub.is_dir():
                continue
            if _is_evaluable_run(sub):
                found.append(sub.resolve())
            elif (sub / "config.yaml").exists():
                print(f"[eval] skip {sub.name}: config.yaml/model 누락")
    return found


def _expand_run_paths(patterns: list[str]) -> list[Path]:
    """디렉토리/글로브 패턴 목록을 평가 가능한 run 디렉토리 리스트로 확장.

    - 단일 run 디렉토리, ``experiments/for_eval/`` 같은 그룹 폴더, 글로브 패턴 지원.
    - ``config.yaml`` 과 모델(zip)이 모두 있는 디렉토리만 통과, 중복은 제거.
    """
    seen: dict[Path, None] = {}
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            matches = _scan_run_dirs(p)
            if not matches:
                print(f"[eval] warning: no evaluable run under {p}")
        else:
            matches = [Path(m) for m in sorted(glob.glob(pat))]
            if not matches:
                print(f"[eval] warning: no match for {pat!r}")
        for m in matches:
            mr = m.resolve()
            if mr.is_dir() and _is_evaluable_run(mr):
                seen.setdefault(mr, None)
            elif mr.is_dir():
                print(f"[eval] skip {mr.name}: config.yaml/model 누락")
    return list(seen.keys())


def _print_leaderboard(results: list[tuple[str, EvalSummary]]) -> None:
    """평균 보상 내림차순 leaderboard 를 stdout 에 출력."""
    if not results:
        print("[eval] 평가된 run 이 없습니다.")
        return
    ranked = sorted(results, key=lambda kv: kv[1].mean, reverse=True)
    name_w = max(len(n) for n, _ in ranked)
    header = (
        f"{'#':<4}{'run':<{name_w}}  {'mean':>8}  {'std':>7}  "
        f"{'median':>7}  {'min':>5}  {'max':>6}  {'95% CI':>17}"
    )
    print("\n===== leaderboard (mean reward ↓) =====")
    print(header)
    print("-" * len(header))
    for i, (name, s) in enumerate(ranked, 1):
        print(
            f"{i:<4}{name:<{name_w}}  {s.mean:>8.2f}  {s.std:>7.2f}  "
            f"{s.median:>7.1f}  {s.min:>5.1f}  {s.max:>6.1f}  "
            f"[{s.ci95_low:>6.2f}, {s.ci95_high:>6.2f}]"
        )


def _write_compare_csv(path: Path, results: list[tuple[str, EvalSummary]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda kv: kv[1].mean, reverse=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["rank", "run", "n_episodes", "mean", "std", "median",
             "min", "max", "ci95_low", "ci95_high"]
        )
        for rank, (name, s) in enumerate(ranked, 1):
            w.writerow(
                [rank, name, s.n_episodes, f"{s.mean:.4f}", f"{s.std:.4f}",
                 f"{s.median:.4f}", f"{s.min:.4f}", f"{s.max:.4f}",
                 f"{s.ci95_low:.4f}", f"{s.ci95_high:.4f}"]
            )


def evaluate_runs(
    run_dirs: list[Path],
    *,
    n_eval_episodes: int,
    eval_seed: int,
    deterministic: bool,
    device: str,
    compare_csv: Path,
) -> None:
    """여러 run 을 순차 평가하고 leaderboard + 비교 CSV 를 생성."""
    results: list[tuple[str, EvalSummary]] = []
    for rd in run_dirs:
        print("=" * 70)
        try:
            summary = evaluate_run(
                rd,
                n_eval_episodes=n_eval_episodes,
                eval_seed=eval_seed,
                deterministic=deterministic,
                device=device,
            )
        except Exception as e:  # noqa: BLE001 — 한 run 실패가 전체를 막지 않게
            print(f"[eval] FAILED {rd.name}: {type(e).__name__}: {e}")
            continue
        results.append((rd.name, summary))

    print("=" * 70)
    _print_leaderboard(results)
    if results:
        _write_compare_csv(compare_csv, results)
        print(f"\n[eval] comparison table = {compare_csv}")
    print(f"[eval] evaluated {len(results)}/{len(run_dirs)} runs")


def _compare_csv_path(args: argparse.Namespace, *, label: str | None = None) -> Path:
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = "comparison.csv" if label is None else f"comparison_{label}.csv"
        return args.output_dir / name
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    return ROOT / "reports" / "tables" / f"eval_compare{suffix}_{ts}.csv"


def main() -> None:
    args = parse_args()

    batch_patterns: list[str] | None = None
    batch_label: str | None = None
    if args.runs:
        batch_patterns = args.runs
    elif args.run is not None:
        run = args.run.resolve()
        if not _is_evaluable_run(run):
            batch_patterns = [str(run)]
            batch_label = _slugify(run.name)

    if batch_patterns:
        run_dirs = _expand_run_paths(batch_patterns)
        if not run_dirs:
            raise SystemExit("[eval] 평가 가능한 run 디렉토리를 찾지 못했습니다.")
        compare_csv = _compare_csv_path(args, label=batch_label)
        print(f"[eval] evaluating {len(run_dirs)} runs "
              f"(n_eval_episodes={args.n_eval_episodes}, seed={args.seed})")
        evaluate_runs(
            run_dirs,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed=args.seed,
            deterministic=args.deterministic,
            device=args.device,
            compare_csv=compare_csv,
        )
        return

    model_path, cfg_path, run_dir = _resolve_run_artifacts(args)
    cfg = load_config(cfg_path)

    out_dir = args.output_dir
    if out_dir is None and args.run is not None:
        out_dir = args.run / "eval_runs" / f"seed{args.seed}"
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model  = {model_path}")
    print(f"[eval] config = {cfg_path}")

    eval_env = build_eval_env(
        cfg["env"], cfg.get("eval_env", {}), seed=args.seed + 10_000
    )
    model = load_model(cfg, str(model_path), env=eval_env, device=args.device)

    summary = evaluate_model(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic,
        seed=args.seed,
    )
    eval_env.close()

    print(format_summary(summary, name=cfg["algo"]["name"]))

    if out_dir is not None:
        json_path, csv_path = _save_eval_artifacts(
            out_dir,
            summary,
            cfg,
            run_dir=run_dir,
            model_path=model_path,
            eval_seed=args.seed,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=args.deterministic,
        )
        print(f"[eval] saved  = {json_path}")
        print(f"[eval] saved  = {csv_path}")


if __name__ == "__main__":
    main()
