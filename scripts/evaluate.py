#!/usr/bin/env python3
"""Breakout v5 — 학습된 정책 평가."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algos.registry import load_model
from src.envs import build_eval_env
from src.eval import evaluate_model, format_summary
from src.utils.config import load_config
from src.utils.logging import write_episode_csv

SUMMARY_JSON = "summary.json"
EPISODES_CSV = "episodes.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained agent on ALE/Breakout-v5.")
    p.add_argument("--run", type=Path, default=None, help="실험 디렉토리")
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
) -> None:
    """완료된 학습 run 을 평가하고 ``eval_runs/seed<N>/`` 에 summary/CSV 저장.

    ``train.py`` 의 ``--evaluate-after-train`` 경로와 ``evaluate.py`` 의
    ``--run`` 경로가 공유한다.
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


def main() -> None:
    args = parse_args()
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
