#!/usr/bin/env python3
"""Breakout v5 — 학습 엔트리포인트.

사용 예:
    conda activate breakout
    python scripts/train.py --config configs/dqn_baseline.yaml --seed 0
    python scripts/train.py --config configs/ddqn.yaml --seed 1 --name ddqn
    python scripts/train.py --config configs/ppo.yaml --seed 0
    python scripts/train.py --config configs/ablations/framestack_1.yaml --seed 0 --total-timesteps 500000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.algos.registry import build_model
from src.envs import build_eval_env, build_train_env
from src.utils.config import load_config, save_config
from src.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an RL agent on ALE/Breakout-v5 from a YAML config.")
    p.add_argument("--config", type=Path, required=True, help="YAML 설정 파일 경로")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default=None, help="실험 디렉토리에 들어갈 라벨(없으면 config 파일명)")
    p.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="config 의 train.total_timesteps 를 override (스모크 테스트용)",
    )
    p.add_argument(
        "--experiments-root",
        type=Path,
        default=ROOT / "experiments",
        help="실험 결과 루트 디렉토리",
    )
    p.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="rich 진행바 비활성화 (CI/원격 로그용)",
    )
    p.add_argument(
        "--evaluate-after-train",
        action="store_true",
        help="학습 완료 후 자동으로 100-episode 평가 실행 (best_model 우선)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_global_seed(args.seed)

    name = args.name or args.config.stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.experiments_root / f"{timestamp}_{name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg.setdefault("train", {})
    total_timesteps = int(
        args.total_timesteps or train_cfg.get("total_timesteps", 2_000_000)
    )
    train_cfg["total_timesteps"] = total_timesteps
    save_config(cfg, run_dir / "config.yaml")

    tb_dir = run_dir / "tensorboard"
    ckpt_dir = run_dir / "checkpoints"
    monitor_dir = run_dir / "monitor_train"
    eval_dir = run_dir / "eval"
    best_model_dir = run_dir / "best_model"
    for d in (tb_dir, ckpt_dir, monitor_dir, eval_dir, best_model_dir):
        d.mkdir(parents=True, exist_ok=True)

    checkpoint_freq = int(train_cfg.get("checkpoint_freq", 200_000))
    eval_freq = int(train_cfg.get("eval_freq", 50_000))
    n_eval_episodes = int(train_cfg.get("n_eval_episodes", 5))

    env_cfg = cfg["env"]
    eval_env_cfg = cfg.get("eval_env", {})
    n_envs = int(env_cfg.get("n_envs", 8))

    print(f"[train] run_dir = {run_dir}")
    print(f"[train] algo    = {cfg['algo']['name']} (features={cfg['algo'].get('features', {})})")
    print(f"[train] env     = {env_cfg['env_id']} (n_envs={n_envs}, frame_stack={env_cfg.get('frame_stack')})")
    print(f"[train] steps   = {total_timesteps:,}")

    train_env = build_train_env(env_cfg, seed=args.seed, monitor_dir=monitor_dir)
    eval_env = build_eval_env(env_cfg, eval_env_cfg, seed=args.seed + 1000)

    model = build_model(
        cfg,
        env=train_env,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
        device=args.device,
    )

    # 체크포인트는 환경 step 기준으로 환산
    periodic_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=cfg["algo"]["name"],
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_dir),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[periodic_cb, eval_cb],
            progress_bar=not args.no_progress_bar,
            tb_log_name=cfg["algo"]["name"],
        )
    finally:
        final_path = run_dir / "final_model.zip"
        model.save(str(final_path))
        train_env.close()
        eval_env.close()
        print(f"[train] saved   : {final_path}")
        print(f"[train] best at : {best_model_dir / 'best_model.zip'}")
        print(f"[train] tensorboard: tensorboard --logdir {tb_dir}")

    if args.evaluate_after_train:
        from scripts.evaluate import evaluate_run

        print("[train] running post-train evaluation (n_eval_episodes=100)...")
        evaluate_run(
            run_dir,
            n_eval_episodes=100,
            eval_seed=0,
            deterministic=True,
            device=args.device,
        )


if __name__ == "__main__":
    main()
