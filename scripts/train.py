#!/usr/bin/env python3
"""Breakout v5 — training entry point.

External library: Stable-Baselines3 provides ``model.learn`` plus the
``EvalCallback`` / ``CheckpointCallback`` used here.

MY ORIGINAL CONTRIBUTION: the experiment harness wrapped around SB3 —
  * deterministic, timestamped, per-seed run directories with a frozen
    config.yaml snapshot (reproducibility);
  * EvalCallback on a SEPARATE eval env (seed offset +1000) that auto-saves
    best_model by true-episode score, decoupled from the training proxy reward;
  * resume-from-checkpoint logic (`--run`) and directory-batch sweeps
    (`--config <dir>`) for running whole ablation suites in one command;
  * optional post-training 100-episode evaluation.

Usage:
    conda activate breakout
    python scripts/train.py --config configs/dqn_baseline.yaml --seed 0
    python scripts/train.py --config configs/ddqn.yaml --seed 1 --name ddqn
    python scripts/train.py --config configs/ppo.yaml --seed 0
    python scripts/train.py --config configs/ablations --seed 7   # batch sweep
    python scripts/train.py --run experiments/2026-06-02_170457_dqn_baseline_seed7
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.algos.registry import build_model, load_model
from src.envs import build_eval_env, build_train_env
from src.utils.config import load_config, save_config
from src.utils.seeding import set_global_seed

_CKPT_STEPS_RE = re.compile(r"_(\d+)_steps$")


def _training_seed(run_dir: Path) -> int:
    for token in run_dir.name.split("_"):
        if token.startswith("seed"):
            try:
                return int(token.replace("seed", ""))
            except ValueError as exc:
                raise ValueError(
                    f"run 디렉토리 이름에서 seed를 파싱할 수 없습니다: {run_dir.name}"
                ) from exc
    raise ValueError(
        f"run 디렉토리 이름에 seed<N> 토큰이 없습니다: {run_dir.name}"
    )


def _checkpoint_steps(path: Path) -> int:
    m = _CKPT_STEPS_RE.search(path.stem)
    return int(m.group(1)) if m else -1


def _resolve_resume_checkpoint(run_dir: Path, algo_name: str) -> Path:
    """이어 학습용 체크포인트: periodic > final > best."""
    ckpt_dir = run_dir / "checkpoints"
    periodic = sorted(
        ckpt_dir.glob(f"{algo_name}_*_steps.zip"),
        key=_checkpoint_steps,
    )
    if periodic:
        return periodic[-1]

    final = run_dir / "final_model.zip"
    if final.exists():
        return final

    best = run_dir / "best_model" / "best_model.zip"
    if best.exists():
        return best

    raise FileNotFoundError(
        f"{run_dir} 에서 재개할 모델을 찾지 못했습니다 "
        f"(checkpoints/{algo_name}_*_steps.zip, final_model.zip, best_model/best_model.zip)."
    )


def _resolve_config_paths(config: Path) -> list[Path]:
    """단일 YAML 파일 또는 디렉터리 안의 모든 *.yaml (이름순)."""
    path = config.resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        configs = sorted(path.glob("*.yaml"))
        if not configs:
            raise SystemExit(f"[train] {path} 에 *.yaml 설정이 없습니다.")
        return configs
    raise SystemExit(f"[train] config 경로가 없습니다: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an RL agent on ALE/Breakout-v5 from a YAML config.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 설정 파일 또는 설정 디렉터리 (디렉터리면 *.yaml 전부 순차 학습)",
    )
    p.add_argument(
        "--run",
        type=Path,
        default=None,
        help="기존 실험 디렉토리에서 이어 학습 (config.yaml·checkpoints 재사용)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="실험 디렉토리 라벨 (단일 config 전용; 디렉터리 배치 시 각 yaml stem 사용)",
    )
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
    args = p.parse_args()
    if args.run is not None and args.config is not None:
        p.error("--run 과 --config 는 동시에 지정할 수 없습니다.")
    if args.run is None and args.config is None:
        p.error("--config (신규 학습) 또는 --run (이어 학습) 중 하나만 지정하세요.")
    return args


def _train_new(
    args: argparse.Namespace,
    *,
    config_path: Path,
    run_label: str,
) -> Path:
    cfg = load_config(config_path)
    seed = args.seed
    set_global_seed(seed)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.experiments_root / f"{timestamp}_{run_label}_seed{seed}"
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
    algo_name = str(cfg["algo"]["name"]).lower()

    print(f"[train] config  = {config_path}")
    print(f"[train] run_dir = {run_dir}")
    print(f"[train] algo    = {algo_name} (features={cfg['algo'].get('features', {})})")
    print(f"[train] env     = {env_cfg['env_id']} (n_envs={n_envs}, frame_stack={env_cfg.get('frame_stack')})")
    print(f"[train] seed    = {seed}")
    print(f"[train] target  = {total_timesteps:,} timesteps")

    train_env = build_train_env(env_cfg, seed=seed, monitor_dir=monitor_dir)
    # Eval env uses a DIFFERENT seed (seed+1000) so "best model" is selected on
    # episodes the agent didn't train on -> a cleaner generalization signal.
    eval_env = build_eval_env(env_cfg, eval_env_cfg, seed=seed + 1000)

    model = build_model(
        cfg,
        env=train_env,
        tensorboard_log=str(tb_dir),
        seed=seed,
        device=args.device,
    )

    # SB3 callbacks count *calls* (one per vec-env step), not env steps, so we
    # divide the desired env-step frequency by n_envs to get the real period.
    periodic_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=algo_name,
    )
    # EvalCallback periodically evaluates on the held-out eval_env and saves the
    # best-scoring policy. deterministic=True -> greedy policy, reproducible score.
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
            tb_log_name=algo_name,
            reset_num_timesteps=True,
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

    return run_dir


def _train_resume(args: argparse.Namespace) -> Path | None:
    run_dir = args.run.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"[train] run 디렉토리가 없습니다: {run_dir}")
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"[train] {cfg_path} 가 없습니다.")

    cfg = load_config(cfg_path)
    seed = _training_seed(run_dir)
    ckpt_path = _resolve_resume_checkpoint(run_dir, str(cfg["algo"]["name"]).lower())
    set_global_seed(seed)

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
    algo_name = str(cfg["algo"]["name"]).lower()

    print(f"[train] run_dir = {run_dir}")
    print(f"[train] algo    = {algo_name} (features={cfg['algo'].get('features', {})})")
    print(f"[train] env     = {env_cfg['env_id']} (n_envs={n_envs}, frame_stack={env_cfg.get('frame_stack')})")
    print(f"[train] seed    = {seed}")
    print(f"[train] target  = {total_timesteps:,} timesteps")

    train_env = build_train_env(env_cfg, seed=seed, monitor_dir=monitor_dir)
    eval_env = build_eval_env(env_cfg, eval_env_cfg, seed=seed + 1000)

    print(f"[train] resume  = {ckpt_path}")
    model = load_model(cfg, str(ckpt_path), env=train_env, device=args.device)
    done = int(model.num_timesteps)
    remaining = total_timesteps - done
    print(f"[train] progress= {done:,} / {total_timesteps:,} ({remaining:,} remaining)")
    if algo_name == "dqn":
        print(
            "[train] note    : periodic checkpoint에는 replay buffer가 없습니다. "
            "재개 후 buffer를 다시 채웁니다 (learning_starts까지 수집만)."
        )
    if remaining <= 0:
        train_env.close()
        eval_env.close()
        print("[train] 이미 목표 timesteps에 도달했습니다. 학습을 건너뜁니다.")
        return None

    periodic_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=algo_name,
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
            total_timesteps=remaining,
            callback=[periodic_cb, eval_cb],
            progress_bar=not args.no_progress_bar,
            tb_log_name=algo_name,
            reset_num_timesteps=False,
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

    return run_dir


def main() -> None:
    args = parse_args()

    if args.run is not None:
        _train_resume(args)
        return

    config_paths = _resolve_config_paths(args.config)
    batch = len(config_paths) > 1
    if batch and args.name is not None:
        print("[train] warning: --name 은 디렉터리 배치 시 무시됩니다 (각 yaml stem 사용).")

    run_dirs: list[Path] = []
    for i, config_path in enumerate(config_paths, start=1):
        if batch:
            print(f"\n[train] ===== batch {i}/{len(config_paths)}: {config_path.name} =====")
        run_label = config_path.stem if batch else (args.name or config_path.stem)
        run_dirs.append(_train_new(args, config_path=config_path, run_label=run_label))

    if batch:
        print(f"\n[train] batch 완료: {len(run_dirs)} runs")
        for rd in run_dirs:
            print(f"  - {rd}")


if __name__ == "__main__":
    main()
