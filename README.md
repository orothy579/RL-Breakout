# Breakout v5 Reinforcement Learning — Final Project

Train and evaluate five deep reinforcement learning algorithms on the
`ALE/Breakout-v5` environment and analyze performance differences through RL
theory. The goal is **rigorous experimentation, fair comparison, and
theory-grounded analysis** — not just the single highest score.

> **Full analysis:** See [`reports/apa_paper.md`](reports/apa_paper.md) for the
> complete APA-format paper covering algorithm comparison, cross-seed analysis,
> hyperparameter ablation, and statistical significance tests.
>
> **Deeper data access:** Raw results, analysis logic, and all figure/table
> generation code live in [`src/analysis.py`](src/analysis.py) and
> [`scripts/plot_results.py`](scripts/plot_results.py) — every number in the
> paper is fully reproducible from the experiment artifacts.

---

## Key Results

### Algorithm Ranking (10M steps, cross-seed average, seeds 7/77/777)

| Algorithm | Cross-seed Mean | Seed 7 | Seed 77 | Seed 777 | Seed Std |
|---|---|---|---|---|---|
| **A2C** | **207.4** | 386 | 113 | 123 | 126.4 |
| Dueling DQN | 126.5 | 269 | 74 | 37 | 101.7 |
| PPO | 102.8 | 262 | 24 | 23 | 112.4 |
| DQN | 55.0 | 76 | 67 | 22 | 23.5 |
| Double DQN | 49.8 | 27 | 33 | 90 | 28.7 |

### Budget Scaling — Cross-seed Mean

| Algorithm | 1M | 10M | 50M |
|---|---|---|---|
| A2C | 76.3 | 207.4 | 378.3 |
| PPO | 18.7 | 102.8 | 338.1 |
| DQN | 5.9 | 55.0 | **368.7** |

At 50M steps, off-policy DQN (369 pts) converges to A2C (378 pts) — the
short-budget on-policy advantage is not permanent.

### Most Dramatic Ablation Results

| Experiment | Change | Before → After |
|---|---|---|
| DQN learning rate | 1.0e-4 → 1.5e-4 | 75.7 → 191.2 pts (+153%) |
| A2C normalize_advantage | False → True | 386 → 10.5 pts (−97%) |
| DQN target_update_interval | 1,000 → 5,000 | 75.7 → 163.3 pts (+116%) |
| PPO clip_range | 0.1 → 0.3 | 261.8 → 361.7 pts (+38%) |

---

## Result Figures

All figures are in [`reports/figures/`](reports/figures/). Key plots:

| Figure | File | Description |
|---|---|---|
| Cross-seed comparison | `figures/notebook/seed_aggregate.png` | Bar chart with 95% CI + IQM across seeds |
| Learning curves (10M) | `figures/notebook/learning_curves_10m.png` | Per-algorithm training progress |
| Learning curves (50M) | `figures/notebook/learning_curves_50m.png` | Long-budget convergence |
| Sample efficiency | `figures/notebook/sample_efficiency.png` | AUC + steps-to-threshold |
| ECDF (10M) | `figures/notebook/ecdf_10m.png` | Full distribution comparison |
| Significance heatmap | `figures/notebook/significance.png` | P(A > B) pairwise matrix |
| Violin plots | `figures/notebook/violin_10m.png` | Score distribution per algorithm |
| Hyperparameter response | `figures/notebook/response_curves.png` | All ablation knobs at a glance |

---

## Project Constraints (Mandatory Baseline)

For evaluation fairness, the assignment fixes the following package versions:

```bash
gymnasium[atari]==1.3.0
ale-py==0.11.2
autorom[accept-rom-license]==0.6.1
```

Stable-Baselines3 on PyPI (2.8.0) pins `gymnasium<1.3.0`, which conflicts with
the required `gymnasium==1.3.0`. This project installs SB3 from the GitHub
`master` branch (cap relaxed to `gymnasium<2.0`); see
[requirements.txt](requirements.txt).

---

## Original Contributions vs. Stable-Baselines3

| Contribution | Where | What was modified / added |
|---|---|---|
| **Double DQN target** | [src/algos/dqn.py](src/algos/dqn.py) | Subclass `DoubleDQN` overrides SB3's `DQN.train`; decouples action *selection* (online net) from *evaluation* (target net) |
| **Dueling Q-head** | [src/algos/dueling.py](src/algos/dueling.py) | Replaces SB3's single-stream Q-head with `V(s)` + `A(s,a)` streams over shared NatureCNN trunk |
| **Unified algorithm factory** | [src/algos/registry.py](src/algos/registry.py) | One config-driven path produces vanilla/Double/Dueling/Double+Dueling DQN |
| **Train/eval env separation** | [src/envs.py](src/envs.py) | Train: `terminal_on_life_loss=True` + clipped reward; Eval: full 5-life episode with true unclipped score |
| **Config inheritance** | [src/utils/config.py](src/utils/config.py) | `inherits:` deep-merge so each ablation overrides exactly one key |
| **Bootstrap-CI evaluation** | [src/eval.py](src/eval.py) | Non-parametric 95% bootstrap CI of the mean over 100 episodes |
| **Multi-axis analysis** | [src/analysis.py](src/analysis.py) | 8 analysis axes: algorithm, seed, budget, hyperparam response, sample efficiency, compute, behavioral, significance |
| **Experiment harness** | [scripts/](scripts/) | Reproducible per-seed run dirs, resume-from-checkpoint, directory-batch ablation sweeps, leaderboard evaluation |

---

## Repository Structure

```text
RL-Breakout/
├── README.md
├── requirements.txt
├── verify_env.ipynb            # environment self-check (notebook)
├── verify_env.py               # environment self-check (script)
├── configs/
│   ├── dqn_baseline.yaml       # DQN baseline (all DQN ablations inherit this)
│   ├── ddqn.yaml               # Double DQN  (inherits dqn_baseline)
│   ├── dueling_dqn.yaml        # Dueling DQN (inherits dqn_baseline)
│   ├── ppo.yaml                # PPO baseline
│   ├── a2c.yaml                # A2C baseline
│   └── ablations/              # 27 single-variable ablations
│       ├── README.md           # per-file knob + justification tables
│       ├── dqn_*.yaml          # lr / buffer / target / batch / expl / frame-stack
│       ├── ppo_*.yaml          # clip / ent / lr / n_epochs / gae / gamma
│       └── a2c_*.yaml          # lr / n_steps / gae / optimizer / vf / norm-adv
├── src/
│   ├── envs.py                 # env factory (train vs. eval preprocessing)
│   ├── eval.py                 # evaluation metrics + bootstrap CI
│   ├── analysis.py             # 8-axis analysis pipeline (discover → dedup → compute)
│   ├── render.py               # rendering helpers for play.py
│   ├── algos/
│   │   ├── registry.py         # name -> builder dispatch
│   │   ├── dqn.py              # DQN + custom DoubleDQN (Double Q-target override)
│   │   ├── dueling.py          # custom Dueling Q-network / DuelingCnnPolicy
│   │   ├── a2c.py              # A2C builder (SB3 with YAML injection)
│   │   └── ppo.py              # PPO builder (SB3 with YAML injection)
│   └── utils/
│       ├── config.py           # YAML loader with `inherits:` deep-merge
│       ├── logging.py          # episode CSV writer
│       └── seeding.py          # global seeding (Python / NumPy / PyTorch)
├── scripts/
│   ├── train.py                # training entry point (+ resume / batch sweep)
│   ├── evaluate.py             # single / multi-run evaluation + leaderboard
│   ├── play.py                 # watch a trained agent play
│   └── plot_results.py         # aggregate runs → figures & tables
├── experiments/                # per-run outputs (gitignored — see Training section)
└── reports/
    ├── apa_paper.md            # full analysis paper (APA 7th edition, Korean)
    ├── figures/
    │   ├── notebook/           # 25 analysis figures (learning curves, violin, ECDF, …)
    │   │   ├── seed_aggregate.png
    │   │   ├── learning_curves_{1m,2m,10m,50m}.png
    │   │   ├── violin_{1m,2m,10m,50m}.png
    │   │   ├── ecdf_{1m,2m,10m,50m}.png
    │   │   ├── sample_efficiency{,_*}.png
    │   │   ├── compute_tradeoff{,_*}.png
    │   │   ├── response_curves.png
    │   │   └── significance.png
    │   └── eval_distribution_{1m,2m,10m,50m}_*.png
    └── tables/                 # summary / comparison CSVs
```

Each `experiments/<timestamp>_<name>_seed<N>/` run dir contains:

```text
config.yaml          # frozen config snapshot (exact hyperparameters)
tensorboard/         # TensorBoard logs
monitor_train/       # raw training episode returns / lengths
eval/                # EvalCallback outputs (evaluations.npz — training curves)
best_model/          # best_model.zip (selected by held-out eval score)
checkpoints/         # periodic checkpoints every 200k steps
final_model.zip      # weights at the last training step
```

---

## Environment Setup

```bash
conda create -n breakout python=3.11 -y
conda activate breakout
pip install -r requirements.txt
AutoROM --accept-license
```

Install PyTorch separately to match your CUDA version. Verify the installation:

```bash
python verify_env.py
```

---

## Algorithms

| Algorithm | Config | Type | Key Idea |
|---|---|---|---|
| DQN | [configs/dqn_baseline.yaml](configs/dqn_baseline.yaml) | value-based, off-policy | Replay buffer + target network |
| Double DQN | [configs/ddqn.yaml](configs/ddqn.yaml) | value-based, off-policy | Decouple action selection from Q-evaluation |
| Dueling DQN | [configs/dueling_dqn.yaml](configs/dueling_dqn.yaml) | value-based, off-policy | Separate `V(s)` and `A(s,a)` streams |
| A2C | [configs/a2c.yaml](configs/a2c.yaml) | actor-critic, on-policy | Synchronous A3C; RMSProp; n_steps=5 |
| PPO | [configs/ppo.yaml](configs/ppo.yaml) | actor-critic, on-policy | Clipped surrogate objective; GAE |

---

## Configuration System

Each ablation overrides exactly one hyperparameter via `inherits:` deep-merge:

```yaml
# configs/ablations/dqn_lr_1.5e-4.yaml
inherits: ../dqn_baseline.yaml
algo:
  learning_rate: 1.5e-4   # the only change
```

The `configs/ablations/` folder contains **27 single-variable ablations**.
Knob choices and rationale are documented in
[configs/ablations/README.md](configs/ablations/README.md).

> **YAML gotcha:** write scientific notation with a decimal point (`1.0e-4`, not
> `1e-4`). PyYAML parses `1e-4` as a *string*, which SB3 then rejects.

---

## Training

```bash
# Single run
python scripts/train.py --config configs/a2c.yaml --seed 7

# Multiple seeds
for s in 7 77 777; do
  for c in dqn_baseline ddqn dueling_dqn a2c ppo; do
    python scripts/train.py --config configs/$c.yaml --seed $s
  done
done

# Full ablation sweep (all *.yaml in the directory, sequentially)
python scripts/train.py --config configs/ablations --seed 7

# Resume an interrupted run
python scripts/train.py --run experiments/<run_dir>

# Override step budget (smoke test)
python scripts/train.py --config configs/ppo.yaml --seed 7 --total-timesteps 200000

# Auto-evaluate after training
python scripts/train.py --config configs/a2c.yaml --seed 7 --evaluate-after-train
```

---

## Evaluation

```bash
# Single run — prefers best_model.zip, falls back to final_model.zip
python scripts/evaluate.py --run experiments/<run_dir> --n-eval-episodes 100

# Many runs → leaderboard + comparison CSV
python scripts/evaluate.py --runs "experiments/*" --n-eval-episodes 100
python scripts/evaluate.py --runs experiments/*ppo* --n-eval-episodes 100 \
    --output-dir reports/tables/ppo_ablation_compare
```

The evaluation environment differs from training intentionally:

| Setting | Train | Evaluation |
|---|---|---|
| `terminal_on_life_loss` | `true` | `false` |
| `clip_reward` | `true` | `false` |
| Policy | ε-greedy (exploring) | deterministic (greedy) |
| Parallel envs | 8 | 1 |

---

## Visualization

```bash
# Watch a trained agent
python scripts/play.py --run experiments/<run_dir> --n-episodes 3 --window-scale 4

# Generate figures and tables from experiment artifacts
python scripts/plot_results.py                             # scan all experiments/
python scripts/plot_results.py --experiments experiments/10m
python scripts/plot_results.py --experiments experiments/10m experiments/1m
```

Output is written to `reports/figures/` and `reports/tables/`.

---

## Reproducibility

Every run seeds Python, NumPy, and PyTorch (CPU + CUDA) via
[src/utils/seeding.py](src/utils/seeding.py). The eval environment uses a
different seed (`train_seed + 1000`) so `best_model` is selected on episodes
the agent never trained on.

---

## References

### Environment and Tooling

- Bellemare et al. (2013). The Arcade Learning Environment. *JAIR*, 47, 253–279.
- Towers et al. (2024). Gymnasium. Farama Foundation.
- Raffin et al. (2021). Stable-Baselines3. *JMLR*, 22(268), 1–8.

### DQN Family

- Mnih et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- van Hasselt et al. (2016). Deep RL with Double Q-learning. *AAAI*.
- Wang et al. (2016). Dueling Network Architectures for Deep RL. *ICML*.

### Actor-Critic and Policy Optimization

- Mnih et al. (2016). Asynchronous Methods for Deep RL (A3C). *ICML*.
- Schulman et al. (2016). High-Dimensional Continuous Control Using GAE. *ICLR*.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

### Additional

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL. *AAAI*.
