# Breakout v5 Reinforcement Learning — Final Project

Train and evaluate several reinforcement-learning algorithms on the
`ALE/Breakout-v5` environment and interpret the performance differences through
RL theory. Following the assignment brief, the goal is **not** the single
highest score but **rigorous experimentation, fair comparison, and
theory-grounded analysis**. Every use of an external library (Stable-Baselines3)
is paired with explicit justification of the chosen library, architecture, and
hyperparameters — both in this README and as in-code comments.

## Project Constraints (Mandatory Baseline)

For evaluation fairness, the assignment fixes the following package versions
(see [docs/readme.md](docs/readme.md)):

```bash
gymnasium[atari]==1.3.0
ale-py==0.11.2
autorom[accept-rom-license]==0.6.1
```

The only hard constraints are these versions and the `ALE/Breakout-v5`
environment. There is **no fixed `batch_size` constraint**, so `batch_size` is
set per algorithm convention: DQN's `batch_size=32` is the number of transitions
sampled from the replay buffer, whereas PPO's `batch_size=256` is the minibatch
size used to split a rollout — different meanings, different values.

Stable-Baselines3 on PyPI (2.8.0) pins `gymnasium<1.3.0`, which conflicts with
the required `gymnasium==1.3.0`. This project therefore installs SB3 from the
GitHub `master` branch (cap relaxed to `gymnasium<2.0`); see
[requirements.txt](requirements.txt).

## Original Contributions vs. Stable-Baselines3

The assignment explicitly penalizes "running an external library without
original analysis." The library is used for the training scaffolding only; the
RL-substantive pieces below are original and are documented inline in the code:

| Contribution | Where | What was modified / added |
|---|---|---|
| **Double DQN target** (architectural modification) | [src/algos/dqn.py](src/algos/dqn.py) | Subclass `DoubleDQN` overrides SB3's `DQN.train`; only the next-Q target is changed to decouple action *selection* (online net) from *evaluation* (target net), reducing Q-value over-estimation (van Hasselt et al., 2016). |
| **Dueling Q-head** (architectural modification) | [src/algos/dueling.py](src/algos/dueling.py) | Replaces SB3's single-stream Q-head with two streams `V(s)` and `A(s,a)`, recombined as `Q = V + (A − mean A)` over the shared Nature-CNN trunk (Wang et al., 2016). |
| **Unified algorithm factory** | [src/algos/registry.py](src/algos/registry.py), [src/algos/dqn.py](src/algos/dqn.py) | One config-driven path produces vanilla/Double/Dueling/Double+Dueling DQN, so variants differ by *exactly one* component. |
| **Shared env factory + deliberate train/eval mismatch** | [src/envs.py](src/envs.py) | Train uses `terminal_on_life_loss=True` + `clip_reward=True` (denser signal); eval uses the full 5-life episode with the **true unclipped score** — the number actually reported. |
| **Single-variable config inheritance** | [src/utils/config.py](src/utils/config.py) | `inherits:` deep-merge so each ablation overrides one key — every comparison is a controlled experiment. |
| **Bootstrap-CI evaluation** | [src/eval.py](src/eval.py) | Reports mean/std/median/min/max plus a non-parametric 95% bootstrap CI of the mean (Atari returns are skewed). |
| **Experiment harness** | [scripts/](scripts/) | Reproducible per-seed run dirs with frozen config snapshots, resume-from-checkpoint, directory-batch ablation sweeps, leaderboard evaluation, and figure/table generation. |

## Repository Structure

```text
Breakout_Final_Project_Pack/
├── README.md
├── requirements.txt
├── verify_env.ipynb            # environment self-check (notebook)
├── verify_env.py               # environment self-check (script)
├── configs/
│   ├── dqn_baseline.yaml       # vanilla DQN baseline (all DQN ablations inherit this)
│   ├── ddqn.yaml               # Double DQN  (inherits dqn_baseline)
│   ├── dueling_dqn.yaml        # Dueling DQN (inherits dqn_baseline)
│   ├── ppo.yaml                # PPO baseline
│   ├── a2c.yaml                # A2C baseline
│   └── ablations/              # 27 single-variable ablations (see its README.md)
│       ├── README.md           # per-file knob + justification tables
│       ├── dqn_*.yaml          # buffer / target / lr / batch / expl / frame-stack
│       ├── ppo_*.yaml          # clip / ent / lr / n_epochs / gae / gamma
│       └── a2c_*.yaml          # lr / n_steps / gae / optimizer / vf / norm-adv
├── src/
│   ├── envs.py                 # env factory (train vs. eval preprocessing)
│   ├── eval.py                 # evaluation metrics + bootstrap CI
│   ├── render.py               # rendering helpers for play.py
│   ├── algos/
│   │   ├── registry.py         # name -> builder dispatch
│   │   ├── dqn.py              # DQN + custom DoubleDQN
│   │   ├── dueling.py          # custom Dueling Q-network/policy
│   │   ├── a2c.py              # A2C builder (SB3 as-is)
│   │   └── ppo.py              # PPO builder (SB3 as-is)
│   └── utils/
│       ├── config.py           # YAML loader with `inherits:` deep-merge
│       ├── logging.py          # episode CSV writer
│       └── seeding.py          # global seeding for reproducibility
├── scripts/
│   ├── train.py                # training entry point (+ resume / batch sweep)
│   ├── evaluate.py             # single / multi-run evaluation + leaderboard
│   ├── play.py                 # watch a trained agent
│   └── plot_results.py         # aggregate runs -> figures & tables
├── experiments/                # per-run outputs (gitignored)
└── reports/
    ├── figures/                # learning curves, eval distributions
    └── tables/                 # summary / comparison CSVs
```

## Environment Setup

```bash
conda create -n breakout python=3.11 -y
conda activate breakout
pip install -r requirements.txt
AutoROM --accept-license
```

Install PyTorch separately to match your CUDA version (see the Installation
Manual in [docs/](docs/)).

Verify the installation:

```bash
python verify_env.py
```

or open [verify_env.ipynb](verify_env.ipynb) and run the install + verification
cells in order. The check imports Gymnasium/ale-py, registers and creates
`ALE/Breakout-v5`, and runs 100 random steps.

## Algorithms

| Algorithm | Config | Type | Purpose |
|---|---|---|---|
| DQN | [configs/dqn_baseline.yaml](configs/dqn_baseline.yaml) | value-based, off-policy | Baseline for Atari pixel control |
| Double DQN | [configs/ddqn.yaml](configs/ddqn.yaml) | value-based, off-policy | Mitigate DQN's Q-value over-estimation |
| Dueling DQN | [configs/dueling_dqn.yaml](configs/dueling_dqn.yaml) | value-based, off-policy | Separate state value `V(s)` from advantage `A(s,a)` |
| A2C | [configs/a2c.yaml](configs/a2c.yaml) | actor-critic, on-policy | Synchronous A3C variant; simplest policy-gradient control |
| PPO | [configs/ppo.yaml](configs/ppo.yaml) | actor-critic, on-policy | Stable policy optimization via a clipped objective |

All five share an identical preprocessing pipeline, seeding, 10M-step budget,
and evaluation protocol, so differences are attributable to the algorithm.

## Configuration System

Configs use single-inheritance deep-merge. A child file pulls in a parent via
`inherits:` and overrides only the keys it changes:

```yaml
# configs/ddqn.yaml
inherits: dqn_baseline.yaml
algo:
  features:
    double_q: true   # the only change vs. the baseline
```

The `configs/ablations/` folder contains **27 single-variable ablations** of the
DQN/PPO/A2C baselines. Every value is anchored to a published standard or a
deliberate ×0.5/×2 bracket; the per-file knob and rationale are tabulated in
[configs/ablations/README.md](configs/ablations/README.md).

> **YAML gotcha:** write scientific notation with a decimal point (`1.0e-4`, not
> `1e-4`). PyYAML parses `1e-4` as a *string*, which SB3 then rejects.

## Training

Single run:

```bash
python scripts/train.py --config configs/dqn_baseline.yaml --seed 7
python scripts/train.py --config configs/ddqn.yaml         --seed 7
python scripts/train.py --config configs/dueling_dqn.yaml  --seed 7
python scripts/train.py --config configs/a2c.yaml          --seed 7
python scripts/train.py --config configs/ppo.yaml          --seed 7
```

Multiple seeds:

```bash
for s in 7 77 777; do
  for c in dqn_baseline ddqn dueling_dqn a2c ppo; do
    python scripts/train.py --config configs/$c.yaml --seed $s
  done
done
```

Each run is written to `experiments/<timestamp>_<name>_seed<seed>/`:

```text
config.yaml          # frozen config snapshot for this run
tensorboard/         # TensorBoard logs
monitor_train/       # raw training episode returns/lengths
eval/                # EvalCallback outputs (evaluations.npz)
best_model/          # best_model.zip (selected on held-out eval score)
checkpoints/         # periodic checkpoints
final_model.zip      # final checkpoint
```

Temporarily override the step budget (handy for smoke tests):

```bash
python scripts/train.py --config configs/ppo.yaml --seed 7 --total-timesteps 200000
```

Point `--config` at a **directory** to train every `*.yaml` in it sequentially
(ablation sweep):

```bash
python scripts/train.py --config configs/ablations --seed 7
```

Each config then gets its own `experiments/<timestamp>_<yaml_stem>_seed<seed>/`.

Resume an interrupted run from its **own** directory (reuses `config.yaml` and
`checkpoints/`; the seed is parsed from the directory name):

```bash
python scripts/train.py --run experiments/<run_dir>
```

Resuming loads the **latest** `checkpoints/<algo>_*_steps.zip`. Periodic
checkpoints don't store the replay buffer, so DQN refills it before resuming.

Add `--evaluate-after-train` to run a 100-episode evaluation automatically when
training finishes.

## Evaluation

Evaluate a trained model. With `--run`, `best_model/best_model.zip` is preferred,
falling back to `final_model.zip`:

```bash
python scripts/evaluate.py --run experiments/<run_dir> --n-eval-episodes 100
```

The evaluation environment differs from training on purpose:

| Setting | Train | Evaluation |
|---|---|---|
| `terminal_on_life_loss` | `true` | `false` |
| `clip_reward` | `true` | `false` |
| policy | exploration during learning | deterministic (greedy) |

So the eval score approximates the true episode return — playing until all lives
are lost, with the unclipped game score.

Reported metrics:

| Metric | Meaning |
|---|---|
| `mean`, `std` | mean and standard deviation of episode return |
| `median` | median episode return |
| `min`, `max` | lowest / highest episode return |
| `95% CI` | bootstrap 95% confidence interval of the mean |
| `lengths` | step count per episode |

Evaluate **many runs at once** → leaderboard + comparison CSV. `--runs` accepts
directories and/or glob patterns:

```bash
python scripts/evaluate.py --runs "experiments/*" --n-eval-episodes 100
python scripts/evaluate.py --runs experiments/*ppo* --n-eval-episodes 100 \
    --output-dir reports/tables/ppo_ablation_compare
```

Runs missing a `config.yaml` or model are skipped with a warning; one failure
does not abort the rest.

## Visualization

Watch a trained agent play:

```bash
python scripts/play.py --run experiments/<run_dir> --n-episodes 3
python scripts/play.py --run experiments/<run_dir> --n-episodes 3 --window-scale 4
```

`--window-scale` renders an upscaled OpenCV window instead of the native
Stella window.

## Plotting Results

Aggregate multiple runs into learning curves, final-evaluation distributions,
and a summary CSV:

```bash
# one experiment group
python scripts/plot_results.py --experiments experiments/2m --reports-dir reports

# compare several groups
python scripts/plot_results.py --experiments experiments/2m experiments/1m

# default: scan all of experiments/
python scripts/plot_results.py
```

Output filenames are suffixed with the group name (the directory directly under
`experiments/`, e.g. `2m`):

```text
reports/figures/learning_curves_2m.png
reports/figures/eval_distribution_2m.png   # CI / std / median / min-max from summary.json
reports/tables/runs_summary_2m.csv
```

Filter to a single training seed with `--training-seed 7` (adds `_seed7` to the
slug), or set the suffix manually with `--slug my_label`.

## Reproducibility

Every run seeds Python, NumPy, and PyTorch (CPU + CUDA) via
[src/utils/seeding.py](src/utils/seeding.py), and SB3's per-algorithm `seed=` is
forwarded for env/policy sampling. The eval environment uses a different seed
(`train_seed + 1000`) so "best model" is selected on episodes the agent did not
train on.

## Suggested Experiment Plan

1. Verify the environment: `python verify_env.py`
2. Train the core algorithms: DQN, Double DQN, Dueling DQN, A2C, PPO
3. Repeat each with seeds `7`, `77`, `777`
4. Evaluate every run with `--n-eval-episodes 100`
5. Run ablations from `configs/ablations/` (e.g. frame-stack, replay buffer
   size, target-update interval, PPO clip range / entropy, A2C optimizer)
6. Generate figures and tables with `plot_results.py`
7. Interpret the results through RL theory:
   - value-based vs. actor-critic
   - off-policy vs. on-policy sample efficiency
   - stabilizing effect of the replay buffer and target network
   - PPO's clipped objective vs. A2C's plain actor-critic
   - how preprocessing changes affect partial observability and reward scale

## References

### Environment and Tooling

- Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The Arcade Learning Environment: An Evaluation Platform for General Agents. *Journal of Artificial Intelligence Research*, 47, 253-279.
- Towers, M., Terry, J. K., Kwiatkowski, A., et al. (2024). Gymnasium: A Standard Interface for Reinforcement Learning Environments. Farama Foundation.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *Journal of Machine Learning Research*, 22(268), 1-8.

### DQN Family

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI Conference on Artificial Intelligence*.
- Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *International Conference on Machine Learning (ICML)*.

### Actor-Critic and Policy Optimization

- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *International Conference on Machine Learning (ICML)*. A3C paper; A2C is the synchronous, batched variant.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *International Conference on Learning Representations (ICLR)*.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

### Additional References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Hessel, M., Modayil, J., van Hasselt, H., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. *AAAI Conference on Artificial Intelligence*.
