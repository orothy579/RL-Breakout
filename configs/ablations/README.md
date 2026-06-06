# Ablation configs — design & rationale

This folder holds **single‑variable ablations** of the three baseline agents
(`configs/dqn_baseline.yaml`, `configs/ppo.yaml`, `configs/a2c.yaml`).

## Design rules

1. **One knob at a time.** Every file uses `inherits: ../<base>.yaml` and overrides
   *exactly one* hyperparameter. A measured difference can therefore be attributed to
   that knob alone (no confounds).
2. **Same budget & protocol.** All ablations inherit the base's `total_timesteps`
   (10M), env preprocessing, seeds and eval protocol, so runs are directly comparable.
3. **Numbers are anchored, not arbitrary.** Each value is either (a) a *published
   standard* (Mnih 2015/2016, Schulman 2017, GAE 2016, SB3 / RL‑Baselines3‑Zoo
   defaults), or (b) a deliberate *bracket* (½× / 2×, or one notch above/below the
   baseline) chosen to span "too low → baseline → too high" so the response curve is
   visible. The tables below give the source for every number.
4. **YAML float gotcha.** Scientific notation must include a decimal point
   (`1.0e-4`, not `1e-4`) — PyYAML otherwise parses it as a *string* and SB3 rejects it.

> Baselines for reference
> - **DQN**: lr `1.0e-4`, buffer `250k`, batch `32`, target_update `1000`, expl_frac `0.1`, γ `0.99`
> - **PPO**: lr `2.5e-4`, n_steps `128`, batch `256`, n_epochs `4`, γ `0.99`, λ `0.95`, clip `0.1`, ent `0.01`, vf `0.5`
> - **A2C**: lr `7.0e-4`, n_steps `5`, γ `0.99`, λ `1.0`, ent `0.01`, vf `0.25`, RMSProp, normalize_adv `false`

---

## DQN ablations

| File | Knob | Base → value | Why this number |
|---|---|---|---|
| `dqn_buffer_100k.yaml` | `buffer_size` | 250k → **100k** | Lower bound of a ×0.4/×1.2 bracket around the baseline. Smaller buffer ⇒ more *recent* transitions (more on‑policy), less decorrelation ⇒ variance↑. ~2.6 GiB RAM. |
| `dqn_buffer_300k.yaml` | `buffer_size` | 250k → **300k** | Upper end of the bracket; moves toward Mnih 2015's 1M replay (RAM‑bounded here). More sample diversity / off‑policyness. ~7.9 GiB RAM. |
| `dqn_target_10000.yaml` | `target_update_interval` | 1000 → **10000** | **10k is the canonical value** (Mnih 2015 *Nature*, SB3 Atari default). The baseline's 1k is intentionally aggressive; this tests stability from a slower, less‑stale target. |
| `dqn_lr_2.5e-4.yaml` | `learning_rate` | 1.0e‑4 → **2.5e‑4** | The *other* published anchor: 2.5e‑4 is Nature‑DQN's RMSProp LR; SB3's default 1e‑4 is the conservative one. Both endpoints are standards → clean speed‑vs‑stability test. |
| `dqn_expl_frac_0.2.yaml` | `exploration_fraction` | 0.1 → **0.2** | Exactly ×2 (a clean bracket). ε anneals over 20 % instead of 10 % of training — more early exploration, while still leaving 80 % greedy. 0.2 is a common Zoo alternative. |
| `dqn_batch_64.yaml` | `batch_size` | 32 → **64** | Next power of two above the Nature value (32). Halves gradient‑estimate variance at ×2 compute/update — the standard stability↔throughput axis. |

---

## PPO ablations

| File | Knob | Base → value | Why this number |
|---|---|---|---|
| `ppo_clip_0.05.yaml` | `clip_range` | 0.1 → **0.05** | Tight end of the clip sweep — very conservative trust region. |
| `ppo_clip_0.2.yaml` | `clip_range` | 0.1 → **0.2** | **Schulman 2017's original recommended value.** |
| `ppo_clip_0.3.yaml` | `clip_range` | 0.1 → **0.3** | Loose end — clipping barely binds, approaching vanilla PG (divergence risk). |
| `ppo_lr_1e-4.yaml` | `learning_rate` | 2.5e‑4 → **1.0e‑4** | ~×0.4 of the standard Atari‑PPO LR (2.5e‑4) — a common lower setting; tests stability vs speed. (clip stays at baseline 0.1.) |
| `ppo_ent_0.0.yaml` | `ent_coef` | 0.01 → **0.0** | Removes the entropy bonus entirely → measures its contribution to exploration. |
| `ppo_ent_0.05.yaml` | `ent_coef` | 0.01 → **0.05** | ×5 stronger exploration bonus → keeps the policy stochastic longer. |
| `ppo_ne_10.yaml` | `n_epochs` | 4 → **10** | 4 = SB3 Atari default; 10 is the common high setting (original PPO used 3–10). Tests data‑reuse intensity vs off‑policy drift. |
| `ppo_gae_0.9.yaml` | `gae_lambda` | 0.95 → **0.90** | Lower λ → more bias / less variance in the advantage. |
| `ppo_gae_1.0.yaml` | `gae_lambda` | 0.95 → **1.0** | λ=1 = Monte‑Carlo advantage (unbiased, high variance) — the GAE extreme. |
| `ppo_gamma_0.995.yaml` | `gamma` | 0.99 → **0.995** | One notch up: effective horizon ~100 → ~200 steps, for Breakout's long brick‑clearing credit assignment. |

**Designed sweeps** (vary one axis across several files for a response curve):
- **clip_range**: `0.05 · 0.1(base) · 0.2 · 0.3`
- **ent_coef**: `0.0 · 0.01(base) · 0.05`
- **gae_lambda**: `0.90 · 0.95(base) · 1.0`

---

## A2C ablations

| File | Knob | Base → value | Why this number |
|---|---|---|---|
| `a2c_lr_2.5e-4.yaml` | `learning_rate` | 7.0e‑4 → **2.5e‑4** | Cross‑algo anchor = PPO's LR. Tests whether the RMSProp standard 7e‑4 is too hot for short‑rollout A2C. |
| `a2c_nstep_16.yaml` | `n_steps` | 5 → **16** | Longer rollout → less n‑step‑return bias / more variance; also brackets toward PPO‑like horizons. |
| `a2c_gae_0.95.yaml` | `gae_lambda` | 1.0 → **0.95** | Standard A2C uses pure n‑step return (λ=1). 0.95 introduces GAE smoothing (PPO's value) → bias↑/variance↓. |
| `a2c_nstep16_gae095.yaml` | `n_steps`+`gae_lambda` | 5,1.0 → **16, 0.95** | *Intentional 2‑knob combo*: together they approximate PPO's advantage‑estimation regime (longer rollout + exponential weighting). |
| `a2c_adam.yaml` | `use_rms_prop` | true → **false (Adam)** | Optimizer swap with LR fixed at 7e‑4. A3C/A2C used RMSProp; modern code often uses Adam → direct head‑to‑head. (7e‑4 may be hot for Adam — that sensitivity is the point.) |
| `a2c_normadv.yaml` | `normalize_advantage` | false → **true** | Isolates PPO's per‑batch advantage normalization → steadier gradient scale. |
| `a2c_vf_0.5.yaml` | `vf_coef` | 0.25 → **0.5** | Matches PPO's value‑loss weight (A2C's Atari default is 0.25) → stronger value head. |

---

## Running

Train one ablation (full 10M budget, auto‑evaluate at the end):

```bash
conda activate breakout
python scripts/train.py --config configs/ablations/ppo_gae_0.9.yaml --seed 0 --evaluate-after-train
```

Quick smoke test (tiny budget, no progress bar):

```bash
python scripts/train.py --config configs/ablations/ppo_gae_0.9.yaml --seed 0 \
    --total-timesteps 50000 --no-progress-bar
```
> **DQN smoke tests**: the real 250k buffer needs ~6.6 GiB. On low‑RAM machines make a
> temp config that `inherits` the ablation and shrinks `buffer_size`/`learning_starts`.

### Evaluate **multiple** runs at once → leaderboard + comparison CSV

`evaluate.py` accepts `--runs` with directories and/or glob patterns. Each run is
evaluated (artifacts saved under its own `eval_runs/seed<N>/`), then a leaderboard is
printed (sorted by mean reward) and a combined CSV is written:

```bash
# all finished experiments, 100 episodes each
python scripts/evaluate.py --runs "experiments/*" --n-eval-episodes 100

# only the PPO ablations, custom output location
python scripts/evaluate.py --runs experiments/*ppo* --n-eval-episodes 100 \
    --output-dir reports/tables/ppo_ablation_compare
```

- Default combined CSV: `reports/tables/eval_compare_<timestamp>.csv`
  (or `<output-dir>/comparison.csv` when `--output-dir` is given).
- A run is skipped (with a warning) if it lacks `config.yaml` or a model `.zip`;
  one failing run won't abort the rest.
- The single‑run form (`--run <dir>`) and `--model/--config` form are unchanged.
