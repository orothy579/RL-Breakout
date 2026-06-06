"""DQN / Double DQN / Dueling DQN builder.

================================ ATTRIBUTION ================================
External library: Stable-Baselines3 (SB3). We reuse SB3's ``DQN`` class,
replay buffer, Nature-CNN feature extractor and training loop scaffolding.

MY ORIGINAL CONTRIBUTIONS IN THIS FILE
  1. ``DoubleDQN`` (architectural modification): a subclass that overrides
     SB3's ``DQN.train`` so the bootstrap target uses the Double Q-learning
     decoupling (van Hasselt et al., 2016). SB3's stock ``DQN`` only
     implements the vanilla single-network target.
  2. ``build_dqn`` (original glue): a single factory that turns one YAML
     ``algo`` block into vanilla DQN, Double DQN, Dueling DQN, or
     Double+Dueling DQN, so all four variants share one code path and one
     set of hyper-parameters (only the target rule / Q-head change).
  3. YAML-friendliness helpers (``_coerce_train_freq``).
============================================================================

Config example (configs/*.yaml):

    algo:
      name: dqn
      features:
        double_q: true        # -> use the DoubleDQN target rule
        dueling: true         # -> use the Dueling Q-head
      kwargs:
        policy: CnnPolicy
        learning_rate: 1.0e-4
        ...
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv

from src.algos.dueling import DuelingCnnPolicy


class DoubleDQN(DQN):
    """Double Q-learning target (van Hasselt et al., AAAI 2016).

    ARCHITECTURAL MODIFICATION over SB3's ``DQN``
    ----------------------------------------------
    Vanilla DQN target:  y = r + γ * max_a' Q_target(s', a')
    Double DQN target:   y = r + γ * Q_target(s', argmax_a' Q_online(s', a'))

    Vanilla DQN uses the *same* target network both to pick the next action
    and to evaluate it; the ``max`` operator then systematically over-estimates
    Q-values. Double DQN decouples the two roles — the *online* network selects
    the greedy next action, the *target* network only evaluates it — which
    reduces the maximization bias and tends to stabilise learning.

    Implementation note: SB3 does not ship a Double DQN. We copy SB3's
    ``DQN.train`` loop verbatim and change ONLY the next-Q computation
    (the ``with th.no_grad()`` block). Everything else (replay sampling,
    Huber/smooth-L1 loss, gradient clipping, optimizer step, logging) is the
    upstream behaviour so the comparison against vanilla DQN stays fair.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # noqa: D401
        # --- identical to SB3 DQN.train: put nets in train mode, anneal LR ---
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        for _ in range(gradient_steps):
            # --- identical to SB3: sample a minibatch of transitions ---
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts if replay_data.discounts is not None else self.gamma
            )

            with th.no_grad():
                # ===== THE Double DQN MODIFICATION (my contribution) =====
                # Step 1: ONLINE network chooses the greedy next action.
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                # Step 2: TARGET network only *evaluates* that chosen action.
                #         (Vanilla SB3 DQN would do next_q_target.max(dim=1)
                #          here, coupling selection and evaluation.)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions)

                # Standard 1-step TD target; (1 - dones) zeroes out terminals.
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                )

            # --- identical to SB3: current Q(s,a) for the taken actions ---
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Smooth-L1 (Huber) loss: SB3/DeepMind default, robust to outliers.
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # --- identical to SB3: clip-grad then optimizer step ---
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)) if losses else 0.0)


def _coerce_train_freq(value: Any) -> Any:
    """Convert YAML's ``[4, step]`` list into SB3's ``(4, "step")`` tuple.

    YAML has no tuple type, so ``train_freq`` arrives as a list; SB3 expects a
    ``(int, str)`` tuple. This keeps the config human-readable while staying
    compatible with the upstream API.
    """
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), str(value[1]))
    return value


def build_dqn(
    cfg: dict[str, Any],
    *,
    env: VecEnv,
    tensorboard_log: str | None,
    seed: int,
    device: str = "auto",
) -> DQN:
    """Build a DQN / DDQN / Dueling-DQN instance from a parsed config.

    ORIGINAL CONTRIBUTION: one factory dispatches all four DQN variants by
    reading two boolean feature flags, so every variant inherits the exact
    same baseline hyper-parameters (configs/dqn_baseline.yaml) and only the
    studied component (target rule and/or Q-head) changes between runs. This
    is what makes the algorithm comparison an apples-to-apples ablation.
    """
    algo_cfg = cfg["algo"]
    features = dict(algo_cfg.get("features", {}) or {})
    # deepcopy so we never mutate the caller's config dict (it gets saved to disk).
    kwargs = copy.deepcopy(algo_cfg.get("kwargs", {}) or {})

    use_double = bool(features.get("double_q", False))    # -> DoubleDQN target rule
    use_dueling = bool(features.get("dueling", False))    # -> Dueling Q-head

    # Fix up the YAML list -> SB3 tuple for train_freq.
    if "train_freq" in kwargs:
        kwargs["train_freq"] = _coerce_train_freq(kwargs["train_freq"])

    # Policy selection: swap in our custom Dueling policy when requested.
    # (DuelingCnnPolicy keeps SB3's Nature-CNN body and only rewires the head.)
    if use_dueling:
        kwargs.pop("policy", None)
        policy: Any = DuelingCnnPolicy
    else:
        policy = kwargs.pop("policy", "CnnPolicy")

    # ``policy_kwargs.net_arch`` etc. pass straight through to SB3.
    # Class selection: our DoubleDQN target rule vs. SB3's vanilla DQN.
    DqnClass: type[DQN] = DoubleDQN if use_double else DQN

    model = DqnClass(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        seed=seed,          # forwarded for reproducibility (see utils/seeding.py)
        device=device,
        verbose=0,
        **kwargs,
    )
    return model
