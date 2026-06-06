"""Dueling DQN (Wang et al., ICML 2016) policy / network.

================================ ATTRIBUTION ================================
External library: Stable-Baselines3 (SB3). We reuse SB3's ``QNetwork``,
``CnnPolicy``, the ``NatureCNN`` feature extractor and ``create_mlp`` helper.

MY ORIGINAL CONTRIBUTION / ARCHITECTURAL MODIFICATION
  SB3 ships only a single-stream Q-head (one MLP mapping CNN features -> Q).
  Here I subclass that head into a two-stream "dueling" head:

      V(s)    : scalar state value          (1 output)
      A(s,a)  : per-action advantage        (n_actions outputs)
      Q(s,a)  = V(s) + ( A(s,a) - mean_a A(s,a) )

  Subtracting the mean advantage fixes the identifiability problem: without
  it, V and A can drift by an arbitrary constant (only their sum Q is
  observed), so training is ill-posed. Mean-centering pins down a unique
  decomposition (Wang et al. 2016, Eq. 9). The shared NatureCNN trunk is
  left untouched, so this is a drop-in head replacement, not a new network.
============================================================================
"""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, QNetwork


class DuelingQNetwork(QNetwork):
    """Q = V + (A - mean_a A) dueling head, replacing SB3's single-stream head."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        # We let SB3's QNetwork.__init__ run fully (it builds the standard
        # single-stream ``self.q_net``), then we DISCARD that head and attach
        # our own value/advantage streams. Reusing the parent init keeps the
        # feature extractor, image-normalization flags, etc. exactly as SB3
        # expects them.
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch if net_arch else [512],  # 512 hidden units (Wang 2016)
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        hidden = self.net_arch
        action_dim = int(self.action_space.n)
        # Disable the inherited single-stream head, but keep it as an
        # Identity module so SB3's state_dict save/load stays structurally
        # compatible (the key still exists, it just maps to a no-op).
        self.q_net = nn.Identity()
        # Two parallel MLP streams off the shared CNN features:
        #   value_net:     features -> hidden -> 1            (state value V)
        #   advantage_net: features -> hidden -> n_actions    (advantages A)
        self.value_net = nn.Sequential(*create_mlp(self.features_dim, 1, hidden, self.activation_fn))
        self.advantage_net = nn.Sequential(
            *create_mlp(self.features_dim, action_dim, hidden, self.activation_fn)
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        value = self.value_net(features)            # (B, 1)
        advantage = self.advantage_net(features)    # (B, n_actions)
        # Mean-centered aggregation -> resolves V/A identifiability (Wang 2016).
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingCnnPolicy(CnnPolicy):
    """SB3's ``CnnPolicy`` with only ``make_q_net`` overridden to build a DuelingQNetwork.

    Everything else (Nature-CNN trunk, optimizer, epsilon-greedy action
    selection, save/load) is inherited unchanged from SB3.
    """

    def make_q_net(self) -> QNetwork:
        # Same plumbing SB3 uses, but instantiating OUR dueling head instead
        # of the default QNetwork.
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


# Alias so SB3 can also resolve the policy by string if ever needed.
DuelingPolicy = DuelingCnnPolicy


def patch_default_features_extractor(policy_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Ensure the Dueling policy defaults to the NatureCNN feature extractor."""
    pk = dict(policy_kwargs or {})
    pk.setdefault("features_extractor_class", NatureCNN)
    return pk


# Re-export so ``BasePolicy._dummy_schedule`` / ``Schedule`` stay importable.
_ = (BasePolicy, Schedule)
