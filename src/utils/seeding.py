"""Global seed setup for reproducibility.

MY CONTRIBUTION: a single entry point that seeds every RNG that can affect a
run — Python ``random``, NumPy, and (if installed) PyTorch CPU + all CUDA
devices. Called once at the start of training so results are reproducible
across runs with the same ``--seed``. SB3's per-algorithm ``seed=`` argument
seeds the env/policy sampling separately; this covers everything outside SB3.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)        # Python stdlib RNG
    np.random.seed(seed)     # NumPy global RNG
    try:
        import torch

        torch.manual_seed(seed)                 # PyTorch CPU RNG
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)    # all visible GPUs
    except ImportError:
        # torch is optional for some tooling paths; seeding it is best-effort.
        pass
