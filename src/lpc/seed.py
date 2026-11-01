import os
import random

import numpy as np


def set_seed(seed: int = 0) -> np.random.Generator:
    """Seed python/numpy (and torch if installed). Returns a numpy Generator."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)
