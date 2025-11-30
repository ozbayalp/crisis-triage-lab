"""
CrisisTriage AI - Reproducibility Utilities

Helpers for setting random seeds and ensuring deterministic behavior.
"""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 1.8+ deterministic algorithms
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError:
                    # Some operations don't have deterministic implementations
                    pass
    except ImportError:
        pass


def get_seed_worker(worker_id: int) -> None:
    """
    Worker init function for DataLoader to ensure reproducibility.
    
    Usage:
        DataLoader(..., worker_init_fn=get_seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# For backwards compatibility
try:
    import torch
except ImportError:
    torch = None
