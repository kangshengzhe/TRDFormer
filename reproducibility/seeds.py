"""Global seed management for reproducible experiments.

Sets Python, NumPy, PyTorch, and CUDA random seeds, and optionally enables
deterministic algorithm mode for full bit-level reproducibility.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set the global random seed across all relevant libraries.

    Args:
        seed: Integer seed value to use for all random number generators.
        deterministic: If True, enables PyTorch deterministic mode including
            cudnn deterministic, disabling cudnn benchmark, enabling
            use_deterministic_algorithms (warn_only), and setting
            CUBLAS_WORKSPACE_CONFIG for reproducible cuBLAS operations.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU and all CUDA devices
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch deterministic algorithms (warn_only=True allows fallback
        # for ops without deterministic implementation)
        torch.use_deterministic_algorithms(True, warn_only=True)

        # cuBLAS workspace configuration for reproducible results
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
