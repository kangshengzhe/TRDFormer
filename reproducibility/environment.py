"""Environment snapshot capture for reproducibility records.

Captures Python, NumPy, PyTorch, and CUDA versions along with device
information, execution location, hostname, and git SHA.
"""

import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class EnvSnapshot:
    """Immutable snapshot of the software and hardware environment for a run."""

    python: str
    numpy: str
    torch: str
    cuda: Optional[str]
    device: str
    execution_location: str
    hostname: str
    git_sha: Optional[str]

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvSnapshot":
        """Reconstruct an EnvSnapshot from a dictionary."""
        return cls(**d)


def _get_git_sha() -> Optional[str]:
    """Attempt to retrieve the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _get_device() -> str:
    """Determine the active compute device."""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def _get_cuda_version() -> Optional[str]:
    """Get the CUDA runtime version if available."""
    if torch.cuda.is_available():
        return torch.version.cuda
    return None


def capture_environment(execution_location: str) -> EnvSnapshot:
    """Capture a snapshot of the current software/hardware environment.

    Args:
        execution_location: Identifier for where this run is executed,
            typically 'local_cpu' or 'kaggle_gpu'.

    Returns:
        An EnvSnapshot dataclass containing version and device information.
    """
    return EnvSnapshot(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        cuda=_get_cuda_version(),
        device=_get_device(),
        execution_location=execution_location,
        hostname=socket.gethostname(),
        git_sha=_get_git_sha(),
    )
