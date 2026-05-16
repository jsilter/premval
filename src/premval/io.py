from pathlib import Path

import numpy as np

# Type aliases used across the package
XyzArray = np.ndarray  # shape (..., 3), float32, nm
IndexArray = np.ndarray  # shape (n,), int64


def ensure_dir(path: Path) -> Path:
    """Create directory and parents if missing, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
