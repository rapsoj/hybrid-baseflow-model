"""Reference forecasts the transformer must beat. Both return quantile predictions shaped (N, H, Q)."""
from typing import Sequence

import numpy as np


def climatology(train_y: np.ndarray, n: int, quantiles: Sequence[float]) -> np.ndarray:
    """Training-set flow quantiles (pooled over samples and lead times), identical for every sample and step."""
    qs = np.quantile(train_y.ravel(), list(quantiles))
    return np.broadcast_to(qs, (n, train_y.shape[1], len(qs))).copy()


def persistence(last_flow: np.ndarray, horizon_steps: int, n_quantiles: int) -> np.ndarray:
    """Latest observed flow at the forecast origin held constant over the horizon (deterministic: all quantiles equal)."""
    return np.broadcast_to(last_flow[:, None, None], (len(last_flow), horizon_steps, n_quantiles)).copy()
