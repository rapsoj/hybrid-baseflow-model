"""
Target modes and the seasonal climatology.

The network predicts a scaled target  z = (log1p(flow) - offset - mean) / std  and the prediction is inverted with
flow = expm1(z * std + mean + offset). The per-sample, per-step `offset` (log space) is
  absolute        0
  delta_log_last  log1p(anchor)  : anchor = mean observed flow over the last `anchor_hours` before the origin
  anomaly         log1p(seasonal climatology median at the target day of year)
  delta_log_last_anomaly  anomaly + the anomaly of the recent flow carried forward:
                  log1p(clim at target day) + log1p(anchor) - log1p(clim at the origin day)
Offsets only use information available at the origin (anchor) or training data (climatology).
"""
from typing import Sequence

import numpy as np
import pandas as pd

TARGET_MODES = ("absolute", "delta_log_last", "anomaly", "delta_log_last_anomaly")
ANCHOR_MODES = ("delta_log_last", "delta_log_last_anomaly")  # modes that also feed the anchor to the decoder


def seasonal_table(
    hde: pd.DataFrame, end_time: pd.Timestamp, out_step_minutes: int, quantiles: Sequence[float], window_days: int
) -> np.ndarray:
    """
    Flow quantiles by day of year: (367, Q), indexed by dayofyear. Every output-step mean flow observed up to
    `end_time` (the end of the training windows) within +/- `window_days` of the day of year (circular) is pooled.
    """
    flow = hde.loc[:end_time, "flow"].resample(f"{out_step_minutes}min").mean().dropna()
    if flow.empty:
        raise RuntimeError("No historic flow available for the seasonal climatology")
    doy = flow.index.dayofyear.to_numpy()
    vals = flow.to_numpy()
    table = np.full((367, len(quantiles)), np.nan)
    for d in range(1, 367):
        dist = np.abs(doy - d)
        sel = np.minimum(dist, 365 - dist) <= window_days
        if sel.any():
            table[d] = np.quantile(vals[sel], list(quantiles))
    table[0] = table[1]
    return np.where(np.isnan(table), np.nanmedian(table, axis=0), table)


def target_doy(origins: pd.DatetimeIndex, n_steps: int, out_step_minutes: int) -> np.ndarray:
    """Day of year of the middle of every target bin: (N, n_steps)."""
    step = pd.Timedelta(minutes=out_step_minutes)
    mid = [(origins + (j + 0.5) * step).dayofyear.to_numpy() for j in range(n_steps)]
    return np.stack(mid, axis=1)


def seasonal_climatology(table: np.ndarray, origins: pd.DatetimeIndex, n_steps: int, out_step_minutes: int) -> np.ndarray:
    """Seasonal climatology quantile forecast (N, n_steps, Q)."""
    return table[target_doy(origins, n_steps, out_step_minutes)]


def target_offset(mode: str, samples, table: np.ndarray, quantiles: Sequence[float], out_step_minutes: int) -> np.ndarray:
    """Log-space offset (N, H) for a target mode."""
    n, h = samples.y.shape
    if mode == "absolute":
        return np.zeros((n, h), dtype=np.float32)
    if mode == "delta_log_last":
        return np.broadcast_to(np.log1p(samples.anchor_flow)[:, None], (n, h)).astype(np.float32).copy()
    if mode in ("anomaly", "delta_log_last_anomaly"):
        med = int(np.argmin(np.abs(np.asarray(list(quantiles)) - 0.5)))
        clim = seasonal_climatology(table, samples.origins, h, out_step_minutes)[..., med]
        off = np.log1p(np.clip(clim, 0, None))
        if mode == "delta_log_last_anomaly":
            origin_clim = table[samples.origins.dayofyear.to_numpy(), med]
            off = off + (np.log1p(samples.anchor_flow) - np.log1p(np.clip(origin_clim, 0, None)))[:, None]
        return off.astype(np.float32)
    raise ValueError(f"Unknown data.target_mode '{mode}' (choose from {TARGET_MODES})")
