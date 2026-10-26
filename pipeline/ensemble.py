"""
GeoSmart ensemble handling. The model sees one member trajectory at a time:
  * training: every time a sample is used, one of its members is drawn at random (also an augmentation);
  * inference: the model is run once per member and the member predictive distributions are pooled into one
    mixture, so the output spread contains both forcing (member) uncertainty and model (quantile) uncertainty.
"""
from typing import Optional, Sequence

import numpy as np
import torch

from .evaluate import predict_scaled


def draw_members(geo_n: np.ndarray, rng: np.random.Generator, k: int = 1) -> np.ndarray:
    """k random member indices per sample: (N, k)."""
    return np.floor(rng.random((len(geo_n), k)) * geo_n[:, None]).astype(int)


def with_member(dec: np.ndarray, members: np.ndarray, idx: np.ndarray, pos: int) -> np.ndarray:
    """Copy of dec (N, T, C) whose geosmart channel `pos` holds member idx (N,) of members (N, M, T)."""
    out = dec.copy()
    out[:, :, pos] = members[np.arange(len(dec)), idx]
    return out


def expand_validation(tensors, members: np.ndarray, geo_n: np.ndarray, pos: int, k: int, seed: int):
    """Validation set with k fixed random members per sample, so early stopping sees a stable, ensemble-aware loss."""
    enc, dec, y = (t.numpy() for t in tensors)
    idx = draw_members(geo_n, np.random.default_rng(seed), k)
    decs = [with_member(dec, members, idx[:, j], pos) for j in range(k)]
    return tuple(torch.from_numpy(np.concatenate(a)) for a in ([enc] * k, decs, [y] * k))


def pool_quantiles(member_q: np.ndarray, taus: Sequence[float], n_draws: int) -> np.ndarray:
    """
    Equal-weight mixture of M predictive distributions, each given by its quantiles. member_q (M, S, Q) -> (S, Q).
    Each distribution is drawn from at `n_draws` evenly spaced probability levels; its quantile function is
    interpolated linearly between the predicted quantiles and extended linearly to probability 0 and 1 (flow >= 0).
    """
    taus = np.asarray(taus, dtype=float)
    lo = member_q[..., :1] - (member_q[..., 1:2] - member_q[..., :1]) / (taus[1] - taus[0]) * taus[0]
    hi = member_q[..., -1:] + (member_q[..., -1:] - member_q[..., -2:-1]) / (taus[-1] - taus[-2]) * (1 - taus[-1])
    ext_v = np.clip(np.concatenate([lo, member_q, hi], axis=-1), 0, None)
    ext_t = np.concatenate([[0.0], taus, [1.0]])
    u = (np.arange(n_draws) + 0.5) / n_draws
    j = np.clip(np.searchsorted(ext_t, u, side="right") - 1, 0, len(ext_t) - 2)
    w = (u - ext_t[j]) / (ext_t[j + 1] - ext_t[j])
    draws = ext_v[..., j] * (1 - w) + ext_v[..., j + 1] * w            # (M, S, n_draws)
    pooled = np.moveaxis(draws, 0, -1).reshape(draws.shape[1], -1)     # (S, n_draws * M)
    return np.quantile(pooled, taus, axis=1).T


def predict_flow(
    model, scalers, enc: np.ndarray, dec: np.ndarray, device, taus: Sequence[float],
    offset: np.ndarray, members: Optional[np.ndarray] = None, geo_n: Optional[np.ndarray] = None,
    pos: Optional[int] = None, n_draws: int = 64,
) -> np.ndarray:
    """Quantile predictions in m3/s (N, horizon_steps, Q). With `members`, runs every member and pools."""
    if members is None:
        return scalers.inverse_flow(predict_scaled(model, enc, dec, device), offset)
    out = []
    for i in range(len(enc)):
        n = int(geo_n[i])
        d = np.repeat(dec[i:i + 1], n, axis=0)
        d[:, :, pos] = members[i, :n]
        pred = scalers.inverse_flow(
            predict_scaled(model, np.repeat(enc[i:i + 1], n, axis=0), d, device, batch_size=64), offset[i])
        out.append(pool_quantiles(pred, taus, n_draws))
    return np.stack(out)


def predict_member_flow(
    model, scalers, enc: np.ndarray, dec: np.ndarray, device, offset: np.ndarray,
    members: np.ndarray, geo_n: np.ndarray, pos: int,
) -> list[np.ndarray]:
    """
    Per-sample member forecasts in m3/s.

    Returns a list of length N where each item is (n_members_i, horizon_steps, n_quantiles).
    """
    out = []
    for i in range(len(enc)):
        n = int(geo_n[i])
        if n <= 0:
            out.append(np.empty((0, 0, len(model.quantiles)), dtype=np.float32))
            continue
        d = np.repeat(dec[i:i + 1], n, axis=0)
        d[:, :, pos] = members[i, :n]
        pred = scalers.inverse_flow(
            predict_scaled(model, np.repeat(enc[i:i + 1], n, axis=0), d, device, batch_size=64), offset[i]
        )
        out.append(pred)
    return out
