"""Prediction and evaluation: quantile loss and extreme-event detection."""
from typing import Dict, Optional, Sequence

import numpy as np
import torch


@torch.no_grad()
def predict_scaled(model, enc: np.ndarray, dec: np.ndarray, device, batch_size: int = 32) -> np.ndarray:
    """Quantile predictions in scaled (log1p, standardised) space: (N, horizon_steps, n_quantiles)."""
    model.eval()
    out = []
    for i in range(0, len(enc), batch_size):
        e = torch.from_numpy(enc[i:i + batch_size]).to(device)
        d = torch.from_numpy(dec[i:i + batch_size]).to(device)
        out.append(model(e, d).cpu().numpy())
    return np.concatenate(out, axis=0)


def pinball(y: np.ndarray, pred: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Elementwise-averaged pinball loss per quantile. y (N, H), pred (N, H, Q) -> (Q,)."""
    q = np.asarray(quantiles)
    err = y[..., None] - pred
    return np.maximum(q * err, (q - 1) * err).mean(axis=(0, 1))


def quantile_metrics(y: np.ndarray, pred: np.ndarray, quantiles: Sequence[float]) -> dict:
    """Quantile loss (original units, m3/s) plus median error and central-interval coverage."""
    per_q = pinball(y, pred, quantiles)
    q = list(quantiles)
    median = pred[..., int(np.argmin(np.abs(np.asarray(q) - 0.5)))]
    lo, hi = pred[..., 0], pred[..., -1]
    return {
        "quantile_loss": float(per_q.mean()),
        "quantile_loss_by_quantile": {str(k): float(v) for k, v in zip(q, per_q)},
        "median_mae": float(np.abs(y - median).mean()),
        f"coverage_{q[0]}_{q[-1]}": float(((y >= lo) & (y <= hi)).mean()),
        "nominal_coverage": float(q[-1] - q[0]),
    }


def _safe_div(a: float, b: float) -> Optional[float]:
    return float(a / b) if b > 0 else None


def classification_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Binary event metrics; ratios whose denominator is zero are reported as None."""
    truth, pred = truth.astype(bool).ravel(), pred.astype(bool).ravel()
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_negative_rate": _safe_div(fn, fn + tp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "precision": precision,
        "recall": recall,
        "accuracy": _safe_div(tp + tn, tp + fp + fn + tn),
        "f1": f1,
        "event_rate": _safe_div(tp + fn, tp + fp + fn + tn),
    }


def extreme_event_metrics(y: np.ndarray, pred: np.ndarray, quantiles: Sequence[float],
                          threshold: float, prediction_quantile: float) -> dict:
    """
    Event = flow above `threshold` (top 5 % of historic flow). A step is flagged when the chosen predicted
    quantile exceeds the threshold. Reported per time step (pooled) and per forecast window (any step in window).
    """
    qi = int(np.argmin(np.abs(np.asarray(list(quantiles)) - prediction_quantile)))
    pred_flag = pred[..., qi] > threshold
    truth_flag = y > threshold
    return {
        "threshold_m3s": float(threshold),
        "prediction_quantile": float(quantiles[qi]),
        "per_step": classification_metrics(truth_flag, pred_flag),
        "per_window": classification_metrics(truth_flag.any(axis=1), pred_flag.any(axis=1)),
    }


def _metrics(y: np.ndarray, pred: np.ndarray, quantiles, threshold: float, prediction_quantile: float) -> dict:
    out = quantile_metrics(y, pred, quantiles)
    out["extreme_events"] = extreme_event_metrics(y, pred, quantiles, threshold, prediction_quantile)
    out["n_samples"] = int(len(y))
    return out


def week_slices(n_steps: int, steps_per_day: int) -> Dict[str, slice]:
    """{'week_1': slice, ...} over the forecast horizon (a trailing part-week is not scored separately)."""
    per_week = 7 * steps_per_day
    return {f"week_{w + 1}": slice(w * per_week, (w + 1) * per_week) for w in range(n_steps // per_week)}


def regime_metrics(y: np.ndarray, pred: np.ndarray, quantiles, steps_per_day: int,
                   bfi: np.ndarray, bfi_threshold: float) -> dict:
    """
    Quantile metrics separately for baseflow-dominated target steps (baseflow index >= threshold) and runoff-dominated
    steps (below it), pooled over samples and steps, overall and per forecast week. Steps with unknown BFI are skipped.
    """
    known = ~np.isnan(bfi)
    regimes = {"baseflow_dominated": known & (bfi >= bfi_threshold), "runoff_dominated": known & (bfi < bfi_threshold)}

    def one(sl: slice, mask: np.ndarray) -> dict:
        n = int(mask[:, sl].sum())
        if n == 0:
            return {"n_steps": 0}
        m = mask[:, sl]
        out = quantile_metrics(y[:, sl][m][None], pred[:, sl][m][None], quantiles)
        out["n_steps"] = n
        out["share_of_steps"] = float(n / known[:, sl].sum())
        return out

    result = {}
    for name, mask in regimes.items():
        result[name] = one(slice(None), mask)
        result[name]["by_week"] = {
            wk: one(sl, mask) for wk, sl in week_slices(y.shape[1], steps_per_day).items()
        }
    return result


def evaluate_split(y: np.ndarray, pred: np.ndarray, quantiles, steps_per_day: int,
                   threshold: float, prediction_quantile: float) -> dict:
    """All metrics over the whole horizon, plus the same metrics for each forecast week under `by_week`."""
    out = _metrics(y, pred, quantiles, threshold, prediction_quantile)
    out["by_week"] = {
        name: _metrics(y[:, s], pred[:, s], quantiles, threshold, prediction_quantile)
        for name, s in week_slices(y.shape[1], steps_per_day).items()
    }
    return out
