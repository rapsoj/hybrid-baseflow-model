"""
Baseflow separation of observed flow with the Boughton recursive digital filter, used to stratify the evaluation
into baseflow-dominated and runoff-dominated periods, and to define the baseflow and runoff prediction targets
(`data.target`). Applied to daily mean flow.

The filter is causal (each day depends only on earlier days) and is applied to observed flow only.
"""
import numpy as np
import pandas as pd


def boughton_filter(q: np.ndarray, k: float, C: float) -> np.ndarray:
    """bf[i] = min(k / (1 + C) * bf[i-1] + C / (1 + C) * q[i], q[i]), started at bf[0] = q[0]."""
    bf = np.empty(len(q), dtype=float)
    bf[0] = q[0]
    for i in range(1, len(q)):
        bf[i] = min(k / (1 + C) * bf[i - 1] + C / (1 + C) * q[i], q[i])
    return bf


def daily_baseflow(flow: pd.Series, k: float, C: float) -> pd.DataFrame:
    """
    Daily-mean flow, baseflow and baseflow index (BFI = baseflow / flow), indexed by day. Days without
    observations are dropped and the filter is restarted after every gap so it never bridges missing data.
    """
    daily = flow.resample("D").mean().dropna()
    out = pd.DataFrame({"flow": daily, "baseflow": np.nan})
    if daily.empty:
        out["bfi"] = np.nan
        return out
    segment = (daily.index.to_series().diff() != pd.Timedelta(days=1)).cumsum().to_numpy()
    for seg in np.unique(segment):
        sel = segment == seg
        out.loc[sel, "baseflow"] = boughton_filter(daily.to_numpy()[sel], k, C)
    out["bfi"] = (out["baseflow"] / out["flow"].where(out["flow"] > 0)).clip(0, 1)
    return out



TARGETS = ("total_flow", "baseflow", "runoff")


def target_series(flow: pd.Series, target: str, k: float, C: float) -> pd.DataFrame:
    """
    The flow component to predict, on the 15-min grid of `flow`, in two forms:
      target  the component being predicted. Baseflow is the Boughton baseflow of the day (constant within the day);
              runoff is the observed flow minus that baseflow, clipped at 0.
      causal  the same component built only from days that are complete at any time of the day, i.e. every step takes the
              previous day's baseflow. Used for the anchor, the persistence baseline and anything else read at the
              forecast origin, because the daily filter value of the origin day uses flow after the origin.
    Steps without an observation stay NaN in both, so every target uses the same samples as total flow.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if target == "total_flow":
        return pd.DataFrame({"target": flow, "causal": flow})
    bf = daily_baseflow(flow, k, C)["baseflow"]
    day = flow.index.normalize()
    bf_day = pd.Series(bf.reindex(day).to_numpy(), index=flow.index)
    bf_prev = pd.Series(bf.reindex(day - pd.Timedelta(days=1)).to_numpy(), index=flow.index)
    if target == "baseflow":
        out = pd.DataFrame({"target": bf_day, "causal": bf_prev})
    else:
        out = pd.DataFrame({"target": (flow - bf_day).clip(lower=0), "causal": (flow - bf_prev).clip(lower=0)})
    return out.where(flow.notna())
