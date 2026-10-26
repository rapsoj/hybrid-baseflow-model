"""
Data loading, forecast alignment and sample construction.

Forecast origin ("t0")
    One sample per day at data.origin_time (UTC). Everything the model sees must be knowable at t0:
      * encoder: HDE rainfall + groundwater with timestamps in (t0 - history, t0]
      * decoder: the latest GEFS 00Z init with init_time + availability_lag <= t0, and the latest GeoSmart
        file whose last historic (TimeSeriesID 0) timestamp (+ lag) <= t0
      * target : HDE flow with timestamps in (t0, t0 + horizon], averaged into output_step_minutes bins
    Forecasts are matched to decoder tokens by *valid time* (GEFS: init_time + lead_time), never by lead index,
    so a stale forecast can never be silently mis-aligned.

Decoder tokens
    Token j (j = 1..N) covers (t0 + (j-1)*bin, t0 + j*bin] and carries the value at its end time (valid time).
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import geosmart_store
from .baseflow import daily_baseflow, target_series

logger = logging.getLogger(__name__)

ENC_CHANNELS = ["rain_log1p", "groundwater", "rain_missing", "groundwater_missing"]
# "geosmart_member" is one ensemble member's level trajectory. In `SampleSet.dec` it holds the ensemble mean as a
# placeholder; training and inference substitute individual members (see pipeline/ensemble.py).
# "anchor_log_flow" (log1p of the recent observed flow, constant over the horizon) is only used by target_modes delta_log_last and delta_log_last_anomaly
DEC_CHANNELS = ["gefs_mean_log1p", "gefs_std_log1p", "geosmart_member", "season_sin", "season_cos", "anchor_log_flow"]
DEC_GROUPS = {"gefs": [0, 1], "geosmart": [2], "season": [3, 4], "anchor": [5]}
GEO_CH = DEC_GROUPS["geosmart"][0]


# ---------------------------------------------------------------------------
# HDE
# ---------------------------------------------------------------------------
def _read_hde_csv(hde_dir: Path, suffix: str, step_minutes: int) -> pd.Series:
    matches = sorted(Path(hde_dir).glob(f"*_{suffix}.csv"))
    if not matches:
        raise FileNotFoundError(f"No '*_{suffix}.csv' in {hde_dir}. Run the download stage first.")
    if len(matches) > 1:
        logger.warning("Several files match *_%s.csv, using %s", suffix, matches[0].name)
    df = pd.read_csv(matches[0], parse_dates=["dateTime"])
    s = df.set_index(df["dateTime"].dt.round(f"{step_minutes}min"))["value"].astype(float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_hde(hde_dir, step_minutes: int, data_cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Rainfall (mm/step), groundwater level and flow on one regular grid; gaps stay NaN.

    With `data_cfg` (the config's `data` section) and `data.target` other than total_flow, the `flow` column holds the
    predicted component (baseflow or runoff) and `total_flow` the observed flow; `flow_causal` is the origin-safe form
    of the component (see baseflow.target_series).
    """
    series = {
        "rain": _read_hde_csv(hde_dir, "rainfall_15min", step_minutes),
        "gw": _read_hde_csv(hde_dir, "groundwaterLevel_15min", step_minutes),
        "flow": _read_hde_csv(hde_dir, "waterFlow_15min", step_minutes),
    }
    start = min(s.index.min() for s in series.values())
    end = max(s.index.max() for s in series.values())
    grid = pd.date_range(start, end, freq=f"{step_minutes}min")
    hde = pd.DataFrame({k: s.reindex(grid) for k, s in series.items()})
    target = (data_cfg or {}).get("target", "total_flow")
    hde["total_flow"] = hde["flow"]
    bf = (data_cfg or {}).get("baseflow", {"boughton_k": 0.95, "boughton_C": 0.5})
    comp = target_series(hde["total_flow"], target, bf["boughton_k"], bf["boughton_C"])
    hde["flow"], hde["flow_causal"] = comp["target"], comp["causal"]
    return hde


# ---------------------------------------------------------------------------
# GEFS
# ---------------------------------------------------------------------------
def build_gefs_bins(df: pd.DataFrame, bin_hours: int) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Convert the GEFS ensemble mean/variance of precipitation *rate* (mm/s, averaged since the previous step)
    into accumulations (mm) per `bin_hours` window, keyed by init_time.

    Result per init: DataFrame indexed by valid_time = end of bin (init_time + lead_time) with columns
    mean_mm and std_mm. Sub-steps inside a bin are assumed independent when combining variances.
    Bins that are not completely covered by sub-steps are dropped.
    """
    df = df[["init_time", "lead_hours", "ensemble_mean", "ensemble_variance"]].copy()
    df["init_time"] = pd.to_datetime(df["init_time"])
    df = df.sort_values(["init_time", "lead_hours"])
    df["interval_h"] = df.groupby("init_time")["lead_hours"].diff()
    df = df.dropna(subset=["interval_h"])  # lead 0 has no accumulation window

    seconds = df["interval_h"] * 3600.0
    df["mean_mm"] = df["ensemble_mean"] * seconds
    df["var_mm"] = df["ensemble_variance"] * seconds ** 2
    df["bin_lead"] = np.ceil(df["lead_hours"] / bin_hours) * bin_hours

    agg = df.groupby(["init_time", "bin_lead"]).agg(
        mean_mm=("mean_mm", "sum"), var_mm=("var_mm", "sum"), span_h=("interval_h", "sum")
    ).reset_index()
    agg = agg[np.isclose(agg["span_h"], bin_hours)].dropna(subset=["mean_mm", "var_mm"])
    agg["std_mm"] = np.sqrt(agg["var_mm"].clip(lower=0))
    agg["valid_time"] = agg["init_time"] + pd.to_timedelta(agg["bin_lead"], unit="h")

    return {
        init: grp.set_index("valid_time")[["mean_mm", "std_mm"]]
        for init, grp in agg.groupby("init_time")
    }


def load_gefs(csv_path, bin_hours: int) -> Dict[pd.Timestamp, pd.DataFrame]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"GEFS file {path} not found. Run the download stage first.")
    df = pd.read_csv(path, parse_dates=["init_time", "valid_time"])
    # valid_time must equal init_time + lead_time; guards against a corrupted/misaligned export
    delta = (df["valid_time"] - df["init_time"]).dt.total_seconds() / 3600.0
    if not np.allclose(delta.to_numpy(), df["lead_hours"].to_numpy()):
        raise ValueError("GEFS csv has valid_time != init_time + lead_hours")
    return build_gefs_bins(df, bin_hours)


# ---------------------------------------------------------------------------
# GeoSmart
# ---------------------------------------------------------------------------
@dataclass
class GeosmartForecast:
    issue_time: pd.Timestamp            # last historic (ID 0) timestamp in the file + availability lag
    anchor_time: pd.Timestamp           # last historic timestamp (observation known at issue time)
    anchor_level: float
    knot_times: np.ndarray              # datetime64[ns], daily forecast times
    mean: np.ndarray                    # over `members`
    std: np.ndarray
    members: np.ndarray                 # (n_knots, n_members): members present on every knot

    def _x(self, times: pd.DatetimeIndex):
        xk = np.concatenate([[self.anchor_time.value], self.knot_times.astype("datetime64[ns]").astype(np.int64)])
        return times.as_unit("ns").asi8.astype(np.float64), xk.astype(np.float64)  # pandas>=3 may hold s/us units

    def at(self, times: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """Linear interpolation of ensemble mean/std; flat outside the knots. Std is 0 at the anchor."""
        x, xk = self._x(times)
        mean = np.interp(x, xk, np.concatenate([[self.anchor_level], self.mean]))
        std = np.interp(x, xk, np.concatenate([[0.0], self.std]))
        return mean, std

    def members_at(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Each member's trajectory interpolated to `times`, all starting at the observed anchor: (n_members, len(times))."""
        x, xk = self._x(times)
        return np.stack([
            np.interp(x, xk, np.concatenate([[self.anchor_level], self.members[:, j]]))
            for j in range(self.members.shape[1])
        ])


def parse_geosmart_file(
    df: pd.DataFrame, min_member_fraction: float, min_members: int, availability_lag_hours: float
) -> Optional[GeosmartForecast]:
    """
    Ensemble statistics from one GeoSmart file. TimeSeriesID 0 is historic data; every other ID is an ensemble
    member. Statistics use the members that persist to the last forecast day (a subset of the full ensemble, e.g.
    44 of 75), so there is no discontinuity when the ensemble shrinks. If that subset is smaller than
    `min_member_fraction` of the full ensemble, only the leading run of days with (nearly) the full ensemble is kept.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M")
    hist = df[(df["TimeSeriesID"] == 0) & df["Level"].notna()].sort_values("Date")  # last *valid* observation
    if hist.empty:
        return None
    anchor_time = hist["Date"].iloc[-1]
    anchor_level = float(hist["Level"].iloc[-1])

    members = df[(df["TimeSeriesID"] != 0) & (df["Date"] > anchor_time)]
    if members.empty:
        return None
    counts = members.groupby("Date")["Level"].count().sort_index()
    # Later days come from a smaller subset of members. If the members present on the last day are a large enough
    # subset, use only them on every day, so the statistics stay consistent across the whole horizon.
    persistent = set(members.loc[members["Date"] == counts.index[-1], "TimeSeriesID"])
    if len(persistent) >= max(min_members, min_member_fraction * counts.max()):
        members = members[members["TimeSeriesID"].isin(persistent)]
    g = members.groupby("Date")["Level"]
    stats = pd.DataFrame({"n": g.count(), "mean": g.mean(), "std": g.std(ddof=1)}).sort_index()
    ok = (stats["n"] >= max(min_members, min_member_fraction * stats["n"].max())) & stats["std"].notna()
    keep = ok.to_numpy().cumprod().astype(bool)  # leading run only (fallback when no persistent subset exists)
    stats = stats[keep]
    if stats.empty:
        return None
    # members complete on every kept day; their trajectories are what the model is trained and run on
    wide = members.pivot_table(index="Date", columns="TimeSeriesID", values="Level").reindex(stats.index)
    wide = wide.dropna(axis=1)
    if wide.shape[1] < min_members:
        return None
    stats = pd.DataFrame({"mean": wide.mean(axis=1), "std": wide.std(axis=1, ddof=1)})
    return GeosmartForecast(
        issue_time=anchor_time + pd.Timedelta(hours=availability_lag_hours),
        anchor_time=anchor_time,
        anchor_level=anchor_level,
        knot_times=stats.index.to_numpy(),
        mean=stats["mean"].to_numpy(),
        std=stats["std"].to_numpy(),
        members=wide.to_numpy(np.float64),
    )


def load_geosmart(geosmart_dir, prefix: str, cfg_geo: dict) -> List[GeosmartForecast]:
    """Forecasts from the consolidated store (`pipeline/geosmart_store.py`) plus any loose csvs not yet consolidated."""
    frames = geosmart_store.frames_by_issue_date(geosmart_dir, prefix)
    loose = {d: f for d, f in geosmart_store.loose_files(geosmart_dir, prefix).items() if d not in frames}
    if not frames and not loose:
        raise FileNotFoundError(
            f"No GeoSmart forecasts for '{prefix}' in {geosmart_dir} (run python -m processing.download_geosmart {prefix})"
        )
    sources = [(d, df) for d, df in frames.items()] + [(d, f) for d, f in loose.items()]
    out = []
    for d, src in sources:
        df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src)
        fc = parse_geosmart_file(df, cfg_geo["min_member_fraction"], cfg_geo["min_members"], cfg_geo["availability_lag_hours"])
        if fc is None:
            logger.warning("Skipping unusable GeoSmart forecast %s %s", prefix, d)
            continue
        out.append(fc)
    out.sort(key=lambda x: x.issue_time)
    return out


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
@dataclass
class SampleSet:
    origins: pd.DatetimeIndex
    enc: np.ndarray                  # (N, history_steps, len(ENC_CHANNELS)), rain log1p, gw raw, masks; NaN = missing
    dec: np.ndarray                  # (N, n_decoder_tokens, len(DEC_CHANNELS)); NaN where a source is unavailable
    y: np.ndarray                    # (N, horizon_steps) mean flow per output bin (m3/s)
    last_flow: np.ndarray            # (N,) latest observed flow at the origin (m3/s); used by the persistence baseline
    anchor_flow: np.ndarray          # (N,) mean observed flow over the last `anchor_hours` before the origin (m3/s)
    has_gefs: np.ndarray
    has_geosmart: np.ndarray
    window_start: pd.DatetimeIndex   # first encoder timestamp
    window_end: pd.DatetimeIndex     # last target timestamp
    bfi: np.ndarray                  # (N, horizon_steps) baseflow index of the day of each target bin (NaN = unknown); reporting only
    geo_members: np.ndarray          # (N, max_members, n_decoder_tokens) GeoSmart member trajectories; NaN padded
    geo_n: np.ndarray                # (N,) number of members per sample (0 without GeoSmart)
    meta: dict = field(default_factory=dict)
    geo_origin: Optional[np.ndarray] = None  # (N, max_members) each member's level at the forecast origin; NaN padded

    def __len__(self):
        return len(self.origins)

    def subset(self, idx) -> "SampleSet":
        idx = np.asarray(idx)
        return SampleSet(
            self.origins[idx], self.enc[idx], self.dec[idx], self.y[idx], self.last_flow[idx], self.anchor_flow[idx], self.has_gefs[idx],
            self.has_geosmart[idx], self.window_start[idx], self.window_end[idx], self.bfi[idx], self.geo_members[idx], self.geo_n[idx],
            dict(self.meta),
            None if self.geo_origin is None else self.geo_origin[idx],
        )


def geosmart_representation(s: SampleSet, mode: str = "level", ensemble: str = "members") -> SampleSet:
    """
    How the model sees the GeoSmart forecast.
      mode     'level'   : absolute groundwater level (default)
               'anomaly' : level minus the member's own level at the forecast origin, so only the trajectory is seen
      ensemble 'members' : every member is a separate input (one drawn per sample in training, pooled at inference)
               'mean'    : a single input, the ensemble mean
    The decoder placeholder channel always holds the ensemble mean of the chosen representation.
    """
    if mode not in ("level", "anomaly") or ensemble not in ("members", "mean"):
        raise ValueError(f"Unknown GeoSmart representation: mode={mode!r}, ensemble={ensemble!r}")
    if mode == "level" and ensemble == "members":
        return s
    members = s.geo_members.copy()
    if mode == "anomaly":
        members = members - s.geo_origin[:, :, None]
    with np.errstate(all="ignore"):
        mean = np.nanmean(members, axis=1) if members.shape[1] else np.full(s.dec.shape[:2], np.nan, np.float32)
    dec = s.dec.copy()
    has = s.geo_n > 0
    dec[has, :, GEO_CH] = mean[has]
    geo_n = s.geo_n
    if ensemble == "mean":
        members = mean[:, None, :].astype(np.float32) if members.shape[1] else members
        members[~has] = np.nan
        geo_n = np.where(has, 1, 0)
    return SampleSet(
        s.origins, s.enc, dec, s.y, s.last_flow, s.anchor_flow, s.has_gefs, s.has_geosmart, s.window_start,
        s.window_end, s.bfi, members.astype(np.float32), geo_n, dict(s.meta), s.geo_origin,
    )


def _latest_available(avail_times: np.ndarray, t0: pd.Timestamp) -> int:
    """Index of the latest entry whose availability time is <= t0, or -1."""
    return int(np.searchsorted(avail_times, t0.to_datetime64(), side="right")) - 1


def build_samples(cfg: dict, hde: pd.DataFrame, gefs: Optional[dict], geosmart: Optional[List[GeosmartForecast]]) -> SampleSet:
    """Build every candidate daily-origin sample; availability of each forecast source is recorded per sample."""
    d, dv = cfg["data"], cfg["derived"]
    step = pd.Timedelta(minutes=d["step_minutes"])
    bin_td = pd.Timedelta(hours=d["gefs"]["bin_hours"])
    H, T, N_tok = dv["horizon_fine_steps"], dv["history_steps"], dv["n_decoder_tokens"]
    ratio = dv["output_ratio"]
    origin_offset = pd.Timedelta(minutes=dv["origin_minutes"])

    grid = hde.index
    rain = hde["rain"].to_numpy(np.float64)
    gw = hde["gw"].to_numpy(np.float64)
    flow = hde["flow"].to_numpy(np.float64)

    # gw forward-filled over short gaps only, tracking which values are real
    gw_ff = hde["gw"].ffill(limit=int(d["fill_limit_steps"])).to_numpy(np.float64)
    gw_real = ~np.isnan(gw)
    causal = hde["flow_causal"] if "flow_causal" in hde else hde["flow"]  # what is known at the origin
    flow_causal = causal.to_numpy(np.float64)
    flow_ff = causal.ffill(limit=int(d["fill_limit_steps"])).to_numpy(np.float64)

    if gefs:
        gefs_inits = np.array(sorted(gefs), dtype="datetime64[ns]")
        gefs_avail = gefs_inits + np.timedelta64(int(d["gefs"]["availability_lag_hours"] * 3600), "s")
    if geosmart:
        geo_avail = np.array([f.issue_time.to_datetime64() for f in geosmart], dtype="datetime64[ns]")

    anchor_steps = max(1, int(round(d["anchor_hours"] * 60 / d["step_minutes"])))
    bf_cfg = d["baseflow"]
    day_bfi = daily_baseflow(hde["total_flow"] if "total_flow" in hde else hde["flow"], bf_cfg["boughton_k"], bf_cfg["boughton_C"])["bfi"]
    out_td = pd.Timedelta(minutes=d["output_step_minutes"])
    first_day = (grid[0] + pd.Timedelta(days=d["history_days"])).normalize()
    last_day = (grid[-1] - pd.Timedelta(days=d["horizon_days"])).normalize()
    days = pd.date_range(first_day, last_day, freq="D")

    max_missing = d["max_history_missing_fraction"]
    rows = dict(origin=[], enc=[], dec=[], y=[], last=[], anchor=[], g=[], s=[], w0=[], w1=[], gm=[], go=[], bfi=[])
    stats = {"candidate_days": len(days), "no_history": 0, "no_target": 0, "no_last_flow": 0}

    for day in days:
        t0 = day + origin_offset
        i0 = grid.get_indexer([t0])[0]
        if i0 < T - 1 or i0 + H >= len(grid):
            continue
        sl = slice(i0 - T + 1, i0 + 1)
        r, g_fill, g_real = rain[sl], gw_ff[sl], gw_real[sl]
        if max(np.isnan(r).mean(), np.isnan(g_fill).mean()) > max_missing:
            stats["no_history"] += 1
            continue
        y = flow[i0 + 1: i0 + 1 + H]
        if np.isnan(y).any():
            stats["no_target"] += 1
            continue
        if np.isnan(flow_ff[i0]):  # persistence baseline needs the latest observed flow; keep all variants on the same samples
            stats["no_last_flow"] += 1
            continue
        anchor = float(np.nanmean(flow_causal[max(0, i0 - anchor_steps + 1): i0 + 1]))  # last real observations (>= 1 exists)
        y = y.reshape(-1, ratio).mean(axis=1)  # mean flow over each output bin, labelled by the bin end

        enc = np.stack([
            np.log1p(np.clip(r, 0, None)),
            g_fill,
            np.isnan(r).astype(np.float64),
            (~g_real | np.isnan(g_fill)).astype(np.float64),  # 1 if the value is missing or only forward-filled
        ], axis=1)

        # ---- decoder tokens ----
        token_ends = pd.DatetimeIndex([t0 + (j + 1) * bin_td for j in range(N_tok)])
        dec = np.full((N_tok, len(DEC_CHANNELS)), np.nan)
        doy = token_ends.dayofyear.to_numpy() / 365.25
        dec[:, DEC_GROUPS["anchor"][0]] = np.log1p(anchor)
        dec[:, DEC_GROUPS["season"][0]] = np.sin(2 * np.pi * doy)
        dec[:, DEC_GROUPS["season"][1]] = np.cos(2 * np.pi * doy)

        has_g = False
        if gefs:
            k = _latest_available(gefs_avail, t0)
            if k >= 0 and t0 - pd.Timestamp(gefs_inits[k]) <= pd.Timedelta(hours=d["gefs"]["max_age_hours"]):
                bins = gefs[pd.Timestamp(gefs_inits[k])].reindex(token_ends)
                if not bins.isna().any().any():
                    dec[:, 0] = np.log1p(bins["mean_mm"].to_numpy())
                    dec[:, 1] = np.log1p(bins["std_mm"].to_numpy())
                    has_g = True

        has_s = False
        members = None
        origin_levels = None
        if geosmart:
            k = _latest_available(geo_avail, t0)
            if k >= 0:
                fc = geosmart[k]
                age_ok = t0 - fc.issue_time <= pd.Timedelta(hours=d["geosmart"]["max_age_hours"])
                # daily knots: the last token may hold the last knot flat, but not extrapolate further than a day
                covered = pd.Timestamp(fc.knot_times[-1]) >= token_ends[-1] - pd.Timedelta(days=1)
                if age_ok and covered and fc.anchor_time <= t0:
                    m, _ = fc.at(token_ends)
                    dec[:, GEO_CH] = m
                    members = fc.members_at(token_ends)
                    origin_levels = fc.members_at(pd.DatetimeIndex([t0]))[:, 0]
                    has_s = True

        bin_mid = pd.DatetimeIndex([t0 + (j + 0.5) * out_td for j in range(len(y))])
        bfi = day_bfi.reindex(bin_mid.normalize()).to_numpy()

        for key, val in zip(rows, (t0, enc, dec, y, flow_ff[i0], anchor, has_g, has_s, grid[i0 - T + 1], grid[i0 + H], members, origin_levels, bfi)):
            rows[key].append(val)

    if not rows["origin"]:
        raise RuntimeError(f"No usable samples were found ({stats}). Check the downloaded data coverage.")

    n_max = max((m.shape[0] for m in rows["gm"] if m is not None), default=0)
    geo_members = np.full((len(rows["origin"]), n_max, N_tok), np.nan, dtype=np.float32)
    geo_n = np.zeros(len(rows["origin"]), dtype=int)
    geo_origin = np.full((len(rows["origin"]), n_max), np.nan, dtype=np.float32)
    for i, (m, o) in enumerate(zip(rows["gm"], rows["go"])):
        if m is not None:
            geo_members[i, : m.shape[0]] = m
            geo_origin[i, : m.shape[0]] = o
            geo_n[i] = m.shape[0]

    out = SampleSet(
        origins=pd.DatetimeIndex(rows["origin"]),
        enc=np.stack(rows["enc"]).astype(np.float32),
        dec=np.stack(rows["dec"]).astype(np.float32),
        y=np.stack(rows["y"]).astype(np.float32),
        last_flow=np.array(rows["last"], dtype=np.float32),
        anchor_flow=np.array(rows["anchor"], dtype=np.float32),
        has_gefs=np.array(rows["g"]),
        has_geosmart=np.array(rows["s"]),
        window_start=pd.DatetimeIndex(rows["w0"]),
        window_end=pd.DatetimeIndex(rows["w1"]),
        bfi=np.stack(rows["bfi"]).astype(np.float32),
        geo_members=geo_members,
        geo_n=geo_n,
        meta=stats,
        geo_origin=geo_origin,
    )
    logger.info(
        "Built %d samples (%s .. %s); GEFS available for %d, GeoSmart for %d",
        len(out), out.origins[0].date(), out.origins[-1].date(), out.has_gefs.sum(), out.has_geosmart.sum(),
    )
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def purged_chronological_split(
    origins: pd.DatetimeIndex, gap: pd.Timedelta, train_fraction: float, val_fraction: float
) -> Dict[str, np.ndarray]:
    """
    Train < val < test in time. Samples cover [t0 - history, t0 + horizon], so a split may only start after
    the previous split's last window has ended: next_t0 > prev_last_t0 + gap with gap = history + horizon.
    """
    origins = pd.DatetimeIndex(origins)
    if not origins.is_monotonic_increasing:
        raise ValueError("origins must be sorted")
    if len(origins) == 0:
        return {k: np.array([], int) for k in ("train", "val", "test")}
    # Fractions apply to the span left after reserving the two purge gaps, so the gaps never starve a split.
    span = origins[-1] - origins[0]
    usable = span - 2 * gap
    if usable <= pd.Timedelta(0):
        return {k: np.array([], int) for k in ("train", "val", "test")}
    train_end = origins[0] + usable * train_fraction
    val_start = train_end + gap
    val_end = val_start + usable * val_fraction
    test_start = val_end + gap
    return {
        "train": np.flatnonzero(origins <= train_end),
        "val": np.flatnonzero((origins > val_start) & (origins <= val_end)),
        "test": np.flatnonzero(origins > test_start),
    }


def assert_no_temporal_overlap(samples: SampleSet, splits: Dict[str, np.ndarray]) -> None:
    """Hard check: the [window_start, window_end] intervals of different splits never intersect."""
    names = [n for n in ("train", "val", "test") if len(splits[n])]
    spans = {n: (samples.window_start[splits[n]].min(), samples.window_end[splits[n]].max()) for n in names}
    for a, b in zip(names[:-1], names[1:]):
        if spans[a][1] >= spans[b][0]:
            raise AssertionError(f"Temporal overlap between {a} {spans[a]} and {b} {spans[b]}")


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
@dataclass
class Scalers:
    """Standardisation statistics fitted on the training split only."""
    enc_mean: np.ndarray
    enc_std: np.ndarray
    dec_mean: np.ndarray
    dec_std: np.ndarray
    y_mean: float
    y_std: float

    @staticmethod
    def fit(train: SampleSet, offset: np.ndarray) -> "Scalers":
        def ms(x, axes):
            mean = np.nanmean(x, axis=axes)
            std = np.nanstd(x, axis=axes)
            return np.nan_to_num(mean), np.where(np.nan_to_num(std) < 1e-6, 1.0, np.nan_to_num(std))

        enc_mean, enc_std = ms(train.enc, (0, 1))
        enc_mean[2:], enc_std[2:] = 0.0, 1.0  # mask channels are left as 0/1

        dec_mean, dec_std = np.zeros(len(DEC_CHANNELS)), np.ones(len(DEC_CHANNELS))
        for name, ch in (("gefs", DEC_GROUPS["gefs"]), ("geosmart", DEC_GROUPS["geosmart"])):
            sel = train.has_gefs if name == "gefs" else train.has_geosmart
            if sel.any():
                m, s = ms(train.dec[sel][:, :, ch], (0, 1))
                dec_mean[ch], dec_std[ch] = m, s
        # one scale for every GeoSmart member, fitted on the members themselves
        if train.has_geosmart.any():
            gm = train.geo_members[train.has_geosmart]
            dec_mean[GEO_CH], dec_std[GEO_CH] = float(np.nanmean(gm)), max(float(np.nanstd(gm)), 1e-6)

        anc = DEC_GROUPS["anchor"][0]
        m, s = ms(train.dec[:, :, [anc]], (0, 1))
        dec_mean[anc], dec_std[anc] = m[0], s[0]
        ty = np.log1p(np.clip(train.y, 0, None)) - offset
        return Scalers(enc_mean, enc_std, dec_mean, dec_std, float(ty.mean()), float(max(ty.std(), 1e-6)))

    def transform(self, s: SampleSet, dec_channels: List[int], offset: np.ndarray):
        enc = (s.enc - self.enc_mean) / self.enc_std
        enc = np.nan_to_num(enc, nan=0.0)
        dec = (s.dec - self.dec_mean) / self.dec_std
        dec = np.nan_to_num(dec, nan=0.0)[:, :, dec_channels]
        y = (np.log1p(np.clip(s.y, 0, None)) - offset - self.y_mean) / self.y_std
        return enc.astype(np.float32), dec.astype(np.float32), y.astype(np.float32)

    def transform_members(self, s: SampleSet) -> np.ndarray:
        """Scaled GeoSmart member trajectories (N, max_members, T); padding is 0 and is never drawn (see geo_n)."""
        g = (s.geo_members - self.dec_mean[GEO_CH]) / self.dec_std[GEO_CH]
        return np.nan_to_num(g, nan=0.0).astype(np.float32)

    def inverse_flow(self, y_scaled: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Scaled predictions (..., H, Q) back to m3/s; offset (H,) or (N, H) is the log-space target offset."""
        off = np.asarray(offset)[..., None]
        return np.clip(np.expm1(y_scaled * self.y_std + self.y_mean + off), 0, None)

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def variant_dec_channels(use_gefs: bool, use_geosmart: bool, use_anchor: bool = False) -> List[int]:
    ch = list(DEC_GROUPS["season"])
    if use_anchor:
        ch = ch + DEC_GROUPS["anchor"]
    if use_gefs:
        ch = DEC_GROUPS["gefs"] + ch
    if use_geosmart:
        ch = ch + DEC_GROUPS["geosmart"]
    return sorted(ch)


def historic_flow_threshold(hde: pd.DataFrame, end_time: pd.Timestamp, quantile: float) -> float:
    """Flow quantile over all observations up to and including end_time (never looks past the train split)."""
    flow = hde.loc[:end_time, "flow"].dropna()
    if flow.empty:
        raise RuntimeError("No historic flow available to compute the extreme-event threshold")
    return float(flow.quantile(quantile))
