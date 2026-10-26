import pandas as pd
import xarray as xr
import dynamical_catalog


_COLUMNS = ["init_time", "lead_time", "valid_time", "ensemble_mean", "ensemble_variance", "lead_hours"]


def export_gefs_precip_forecasts_multi(sites, start="2020-01-01", end=None, max_lead_hours=None, workers=1):
    """
    Export GEFS surface precipitation ensemble statistics (mean and variance per init_time / valid_time) for several
    sites at once: {key: (lat, lon)} -> {key: DataFrame}.

    The archive is chunked 1 init x 31 members x 64 leads x 17 lat x 16 lon, so sites that share a spatial chunk are
    served by the same download. Each (init, chunk) block is read once and every site in it is cut from that block,
    which makes N nearby boreholes cost roughly the same as one.
    """
    import numpy as np

    ds = dynamical_catalog.open("noaa-gefs-forecast-35-day", chunks=None)
    init_times = pd.DatetimeIndex(ds.init_time.values)
    keep = (init_times >= pd.Timestamp(start)) & (init_times.hour == 0)
    if end is not None:
        keep &= init_times <= pd.Timestamp(end)
    inits = list(init_times[keep])

    precip = ds["precipitation_surface"]
    if max_lead_hours is not None:
        precip = precip.sel(lead_time=precip.lead_time <= pd.Timedelta(hours=max_lead_hours))

    lats, lons = ds.latitude.values, ds.longitude.values
    cells = {k: (int(np.abs(lats - la).argmin()), int(np.abs(lons - lo).argmin())) for k, (la, lo) in sites.items()}
    groups = {}
    for k, (i, j) in cells.items():
        groups.setdefault((i // 17, j // 16), []).append(k)

    def _fetch(init):
        frames = {k: [] for k in sites}
        for keys in groups.values():
            ii = [cells[k][0] for k in keys]
            jj = [cells[k][1] for k in keys]
            block = precip.sel(init_time=[init]).isel(
                latitude=slice(min(ii), max(ii) + 1), longitude=slice(min(jj), max(jj) + 1)
            ).load()
            for k, i, j in zip(keys, ii, jj):
                p = block.isel(latitude=i - min(ii), longitude=j - min(jj))
                stats = xr.Dataset({
                    "ensemble_mean": p.mean("ensemble_member"),
                    "ensemble_variance": p.var("ensemble_member"),
                })
                frames[k].append(stats.to_dataframe().reset_index())
        return {k: v[0] for k, v in frames.items()}

    if workers > 1 and len(inits) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_init = list(pool.map(_fetch, inits))
    else:
        per_init = [_fetch(i) for i in inits]

    result = {}
    for k in sites:
        if not per_init:
            result[k] = pd.DataFrame(columns=_COLUMNS)
            continue
        df = pd.concat([f[k] for f in per_init], ignore_index=True)
        df["lead_hours"] = (df["valid_time"] - df["init_time"]).dt.total_seconds() / 3600
        keep_cols = [c for c in _COLUMNS if c in df.columns]
        result[k] = df.sort_values(["init_time", "valid_time"]).reset_index(drop=True)[keep_cols]
    return result

