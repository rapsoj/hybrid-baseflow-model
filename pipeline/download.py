"""
Download stage: HDE (groundwater, rainfall, flow) and GEFS forecasts for one borehole.

Data is stored locally (paths in config.yaml). If the local file already exists the download is skipped without
any network call, unless download.update is true (incremental top-up) or download.gefs.refresh is true (full redo).
"""
import logging
import math
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def download_hde(cfg: dict) -> None:
    """Groundwater and rainfall (15 min) plus flow (15 min) for the borehole's configured stations."""
    from processing.download_hde import (
        download_hde_groundwater_15min,
        download_hde_rainfall_15min,
        download_hde_river_data_15min,
    )

    key = cfg["borehole_key"]
    expected = os.path.normpath(os.path.join("data", "hde", key))
    if os.path.normpath(cfg["paths"]["hde_dir"]) != expected:
        raise ValueError(f"paths.hde_dir must be '{expected}' (processing/download_hde.py writes there)")

    update = bool(cfg["download"].get("update", False))
    hde_dir = Path(cfg["paths"]["hde_dir"])
    stations = cfg["borehole"]["hde_stations"]
    from processing.download_hde import make_session

    with make_session() as sess:
        jobs = (
            ("groundwater", "groundwaterLevel_15min",
             lambda g: download_hde_groundwater_15min(g, output_prefix=key, session=sess, update_existing=update)),
            ("rainfall", "rainfall_15min",
             lambda g: download_hde_rainfall_15min(g, output_prefix=key, session=sess, update_existing=update)),
            ("flow", "waterFlow_15min",
             lambda g: download_hde_river_data_15min(g, "waterFlow", output_prefix=key, session=sess, update_existing=update)),
        )
        for label, suffix, fn in jobs:
            have = sorted(hde_dir.glob(f"*_{suffix}.csv"))
            if have and not update:
                logger.info("HDE %s: already downloaded (%s), skipping", label, have[0])
                continue
            logger.info("HDE %s: %s (%s)", label, "updating" if have else "downloading", stations[label])
            fn(stations[label])


def download_gefs(cfg: dict) -> None:
    """GEFS precipitation ensemble mean/variance at the borehole, appended incrementally."""
    download_gefs_many([cfg])


def _gefs_plan(cfg: dict):
    """Returns (existing, start) for a borehole, or None if its GEFS file is already complete (no network needed)."""
    g = cfg["download"]["gefs"]
    out = Path(cfg["paths"]["gefs_csv"])
    max_lead = math.ceil(
        (cfg["derived"]["origin_minutes"] / 60 + cfg["data"]["decoder_days"] * 24)
        / cfg["data"]["gefs"]["bin_hours"]
    ) * cfg["data"]["gefs"]["bin_hours"]

    update = bool(cfg["download"].get("update", False))
    start = pd.Timestamp(g["start"])
    existing = None
    if out.exists() and not g["refresh"]:
        # a quick check reads just two columns; the full file is only loaded when it will be appended to
        head = pd.read_csv(out, usecols=["init_time", "lead_hours"], parse_dates=["init_time"])
        if head.empty or head["lead_hours"].max() < max_lead:
            logger.info("GEFS: existing file has too short a lead range, re-downloading")
        elif not update:
            logger.info("GEFS: already downloaded (%s, %d inits), skipping", out, head["init_time"].nunique())
            return None
        else:
            existing = pd.read_csv(out, parse_dates=["init_time", "valid_time"])
            start = max(start, head["init_time"].max() + pd.Timedelta(days=1))

    end = g["end"]
    if end is not None and pd.Timestamp(end) < start:
        logger.info("GEFS: already up to date (%s)", out)
        return None
    return existing, start, max_lead


def download_gefs_many(cfgs: list) -> None:
    """
    GEFS for several boreholes. Boreholes that need data are fetched together (grouped by start date), so boreholes
    that fall in the same archive chunk share one download instead of repeating it.
    """
    from processing.download_gefs import export_gefs_precip_forecasts_multi

    plans = {}
    for cfg in cfgs:
        plan = _gefs_plan(cfg)
        if plan is not None:
            plans[cfg["borehole_key"]] = (cfg, *plan)

    groups = {}
    for key, (cfg, existing, start, max_lead) in plans.items():
        groups.setdefault((start, cfg["download"]["gefs"]["end"], max_lead), []).append(key)

    for (start, end, max_lead), keys in groups.items():
        workers = max(int(plans[k][0]["download"]["gefs"].get("workers", 1)) for k in keys)
        sites = {k: (plans[k][0]["borehole"]["latitude"], plans[k][0]["borehole"]["longitude"]) for k in keys}
        logger.info("GEFS: downloading %s from %s to %s (leads <= %dh)", ", ".join(keys), start.date(), end or "latest", max_lead)
        results = export_gefs_precip_forecasts_multi(sites, start=start, end=end, max_lead_hours=max_lead, workers=workers)
        for k in keys:
            cfg, existing, _, _ = plans[k]
            out = Path(cfg["paths"]["gefs_csv"])
            out.parent.mkdir(parents=True, exist_ok=True)
            new = results[k]
            if existing is not None and not new.empty:
                new = pd.concat([existing, new], ignore_index=True)
            elif existing is not None:
                new = existing
            if new.empty:
                logger.warning("GEFS %s: no forecasts returned", k)
                continue
            new = new.drop_duplicates(["init_time", "lead_hours"]).sort_values(["init_time", "valid_time"])
            new.to_csv(out, index=False)
            logger.info("GEFS %s: %d rows, %d inits -> %s", k, len(new), new["init_time"].nunique(), out)
