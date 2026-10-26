"""
Match chalk-river flow stations to a borehole, rainfall station and GeoSmart forecast.

For every Environment Agency Hydrology flow station on a Natural England chalk river (15-min flow data):
  - catchment  : the HydroBASINS polygon containing the flow station
  - groundwater: the nearest station (to the flow station) in that polygon with 15-min groundwater level data and a
                 GeoSmart forecast (so nearer boreholes without a forecast are skipped)
  - rainfall   : the nearest station (to the borehole) in that polygon with 15-min rainfall data
  - GeoSmart   : the forecast prefix for the borehole, via `geosmart_name_lookup.csv` and normalised-name matching
A catchment survives only when all four are found. Each borehole then keeps one flow station, the closest to it (stations
sharing a borehole have identical model inputs and differ only in the target), and each catchment keeps one borehole.
Nothing is downloaded except the station list and the GeoSmart folder listing (names only).

Run from the repository root:
    python -m processing.match_stations
Outputs (data/matching/): chalk_matches.csv (survivors, one flow station per borehole and catchment),
chalk_candidates.csv (every flow station with its match status; 'matched' rows include those later not selected) and
geosmart_boreholes.csv (the GeoSmart forecast prefixes found on S3).
"""
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd

from processing.get_chalk_rivers import (
    DEFAULT_BUFFER_M,
    DEFAULT_TARGET_CRS,
    download_stations_csv,
    find_chalk_stations,
    load_shapefile,
    stations_df_to_gdf,
)

CHALK_SHP = "data/ea/Chalk_Rivers_(England)___Natural_England.shp"
CATCHMENTS_SHP = "data/hydrobasins/uk_catchments.shp"
STATIONS_URL = "https://environment.data.gov.uk/hydrology/id/stations.csv?_limit=20000"
LOOKUP_CSV = Path("processing/geosmart_name_lookup.csv")
OUT_DIR = Path("data/matching")

BUCKET = "gwforecasting.geosmartinfo.co.uk"
AWS_PROFILE = "gwforecasting"
AWS_REGION = "eu-west-1"
# One folder near the start of the archive plus later ones, so boreholes added after 2020 are not missed.
GEOSMART_LIST_DATES = ["2020-01-01", "2023-01-01", "2026-02-27"]
FORECAST_SUFFIX = "_headTimeSeries"

STEP_SECONDS = 900  # 15-minute data in the Hydrology API `measures.period`


def normalise(name: str) -> str:
    """Lower-case alphanumerics only, so 'Chilgrove House' == 'ChilgroveHouse'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def list_geosmart_prefixes(dates: Iterable[str] = GEOSMART_LIST_DATES) -> pd.DataFrame:
    """Forecast prefixes ('<prefix>_headTimeSeries <date>.csv') found in the given S3 daily folders."""
    rows = []
    for d in dates:
        y, m, day = d.split("-")
        month = pd.Timestamp(d).strftime("%m_%B")
        uri = f"s3://{BUCKET}/{y}/{month}/{day}/"
        out = subprocess.run(
            ["aws", "s3", "ls", uri, "--profile", AWS_PROFILE, "--region", AWS_REGION],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            raise RuntimeError(f"aws s3 ls {uri} failed: {out.stderr.strip()}")
        for line in out.stdout.splitlines():
            m_ = re.search(rf"(\S.*?){FORECAST_SUFFIX} {re.escape(d)}\.csv$", line.split(None, 3)[-1])
            if m_:
                rows.append({"geosmart_prefix": m_.group(1), "listed_on": d})
    df = pd.DataFrame(rows)
    return df.groupby("geosmart_prefix", as_index=False).agg(
        first_listed=("listed_on", "min"), last_listed=("listed_on", "max")
    )


def explode_measures(stations: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, measure parameter, period); the CSV stores them as parallel '|'-joined strings."""
    rows = []
    for rec in stations.itertuples(index=False):
        params = str(rec[stations.columns.get_loc("measures.parameter")]).split("|")
        periods = str(rec[stations.columns.get_loc("measures.period")]).split("|")
        for p, per in zip(params, periods):
            rows.append((rec[stations.columns.get_loc("stationGuid")], p, per))
    out = pd.DataFrame(rows, columns=["stationGuid", "parameter", "period"])
    out["period"] = pd.to_numeric(out["period"], errors="coerce")
    return out


def classify_15min(stations: pd.DataFrame) -> pd.DataFrame:
    """Add has_flow_15min / has_rainfall_15min / has_groundwater_15min boolean columns."""
    m = explode_measures(stations)
    m = m[m["period"] == STEP_SECONDS]
    flow = set(m.loc[m["parameter"] == "flow", "stationGuid"])
    rain = set(m.loc[m["parameter"] == "rainfall", "stationGuid"])
    level = set(m.loc[m["parameter"] == "level", "stationGuid"])
    is_gw = stations["type"].astype(str).str.contains("Groundwater", case=False)
    out = stations.copy()
    out["has_flow_15min"] = out["stationGuid"].isin(flow)
    out["has_rainfall_15min"] = out["stationGuid"].isin(rain)
    out["has_groundwater_15min"] = out["stationGuid"].isin(level) & is_gw
    return out


def live_flow_ok(guids: pd.Series, workers: int = 8) -> pd.Series:
    """Boolean mask: the Hydrology API currently serves a 15 min waterFlow measure for the station."""
    from concurrent.futures import ThreadPoolExecutor

    from processing.download_hde import make_session, station_has_15min_flow

    with make_session() as sess, ThreadPoolExecutor(max_workers=workers) as pool:
        ok = list(pool.map(lambda g: station_has_15min_flow(g, session=sess, timeout=30), guids))
    dropped = [g for g, o in zip(guids, ok) if not o]
    if dropped:
        print(f"dropped {len(dropped)} flow stations without a live 15 min waterFlow measure: {dropped}")
    return pd.Series(ok, index=guids.index)


def load_lookup(path: Path = LOOKUP_CSV) -> dict:
    """Manual overrides {normalised HDE groundwater label: GeoSmart prefix}. Rows with an empty prefix mean 'no forecast'."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, keep_default_na=False)
    return {normalise(r.hde_label): r.geosmart_prefix for r in df.itertuples()}


def match_geosmart(label: str, prefixes: pd.DataFrame, lookup: dict) -> Optional[str]:
    """Manual lookup first (must exist on S3), then exact match on the normalised name."""
    available = set(prefixes["geosmart_prefix"])
    key = normalise(label)
    if key in lookup:
        return lookup[key] if lookup[key] in available else None
    by_norm = {normalise(p): p for p in available}
    return by_norm.get(key)


def nearest(station: pd.Series, candidates: gpd.GeoDataFrame):
    """Nearest candidate (projected CRS) to the given geometry; returns (row, distance_m) or (None, None)."""
    if candidates.empty:
        return None, None
    d = candidates.geometry.distance(station)
    idx = d.idxmin()
    return candidates.loc[idx], float(d.loc[idx])


def match_stations(
    stations_df: pd.DataFrame,
    chalk_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    prefixes: pd.DataFrame,
    lookup: dict,
    buffer_m: int = DEFAULT_BUFFER_M,
    crs: int = DEFAULT_TARGET_CRS,
) -> pd.DataFrame:
    """One row per chalk flow station with its matched stations and `status` ('matched' or the reason for dropping)."""
    stations_df = classify_15min(stations_df.drop_duplicates("stationGuid"))
    gdf = stations_df_to_gdf(stations_df, "long", "lat")

    _, chalk_stations = find_chalk_stations(chalk_gdf, gdf, buffer_m=buffer_m)
    flow = chalk_stations[chalk_stations["has_flow_15min"]]
    flow = flow[live_flow_ok(flow["stationGuid"])]  # the stations CSV can list measures the API no longer serves

    proj = gdf.to_crs(epsg=crs)
    cat = catchments_gdf.to_crs(epsg=crs).reset_index(drop=True)
    cat_id = next((c for c in ("HYBAS_ID", "OBJECTID", "FID", "id") if c in cat.columns), None)
    in_catch = gpd.sjoin(proj[["geometry"]], cat[["geometry"]], how="inner", predicate="intersects")
    # a station on a polygon boundary can fall in two; keep the first
    station_catch = in_catch[~in_catch.index.duplicated()]["index_right"]

    rows = []
    for idx in flow.index:
        f = gdf.loc[idx]
        row = {
            "catchment_id": None, "flow_name": f["label"], "flow_guid": f["stationGuid"], "river": f.get("riverName"),
            "flow_lat": f["lat"], "flow_long": f["long"],
            "groundwater_name": None, "groundwater_guid": None, "borehole_lat": None, "borehole_long": None,
            "groundwater_distance_m": None, "rainfall_name": None, "rainfall_guid": None, "rainfall_distance_m": None,
            "geosmart_prefix": None, "status": None,
        }
        rows.append(row)
        if idx not in station_catch.index:
            row["status"] = "no catchment polygon"
            continue
        c = int(station_catch.loc[idx])
        row["catchment_id"] = cat.loc[c, cat_id] if cat_id else c

        members = station_catch[station_catch == c].index
        pool = proj.loc[members]

        gw_pool = pool[pool["has_groundwater_15min"]].copy()
        gw_pool["geosmart_prefix"] = [match_geosmart(n, prefixes, lookup) for n in gw_pool["label"]]
        gw_pool = gw_pool[gw_pool["geosmart_prefix"].notna()]
        gw, gw_d = nearest(proj.loc[idx].geometry, gw_pool)
        if gw is None:
            row["status"] = "no 15-min groundwater station with a GeoSmart forecast in catchment"
            continue
        row.update(groundwater_name=gw["label"], groundwater_guid=gw["stationGuid"], borehole_lat=gw["lat"],
                   borehole_long=gw["long"], groundwater_distance_m=round(gw_d), geosmart_prefix=gw["geosmart_prefix"])

        rain, rain_d = nearest(gw.geometry, pool[pool["has_rainfall_15min"]])
        if rain is None:
            row["status"] = "no 15-min rainfall station in catchment"
            continue
        row.update(rainfall_name=rain["label"], rainfall_guid=rain["stationGuid"], rainfall_distance_m=round(rain_d),
                   status="matched")
    out = pd.DataFrame(rows)
    out["catchment_id"] = out["catchment_id"].astype("Int64")
    return out.sort_values(["status", "catchment_id", "flow_name"]).reset_index(drop=True)


def one_flow_station_per_borehole(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the flow station closest to its borehole, then the closest pair per catchment, so each borehole and each
    catchment appears once. Ties are broken by flow station name for a deterministic result.
    """
    m = matches.sort_values(["groundwater_distance_m", "flow_name"])
    m = m.drop_duplicates("groundwater_guid")
    m = m.drop_duplicates("catchment_id")
    return m.sort_values("catchment_id").reset_index(drop=True)


def main() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefixes = list_geosmart_prefixes()
    prefixes.to_csv(OUT_DIR / "geosmart_boreholes.csv", index=False)

    result = match_stations(
        download_stations_csv(STATIONS_URL),
        load_shapefile(CHALK_SHP),
        load_shapefile(CATCHMENTS_SHP),
        prefixes,
        load_lookup(),
    )
    result.to_csv(OUT_DIR / "chalk_candidates.csv", index=False)
    matched = one_flow_station_per_borehole(result[result["status"] == "matched"])
    summary = matched[["catchment_id", "flow_name", "rainfall_name", "groundwater_name", "geosmart_prefix"]]
    summary = summary.rename(columns={"flow_name": "hde_streamflow", "rainfall_name": "hde_rainfall",
                                      "groundwater_name": "hde_groundwater", "geosmart_prefix": "geosmart_forecast"})
    matched.to_csv(OUT_DIR / "chalk_matches.csv", index=False)
    print(result["status"].value_counts().to_string())
    print()
    print(summary.to_string(index=False))
    return matched


if __name__ == "__main__":
    main()
