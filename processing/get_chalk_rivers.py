# get_chalk_rivers.py
"""
Identify Hydrology stations on/near Natural England Chalk Rivers, then find all rainfall and
groundwater stations in the same catchment polygons. Adds distance (metres) from each linked
station to the chalk station and writes CSV to out_csv_path (index=False).

Primary function:
    get_chalk_rivers(chalk_shp_path, stations_csv_url, catchments_shp_path, buffer_m, out_csv_path)
"""
from typing import Tuple, Optional, List
import os
import io
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DEFAULT_BUFFER_M = 50
DEFAULT_OUT_CSV = "analysis/results/chalk_rivers.csv"
DEFAULT_TARGET_CRS = 27700  # British National Grid for metre distances


# -------------------------
# Basic I/O / helpers
# -------------------------
def load_shapefile(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    return gdf


def download_stations_csv(url: str) -> pd.DataFrame:
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))


def detect_lon_lat_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols_lower = {c.lower(): c for c in df.columns}
    lon_candidates = ["longitude", "lon", "long", "x", "easting"]
    lat_candidates = ["latitude", "lat", "y", "northing"]

    def find(candidates):
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        for cand in candidates:
            for original in df.columns:
                if cand in original.lower():
                    return original
        return None

    lon_col = find(lon_candidates)
    lat_col = find(lat_candidates)
    if lon_col is None or lat_col is None:
        raise RuntimeError(
            "Could not find lon/lat columns in stations CSV. Available columns: "
            + ", ".join(df.columns)
        )
    return lon_col, lat_col


def detect_station_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["stationid", "station_id", "stationGuid", "stationguid", "id", "@id", "notation", "station"]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        if c.lower() in ("id", "station_id", "stationid"):
            return c
    return None


# -------------------------
# Core geoprocessing
# -------------------------
def stations_df_to_gdf(stations_df: pd.DataFrame, lon_col: str, lat_col: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    df = stations_df.copy()
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


def find_chalk_stations(
    chalk_gdf: gpd.GeoDataFrame,
    stations_gdf: gpd.GeoDataFrame,
    buffer_m: int = DEFAULT_BUFFER_M,
    target_crs_epsg: int = DEFAULT_TARGET_CRS,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Return (stations_annotated, chalk_stations).
    stations_annotated: original stations_gdf with boolean 'is_chalk'.
    chalk_stations: subset where is_chalk == True.
    """
    stations_proj = stations_gdf.to_crs(epsg=target_crs_epsg)
    chalk_proj = chalk_gdf.to_crs(epsg=target_crs_epsg)

    chalk_buffered = gpd.GeoDataFrame(geometry=chalk_proj.geometry.buffer(buffer_m), crs=chalk_proj.crs)

    matched = gpd.sjoin(stations_proj, chalk_buffered, how="inner", predicate="intersects")
    matched_idxs = matched.index.unique()

    stations_annotated = stations_gdf.copy()
    stations_annotated["is_chalk"] = stations_gdf.index.isin(matched_idxs).tolist()
    chalk_stations = stations_annotated[stations_annotated["is_chalk"]].copy()
    return stations_annotated, chalk_stations


def find_stations_within_catchments(
    all_stations_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    target_crs_epsg: int = DEFAULT_TARGET_CRS,
) -> gpd.GeoDataFrame:
    """
    Robust: spatial-join stations -> catchments using 'intersects'.
    Returns a GeoDataFrame where each station row includes 'station_index' (original stations index)
    and 'catchment_index' (index into catchments_gdf), preserves original geometry/crs, and keeps
    original station attributes (including name and station_type).
    """
    if all_stations_gdf.empty or catchments_gdf.empty:
        return gpd.GeoDataFrame(columns=["station_index", "catchment_index"])

    stations_proj = all_stations_gdf.to_crs(epsg=target_crs_epsg)
    catch_proj = catchments_gdf.to_crs(epsg=target_crs_epsg)

    joined = gpd.sjoin(stations_proj, catch_proj, how="inner", predicate="intersects")
    # joined.index is stations_proj.index (which matches original all_stations_gdf index if not reset)
    joined = joined.rename_axis("station_index").reset_index()
    # attach original geometry from all_stations_gdf (original CRS)
    joined["geometry_orig"] = all_stations_gdf.geometry.loc[joined["station_index"]].values
    joined = gpd.GeoDataFrame(joined, geometry=joined["geometry_orig"], crs=all_stations_gdf.crs)
    joined = joined.rename(columns={"index_right": "catchment_index"})
    return joined


# -------------------------
# High-level linking
# -------------------------
def build_chalk_catchment_links(
    chalk_stations_gdf: gpd.GeoDataFrame,
    all_stations_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    target_crs_epsg: int = DEFAULT_TARGET_CRS,
) -> pd.DataFrame:
    """
    For each chalk station, find its catchment and all stations inside that catchment.
    Returns flattened table including:
      - station_name and station_type (from the linked station)
      - distance_m (metres between chalk station and linked station)
      - catchment_index and catchment_objectid (if available)
      - station_index (index into all_stations_gdf) used as merge key
    """
    # project and join chalk -> catchment (use intersects to be tolerant)
    chalk_proj = chalk_stations_gdf.to_crs(epsg=target_crs_epsg)
    catch_proj = catchments_gdf.to_crs(epsg=target_crs_epsg)
    chalk_indexed = gpd.sjoin(chalk_proj, catch_proj, how="left", predicate="intersects")

    if "index_right" not in chalk_indexed.columns:
        raise RuntimeError("Spatial join failed between chalk and catchments.")

    chalk_indexed = chalk_indexed[["index_right"]].copy()
    chalk_indexed = chalk_indexed.dropna(subset=["index_right"])
    if chalk_indexed.empty:
        return pd.DataFrame(
            columns=[
                "chalk_station_index",
                "chalk_station_id",
                "chalk_station_name",
                "catchment_index",
                "catchment_objectid",
                "station_index",
                "station_id",
                "station_name",
                "station_type",
                "longitude",
                "latitude",
                "distance_m",
            ]
        )

    catchment_indices = chalk_indexed["index_right"].astype(int).unique().tolist()
    selected_catchments = catch_proj.loc[catchment_indices]

    # detect probable catchment ID column name for OBJECTID
    catchment_id_candidates = ["OBJECTID", "OBJECTID_1", "FID", "id", "ID", "objectid", "gid"]
    catchment_id_col = next((c for c in selected_catchments.columns if c in catchment_id_candidates), None)

    # find all stations within these selected catchments
    stations_in_catch = find_stations_within_catchments(all_stations_gdf, selected_catchments, target_crs_epsg=target_crs_epsg)

    # prepare projected geometries for distance calc
    stations_proj = all_stations_gdf.to_crs(epsg=target_crs_epsg)
    chalk_proj_full = chalk_stations_gdf.to_crs(epsg=target_crs_epsg)

    # station id column to extract station identifier
    station_id_col = detect_station_id_column(all_stations_gdf)

    rows = []
    if stations_in_catch.empty:
        return pd.DataFrame(rows)

    grouped = stations_in_catch.groupby("catchment_index")

    for chalk_idx, chalk_row in chalk_indexed.iterrows():
        catch_idx = int(chalk_row["index_right"])
        if catch_idx not in grouped.groups:
            continue

        chalk_point_proj = chalk_proj_full.loc[chalk_idx].geometry

        # catchment OBJECTID value if available
        if catchment_id_col is not None and (catch_idx in selected_catchments.index):
            catch_obj_id = selected_catchments.loc[catch_idx].get(catchment_id_col)
        else:
            catch_obj_id = catch_idx

        grp = grouped.get_group(catch_idx)
        for _, s in grp.iterrows():
            station_idx = int(s["station_index"])
            # station attributes from original all_stations_gdf (preserves station_name and station_type)
            station_rec = all_stations_gdf.loc[station_idx]
            sid = station_rec.get(station_id_col) if station_id_col in station_rec.index else None

            # geometry and basic attrs
            lon = station_rec.geometry.x
            lat = station_rec.geometry.y
            sname = None
            for cand in ("name", "stationName", "station_name", "label"):
                if cand in station_rec.index:
                    sname = station_rec.get(cand)
                    break
            stype = None
            for cand in ("station_type", "type", "stationType"):
                if cand in station_rec.index:
                    stype = station_rec.get(cand)
                    break

            # compute distance in projected CRS
            try:
                station_proj_geom = stations_proj.loc[station_idx].geometry
                distance_m = float(chalk_point_proj.distance(station_proj_geom))
            except Exception:
                distance_m = None

            rows.append(
                {
                    "chalk_station_index": chalk_idx,
                    "chalk_station_id": None,  # filled later if available
                    "chalk_station_name": None,  # filled later if available
                    "catchment_index": catch_idx,
                    "catchment_objectid": catch_obj_id,
                    "station_index": station_idx,
                    "station_id": sid,
                    "station_name": sname,
                    "station_type": stype,  # <-- single type column for linked station
                    "longitude": lon,
                    "latitude": lat,
                    "distance_m": distance_m,
                }
            )

    df_links = pd.DataFrame(rows)

    # fill chalk_station_id and chalk_station_name from chalk_stations_gdf (use available name/id columns)
    chalk_id_col = detect_station_id_column(chalk_stations_gdf)
    name_candidate = next((c for c in ("name", "stationName", "station_name", "label") if c in chalk_stations_gdf.columns), None)
    if not df_links.empty:
        df_links["chalk_station_id"] = None
        df_links["chalk_station_name"] = None
        # chalk_indexed.index are chalk station indices
        for chalk_idx in chalk_indexed.index.unique():
            mask = df_links["chalk_station_index"] == chalk_idx
            if not mask.any():
                continue
            if chalk_id_col and chalk_id_col in chalk_stations_gdf.columns:
                df_links.loc[mask, "chalk_station_id"] = chalk_stations_gdf.loc[chalk_idx].get(chalk_id_col)
            if name_candidate and name_candidate in chalk_stations_gdf.columns:
                df_links.loc[mask, "chalk_station_name"] = chalk_stations_gdf.loc[chalk_idx].get(name_candidate)

    return df_links


# -------------------------
# Station classification helper & main
# -------------------------
def classify_station_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Minimal, robust detection using the known column 'station_type' (falls back if missing).
    Adds boolean columns is_groundwater/is_flow/is_rainfall to gdf (preserves original index).
    Operates on the full station_type string (simple substring checks).
    """
    df = gdf.copy()
    # prefer explicit 'station_type' column
    if "station_type" in df.columns:
        col = "station_type"
    else:
        col = next((c for c in ("type", "stationType") if c in df.columns), None)

    if col is None:
        df["is_groundwater"] = False
        df["is_flow"] = False
        df["is_rainfall"] = False
        return df

    s = df[col].astype(str).fillna("").str.lower().str.strip()
    df["is_groundwater"] = s.str.contains("groundwater", na=False) | s.str.contains("ground", na=False)
    df["is_flow"] = s.str.contains("riverflow", na=False) | s.str.contains(" flow", na=False) | s.str.contains("river", na=False)
    df["is_rainfall"] = s.str.contains("rainfallstation", na=False) | s.str.contains("rainfall", na=False) | s.str.contains("rain", na=False)
    return df


# -------------------------
# Main convenience function
# -------------------------
def get_chalk_rivers(
    chalk_shp_path: str = "data/ea/Chalk_Rivers_(England)___Natural_England.shp",
    stations_csv_url: str = "https://environment.data.gov.uk/hydrology/id/stations.csv?_limit=20000",
    catchments_shp_path: str = "data/hydrobasins/uk_catchments.shp",
    buffer_m: int = DEFAULT_BUFFER_M,
    out_csv_path: str = DEFAULT_OUT_CSV,
) -> pd.DataFrame:
    """
    - Load chalk shapefile and all hydrology stations
    - Identify chalk (river) stations (within buffer of chalk lines)
    - For each chalk station, find the catchment polygon and then all stations inside that catchment
    - Compute distance (metres) from each linked station to the chalk station
    - Add classification flags for linked station only (is_groundwater/is_flow/is_rainfall),
      keep a single station_type and station_name column (for linked station)
    - Save CSV and return links DataFrame
    """
    chalk_gdf = load_shapefile(chalk_shp_path)
    catchments_gdf = load_shapefile(catchments_shp_path)
    stations_df = download_stations_csv(stations_csv_url)

    lon_col, lat_col = detect_lon_lat_columns(stations_df)
    all_stations_gdf = stations_df_to_gdf(stations_df, lon_col, lat_col, crs="EPSG:4326")

    # find chalk stations (annotated + subset)
    stations_annotated, chalk_stations = find_chalk_stations(chalk_gdf, all_stations_gdf, buffer_m=buffer_m)

    # build links (linked station attributes are taken from all_stations_gdf inside build_chalk_catchment_links)
    links_df = build_chalk_catchment_links(chalk_stations, all_stations_gdf, catchments_gdf)

    if links_df.empty:
        # persist empty CSV with expected columns
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        links_df.to_csv(out_csv_path, index=False)
        return links_df

    # classify linked stations using all_stations_gdf and merge flags INTO links_df as single is_* columns
    all_flags = classify_station_types(all_stations_gdf)[["is_groundwater", "is_flow", "is_rainfall"]]
    # prepare flags dataframe keyed by station_index
    flags = all_flags.reset_index().rename(columns={"index": "station_index"})
    # merge; station_index in links_df maps to all_stations_gdf.index
    links_df = links_df.merge(flags, on="station_index", how="left")

    # station_type in links_df is the linked station's type (kept from build_chalk_catchment_links)
    # remove chalk station type duplicates (we intentionally do not add chalk station type to output)
    # keep station_name (linked station) preserved.

    # save csv
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    links_df.to_csv(out_csv_path, index=False)
    print(f"CSV saved to {out_csv_path}")


if __name__ == "__main__":
    df = get_chalk_rivers()
    print(df.head())
