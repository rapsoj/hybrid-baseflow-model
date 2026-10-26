"""
Consolidated storage for GeoSmart forecasts: one gzip csv per borehole instead of one csv per issue day.

A raw file holds ~2,700 history rows (TimeSeriesID 0, 15 min, months back) plus the ensemble members. The model only
uses the last valid history observation (the anchor) and the member rows after it (`pipeline.data.parse_geosmart_file`),
so that is all the store keeps: the anchor row plus every member row after the anchor. Parsing the stored rows gives
exactly the same forecast as parsing the raw file. A file with no valid history keeps a single placeholder row
(TimeSeriesID 0, Date NaT, Level NaN) so the issue day still counts as downloaded.

Layout: <geosmart_dir>/<prefix>_forecasts.csv.gz with columns issue_date, TimeSeriesID, Date (ISO), Level. gzip csv needs no
extra dependency, reads in about a second per borehole and is ~15x smaller than the raw files; values round-trip exactly.
"""
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

DATE_FORMAT = "%d-%m-%Y %H:%M"
COLUMNS = ["issue_date", "TimeSeriesID", "Date", "Level"]
_ISSUE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")


def store_path(geosmart_dir, prefix: str) -> Path:
    return Path(geosmart_dir) / f"{prefix}_forecasts.csv.gz"


def loose_files(geosmart_dir, prefix: str) -> Dict[str, Path]:
    """{issue date 'YYYY-MM-DD': path} of individual '<prefix>_headTimeSeries <date>.csv' files below geosmart_dir."""
    out = {}
    for f in sorted(Path(geosmart_dir).rglob(f"{prefix}_headTimeSeries *.csv")):
        m = _ISSUE_RE.search(f.name)
        if m:
            out[m.group(1)] = f
    return out


def compact_forecast(df: pd.DataFrame, issue_date: str) -> pd.DataFrame:
    """The rows of one raw file that the model uses (see module docstring), as COLUMNS."""
    d = df[["TimeSeriesID", "Date", "Level"]].copy()
    d["Date"] = pd.to_datetime(d["Date"], format=DATE_FORMAT)
    hist = d[(d["TimeSeriesID"] == 0) & d["Level"].notna()].sort_values("Date")
    if hist.empty:
        keep = pd.DataFrame({"TimeSeriesID": [0], "Date": [pd.NaT], "Level": [float("nan")]})
    else:
        anchor = hist.tail(1)
        keep = pd.concat([anchor, d[(d["TimeSeriesID"] != 0) & (d["Date"] > anchor["Date"].iloc[0])]])
    keep = keep.assign(issue_date=issue_date)[COLUMNS]
    keep["TimeSeriesID"] = keep["TimeSeriesID"].astype("int32")
    keep["Date"] = keep["Date"].astype("datetime64[ns]")
    keep["Level"] = keep["Level"].astype("float64")
    return keep.reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(COLUMNS, ["object", "int32", "datetime64[ns]", "float64"])})


def read_store(geosmart_dir, prefix: str) -> pd.DataFrame:
    path = store_path(geosmart_dir, prefix)
    if not path.exists():
        return _empty()
    df = pd.read_csv(path, parse_dates=["Date"], float_precision="round_trip", dtype={"issue_date": str})
    df["TimeSeriesID"] = df["TimeSeriesID"].astype("int32")
    df["Date"] = df["Date"].astype("datetime64[ns]")
    return df


def stored_issue_dates(geosmart_dir, prefix: str) -> Set[str]:
    path = store_path(geosmart_dir, prefix)
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["issue_date"], dtype=str)["issue_date"].unique())


def write_store(geosmart_dir, prefix: str, frame: pd.DataFrame) -> Path:
    path = store_path(geosmart_dir, prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.sort_values(["issue_date", "TimeSeriesID", "Date"]).to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)  # atomic: an interrupted write never corrupts the existing store
    return path


def consolidate(geosmart_dir, prefix: str, delete: bool = True) -> int:
    """
    Add every loose csv below geosmart_dir to the store (files already stored are replaced by the stored copy) and, after
    the store has been written, delete the csvs and the empty folders left behind. Returns the number of files added.
    """
    files = loose_files(geosmart_dir, prefix)
    if not files:
        return 0
    have = stored_issue_dates(geosmart_dir, prefix)
    new = {d: f for d, f in files.items() if d not in have}
    if new:
        add = pd.concat([compact_forecast(pd.read_csv(f), d) for d, f in new.items()], ignore_index=True)
        write_store(geosmart_dir, prefix, pd.concat([read_store(geosmart_dir, prefix), add], ignore_index=True))
    if delete:
        for f in files.values():
            f.unlink()
        root = Path(geosmart_dir)
        for sub in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if not any(sub.iterdir()):
                sub.rmdir()
    return len(new)


def frames_by_issue_date(geosmart_dir, prefix: str) -> Dict[str, pd.DataFrame]:
    """{issue date: rows} for everything in the store (frames are ready for `parse_geosmart_file`)."""
    store = read_store(geosmart_dir, prefix)
    return {d: g.drop(columns="issue_date").reset_index(drop=True) for d, g in store.groupby("issue_date", sort=True)}


def issue_summary(geosmart_dir, prefix: str) -> List[tuple]:
    """[(issue date, has ensemble members)] sorted by date, from the store and any loose csvs not yet consolidated."""
    store = read_store(geosmart_dir, prefix)
    rows = {d: bool(g) for d, g in (store["TimeSeriesID"] != 0).groupby(store["issue_date"]).any().items()}
    for d, f in loose_files(geosmart_dir, prefix).items():
        if d not in rows:
            rows[d] = bool((pd.read_csv(f, usecols=["TimeSeriesID"])["TimeSeriesID"] != 0).any())
    return sorted(rows.items())
