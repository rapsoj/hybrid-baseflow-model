"""
Download GeoSmart groundwater forecasts (one `<prefix>_headTimeSeries <date>.csv` per issue day) with the AWS CLI and
consolidate them into one file per borehole.

Needs the AWS CLI with a `gwforecasting` profile (region eu-west-1). Each chunk of days is copied with `aws s3 cp` into
data/geosmart/<prefix lower case>/<YYYY>/<MM>/, added to data/geosmart/<prefix lower case>/<prefix>_forecasts.csv.gz
(see pipeline/geosmart_store.py: only the rows the model uses are kept) and the individual csvs are then deleted.
Days already in the store are skipped, so the script can be re-run to top up, and csvs left by an interrupted run
are consolidated first.

    python -m processing.download_geosmart ChilgroveHouse
    python -m processing.download_geosmart FollyFarm --start 2022-01-01 --end 2022-03-31 --workers 16
    python -m processing.download_geosmart ChilgroveHouse --keep-files     # do not delete the individual csvs

The prefixes of the matched chalk catchments are in data/matching/chalk_matches.csv (`geosmart_prefix`).
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import geosmart_store  # noqa: E402

BUCKET = "gwforecasting.geosmartinfo.co.uk"
PROFILE = "gwforecasting"
REGION = "eu-west-1"
FIRST_DAY = date(2020, 1, 1)
DEFAULT_ROOT = Path("data/geosmart")
RETRY_DAYS = 30  # a day recorded as missing is only tried again while it is this recent
CHUNK_DAYS = 400  # days downloaded between consolidations, so an interruption loses at most one chunk of downloads


def file_name(prefix: str, day: date) -> str:
    return f"{prefix}_headTimeSeries {day.isoformat()}.csv"


def s3_uri(prefix: str, day: date) -> str:
    return f"s3://{BUCKET}/{day:%Y}/{day:%m_%B}/{day:%d}/{file_name(prefix, day)}"


def borehole_dir(prefix: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / prefix.lower()


def local_path(prefix: str, day: date, root: Path = DEFAULT_ROOT) -> Path:
    return borehole_dir(prefix, root) / f"{day:%Y}" / f"{day:%m}" / file_name(prefix, day)


def missing_path(prefix: str, root: Path = DEFAULT_ROOT) -> Path:
    return borehole_dir(prefix, root) / f"{prefix}_missing_days.txt"


def read_missing(prefix: str, root: Path = DEFAULT_ROOT) -> set:
    f = missing_path(prefix, root)
    return set(f.read_text().split()) if f.exists() else set()


def download_day(prefix: str, day: date, root: Path = DEFAULT_ROOT) -> str:
    """Returns 'downloaded' or 'missing' (no forecast file for that day, or the copy failed)."""
    target = local_path(prefix, day, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")  # a partial download is never mistaken for a finished file
    out = subprocess.run(
        ["aws", "s3", "cp", s3_uri(prefix, day), str(tmp), "--profile", PROFILE, "--region", REGION, "--only-show-errors"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        tmp.unlink(missing_ok=True)
        return "missing"
    tmp.rename(target)
    return "downloaded"


def download_geosmart(
    prefix: str, start: date = FIRST_DAY, end: Optional[date] = None, root: Path = DEFAULT_ROOT, workers: int = 8,
    keep_files: bool = False,
) -> dict:
    end = end or date.today()
    folder = borehole_dir(prefix, root)
    geosmart_store.consolidate(folder, prefix, delete=not keep_files)  # leftovers from an interrupted run
    stored = geosmart_store.stored_issue_dates(folder, prefix)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    known_missing = read_missing(prefix, root)
    recent = (date.today() - timedelta(days=RETRY_DAYS)).isoformat()
    # days S3 did not have on an earlier run are not asked for again, which makes re-runs near-instant
    todo = [d for d in days if d.isoformat() not in stored and (d.isoformat() not in known_missing or d.isoformat() >= recent)]
    if keep_files:
        todo = [d for d in todo if not local_path(prefix, d, root).exists()]

    downloaded, missing = 0, []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i in range(0, len(todo), CHUNK_DAYS):
            chunk = todo[i:i + CHUNK_DAYS]
            for d, r in zip(chunk, pool.map(lambda d: download_day(prefix, d, root), chunk)):
                if r == "downloaded":
                    downloaded += 1
                else:
                    missing.append(d.isoformat())
            geosmart_store.consolidate(folder, prefix, delete=not keep_files)
    all_missing = sorted((known_missing | set(missing)) - geosmart_store.stored_issue_dates(folder, prefix))
    if all_missing or missing_path(prefix, root).exists():
        borehole_dir(prefix, root).mkdir(parents=True, exist_ok=True)
        missing_path(prefix, root).write_text("\n".join(all_missing) + ("\n" if all_missing else ""))
    counts = {"downloaded": downloaded, "exists": len(days) - len(todo), "missing": len(missing)}
    print(f"{prefix}: {counts['downloaded']} downloaded, {counts['exists']} already present, "
          f"{counts['missing']} failed or missing ({start} to {end})")
    if missing:
        print("missing/failed days:", ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    return counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", nargs="?", default="ChilgroveHouse", help="GeoSmart borehole prefix (default: ChilgroveHouse)")
    ap.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY, help="first issue day (default: 2020-01-01)")
    ap.add_argument("--end", type=date.fromisoformat, default=None, help="last issue day (default: today)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="output root (default: data/geosmart)")
    ap.add_argument("--workers", type=int, default=8, help="parallel aws s3 cp calls (default: 8)")
    ap.add_argument("--keep-files", action="store_true", help="keep the individual csvs after consolidating")
    a = ap.parse_args(argv)
    download_geosmart(a.prefix, a.start, a.end, a.root, a.workers, a.keep_files)


if __name__ == "__main__":
    main()
