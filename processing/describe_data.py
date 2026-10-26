"""
Regenerate DATA.md: per-catchment HDE record lengths and GEFS / GeoSmart forecast coverage for every borehole in config.yaml.

    python -m processing.describe_data
"""
import glob
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import geosmart_store  # noqa: E402

KINDS = (("groundwaterLevel", "Groundwater level"), ("rainfall", "Rainfall"), ("waterFlow", "River flow"))


def hde_cell(key: str, kind: str) -> str:
    files = glob.glob(str(ROOT / "data" / "hde" / key / f"*_{kind}_15min.csv"))
    if not files:
        return "—"
    t = pd.read_csv(files[0], usecols=["dateTime"], parse_dates=["dateTime"])["dateTime"]
    return f"{t.min():%Y-%m-%d}–{t.max():%Y-%m-%d}; {len(t):,}"


def station_name(key: str, kind: str) -> str:
    files = glob.glob(str(ROOT / "data" / "hde" / key / f"*_{kind}_15min.csv"))
    return Path(files[0]).name.rsplit(f"_{kind}", 1)[0].replace("_", " ").title() if files else "—"


def gefs_cell(key: str) -> str:
    f = ROOT / "data" / "gefs" / f"{key}_gefs_forecasts.csv"
    if not f.exists():
        return "—"
    t = pd.read_csv(f, usecols=["init_time"], parse_dates=["init_time"])["init_time"].drop_duplicates()
    span = (t.max() - t.min()).days + 1
    return f"{t.min():%Y-%m-%d}–{t.max():%Y-%m-%d}; {len(t):,}" + (" (daily)" if len(t) == span else f" of {span} days")


def geosmart_cell(folder: Path, prefix: str) -> str:
    rows = geosmart_store.issue_summary(folder, prefix)
    if not rows:
        return "—"
    with_members = sum(h for _, h in rows)
    return f"{with_members:,} / {len(rows):,}; file dates {rows[0][0]}–{rows[-1][0]}"


def joint_run_section(items) -> list[str]:
    """Which catchments the joint `all-chalk` run can use, applying the same rules as the pipeline."""
    from pipeline.multi import select_catchments

    states, skipped = select_catchments(str(ROOT / "config.yaml"))
    out = [
        "", "## Catchments usable for joint training", "",
        "`python main.py all-chalk` needs, for every catchment, all three HDE series, GEFS, GeoSmart ensemble members and enough",
        "origins where GEFS and GeoSmart both exist (complete encoder and target windows) for each purged",
        "train/validation/test split, and a gauge that is not dry. Samples are daily forecast origins; the split",
        "sizes below are the numbers of origins after the purge gaps.", "",
        "| Catchment | Usable | Train | Validation | Test | Reason if excluded |", "|---|---|---|---|---|---|",
    ]
    for key, b in items:
        if key in states:
            sz = states[key]["sizes"]
            out.append(f"| {b['name']} | yes | {sz['train']} | {sz['val']} | {sz['test']} | |")
        else:
            out.append(f"| {b['name']} | no | | | | {skipped[key]} |")
    out.append("")
    out.append(f"{len(states)} of {len(items)} catchments are usable.")
    return out


def main() -> None:
    boreholes = yaml.safe_load(open(ROOT / "config.yaml"))["boreholes"]
    items = sorted(boreholes.items(), key=lambda kv: kv[1]["name"])
    out = [
        "# Data coverage by catchment", "",
        f"This is a snapshot of the files in `data/` generated on {date.today()} by `python -m processing.describe_data`.",
        "Record counts and date ranges describe the files on disk; they are not a guarantee that observations are valid",
        "or continuous between the endpoints. Catchments are named after their groundwater borehole.", "",
        "## HDE observations", "",
        "Each cell gives the first and last observation date and the number of CSV rows. `—` means that no matching",
        "file was present locally.", "",
        "| Catchment | Groundwater level | Rainfall | River flow |", "|---|---|---|---|",
    ]
    for key, b in items:
        out.append(f"| {b['name']} | " + " | ".join(hde_cell(key, k) for k, _ in KINDS) + " |")
    out += [
        "", "Stations used (nearest 15-minute stations; see `data/matching/chalk_matches.csv`):", "",
        "| Catchment | Rainfall station | River flow station |", "|---|---|---|",
    ]
    for key, b in items:
        out.append(f"| {b['name']} | {station_name(key, 'rainfall')} | {station_name(key, 'waterFlow')} |")
    out += [
        "",
        "HDE files are named for nominal 15-minute data, but actual sampling is not uniformly 15-minute: groundwater",
        "series in particular include hourly stretches and gaps. The counts above are raw rows, not unique timestamps",
        "or valid-value counts. Inspect and align each series before treating its date span as continuous coverage.",
        "", "## Forecast data", "",
        "GEFS counts are distinct initialization dates in the local files. GeoSmart member coverage counts issue dates",
        "that have at least one ensemble member (`TimeSeriesID` other than 0) after the last history time; the",
        "denominator is the issue dates present in the consolidated file (`data/geosmart/<prefix>/<Prefix>_forecasts.csv.gz`).",
        "Issue dates with history only are not usable ensemble forecasts. Days that S3 does not hold are listed in",
        "`<Prefix>_missing_days.txt` and are not counted.", "",
        "| Catchment | GEFS initialization dates | GeoSmart member forecast dates |", "|---|---|---|",
    ]
    no_members = []
    for key, b in items:
        folder = ROOT / "data" / "geosmart" / b["geosmart_prefix"].lower()
        cell = geosmart_cell(folder, b["geosmart_prefix"])
        if cell.startswith("0 /"):
            no_members.append(b["name"])
        out.append(f"| {b['name']} | {gefs_cell(key)} | {cell} |")
    out += joint_run_section(items)
    out += ["", "## Completeness notes", ""]
    out.append("- No catchment is missing any of the three HDE inputs." if all(
        hde_cell(k, kind) != "—" for k, _ in items for kind, _ in KINDS) else
        "- Catchments with `—` in the HDE table lack that input and cannot run the pipeline.")
    out.append("- Several HDE records end well before the present (stations stopped reporting), which shortens the overlap"
               " with GeoSmart and GEFS; check the end dates in the HDE table.")
    if no_members:
        out.append(f"- GeoSmart files with no ensemble members at all (history only): {', '.join(no_members)}. These catchments"
                   " have no usable GeoSmart forecasts, so they can only be modelled without GeoSmart.")
    out.append("- These counts describe the local snapshot. They do not establish common usable sample dates across input"
               " series; the pipeline must align timestamps and require complete target windows before training or evaluation.")
    (ROOT / "DATA.md").write_text("\n".join(out) + "\n")
    print("wrote DATA.md")


if __name__ == "__main__":
    main()
