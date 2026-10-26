"""
Add the matched chalk catchments (data/matching/chalk_matches.csv, from processing/match_stations.py) to config.yaml as
boreholes, so every pipeline stage can run on them: `python main.py <key> ...`. The key is the lower-case GeoSmart prefix.
Boreholes already in config.yaml are left alone. Appends to the `boreholes:` section, so comments in the file are kept.

    python -m processing.add_chalk_boreholes
"""
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
MATCHES = ROOT / "data" / "matching" / "chalk_matches.csv"


def borehole_block(r) -> str:
    return (
        f"  {r.geosmart_prefix.lower()}:\n"
        f"    name: {r.groundwater_name}\n"
        f"    latitude: {r.borehole_lat}                   # GEFS nearest-grid-point lookup\n"
        f"    longitude: {r.borehole_long}\n"
        f"    geosmart_prefix: {r.geosmart_prefix}\n"
        f"    hde_stations:\n"
        f"      groundwater: {r.groundwater_guid}   # {r.groundwater_name}\n"
        f"      rainfall: {r.rainfall_guid}      # {r.rainfall_name}\n"
        f"      flow: {r.flow_guid}          # {r.flow_name}, {r.river}\n"
    )


def main() -> None:
    text = CONFIG.read_text()
    existing = yaml.safe_load(text)["boreholes"]
    known_prefixes = {b["geosmart_prefix"] for b in existing.values()}
    blocks = [borehole_block(r) for r in pd.read_csv(MATCHES).itertuples()
              if r.geosmart_prefix.lower() not in existing and r.geosmart_prefix not in known_prefixes]
    if not blocks:
        print("config.yaml already has every matched borehole")
        return
    m = re.search(r"^boreholes:\n(?:(?:[ \t]+.*|#.*|)\n)*?(?=^\S)", text + "\n_end:\n", re.M)
    if m is None:
        sys.exit("could not find the boreholes section in config.yaml")
    section_end = len(text) if m.end() > len(text) else m.end()
    body = text[:section_end].rstrip("\n") + "\n" + "".join(blocks)
    CONFIG.write_text(body + "\n" + text[section_end:].lstrip("\n") if text[section_end:].strip() else body)
    print(f"added {len(blocks)} boreholes to config.yaml")


if __name__ == "__main__":
    main()
