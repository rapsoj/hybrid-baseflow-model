"""
Download HDE, GEFS and GeoSmart data for every borehole in config.yaml (or the ones named).
HDE runs for several boreholes in parallel, GEFS is fetched for all boreholes in one pass (shared archive chunks).
Everything is resumable: data already on disk is skipped.

    python -m processing.download_all                  # all boreholes
    python -m processing.download_all compton hursley
    python -m processing.download_all --update         # top up existing data
"""
import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from pipeline.config import load_config  # noqa: E402
from pipeline.download import download_gefs_many, download_hde  # noqa: E402
from processing.download_geosmart import download_geosmart  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boreholes", nargs="*", help="borehole keys (default: all in config.yaml)")
    ap.add_argument("--update", action="store_true", help="top up existing HDE/GEFS data")
    ap.add_argument("--workers", type=int, default=8, help="parallel GeoSmart downloads")
    ap.add_argument("--hde-workers", type=int, default=2, help="boreholes downloaded from HDE in parallel")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    import yaml
    from concurrent.futures import ThreadPoolExecutor

    keys = a.boreholes or list(yaml.safe_load(open(ROOT / "config.yaml"))["boreholes"])
    overrides = ["download.update=true"] if a.update else []
    cfgs = {k: load_config(str(ROOT / "config.yaml"), k, overrides) for k in keys}
    failed = []

    def guarded(stage, key, fn, *args):
        try:
            fn(*args)
        except Exception:
            print(f"\n!!! {stage} failed for {key}", flush=True)
            traceback.print_exc()
            failed.append(f"{key}:{stage}")

    print("\n=== HDE (boreholes in parallel) ===", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.hde_workers)) as pool:
        for k in keys:
            pool.submit(guarded, "hde", k, download_hde, cfgs[k])
    print("\n=== GEFS (one pass for all boreholes that need it) ===", flush=True)
    guarded("gefs", "all", download_gefs_many, list(cfgs.values()))
    for k in keys:
        print(f"\n=== GeoSmart {k} ===", flush=True)
        c = cfgs[k]
        guarded("geosmart", k, lambda: download_geosmart(
            c["borehole"]["geosmart_prefix"], root=Path(c["paths"]["geosmart_dir"]).parent, workers=a.workers))
    print("\nfailed:", failed or "none")


if __name__ == "__main__":
    main()
