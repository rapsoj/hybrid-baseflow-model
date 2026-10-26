#!/usr/bin/env python
"""
Single entry point.

    python main.py chilgrove                    # download HDE + GEFS, then train/evaluate all ablation variants
    python main.py chilgrove --skip-download    # reuse data already on disk
    python main.py all-chalk --skip-download   # jointly train and compare eligible chalk catchments
    python main.py all-chalk-final             # multi-seed with vs without GeoSmart (see final_run in config.yaml)
    python main.py chilgrove --download-only
    python main.py chilgrove --set data.history_days=7 --set training.epochs=5

GeoSmart forecasts are not downloaded here: run processing/download_geosmart.py first (writes data/geosmart/<prefix>/<Prefix>_forecasts.csv.gz).
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import load_config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("borehole", help="borehole key from config.yaml (e.g. chilgrove), all-chalk for joint training, or all-chalk-final for the multi-seed comparison")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. training.epochs=5 (repeatable)")
    p.add_argument("--skip-download", action="store_true", help="do not contact the HDE / GEFS services")
    p.add_argument("--update", action="store_true", help="top up existing local data instead of skipping it")
    p.add_argument("--download-only", action="store_true", help="stop after the download stage")
    p.add_argument("--run-id", default=None, help="name for this run (default: timestamp)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # all relative paths in config.yaml and the processing scripts are relative to the repo root
    import os
    os.chdir(ROOT)
    overrides = args.overrides + (["download.update=true"] if args.update else [])
    if args.borehole == "all-chalk-final":
        from pipeline.final import run_final

        summary = run_final(args.config, overrides, run_id=args.run_id)
        print("\nMulti-seed summary (overall test metrics)\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return
    if args.borehole == "all-chalk":
        from pipeline.multi import run_multi_catchment

        comparison = run_multi_catchment(args.config, overrides, run_id=args.run_id)
        print("\nPer-catchment test-set comparison\n" + comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return

    cfg = load_config(args.config, args.borehole, overrides)

    if not args.skip_download:
        from pipeline.download import download_gefs, download_hde
        download_hde(cfg)
        download_gefs(cfg)
    if args.download_only:
        return

    from pipeline import data as D
    from pipeline.train import run_training

    hde = D.load_hde(cfg["paths"]["hde_dir"], cfg["data"]["step_minutes"], cfg["data"])
    gefs = D.load_gefs(cfg["paths"]["gefs_csv"], cfg["data"]["gefs"]["bin_hours"])
    geosmart = D.load_geosmart(cfg["paths"]["geosmart_dir"], cfg["borehole"]["geosmart_prefix"], cfg["data"]["geosmart"])

    comparison = run_training(cfg, hde, gefs, geosmart, run_id=args.run_id)
    print("\nTest-set comparison (metrics per row, variants as columns)\n" + comparison.T.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
