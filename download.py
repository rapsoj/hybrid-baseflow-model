#!/usr/bin/env python
"""
Stand-alone data download for one borehole (HDE groundwater/rainfall/flow + GEFS).

    python download.py chilgrove            # downloads only what is missing locally
    python download.py chilgrove --update   # top up existing files with newer data

Equivalent to `python main.py <borehole> --download-only`. GeoSmart files are not downloaded here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:] + ["--download-only"])
