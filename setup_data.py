#!/usr/bin/env python
"""Fetch the evaluation data the notebooks use.

    python setup_data.py

Downloads Imagenette-320 (~342 MB) into ./data and extracts it. Imagenette is a
10-class subset of ImageNet-1k; the labels used here are the *ImageNet* indices
of those ten classes, so the pretrained 1000-way heads are used unchanged.

ResNet-50 weights come from torchvision on first use and are cached in the
usual torch hub directory -- nothing to do for those.

Skips work that is already done, so it is safe to re-run.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ROOT = Path(__file__).resolve().parent / "data"
ARCHIVE = ROOT / "imagenette2-320.tgz"
EXTRACTED = ROOT / "imagenette2-320"


def _progress(done: int, block: int, total: int):
    if total <= 0:
        return
    pct = min(100.0, done * block * 100.0 / total)
    mb = done * block / 1e6
    sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f} MB")
    sys.stdout.flush()


def main():
    if (EXTRACTED / "val").is_dir():
        n = sum(1 for _ in (EXTRACTED / "val").rglob("*.JPEG"))
        print(f"already present: {EXTRACTED}  ({n} validation images)")
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE, reporthook=_progress)
        print()

    print(f"extracting into {ROOT}")
    with tarfile.open(ARCHIVE) as t:
        # filter='data' refuses absolute paths and traversal outside the target
        try:
            t.extractall(ROOT, filter="data")
        except TypeError:      # Python < 3.12 has no filter=
            t.extractall(ROOT)

    n = sum(1 for _ in (EXTRACTED / "val").rglob("*.JPEG"))
    print(f"done: {n} validation images under {EXTRACTED}")
    print("\nThe archive can be deleted once extracted:")
    print(f"  {ARCHIVE}")


if __name__ == "__main__":
    main()
