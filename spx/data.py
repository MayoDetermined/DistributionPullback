"""Imagenette validation split -- the paper's primary evaluation set (Sec. 4.1).

Imagenette is a 10-class subset of ImageNet-1k, so the 1000-way pretrained
heads are used unchanged and the labels are the *ImageNet* indices of those
ten classes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = ["IMAGENETTE_WNIDS", "IMAGENETTE_TO_IMAGENET", "load_imagenette"]

#: Imagenette wnids in directory order, with their ImageNet-1k class indices.
IMAGENETTE_WNIDS = [
    ("n01440764", 0, "tench"),
    ("n02102040", 217, "English springer"),
    ("n02979186", 482, "cassette player"),
    ("n03000684", 491, "chain saw"),
    ("n03028079", 497, "church"),
    ("n03394916", 566, "French horn"),
    ("n03417042", 569, "garbage truck"),
    ("n03425413", 571, "gas pump"),
    ("n03445777", 574, "golf ball"),
    ("n03888257", 701, "parachute"),
]
IMAGENETTE_TO_IMAGENET = {w: i for w, i, _ in IMAGENETTE_WNIDS}
IMAGENETTE_NAMES = {i: n for _, i, n in IMAGENETTE_WNIDS}

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "imagenette2-320"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_one(path: Path, res: int) -> np.ndarray:
    """Standard ImageNet eval transform: resize shorter side to ``res * 8/7``,
    centre-crop to ``res``, scale to [0, 1], normalise."""
    img = Image.open(path).convert("RGB")
    short = int(round(res * 256 / 224))
    w, h = img.size
    scale = short / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - res) // 2, (h - res) // 2
    img = img.crop((left, top, left + res, top + res))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return a.transpose(2, 0, 1)


def load_imagenette(
    n: int = 1000,
    res: int = 224,
    split: str = "val",
    seed: int = 0,
    root: os.PathLike | None = None,
    correct_only: bool = False,
    model=None,
    device=None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return ``(x, y, paths)`` for ``n`` images sampled uniformly at random.

    ``y`` holds *ImageNet-1k* indices.  With ``correct_only=True`` the sample is
    filtered to images the given ``model`` classifies correctly (the paper does
    not do this; provided for the target-specificity analyses where an
    explanation of a wrong prediction is not interpretable).
    """
    root = Path(root) if root is not None else DEFAULT_ROOT
    d = root / split
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found -- run `python scripts/00_prepare_data.py` first"
        )

    items = []
    for wnid, cls, _ in IMAGENETTE_WNIDS:
        for p in sorted((d / wnid).glob("*.JPEG")):
            items.append((p, cls))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))

    xs, ys, paths = [], [], []
    buf_x, buf_y, buf_p = [], [], []

    def _flush():
        """Optionally drop misclassified images, then commit the buffer."""
        nonlocal buf_x, buf_y, buf_p
        if not buf_x:
            return
        bx = np.stack(buf_x)
        keep = np.ones(len(bx), dtype=bool)
        if correct_only:
            with torch.no_grad():
                pred = model(torch.as_tensor(bx, device=device)).argmax(-1).cpu().numpy()
            keep = pred == np.asarray(buf_y)
        for i in np.nonzero(keep)[0]:
            if len(xs) < n:
                xs.append(bx[i]), ys.append(buf_y[i]), paths.append(buf_p[i])
        buf_x, buf_y, buf_p = [], [], []

    for j in order:
        if len(xs) >= n:
            break
        p, cls = items[j]
        buf_x.append(_load_one(p, res))
        buf_y.append(cls)
        buf_p.append(str(p))
        if len(buf_x) == batch_size:
            _flush()
    _flush()

    if len(xs) < n:
        raise RuntimeError(f"only {len(xs)}/{n} images available under {d}")
    return np.stack(xs), np.asarray(ys, dtype=np.int64), paths


def denormalise(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Back to displayable [0, 1] HWC."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = x.transpose(1, 2, 0) if x.ndim == 3 else x.transpose(0, 2, 3, 1)
    return np.clip(a * _STD + _MEAN, 0.0, 1.0)

