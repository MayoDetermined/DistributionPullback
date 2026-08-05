"""Sharpness of an attribution map -- and the guard that has to come with it.

The three extensions are argued for partly on their maps being crisper than the
rest of the suite. That needs a number, and it needs a guard, because we have
already been caught by this once: DESIGN_NOTES §7.4 has a sharpness-like
measure putting the raw Gradient first of 19 while its FaithfulnessCorrelation
is a fifth of Soft Pullback's. Saliency is literally |Gradient| and keeps all
the concentration while destroying the direction.

So sharpness_report refuses to rank on sharpness alone and returns the Pareto
frontier against a faithfulness metric instead. A sharpness win only means
something held at equal faithfulness; on its own it rewards degenerate maps.

Everything takes a signed (B, C, H, W) map and first reduces it to a
non-negative spatial map normalised to sum to 1, so the measures are
scale-free -- necessary, since the ascent family returns a perturbation of norm
~alpha*K rather than a differential.

    reduce="abs_sum"  s = sum_c |a_c|, total energy per pixel. Default.
    reduce="sum_abs"  s = |sum_c a_c|, energy of the map as displayed.

Those differ by ~19% of the mass (§6.1), so it's a real choice, not formatting.
abs_sum is the honest one for comparing methods, sum_abs is what a reader of a
figure actually sees.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "spatial_map",
    "gini",
    "spatial_entropy",
    "effective_support",
    "topk_mass",
    "total_variation",
    "channel_cancellation",
    "sharpness_measures",
    "SHARPNESS_DIRECTION",
    "sharpness_report",
]

#: +1 = higher is sharper, -1 = lower is sharper.  Kept in the same shape as
#: ``spx.metrics.METRIC_DIRECTION`` so the two tables can be joined.
SHARPNESS_DIRECTION = {
    "gini": +1,
    "spatial_entropy": -1,
    "effective_support": -1,
    "top1pct_mass": +1,
    "top5pct_mass": +1,
    "total_variation": +1,
    "channel_cancellation": 0,  # descriptive, not a quality axis
}


def spatial_map(a: torch.Tensor, reduce: str = "abs_sum") -> torch.Tensor:
    """``(B, C, H, W) -> (B, H, W)``, non-negative and summing to 1 per sample."""
    if a.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(a.shape)}")
    if reduce == "abs_sum":
        s = a.abs().sum(1)
    elif reduce == "sum_abs":
        s = a.sum(1).abs()
    else:
        raise ValueError(reduce)
    s = s.flatten(1)
    s = s / s.sum(1, keepdim=True).clamp_min(1e-12)
    return s.view(a.shape[0], a.shape[2], a.shape[3])


# --------------------------------------------------------------------------
# the measures
# --------------------------------------------------------------------------


def gini(s: torch.Tensor) -> torch.Tensor:
    """Gini of the pixel mass. 0 = uniform, ->1 = one pixel.

    Least assumption-laden of the lot: no threshold, no scale, no neighbourhood.
    """
    v, _ = s.flatten(1).sort(dim=1)
    n = v.shape[1]
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype).unsqueeze(0)
    return ((2 * idx - n - 1) * v).sum(1) / (n * v.sum(1).clamp_min(1e-12))


def spatial_entropy(s: torch.Tensor, normalise: bool = True) -> torch.Tensor:
    """Shannon entropy of the pixel-mass distribution, in bits.

    ``normalise=True`` divides by ``log2(H*W)`` so a uniform map scores 1 and a
    one-pixel map scores 0.  Lower = sharper.
    """
    p = s.flatten(1).clamp_min(1e-12)
    h = -(p * p.log2()).sum(1)
    return h / np.log2(p.shape[1]) if normalise else h


def effective_support(s: torch.Tensor) -> torch.Tensor:
    """2^H / (H*W): the fraction of pixels the map effectively uses.

    Perplexity of spatial_entropy as a share of the image. 0.02 reads as "this
    map is really about 2% of the pixels".
    """
    h = spatial_entropy(s, normalise=False)
    return torch.pow(2.0, h) / s.flatten(1).shape[1]


def topk_mass(s: torch.Tensor, frac: float = 0.01) -> torch.Tensor:
    """Share of the mass in the top `frac` of pixels -- what a reader of a heat
    map is implicitly applying. Reported at 1% and 5%, since a method can win
    one and lose the other (single hot pixel vs tight blob)."""
    p = s.flatten(1)
    k = max(1, int(round(frac * p.shape[1])))
    top, _ = p.topk(k, dim=1)
    return top.sum(1) / p.sum(1).clamp_min(1e-12)


def total_variation(s: torch.Tensor) -> torch.Tensor:
    """Mean |finite difference| of the normalised map, times H*W.

    Counterweight to the concentration measures: a map can be concentrated and
    smooth (blob) or concentrated and jagged (noise), and only the pair tells
    them apart. Not a quality axis by itself.
    """
    n = s.flatten(1).shape[1]
    dh = (s[:, 1:, :] - s[:, :-1, :]).abs().flatten(1).sum(1)
    dw = (s[:, :, 1:] - s[:, :, :-1]).abs().flatten(1).sum(1)
    return (dh + dw) / 2.0 * n / (s.shape[1] * s.shape[2])


def channel_cancellation(a: torch.Tensor) -> torch.Tensor:
    """``1 - sum|sum_c a_c| / sum sum_c|a_c|`` -- §6.1's measure.

    The share of attribution mass that disappears when the three input channels
    are summed for display.  Descriptive: §6.1 finds ~0.19 for the whole
    pullback family regardless of selector, so a *deviation* from that is the
    interesting event.
    """
    num = a.sum(1).abs().flatten(1).sum(1)
    den = a.abs().flatten(1).sum(1).clamp_min(1e-12)
    return 1.0 - num / den


def sharpness_measures(
    a: torch.Tensor, reduce: str = "abs_sum"
) -> Dict[str, torch.Tensor]:
    """Every measure above, per sample, from one attribution batch."""
    s = spatial_map(a, reduce=reduce)
    return {
        "gini": gini(s),
        "spatial_entropy": spatial_entropy(s),
        "effective_support": effective_support(s),
        "top1pct_mass": topk_mass(s, 0.01),
        "top5pct_mass": topk_mass(s, 0.05),
        "total_variation": total_variation(s),
        "channel_cancellation": channel_cancellation(a),
    }


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def sharpness_report(
    sharp: "object",
    faithfulness: "object",
    faith_metric: str = "FaithfulnessCorrelation",
    sharp_metric: str = "gini",
):
    """Join a per-method sharpness table to a per-method faithfulness column
    and return the frontier, refusing to rank on sharpness alone.

    ``sharp`` and ``faithfulness`` are ``pandas.DataFrame``s indexed by method;
    ``faithfulness`` is the benchmark's own ``results/summary_n*.csv`` pivoted
    to one column per metric.

    Returns a frame with, per method: the sharpness value, the faithfulness
    value, and ``dominated_by`` -- the list of methods that are **both**
    sharper and more faithful.  A method with an empty ``dominated_by`` is on
    the Pareto frontier; that, and not the sharpness column, is the defensible
    claim.

    Why this shape.  The obvious table -- methods sorted by sharpness -- is
    known in advance to be topped by degenerate maps (§7.4: ``Saliency`` is
    ``|Gradient|`` and taking the absolute value destroys the direction while
    leaving the concentration intact).  Sorting by sharpness would therefore
    reward the failure mode.  Pareto dominance answers the question actually
    being asked, "is this method sharper *without* giving anything up".
    """
    import pandas as pd

    if sharp_metric not in sharp.columns:
        raise KeyError(f"{sharp_metric!r} not in sharpness table: {list(sharp.columns)}")
    if faith_metric not in faithfulness.columns:
        raise KeyError(
            f"{faith_metric!r} not in faithfulness table: {list(faithfulness.columns)}"
        )

    from .metrics import METRIC_DIRECTION

    s_dir = SHARPNESS_DIRECTION.get(sharp_metric, +1)
    f_dir = METRIC_DIRECTION.get(faith_metric, +1)
    if s_dir == 0:
        raise ValueError(f"{sharp_metric!r} is descriptive, not a quality axis")

    df = pd.DataFrame({
        sharp_metric: sharp[sharp_metric],
        faith_metric: faithfulness[faith_metric],
    }).dropna()
    sv = df[sharp_metric] * s_dir
    fv = df[faith_metric] * f_dir

    dominated = []
    for m in df.index:
        better = df.index[(sv > sv[m]) & (fv > fv[m])]
        dominated.append(list(better))
    df["dominated_by"] = dominated
    df["on_frontier"] = [len(d) == 0 for d in dominated]
    return df.sort_values(sharp_metric, ascending=(s_dir < 0))
