"""Figures: attribution maps and metric plots.

Everything is signed-symmetric by default -- Semantic Pullbacks are *directions*
in input space, so clipping to the positive part (as most saliency tooling
does) throws away half the information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .data import denormalise

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

#: colour-blind-safe qualitative palette (Okabe-Ito ordering)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#7F7F7F",
           "#8C564B", "#333333", "#17BECF", "#9467BD"]

SIGNED_CMAP = "coolwarm"


#: display gamma.  Raw pullbacks are extremely heavy-tailed -- for ResNet-50 the
#: median |value| is ~11% of the 99.5th percentile -- so a linear colour scale
#: renders all but a few hundred pixels flat grey.  A signed power law
#: ``sign(t)|t|^GAMMA`` lifts the mid-range without touching sign, ordering or
#: the location of the extremes.  Display only: no attribution used for a
#: metric is ever passed through this.
GAMMA = 0.55


def _saturate(m: np.ndarray, v: float, gamma: float = GAMMA) -> np.ndarray:
    """Clip to ``[-v, v]``, rescale to ``[-1, 1]``, apply the display gamma."""
    t = np.clip(m / max(v, 1e-12), -1.0, 1.0)
    return np.sign(t) * np.abs(t) ** gamma


def _to_map(a: np.ndarray, mode: str = "sum", q: float = 99.5,
            gamma: float = GAMMA) -> tuple:
    """Channel-collapse an attribution to a 2-D map in ``[-1, 1]``.

    ``q`` is a robust saturation percentile; the top ``100 - q``% of pixels
    clip rather than dominating the scale.
    """
    a = np.asarray(a)
    m = a.sum(0) if mode == "sum" else a.mean(0)
    v = max(float(np.percentile(np.abs(m), q)), 1e-12)
    return _saturate(m, v, gamma), 1.0


# --------------------------------------------------------------------------
# channel-resolved maps
# --------------------------------------------------------------------------
#
# A Semantic Pullback lives in *input* space, so it has one signed value per
# (channel, pixel).  Every figure above -- and every saliency convention the
# paper inherits -- collapses that to one value per pixel by summing over
# channels.  Summation is not neutral: the three channel attributions can carry
# opposite signs at the same pixel, in which case the sum reports "no evidence"
# where the pullback actually says "push red up and blue down".  The helpers
# below keep the channels apart and measure how much the sum destroys.


CHANNEL_NAMES = ("R", "G", "B")


def channel_maps(
    a: np.ndarray,
    q: float = 99.5,
    gamma: float = GAMMA,
    include_sum: bool = True,
    shared_scale: bool = True,
    scale: Optional[float] = None,
) -> List[tuple]:
    """``(C, H, W)`` attribution -> ``[(name, 2-D map in [-1, 1]), ...]``.

    ``shared_scale`` puts every channel panel on one common saturation value,
    so panel intensity is comparable *between* channels -- which is the whole
    point of not summing.  Per-panel normalisation would hide exactly the
    imbalance we are looking for.  Pass ``scale`` to fix that value externally
    (e.g. one scale for a whole row of images or targets).

    The channel-summed panel keeps its own scale: it is a different quantity,
    up to ``C`` times larger, and putting it on the channel scale would make it
    a solid block of clipped colour.
    """
    a = np.asarray(a)
    v = scale if scale is not None else float(np.percentile(np.abs(a), q))
    out = []
    for i, nm in enumerate(CHANNEL_NAMES[: a.shape[0]]):
        out.append((nm, _saturate(a[i], v if shared_scale else
                                  float(np.percentile(np.abs(a[i]), q)), gamma)))
    if include_sum:
        s = a.sum(0)
        out.append(("sum", _saturate(s, float(np.percentile(np.abs(s), q)), gamma)))
    return out


def channel_stats(a: np.ndarray) -> Dict[str, float]:
    """How much information the channel sum throws away, for one attribution.

    ``cancellation``
        ``1 - mean_pixel |sum_c a_c| / sum_c |a_c|``.  0 = the channels agree
        everywhere and the sum loses nothing; 1 = they cancel exactly and the
        summed map is blank where the pullback is not.
    ``share_R/G/B``
        fraction of total ``l1`` mass in each channel (``1/3`` each = balanced).
    ``cos_RG`` etc.
        cosine between the channel maps: negative means the channels are
        pushing in opposite directions across the image, not just at a pixel.
    """
    a = np.asarray(a, dtype=np.float64)
    absum = np.abs(a).sum(0)
    denom = absum.sum()
    out = {
        "cancellation": float(1.0 - np.abs(a.sum(0)).sum() / max(denom, 1e-12)),
    }
    for i, nm in enumerate(CHANNEL_NAMES[: a.shape[0]]):
        out[f"share_{nm}"] = float(np.abs(a[i]).sum() / max(denom, 1e-12))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if a.shape[0] > max(i, j):
            u, v = a[i].ravel(), a[j].ravel()
            nrm = np.linalg.norm(u) * np.linalg.norm(v)
            out[f"cos_{CHANNEL_NAMES[i]}{CHANNEL_NAMES[j]}"] = float(u @ v / max(nrm, 1e-12))
    return out


def _block_labels(fig, axes, names, first: int, width: int, dy: float = 0.012,
                  fontsize: float = 7.5):
    """Centre a label over each block of ``width`` consecutive columns.

    Called after ``tight_layout`` -- the block centre is read off the laid-out
    axes positions, which is the only way to get it right when the column
    widths are not known in advance.
    """
    top = max(ax.get_position().y1 for ax in axes[0])
    for j, name in enumerate(names):
        cols = axes[0][first + width * j: first + width * (j + 1)]
        x0 = min(ax.get_position().x0 for ax in cols)
        x1 = max(ax.get_position().x1 for ax in cols)
        fig.text(0.5 * (x0 + x1), top + dy, name, ha="center", va="bottom",
                 fontsize=fontsize, fontweight="bold")


def _block_labels(fig, axes, names, first: int, width: int, dy: float = 0.012,
                  fontsize: float = 7.5):
    """Centre a label over each block of ``width`` consecutive columns.

    Called after ``tight_layout`` -- the block centre is read off the laid-out
    axes positions, which is the only way to get it right when the column
    widths are not known in advance.
    """
    top = max(ax.get_position().y1 for ax in axes[0])
    for j, name in enumerate(names):
        cols = axes[0][first + width * j: first + width * (j + 1)]
        x0 = min(ax.get_position().x0 for ax in cols)
        x1 = max(ax.get_position().x1 for ax in cols)
        fig.text(0.5 * (x0 + x1), top + dy, name, ha="center", va="bottom",
                 fontsize=fontsize, fontweight="bold")


def channel_grid(
    images: np.ndarray,
    attributions: Dict[str, np.ndarray],
    titles: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
    suptitle: str = "",
    include_sum: bool = True,
    annotate_cancellation: bool = True,
):
    """Rows = images; per method a block of columns ``[R, G, B, (sum)]``.

    ``attributions`` maps method -> ``(N, C, H, W)``.  The three channel panels
    of one image share a saturation scale, so their relative intensity is
    readable; the scale is per *image*, since a pullback's magnitude varies by
    an order of magnitude between images and a batch-wide scale would black out
    the quiet ones.
    """
    names = list(attributions)
    per = 3 + int(include_sum)
    n_rows, n_cols = len(images), 1 + per * len(names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.42 * n_rows),
                             squeeze=False)
    for r in range(n_rows):
        axes[r][0].imshow(denormalise(images[r]))
        if titles is not None:
            axes[r][0].set_ylabel(titles[r], rotation=0, ha="right", va="center", fontsize=7)
        for j, name in enumerate(names):
            a = attributions[name][r]
            for i, (ch, m) in enumerate(channel_maps(a, include_sum=include_sum)):
                ax = axes[r][1 + per * j + i]
                ax.imshow(m, cmap=SIGNED_CMAP, vmin=-1, vmax=1)
                if r == 0:
                    ax.set_title(ch, fontsize=6.5)
            if annotate_cancellation:
                ax = axes[r][1 + per * j + per - 1]
                ax.text(0.97, 0.03, f"cancel {channel_stats(a)['cancellation']:.2f}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
    axes[0][0].set_title("input", fontsize=6.5)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.tight_layout(pad=0.22)
    _block_labels(fig, axes, names, first=1, width=per)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def channel_class_conditional_grid(
    image: np.ndarray,
    maps: Dict[str, np.ndarray],
    labels: Sequence[str],
    path: Optional[Path] = None,
    suptitle: str = "",
    include_sum: bool = True,
    true_label: Optional[str] = None,
):
    """One image, one column per target class, one row per (method, channel).

    ``maps`` maps method -> ``(n_cls, C, H, W)``.  Unlike
    :func:`class_conditional_grid` the scale is shared across *all* classes of a
    method, so a target whose pullback is genuinely weaker looks weaker instead
    of being renormalised back up -- necessary if the figure is to say anything
    about how the map fluctuates with the target.
    """
    names = list(maps)
    n_cls = len(labels)
    per = 3 + int(include_sum)
    rows = [(n, i) for n in names for i in range(per)]
    scales = {n: float(np.percentile(np.abs(maps[n]), 99.5)) for n in names}

    fig, axes = plt.subplots(len(rows), n_cls + 1,
                             figsize=(1.3 * (n_cls + 1), 1.34 * len(rows)), squeeze=False)
    for r, (name, i) in enumerate(rows):
        axes[r][0].imshow(denormalise(image))
        for c in range(n_cls):
            ch, m = channel_maps(maps[name][c], include_sum=include_sum,
                                 scale=scales[name])[i]
            axes[r][c + 1].imshow(m, cmap=SIGNED_CMAP, vmin=-1, vmax=1)
            if c == 0:
                axes[r][0].set_ylabel(f"{name}\n{ch}", rotation=0, ha="right",
                                      va="center", fontsize=6.5)
    for c, lb in enumerate(labels):
        star = "  *" if true_label is not None and lb == true_label else ""
        axes[0][c + 1].set_title(lb + star, fontsize=6.5,
                                 fontweight="bold" if star else "normal")
    axes[0][0].set_title("input", fontsize=6.5)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.005)
    fig.tight_layout(pad=0.22)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def target_fluctuation_panel(
    maps: Dict[str, np.ndarray],
    labels: Sequence[str],
    path: Optional[Path] = None,
    suptitle: str = "",
    true_idx: Optional[int] = None,
):
    """Quantify how one image's explanation moves as the *target* moves.

    ``maps``: method -> ``(n_cls, C, H, W)`` for a single image.  Three panels:

    1. cross-target cosine similarity of the full (channel-resolved)
       attribution -- a target-invariant method sits near 1 everywhere;
    2. per-channel ``l1`` share per target -- does the channel balance itself
       fluctuate with the target, or only the spatial pattern;
    3. per-pixel dispersion across targets, normalised by the mean magnitude --
       *where* in the image the explanation is target-sensitive.
    """
    names = list(maps)
    fig, axes = plt.subplots(len(names), 3,
                             figsize=(11.4, 3.25 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        a = np.asarray(maps[name], dtype=np.float64)
        n_cls = a.shape[0]
        flat = a.reshape(n_cls, -1)
        nrm = np.linalg.norm(flat, axis=1, keepdims=True)
        cos = (flat / np.clip(nrm, 1e-12, None)) @ (flat / np.clip(nrm, 1e-12, None)).T

        ax = axes[r][0]
        im = ax.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(n_cls)), ax.set_yticks(range(n_cls))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        off = cos[~np.eye(n_cls, dtype=bool)]
        ax.set_title(f"{name}: cross-target cosine\n"
                     f"mean off-diagonal {off.mean():+.3f}", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.75)

        ax = axes[r][1]
        share = np.abs(a).sum(axis=(2, 3))
        share = share / np.clip(share.sum(1, keepdims=True), 1e-12, None)
        pos = np.arange(n_cls)
        for i, nm in enumerate(CHANNEL_NAMES[: a.shape[1]]):
            ax.bar(pos + (i - 1) * 0.27, share[:, i], width=0.26,
                   color=PALETTE[i], label=nm)
        ax.axhline(1 / 3, color="#333", ls=":", lw=0.9)
        ax.set_xticks(pos), ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("share of |attribution|", fontsize=7)
        ax.set_ylim(0, float(share.max()) * 1.32)
        ax.set_title(f"{name}: channel balance per target\n"
                     f"(spread across targets: "
                     f"{share.std(0).max():.4f} max s.d.)", fontsize=8)
        ax.legend(fontsize=6.5, ncol=3, loc="upper center")
        ax.grid(axis="y", lw=0.4, alpha=0.3), ax.set_axisbelow(True)

        ax = axes[r][2]
        s = a.sum(1)                              # (n_cls, H, W), channel-summed
        disp = s.std(0) / np.clip(np.abs(s).mean(0), 1e-12, None)
        v = float(np.percentile(disp, 99.0))
        im = ax.imshow(np.clip(disp, 0, v), cmap="magma")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{name}: per-pixel dispersion across targets\n"
                     f"(std / mean|.|, median {np.median(disp):.2f})", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.75)
        if true_idx is not None:
            axes[r][0].add_patch(mpl.patches.Rectangle(
                (true_idx - 0.5, -0.5), 1, n_cls, fill=False, ec="#0072B2", lw=1.2))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.005)
    fig.tight_layout(pad=0.6)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def attribution_grid(
    images: np.ndarray,
    attributions: Dict[str, np.ndarray],
    titles: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
    mode: str = "sum",
    suptitle: str = "",
):
    """Rows = images, columns = [input] + methods.  The paper's Fig. 3 layout."""
    names = list(attributions)
    n_rows, n_cols = len(images), len(names) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.35 * n_cols, 1.45 * n_rows),
                             squeeze=False)
    for r in range(n_rows):
        axes[r][0].imshow(denormalise(images[r]))
        if titles is not None:
            axes[r][0].set_ylabel(titles[r], rotation=0, ha="right", va="center", fontsize=7)
        for c, name in enumerate(names, start=1):
            m, v = _to_map(attributions[name][r], mode)
            axes[r][c].imshow(m, cmap=SIGNED_CMAP, vmin=-v, vmax=v)
        for c in range(n_cols):
            axes[r][c].set_xticks([]), axes[r][c].set_yticks([])
            for s in axes[r][c].spines.values():
                s.set_visible(False)
    for c, name in enumerate(["input"] + names):
        axes[0][c].set_title(name, fontsize=7)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.tight_layout(pad=0.25)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def class_conditional_grid(
    image: np.ndarray,
    maps: Dict[str, np.ndarray],
    path: Optional[Path] = None,
    suptitle: str = "",
):
    """One image, one row per method, one column per target class (paper Fig. 2).

    ``maps`` maps method -> array of shape ``(n_classes, C, H, W)``; the class
    labels are taken from ``maps['__labels__']`` if present.
    """
    labels = maps.pop("__labels__", None)
    names = list(maps)
    n_cls = len(next(iter(maps.values())))
    fig, axes = plt.subplots(len(names), n_cls + 1,
                             figsize=(1.35 * (n_cls + 1), 1.45 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        axes[r][0].imshow(denormalise(image))
        axes[r][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
        for c in range(n_cls):
            m, v = _to_map(maps[name][c])
            axes[r][c + 1].imshow(m, cmap=SIGNED_CMAP, vmin=-v, vmax=v)
    for c in range(n_cls):
        if labels is not None:
            axes[0][c + 1].set_title(labels[c], fontsize=7)
    axes[0][0].set_title("input", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.tight_layout(pad=0.25)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def counterfactual_grid(
    images: np.ndarray,
    perturbations: Dict[str, np.ndarray],
    path: Optional[Path] = None,
    suptitle: str = "",
):
    """For each ascent method: the perturbation and the counterfactual ``x + d``
    (paper Fig. 4 / Sec. H)."""
    names = list(perturbations)
    n_rows = len(images)
    n_cols = 1 + 2 * len(names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.35 * n_cols, 1.45 * n_rows),
                             squeeze=False)
    for r in range(n_rows):
        axes[r][0].imshow(denormalise(images[r]))
        for j, name in enumerate(names):
            d = perturbations[name][r]
            m, v = _to_map(d)
            axes[r][1 + 2 * j].imshow(m, cmap=SIGNED_CMAP, vmin=-v, vmax=v)
            axes[r][2 + 2 * j].imshow(denormalise(images[r] + d))
    heads = ["input"] + [h for n in names for h in (f"{n}\nperturbation", f"{n}\nx + d")]
    for c, h in enumerate(heads):
        axes[0][c].set_title(h, fontsize=6.5)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.tight_layout(pad=0.25)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def counterfactual_target_grid(
    image: np.ndarray,
    rows: Sequence[dict],
    path: Optional[Path] = None,
    suptitle: str = "",
    include_channels: bool = True,
):
    """One image, one row per *counterfactual target*: what would have to change
    for the network to call this image a ``target``.

    Each entry of ``rows`` is ``{'label', 'delta', 'caption'}`` with ``delta``
    of shape ``(C, H, W)``.  Columns: input, the perturbation channel-summed,
    the three channel components (optional), and the counterfactual
    ``x + delta``.  All rows share one saturation scale so the amount of change
    a target demands is readable off the figure.
    """
    per = 1 + (3 if include_channels else 0)
    n_cols = 2 + per
    deltas = np.stack([r["delta"] for r in rows])
    scale = float(np.percentile(np.abs(deltas), 99.5))

    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(1.42 * n_cols, 1.52 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        axes[r][0].imshow(denormalise(image))
        axes[r][0].set_ylabel(row["label"], rotation=0, ha="right", va="center", fontsize=7)
        panels = channel_maps(row["delta"], include_sum=True, scale=scale)
        order = [panels[-1]] + (panels[:3] if include_channels else [])
        for i, (ch, m) in enumerate(order):
            axes[r][1 + i].imshow(m, cmap=SIGNED_CMAP, vmin=-1, vmax=1)
            if r == 0:
                axes[r][1 + i].set_title("delta (sum)" if ch == "sum" else f"delta {ch}",
                                         fontsize=6.5)
        axes[r][n_cols - 1].imshow(denormalise(image + row["delta"]))
        if row.get("caption"):
            axes[r][n_cols - 1].set_xlabel(row["caption"], fontsize=6, labelpad=1.5)
    axes[0][0].set_title("input", fontsize=6.5)
    axes[0][n_cols - 1].set_title("x + delta", fontsize=6.5)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.005)
    fig.tight_layout(pad=0.25)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def ascent_trajectory_plot(
    traj: Dict[str, Dict[str, Sequence[float]]],
    path: Optional[Path] = None,
    suptitle: str = "",
):
    """Per-step probabilities along an ascent, for several ascent methods.

    ``traj``: method -> ``{'p_target': [...], 'p_true': [...], 'margin': [...]}``,
    each list indexed by ascent step (entry 0 = the unperturbed image).  A
    counterfactual needs *both* halves: the target probability going up and the
    original class going down.  The one-hot selector only asks for the first.
    """
    keys = [("p_target", "p(counterfactual target)", "higher = ascent works"),
            ("p_true", "p(original class)", "lower = a real counterfactual"),
            ("margin", "top-k margin  m_c(x)", "the score Margin Ascent ascends")]
    keys = [k for k in keys if any(k[0] in v for v in traj.values())]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 2.9), squeeze=False)
    for j, (key, title, note) in enumerate(keys):
        ax = axes[0][j]
        for i, (name, v) in enumerate(traj.items()):
            if key not in v:
                continue
            y = np.asarray(v[key], dtype=float)
            ax.plot(np.arange(len(y)), y, "o-", ms=3.5, lw=1.3,
                    color=PALETTE[i % len(PALETTE)], label=name)
        ax.set_xlabel("ascent step")
        ax.set_title(f"{title}\n{note}", fontsize=8)
        ax.grid(lw=0.4, alpha=0.3), ax.set_axisbelow(True)
        if j == 0:
            ax.legend(fontsize=6.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9.5, y=1.02)
    fig.tight_layout(pad=0.5)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def counterfactual_target_grid(
    image: np.ndarray,
    rows: Sequence[dict],
    path: Optional[Path] = None,
    suptitle: str = "",
    include_channels: bool = True,
):
    """One image, one row per *counterfactual target*: what would have to change
    for the network to call this image a ``target``.

    Each entry of ``rows`` is ``{'label', 'delta', 'caption'}`` with ``delta``
    of shape ``(C, H, W)``.  Columns: input, the perturbation channel-summed,
    the three channel components (optional), and the counterfactual
    ``x + delta``.  All rows share one saturation scale so the amount of change
    a target demands is readable off the figure.
    """
    per = 1 + (3 if include_channels else 0)
    n_cols = 2 + per
    deltas = np.stack([r["delta"] for r in rows])
    scale = float(np.percentile(np.abs(deltas), 99.5))

    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(1.42 * n_cols, 1.52 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        axes[r][0].imshow(denormalise(image))
        axes[r][0].set_ylabel(row["label"], rotation=0, ha="right", va="center", fontsize=7)
        panels = channel_maps(row["delta"], include_sum=True, scale=scale)
        order = [panels[-1]] + (panels[:3] if include_channels else [])
        for i, (ch, m) in enumerate(order):
            axes[r][1 + i].imshow(m, cmap=SIGNED_CMAP, vmin=-1, vmax=1)
            if r == 0:
                axes[r][1 + i].set_title("delta (sum)" if ch == "sum" else f"delta {ch}",
                                         fontsize=6.5)
        axes[r][n_cols - 1].imshow(denormalise(image + row["delta"]))
        if row.get("caption"):
            axes[r][n_cols - 1].set_xlabel(row["caption"], fontsize=6, labelpad=1.5)
    axes[0][0].set_title("input", fontsize=6.5)
    axes[0][n_cols - 1].set_title("x + delta", fontsize=6.5)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.005)
    fig.tight_layout(pad=0.25)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def ascent_trajectory_plot(
    traj: Dict[str, Dict[str, Sequence[float]]],
    path: Optional[Path] = None,
    suptitle: str = "",
):
    """Per-step probabilities along an ascent, for several ascent methods.

    ``traj``: method -> ``{'p_target': [...], 'p_true': [...], 'margin': [...]}``,
    each list indexed by ascent step (entry 0 = the unperturbed image).  A
    counterfactual needs *both* halves: the target probability going up and the
    original class going down.  The one-hot selector only asks for the first.
    """
    keys = [("p_target", "p(counterfactual target)", "higher = ascent works"),
            ("p_true", "p(original class)", "lower = a real counterfactual"),
            ("margin", "top-k margin  m_c(x)", "the score Margin Ascent ascends")]
    keys = [k for k in keys if any(k[0] in v for v in traj.values())]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 2.9), squeeze=False)
    for j, (key, title, note) in enumerate(keys):
        ax = axes[0][j]
        for i, (name, v) in enumerate(traj.items()):
            if key not in v:
                continue
            y = np.asarray(v[key], dtype=float)
            ax.plot(np.arange(len(y)), y, "o-", ms=3.5, lw=1.3,
                    color=PALETTE[i % len(PALETTE)], label=name)
        ax.set_xlabel("ascent step")
        ax.set_title(f"{title}\n{note}", fontsize=8)
        ax.grid(lw=0.4, alpha=0.3), ax.set_axisbelow(True)
        if j == 0:
            ax.legend(fontsize=6.5)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9.5, y=1.02)
    fig.tight_layout(pad=0.5)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def metric_bars(
    df,
    metrics: Sequence[str],
    directions: Dict[str, int],
    highlight: Sequence[str] = (),
    log_metrics: Sequence[str] = ("Infidelity",),
    path: Optional[Path] = None,
    suptitle: str = "",
):
    """One horizontal bar panel per metric; methods sorted best-first.

    ``df`` is long-form with columns ``method``, ``metric``, ``mean``, ``std``.
    """
    n = len(metrics)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 0.28 * df.method.nunique() * nrow + 1.1),
                             squeeze=False)
    for i, metric in enumerate(metrics):
        ax = axes[i // ncol][i % ncol]
        sub = df[df.metric == metric].dropna(subset=["mean"])
        asc = directions[metric] > 0  # best (largest) ends up at the top
        sub = sub.sort_values("mean", ascending=asc)
        colors = ["#D55E00" if m in highlight else "#9BB7C9" for m in sub.method]
        pos = np.arange(len(sub))
        ax.barh(pos, sub["mean"], color=colors, height=0.72,
                xerr=None if metric in log_metrics else sub["std"],
                error_kw=dict(lw=0.5, ecolor="#5A5A5A", alpha=0.7))
        ax.set_yticks(pos)
        ax.set_yticklabels(sub.method, fontsize=6.5)
        arrow = "↑ higher better" if directions[metric] > 0 else "↓ lower better"
        ax.set_title(f"{metric}   ({arrow})", fontsize=8)
        if metric in log_metrics:
            ax.set_xscale("symlog", linthresh=1.0)
        ax.grid(axis="x", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.005)
    fig.tight_layout(pad=0.6)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def rank_heatmap(df, metrics, directions, path=None, suptitle=""):
    """Per-metric rank of every method (1 = best).  One glance at the trade-offs."""
    piv = df.pivot_table(index="method", columns="metric", values="mean")
    piv = piv.reindex(columns=[m for m in metrics if m in piv.columns])
    ranks = piv.copy()
    for m in ranks.columns:
        ranks[m] = piv[m].rank(ascending=(directions[m] < 0))
    order = ranks.mean(1).sort_values().index
    ranks = ranks.loc[order]

    fig, ax = plt.subplots(figsize=(0.72 * len(ranks.columns) + 3.0, 0.26 * len(ranks) + 1.2))
    im = ax.imshow(ranks.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(ranks.columns)))
    ax.set_xticklabels(ranks.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(ranks.index, fontsize=7)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            v = ranks.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=6.5)
    ax.set_title(suptitle or "Per-metric rank (1 = best)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label="rank")
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def sweep_plot(sweeps: Dict[str, Dict], xlabel: str, path=None, suptitle="",
               logx=False, ref: Optional[Dict[str, float]] = None):
    """``sweeps``: metric -> {'x': [...], 'mean': [...], 'std': [...]}."""
    names = list(sweeps)
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.4 * nrow), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        s = sweeps[name]
        x, m, sd = np.asarray(s["x"]), np.asarray(s["mean"]), np.asarray(s.get("std", 0))
        ax.plot(x, m, "o-", color=PALETTE[0], ms=3.5, lw=1.3)
        if np.any(sd):
            ax.fill_between(x, m - sd / np.sqrt(s.get("n", 1)), m + sd / np.sqrt(s.get("n", 1)),
                            color=PALETTE[0], alpha=0.18, lw=0)
        if ref and name in ref:
            ax.axhline(ref[name], color=PALETTE[1], ls="--", lw=1.0, label="SoftPullback")
            ax.legend(fontsize=6.5)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
        ax.grid(lw=0.4, alpha=0.3)
        ax.set_axisbelow(True)
    for j in range(len(names), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.01)
    fig.tight_layout(pad=0.5)
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig


def tradeoff_scatter(df, x_metric, y_metric, directions, highlight=(), path=None, suptitle=""):
    """Two metrics against each other, with the Pareto-optimal set connected."""
    piv = df.pivot_table(index="method", columns="metric", values="mean")
    piv = piv.dropna(subset=[x_metric, y_metric])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    for i, (name, row) in enumerate(piv.iterrows()):
        hl = name in highlight
        ax.scatter(row[x_metric], row[y_metric], s=54 if hl else 30,
                   color="#D55E00" if hl else "#9BB7C9",
                   edgecolor="#333", lw=0.5, zorder=3)
        ax.annotate(name, (row[x_metric], row[y_metric]), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"{x_metric} ({'higher' if directions[x_metric] > 0 else 'lower'} better)")
    ax.set_ylabel(f"{y_metric} ({'higher' if directions[y_metric] > 0 else 'lower'} better)")
    if directions[x_metric] < 0:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(suptitle, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig

