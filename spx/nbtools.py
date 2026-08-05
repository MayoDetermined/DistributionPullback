"""Shared plumbing for the notebooks.

They ask a lot of the same questions -- what does the n=200 table say, where
does this method sit on the metric axes, what does its map point at, how sharp
is it -- so the boilerplate lives here and the notebooks stay readable.

Nothing here runs a benchmark. Functions either read results/ or do a handful
of attribution passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .metrics import METRIC_DIRECTION
from .unify import METRIC_ORDER

__all__ = [
    "ROOT",
    "RESULTS",
    "FIGURES",
    "PAPER_METHODS",
    "summary_table",
    "rank_of",
    "unification_tables",
    "resolved_wins",
    "attributions_for",
    "show_maps",
    "sharpness_table",
    "have",
]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

#: The paper's own methods, for "against the base method" comparisons.
PAPER_METHODS = ["SoftPullback", "DoublePullback", "PullbackAscent"]


def have(path) -> bool:
    """Does this precomputed artefact exist?  Notebooks branch on it rather
    than failing, so a fresh clone still runs top to bottom."""
    return Path(path).exists()


# --------------------------------------------------------------------------
# the published tables
# --------------------------------------------------------------------------


def summary_table(model: str = "resnet50", n: int = 200) -> pd.DataFrame:
    """``results/summary_n{n}_paper.csv`` pivoted to methods x metrics (means)."""
    f = RESULTS / f"summary_n{n}_paper.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} -- run scripts/02_run_benchmark.py --n {n}")
    s = pd.read_csv(f)
    s = s[s.model == model]
    piv = s.pivot_table(index="method", columns="metric", values="mean")
    return piv.reindex(columns=[m for m in METRIC_ORDER if m in piv.columns])


def rank_of(method: str, model: str = "resnet50", n: int = 200) -> pd.DataFrame:
    """One method's rank on each metric, out of how many, with the value.

    Ranks are computed with each metric's own direction, so 1 is always best.
    """
    piv = summary_table(model, n)
    out = []
    for m in piv.columns:
        col = piv[m].dropna()
        asc = METRIC_DIRECTION[m] < 0  # lower-better -> ascending sort
        r = col.rank(ascending=asc, method="min")
        out.append({
            "metric": m,
            "value": col.get(method, np.nan),
            "rank": int(r.get(method, np.nan)) if method in r.index else np.nan,
            "of": int(len(col)),
            "best": col.index[r.argmin()] if len(col) else None,
        })
    return pd.DataFrame(out).set_index("metric")


# --------------------------------------------------------------------------
# the unification analysis
# --------------------------------------------------------------------------


def unification_tables(model: str = "resnet50", n: int = 200, budget: str = "paper"):
    """``(variance_decomposition, pca_loadings, pca_scores, pca_explained)``.

    Reads ``results/unification/`` if ``scripts/10_metric_unification.py`` has
    been run, otherwise recomputes from ``results/raw/`` on the spot -- it takes
    seconds and touches no model.
    """
    tag = f"{model}__n{n}__{budget}"
    d = RESULTS / "unification"
    files = {
        "vd": d / f"variance_decomposition__{tag}.csv",
        "load": d / f"pca_loadings__{tag}.csv",
        "score": d / f"pca_scores__{tag}.csv",
        "expl": d / f"pca_explained__{tag}.csv",
    }
    if all(f.exists() for f in files.values()):
        return (
            pd.read_csv(files["vd"], index_col=0),
            pd.read_csv(files["load"], index_col=0),
            pd.read_csv(files["score"], index_col=0),
            pd.read_csv(files["expl"], index_col=0),
        )

    from .unify import factor_analysis, load_raw, variance_decomposition

    df = load_raw(model, n=n, budget=budget)
    fa = factor_analysis(df)
    expl = pd.DataFrame({
        "eigenvalue": fa["eigenvalues"],
        "explained": fa["explained"],
        "cumulative": fa["explained"].cumsum(),
    })
    return variance_decomposition(df), fa["loadings"], fa["scores"], expl


def resolved_wins(
    method: str, model: str = "resnet50", n: int = 200, threshold: float = 0.10
) -> pd.DataFrame:
    """A method's rank on each metric **next to how much that metric resolves**.

    The join that stops a headline being over-read.  A rank-1 finish on a metric
    whose ``var_method`` is 0.046 -- i.e. under 5 % of its variance is about
    which method produced the map -- is a much weaker statement than a rank-1
    finish on one at 0.80, and the published ``mean +/- std`` table cannot show
    the difference.  ``resolved`` flags the metrics above ``threshold``.
    """
    r = rank_of(method, model, n)
    vd, *_ = unification_tables(model, n)
    out = r.join(vd[["var_method", "var_image", "var_resid"]], how="left")
    out["resolved"] = out["var_method"] >= threshold
    return out


# --------------------------------------------------------------------------
# maps
# --------------------------------------------------------------------------


def attributions_for(
    bundle,
    x: torch.Tensor,
    y: torch.Tensor,
    methods: Dict[str, dict],
    batch_size: int = 4,
) -> Dict[str, np.ndarray]:
    """``{name: (B, C, H, W) numpy}`` for a small batch.  Attribution only."""
    from .explainers import build_explainers

    out = {}
    for name, e in build_explainers(bundle, methods).items():
        chunks = []
        for i in range(0, len(x), batch_size):
            chunks.append(
                e.attribute(x[i:i + batch_size], y[i:i + batch_size]).detach().cpu().numpy()
            )
        out[name] = np.concatenate(chunks)
    return out


def show_maps(
    x: torch.Tensor,
    attrs: Dict[str, np.ndarray],
    titles: Optional[Sequence[str]] = None,
    mode: str = "sum",
    suptitle: str = "",
):
    """The paper's Fig. 3 layout, inline.

    ``mode="sum"`` reproduces every attribution figure in the paper and in
    ``04_make_figures.py``; §6.1 of the design notes measures that this throws
    away ~19 % of the mass to sign cancellation between the three input
    channels, so ``mode="absmax"`` is worth a second look for any map that
    seems empty.
    """
    from .viz import attribution_grid

    return attribution_grid(
        x.detach().cpu().numpy(), attrs, titles=titles, mode=mode, suptitle=suptitle
    )


# --------------------------------------------------------------------------
# sharpness
# --------------------------------------------------------------------------


def sharpness_table(
    attrs: Dict[str, np.ndarray], reduce: str = "abs_sum"
) -> pd.DataFrame:
    """Per-method means of every sharpness measure, from maps already computed."""
    from .sharpness import sharpness_measures

    rows = []
    for name, a in attrs.items():
        m = sharpness_measures(torch.as_tensor(a), reduce=reduce)
        rec = {"method": name}
        rec.update({k: float(v.mean()) for k, v in m.items()})
        rows.append(rec)
    return pd.DataFrame(rows).set_index("method")
