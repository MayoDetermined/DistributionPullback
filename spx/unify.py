"""Do the six explanation-quality metrics measure six things, or fewer?

results/raw/*.json keeps the per-sample score of every metric for every method,
which is more than the published tables use they collapse each cell to
mean +/- std. With the per-sample scores there are three separable questions,
and they get different answers:

1. Do the metrics rank methods the same way? Spearman over 19 methods. This
   is what DESIGN_NOTES §7.4 asks of the two candidate metrics. n = 19, so
   almost nothing survives correction.
2. Do they agree about which images are hard? Spearman over the 200 samples,
   inside each method, pooled. n = 200 per method, so this one is actually
   powered and nobody has looked at it.
3. How much of a metric's variance is about the method at all? §4 already warns
   that the per-sample sds are wide for the correlation metrics; if the image
   term dominates then differences between adjacent rows in the published table
   are being read out of image noise.

Runs entirely off data already on disk. No model, no attributions.

Two things to be careful about:

Scale. Infidelity spans fifteen orders of magnitude across methods (8.86 for
SoftPullback, 9.18e15 for Deconvolution) and is heavy-tailed within a method
too. Any Pearson correlation, PCA or mean over that column is a report on
Deconvolution's tail. So everything here is rank-based and every correlation is
Spearman.

Orientation. Three of the six are lower-better. orient() multiplies by
METRIC_DIRECTION so higher is always better, without which half the signs in a
correlation matrix encode nothing but polarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import METRIC_DIRECTION

__all__ = [
    "METRIC_ORDER",
    "load_raw",
    "orient",
    "method_matrix",
    "metric_correlation_methods",
    "metric_correlation_samples",
    "variance_decomposition",
    "factor_analysis",
    "redundancy",
    "position_of",
]

METRIC_ORDER = [
    "Infidelity",
    "FaithfulnessCorrelation",
    "MonotonicityCorrelation",
    "FaithfulnessEstimate",
    "MaxSensitivity",
    "RandomLogit",
]

DEFAULT_RAW = Path(__file__).resolve().parents[1] / "results" / "raw"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_raw(
    model: str = "resnet50",
    n: Optional[int] = 200,
    budget: str = "paper",
    root: Optional[Path] = None,
    metrics: Sequence[str] = tuple(METRIC_ORDER),
) -> pd.DataFrame:
    """Long form: one row per (method, metric, image).

    `sample` indexes the benchmark's own seed=0 draw, so it lines up across
    methods -- that's what makes the paired analyses possible, and why
    02_run_benchmark.py fixes the seed.
    """
    root = Path(root) if root is not None else DEFAULT_RAW
    rows: List[dict] = []
    seen_n = set()
    for f in sorted(root.glob(f"{model}__*.json")):
        d = json.loads(f.read_text())
        if d.get("model") != model or d.get("budget") != budget:
            continue
        if n is not None and d.get("n") != n:
            continue
        seen_n.add(d.get("n"))
        for m, vals in (d.get("scores") or {}).items():
            if m not in metrics:
                continue
            arr = np.asarray(vals, dtype=np.float64).reshape(-1)
            for i, v in enumerate(arr):
                rows.append({"method": d["method"], "metric": m, "sample": i, "score": v})
    if not rows:
        raise FileNotFoundError(
            f"no raw results for model={model!r} n={n} budget={budget!r} under {root}"
        )
    df = pd.DataFrame(rows)
    if len(seen_n) > 1:
        raise ValueError(f"mixed sample sizes {seen_n} -- pass an explicit n")
    return df


def orient(df: pd.DataFrame) -> pd.DataFrame:
    """Flip lower-better metrics so higher is better everywhere."""
    out = df.copy()
    out["score"] = out["score"] * out["metric"].map(METRIC_DIRECTION).astype(float)
    return out


# --------------------------------------------------------------------------
# method-level view
# --------------------------------------------------------------------------


def method_matrix(
    df: pd.DataFrame, agg: str = "rank", oriented: bool = True
) -> pd.DataFrame:
    """``(methods x metrics)``.

    ``agg="rank"`` (default)
        the method's rank on that metric, 1 = best, computed from the per-sample
        **median**.  The only aggregate that survives ``Infidelity``'s fifteen
        decades.
    ``agg="median"``
        the per-sample median, oriented.
    ``agg="mean"``
        the published tables' statistic.  Provided so a notebook can *show* why
        it is not used downstream, not because it should be.
    """
    d = orient(df) if oriented else df
    if agg == "mean":
        piv = d.pivot_table(index="method", columns="metric", values="score", aggfunc="mean")
    else:
        piv = d.pivot_table(index="method", columns="metric", values="score", aggfunc="median")
    piv = piv.reindex(columns=[m for m in METRIC_ORDER if m in piv.columns])
    if agg == "rank":
        # higher (oriented) score = better = rank 1
        piv = piv.rank(ascending=False, method="average")
    return piv


def metric_correlation_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman between metrics over methods (n = 19).

    "Do these agree about which method is better?" Underpowered by
    construction: uncorrected 0.05 sits at |rho| ~ 0.46, Bonferroni over the
    15 pairs at ~0.68. redundancy() reports both.
    """
    piv = method_matrix(df, agg="median")
    return piv.corr(method="spearman")


def metric_correlation_samples(
    df: pd.DataFrame, methods: Optional[Iterable[str]] = None, min_pairs: int = 20
) -> pd.DataFrame:
    """Spearman between metrics over **images**, computed inside each method
    and pooled by Fisher-z average.

    "Do these metrics agree about which *image* is explained well?"  This is the
    powered version of the question -- ``n = 200`` images per method rather than
    19 methods -- and it is a genuinely different one.  Two metrics can rank
    methods identically while disagreeing completely about images, which is the
    signature of their sharing a *method-level* confound (map scale, say) rather
    than a common notion of explanation quality.

    Constant columns (``Deconvolution``'s ``RandomLogit`` is 1.000 +/- 1.3e-5)
    yield undefined correlations and are dropped from the pool rather than
    counted as zero.
    """
    from scipy.stats import spearmanr

    d = orient(df)
    if methods is not None:
        d = d[d["method"].isin(list(methods))]
    mets = [m for m in METRIC_ORDER if m in set(d["metric"])]
    acc = {(a, b): [] for a in mets for b in mets}

    for _, g in d.groupby("method"):
        w = g.pivot_table(index="sample", columns="metric", values="score")
        w = w.reindex(columns=mets)
        for a in mets:
            for b in mets:
                if a == b:
                    continue
                pair = w[[a, b]].dropna()
                if len(pair) < min_pairs:
                    continue
                if pair[a].nunique() < 3 or pair[b].nunique() < 3:
                    continue  # constant column -> undefined, not zero
                r = spearmanr(pair[a], pair[b]).statistic
                if np.isfinite(r):
                    acc[(a, b)].append(np.clip(r, -0.999999, 0.999999))

    out = pd.DataFrame(np.eye(len(mets)), index=mets, columns=mets)
    for a in mets:
        for b in mets:
            if a == b:
                continue
            vals = acc[(a, b)]
            out.loc[a, b] = float(np.tanh(np.mean(np.arctanh(vals)))) if vals else np.nan
    return out


# --------------------------------------------------------------------------
# how much of a metric is even about the method
# --------------------------------------------------------------------------


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Per metric: what share of the score variance is method, image, residual.

    A two-way decomposition on the rank-transformed scores (ranks taken within
    each metric across the whole method x image panel, so ``Infidelity``'s tail
    cannot dominate):

    ``var_method``
        between-method variance of the per-method mean rank.
    ``var_image``
        between-image variance of the per-image mean rank -- how much the metric
        is reporting "this picture is hard" rather than "this method is good".
    ``var_resid``
        the rest, i.e. genuine method x image interaction plus noise.

    The column that matters is ``var_method``.  Where it is small, the
    published ``mean +/- std`` table is comparing method means that sit inside a
    much larger image-driven spread, and the honest read of two adjacent rows is
    "not resolved" -- which §4 of the design notes already flags qualitatively.
    """
    d = orient(df)
    rows = []
    for metric, g in d.groupby("metric"):
        w = g.pivot_table(index="sample", columns="method", values="score")
        w = w.dropna(axis=0, how="any")
        if w.shape[0] < 3 or w.shape[1] < 2:
            continue
        # rank across the whole panel, so heavy tails cannot set the scale
        r = pd.DataFrame(
            w.stack().rank().unstack(), index=w.index, columns=w.columns
        )
        grand = r.values.mean()
        v_method = float(((r.mean(axis=0) - grand) ** 2).mean())
        v_image = float(((r.mean(axis=1) - grand) ** 2).mean())
        v_total = float(((r.values - grand) ** 2).mean())
        rows.append({
            "metric": metric,
            "var_method": v_method / v_total,
            "var_image": v_image / v_total,
            "var_resid": max(0.0, 1.0 - (v_method + v_image) / v_total),
            "n_images": w.shape[0],
            "n_methods": w.shape[1],
        })
    return pd.DataFrame(rows).set_index("metric").reindex(
        [m for m in METRIC_ORDER if m in {r["metric"] for r in rows}]
    )


# --------------------------------------------------------------------------
# the unification question
# --------------------------------------------------------------------------


def factor_analysis(df: pd.DataFrame, n_components: Optional[int] = None) -> Dict:
    """PCA of the ``(19 methods x 6 metrics)`` rank matrix.

    Returns ``{"eigenvalues", "explained", "loadings", "scores", "kaiser",
    "n_for_80", "n_for_90"}``.

    Read it as: *how many independent things do the six metrics measure about a
    method?*  Components are extracted from **ranks oriented so that higher is
    better**, standardised per metric, so the answer cannot be an artefact of
    ``Infidelity``'s scale and a positive loading always means "agrees with
    this metric's notion of good".

    Two honest caveats, both of which belong next to any claim made from this:

    * ``n = 19`` methods against 6 variables.  That is a thin panel for a factor
      model; the eigenvalues are estimated with wide error and the usual advice
      (10+ observations per variable) is not met.  Treat the component *count*
      as descriptive, not inferential.
    * The 19 methods are not a random sample of anything -- 11 of them are
      variants of each other from this repo's own families -- so the covariance
      structure partly reflects which methods were run.  :func:`factor_analysis`
      is therefore most useful re-run on subsets (e.g. Captum baselines only),
      which the notebooks do.
    """
    from sklearn.decomposition import PCA

    piv = method_matrix(df, agg="rank").dropna(axis=1, how="any")
    # ``method_matrix`` returns competition ranks, 1 = best.  Negate so that a
    # larger number is a *better* method on that metric; otherwise every
    # loading and every score below reads backwards, which is exactly the kind
    # of sign trap a reader cannot catch from the output.
    X = -piv.values.astype(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1).clip(1e-12)
    k = n_components or X.shape[1]
    p = PCA(n_components=k).fit(X)
    ev = p.explained_variance_
    return {
        "eigenvalues": pd.Series(ev, index=[f"PC{i+1}" for i in range(k)]),
        "explained": pd.Series(
            p.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(k)]
        ),
        "loadings": pd.DataFrame(
            p.components_.T, index=piv.columns, columns=[f"PC{i+1}" for i in range(k)]
        ),
        "scores": pd.DataFrame(
            p.transform(X), index=piv.index, columns=[f"PC{i+1}" for i in range(k)]
        ),
        "kaiser": int((ev > 1.0).sum()),
        "n_for_80": int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.80) + 1),
        "n_for_90": int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.90) + 1),
    }


def redundancy(df: pd.DataFrame, level: str = "methods") -> pd.DataFrame:
    """Pairwise |rho| with the two significance thresholds spelled out.

    ``level="methods"`` uses :func:`metric_correlation_methods` (``n = 19``),
    ``level="samples"`` uses :func:`metric_correlation_samples` (``n`` images
    per method, pooled).  The ``bonferroni`` column is over the 15 distinct
    pairs; §7.4 of the design notes is explicit that nothing there survived it,
    so reporting the raw ``p`` alone would repeat a mistake this repo has
    already documented.
    """
    from scipy.stats import spearmanr

    if level == "methods":
        piv = method_matrix(df, agg="median")
        n = piv.shape[0]
        C = piv.corr(method="spearman")
    elif level == "samples":
        C = metric_correlation_samples(df)
        n = int(df.groupby(["method", "metric"])["sample"].nunique().median())
    else:
        raise ValueError(level)

    mets = list(C.columns)
    rows = []
    for i, a in enumerate(mets):
        for b in mets[i + 1:]:
            r = float(C.loc[a, b])
            if not np.isfinite(r):
                rows.append({"a": a, "b": b, "rho": np.nan, "abs_rho": np.nan,
                             "p": np.nan, "sig_05": False, "sig_bonf": False, "n": n})
                continue
            # t approximation; exact enough at these n for a threshold call
            t = r * np.sqrt(max(n - 2, 1) / max(1 - r * r, 1e-12))
            from scipy.stats import t as tdist

            p = float(2 * tdist.sf(abs(t), max(n - 2, 1)))
            rows.append({
                "a": a, "b": b, "rho": r, "abs_rho": abs(r), "p": p,
                "sig_05": p < 0.05, "sig_bonf": p < 0.05 / 15, "n": n,
            })
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def position_of(
    fa: Dict, methods: Sequence[str], k: int = 2
) -> pd.DataFrame:
    """Where the named methods sit on the first ``k`` components.

    The point of the factor analysis for this repo: if the Fisher-Rao selector,
    the Margin selector and Entropy-Flow are three answers to one question they
    should cluster; if they are trading different things off they should not.
    """
    s = fa["scores"]
    cols = [f"PC{i+1}" for i in range(k)]
    missing = [m for m in methods if m not in s.index]
    if missing:
        raise KeyError(f"not in the panel: {missing}; have {sorted(s.index)}")
    return s.loc[list(methods), cols]
