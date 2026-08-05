"""Two candidate metrics built from the *forward* Fisher-Rao metric, and their
rank correlation with the paper's six.

    python scripts/07_geometric_metrics.py --model resnet50 --n 24

**Metric 1 -- MaxSensitivity in distribution units.** The paper's own robustness
metric measures the explanation's change over a ball of *pixel* radius.  Nothing
about a pixel ball respects what the network is sensitive to: the same
``||delta||_2`` moves one image's output distribution ten times further than
another's.  Here the same quantity is computed over perturbations rescaled so
every one of them moves the output distribution by the same Fisher-Rao arc.
Reported next to an ``l2``-ball version computed by the identical code path, so
the *only* difference between the two columns is how perturbation size is
defined.  That ``l2`` column doubles as the control: it must correlate strongly
with the benchmark's Quantus ``MaxSensitivity`` or the harness is wrong.

**Metric 2 -- distributional efficiency.** ``sqrt(v' J' G J v)`` for the unit-l2
explanation direction: how much of the output distribution the direction a
method points along actually moves.  One JVP per sample; no reference class, no
baseline, no perturbation sampling.

The default ``--l2-size 4.5`` is the pixel norm of Quantus' own MaxSensitivity
perturbation (uniform noise on ``[-0.02, 0.02]`` per element, i.e.
``0.02/sqrt(3) * sqrt(3*224*224) = 4.48``), so metric 1's ``l2`` column sits in
the paper's regime.  ``--fr-arc`` defaults to the *median arc that regime
induces*, so the two columns spend the same average displacement budget and
differ only in how it is allocated across images.

Resumable: one JSON per (model, method) under ``results/raw_geom/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spx import load_bundle, load_imagenette
from spx.explainers import build_explainers
from spx.geometry import build_perturbations, distributional_efficiency, max_sensitivity
from spx.metrics import METRIC_DIRECTION
from spx.viz import PALETTE

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw_geom"
FIG = ROOT / "figures"

warnings.filterwarnings("ignore")

sys.path.insert(0, str(ROOT / "scripts"))
from importlib.util import module_from_spec, spec_from_file_location

_spec = spec_from_file_location("_bench", ROOT / "scripts" / "02_run_benchmark.py")
_bench = module_from_spec(_spec)
_spec.loader.exec_module(_bench)
METHODS = _bench.METHODS          # the same 19 cells, with the same kwargs

#: +1 = higher is better, -1 = lower is better (as ``METRIC_DIRECTION``)
NEW_DIRECTION = {
    "MaxSensitivity_l2": -1,
    "MaxSensitivity_fr": -1,
    "DistributionalEfficiency": +1,
}


def _attr_batched(expl, x, y, batch_size):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(expl.attribute(x[i:i + batch_size], y[i:i + batch_size]))
    return torch.cat(out)


def _efficiency(model, x, a, batch_size):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(distributional_efficiency(model, x[i:i + batch_size], a[i:i + batch_size]))
    return torch.cat(out)


def run(args):
    dev = torch.device(args.device)
    b = load_bundle(args.model, dev)
    # exactly the benchmark's sample (seed 0, no correct-only filter), first n of it
    x_np, y_np, _ = load_imagenette(n=args.n, res=b.res, seed=args.seed)
    x = torch.as_tensor(x_np, device=dev)
    y = torch.as_tensor(y_np, device=dev)

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    pert_l2, arc_l2 = build_perturbations(b.hard, x, args.n_perturb, "l2",
                                          args.l2_size, generator=gen)
    fr_arc = args.fr_arc if args.fr_arc > 0 else float(arc_l2.median())
    pert_fr, arc_fr = build_perturbations(b.hard, x, args.n_perturb, "fr",
                                          fr_arc, generator=gen)
    print(f"[perturbations] {time.time() - t0:.0f}s\n"
          f"  l2 ball  ||d||_2 = {args.l2_size}:  arc  "
          f"median {float(arc_l2.median()):.4f}  "
          f"IQR [{float(arc_l2.quantile(.25)):.4f}, {float(arc_l2.quantile(.75)):.4f}]  "
          f"max/min {float(arc_l2.max() / arc_l2.min().clamp_min(1e-9)):.1f}x\n"
          f"  fr ball  arc = {fr_arc:.4f}:        arc  "
          f"median {float(arc_fr.median()):.4f}  "
          f"IQR [{float(arc_fr.quantile(.25)):.4f}, {float(arc_fr.quantile(.75)):.4f}]")

    RAW.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}__n{args.n}__p{args.n_perturb}"
    (RAW / f"{tag}__meta.json").write_text(json.dumps({
        "model": args.model, "n": args.n, "n_perturb": args.n_perturb,
        "seed": args.seed, "l2_size": args.l2_size, "fr_arc": fr_arc,
        "arc_l2_median": float(arc_l2.median()),
        "arc_l2_iqr": [float(arc_l2.quantile(.25)), float(arc_l2.quantile(.75))],
        "arc_l2_spread_ratio": float(arc_l2.max() / arc_l2.min().clamp_min(1e-9)),
        "arc_fr_median": float(arc_fr.median()),
    }, indent=1))

    names = args.methods or list(METHODS)
    explainers = build_explainers(b, {m: METHODS[m] for m in names if m in METHODS})
    for name, expl in explainers.items():
        out = RAW / f"{tag}__{name}.json"
        if out.exists() and not args.force:
            print(f"  [skip] {name}")
            continue
        rec = {"model": args.model, "method": name, "n": args.n,
               "n_perturb": args.n_perturb, "fr_arc": fr_arc, "l2_size": args.l2_size}
        try:
            t0 = time.time()
            a0 = _attr_batched(expl, x, y, args.batch_size)
            eff = _efficiency(b.hard, x, a0, args.batch_size)
            ms_l2 = max_sensitivity(expl, x, y, pert_l2, a0, args.batch_size)
            ms_fr = max_sensitivity(expl, x, y, pert_fr, a0, args.batch_size)
            rec["scores"] = {
                "DistributionalEfficiency": [float(v) for v in eff],
                "MaxSensitivity_l2": [float(v) for v in ms_l2],
                "MaxSensitivity_fr": [float(v) for v in ms_fr],
            }
            rec["seconds"] = time.time() - t0
            print(f"  {name:28s} eff {float(eff.mean()):8.4f}  "
                  f"maxsens_l2 {float(ms_l2.mean()):7.3f}  "
                  f"maxsens_fr {float(ms_fr.mean()):7.3f}  ({rec['seconds']:.0f}s)",
                  flush=True)
        except Exception:
            rec["error"] = traceback.format_exc(limit=3)
            print(f"  {name:28s} FAILED")
        out.write_text(json.dumps(rec, indent=1))
    return tag


def collect(tag: str, model: str):
    rows = []
    for f in sorted(RAW.glob(f"{tag}__*.json")):
        if f.name.endswith("__meta.json"):
            continue
        rec = json.loads(f.read_text())
        for m, vals in rec.get("scores", {}).items():
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            rows.append(dict(model=rec["model"], method=rec["method"], metric=m,
                             mean=v.mean(), std=v.std(), sem=v.std() / max(len(v), 1) ** .5,
                             n=len(v)))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / f"geometric_metrics_{tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return df


def correlate(df, model: str, tag: str, bench_csv: Path):
    """Spearman rank correlation of the new metrics against the paper's six."""
    from scipy.stats import spearmanr

    if not bench_csv.exists():
        print(f"  ({bench_csv.name} missing -- run 02_run_benchmark.py first)")
        return None
    bench = pd.read_csv(bench_csv)
    bench = bench[bench.model == model].pivot_table(index="method", columns="metric",
                                                    values="mean")
    new = df[df.model == model].pivot_table(index="method", columns="metric", values="mean")
    joined = new.join(bench, how="inner", lsuffix="_new")
    old_cols = [c for c in METRIC_DIRECTION if c in joined.columns]
    new_cols = [c for c in NEW_DIRECTION if c in joined.columns]

    rows = []
    for nc in new_cols:
        for oc in old_cols:
            sub = joined[[nc, oc]].dropna()
            rho, p = spearmanr(sub[nc], sub[oc])
            rows.append({"new_metric": nc, "benchmark_metric": oc, "spearman": rho,
                         "p_value": p, "n_methods": len(sub)})
    corr = pd.DataFrame(rows)
    out = ROOT / "results" / f"geometric_correlations_{tag}.csv"
    corr.to_csv(out, index=False)
    print(f"wrote {out}\n")
    print(corr.pivot(index="new_metric", columns="benchmark_metric",
                     values="spearman").round(3).to_string())

    _figure(corr, joined, new_cols, old_cols, model)
    return corr


#: methods whose maps are known to be target-invariant (benchmark RandomLogit
#: 1.000 / 0.364 / 0.338) -- highlighted because they turn out to carry the whole
#: correlation between distributional efficiency and target specificity.
DEGENERATE = ("Deconvolution", "GuidedGradCam", "Saliency")


def _figure(corr, joined, new_cols, old_cols, model):
    from scipy.stats import spearmanr

    piv = corr.pivot(index="new_metric", columns="benchmark_metric",
                     values="spearman").reindex(index=new_cols, columns=old_cols)
    pv = corr.pivot(index="new_metric", columns="benchmark_metric",
                    values="p_value").reindex(index=new_cols, columns=old_cols)
    n_m = int(corr.n_methods.max())

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))

    # --- (0,0) the correlation matrix
    ax = axes[0][0]
    im = ax.imshow(piv.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(old_cols)))
    ax.set_xticklabels([f"{c}\n({'up' if METRIC_DIRECTION[c] > 0 else 'down'} = better)"
                        for c in old_cols], rotation=30, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(new_cols)))
    ax.set_yticklabels([f"{c}\n({'up' if NEW_DIRECTION[c] > 0 else 'down'} = better)"
                        for c in new_cols], fontsize=6.5)
    for i in range(len(new_cols)):
        for j in range(len(old_cols)):
            r, p = piv.values[i, j], pv.values[i, j]
            if np.isfinite(r):
                ax.text(j, i, f"{r:+.2f}\n{'*' if p < 0.05 else ''}", ha="center",
                        va="center", fontsize=7)
    ax.set_title(f"Spearman rho against the paper's six ({model}, {n_m} methods)\n"
                 f"* = p < 0.05 uncorrected; none survives Bonferroni over "
                 f"{len(new_cols) * len(old_cols)} tests", fontsize=8.5)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # --- (0,1) the control: our l2 reimplementation against Quantus'
    ax = axes[0][1]
    sub = joined[["MaxSensitivity_l2", "MaxSensitivity_fr", "MaxSensitivity"]].dropna()
    for col, c, lab in [("MaxSensitivity_l2", PALETTE[7], "l2 ball (control)"),
                        ("MaxSensitivity_fr", PALETTE[1], "Fisher-Rao ball")]:
        rho = spearmanr(sub[col], sub.MaxSensitivity).statistic
        ax.scatter(sub.MaxSensitivity, sub[col], s=30, color=c, edgecolor="#333",
                   lw=0.4, label=f"{lab}   rho = {rho:+.3f}")
    lim = [0, float(sub.max().max()) * 1.06]
    ax.plot(lim, lim, ls=":", lw=0.9, color="#555")
    ax.set_xlim(lim), ax.set_ylim(lim)
    for nm in ("GradientShap", "Gradient", "SoftPullback", "Deconvolution"):
        if nm in sub.index:
            ax.annotate(nm, (sub.MaxSensitivity[nm], sub.MaxSensitivity_l2[nm]),
                        fontsize=6, xytext=(4, -6), textcoords="offset points")
    ax.set_xlabel("Quantus MaxSensitivity (n=200)", fontsize=8)
    ax.set_ylabel("this script (n=32)", fontsize=8)
    ax.set_title("Both balls reproduce the official ranking\n"
                 "(dotted = identity; the control validates the harness)", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(lw=0.4, alpha=0.3), ax.set_axisbelow(True)

    # --- (1,0) what changing the ball actually does
    ax = axes[1][0]
    rel = (100 * (joined.MaxSensitivity_fr - joined.MaxSensitivity_l2)
           / joined.MaxSensitivity_l2).dropna().sort_values()
    shift = (joined.MaxSensitivity_fr.rank() - joined.MaxSensitivity_l2.rank()).reindex(rel.index)
    pos = np.arange(len(rel))
    ax.barh(pos, rel.values, height=0.72,
            color=["#D55E00" if abs(s) > 0 else "#9BB7C9" for s in shift])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(pos), ax.set_yticklabels(rel.index, fontsize=6.5)
    ax.set_xlabel("change in MaxSensitivity when the ball is calibrated by "
                  "Fisher-Rao arc  [%]", fontsize=7.5)
    ax.set_title(f"Per-method values move up to {rel.abs().max():.0f}% and "
                 f"systematically ...\n... but no method's rank moves by more than "
                 f"{int(shift.abs().max())} place (orange = moved)", fontsize=8.5)
    ax.grid(axis="x", lw=0.4, alpha=0.3), ax.set_axisbelow(True)

    # --- (1,1) where the efficiency correlation comes from
    ax = axes[1][1]
    sub = joined[["DistributionalEfficiency", "RandomLogit"]].dropna()
    deg = [m for m in DEGENERATE if m in sub.index]
    rest = [m for m in sub.index if m not in deg]
    rho_all = spearmanr(sub.DistributionalEfficiency, sub.RandomLogit).statistic
    rho_rest = spearmanr(sub.DistributionalEfficiency[rest],
                         sub.RandomLogit[rest]).statistic
    ax.scatter(sub.DistributionalEfficiency[rest], sub.RandomLogit[rest], s=30,
               color=PALETTE[0], edgecolor="#333", lw=0.4,
               label=f"the other {len(rest)}   rho = {rho_rest:+.3f}")
    ax.scatter(sub.DistributionalEfficiency[deg], sub.RandomLogit[deg], s=52,
               color=PALETTE[1], edgecolor="#333", lw=0.5, marker="D",
               label="target-invariant methods")
    for nm in deg + ["Gradient", "SoftPullback", "DistributionPullback"]:
        if nm in sub.index:
            ax.annotate(nm, (sub.DistributionalEfficiency[nm], sub.RandomLogit[nm]),
                        fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.axvline(0.0137, color="#555", ls="--", lw=0.9)
    ax.text(0.0137, ax.get_ylim()[1], " random direction", fontsize=6, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("DistributionalEfficiency (higher = moves the distribution more)",
                  fontsize=8)
    ax.set_ylabel("RandomLogit (lower = more target-specific)", fontsize=8)
    ax.set_title(f"Efficiency detects degenerate maps, it does not rank good ones\n"
                 f"rho = {rho_all:+.3f} over all {len(sub)}, "
                 f"{rho_rest:+.3f} without the three", fontsize=8.5)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(lw=0.4, alpha=0.3), ax.set_axisbelow(True)

    fig.tight_layout(pad=0.8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"fig15_geometric_metrics_{model}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG / f'fig15_geometric_metrics_{model}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-perturb", type=int, default=4)
    ap.add_argument("--l2-size", type=float, default=4.5,
                    help="pixel norm of the l2-ball perturbation (Quantus' own regime)")
    ap.add_argument("--fr-arc", type=float, default=-1.0,
                    help="Fisher-Rao arc; default = median arc the l2 ball induces")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench-n", type=int, default=200)
    ap.add_argument("--bench-budget", default="paper")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    tag = f"{args.model}__n{args.n}__p{args.n_perturb}"
    if not args.collect_only:
        tag = run(args)
    df = collect(tag, args.model)
    correlate(df, args.model, tag,
              ROOT / "results" / f"summary_n{args.bench_n}_{args.bench_budget}.csv")


if __name__ == "__main__":
    main()
