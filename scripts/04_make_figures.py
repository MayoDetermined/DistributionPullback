"""Produce every figure: attribution maps and metric plots.

    python scripts/04_make_figures.py --models resnet50 --n 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spx import load_bundle, load_imagenette
from spx.data import IMAGENETTE_NAMES, IMAGENETTE_WNIDS
from spx.explainers import ALL_METHODS
from spx.metrics import METRIC_DIRECTION
from spx.viz import (
    PALETTE, attribution_grid, class_conditional_grid, counterfactual_grid,
    metric_bars, rank_heatmap, sweep_plot, tradeoff_scatter,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OURS = ("DistributionPullback", "DistributionDoublePullback", "DistributionAscent",
        "MarginPullback", "MarginAscent")

MAP_METHODS = {
    "SoftPullback": {},
    "DoublePullback": {"alpha": 2.0},
    "PullbackAscent": {"alpha": 20.0, "steps": 5},
    "DistributionPullback": {},
    "MarginPullback": {"k": 8},
    "MarginAscent": {"k": 8, "alpha": 20.0, "steps": 5},
    "Gradient": {},
    "IntegratedGradients": {},
    "GuidedGradCam": {},
}
ASCENT_METHODS = {
    "PullbackAscent": {"alpha": 20.0, "steps": 5},
    "DistributionAscent": {"alpha": 20.0, "steps": 5},
    "MarginAscent": {"k": 8, "alpha": 20.0, "steps": 5},
    "GradientAscent": {"alpha": 20.0, "steps": 5},
}


def _attr_chunked(expl, x, y, chunk=2):
    """VGG11-bn at 224x224 will not hold six 5-step ascents at once on a 16 GB card."""
    out = []
    for i in range(0, len(x), chunk):
        out.append(expl.attribute(x[i:i + chunk], y[i:i + chunk]).cpu().numpy())
        torch.cuda.empty_cache()
    return np.concatenate(out)


def maps_and_counterfactuals(model_name, n_show=6):
    b = load_bundle(model_name, DEV)
    x_np, y_np, _ = load_imagenette(n=n_show, res=b.res, seed=7,
                                    correct_only=True, model=b.hard, device=DEV)
    x = torch.as_tensor(x_np, device=DEV)
    y = torch.as_tensor(y_np, device=DEV)
    labels = [IMAGENETTE_NAMES[int(v)] for v in y_np]

    attrs = {}
    for name, kw in MAP_METHODS.items():
        if name == "GuidedGradCam" and b.gradcam_layer is None:
            continue
        attrs[name] = _attr_chunked(ALL_METHODS[name](b, **kw), x, y)
    attribution_grid(x_np, attrs, titles=labels,
                     path=FIG / f"fig1_attribution_maps_{model_name}.png",
                     suptitle=f"Attribution maps -- {model_name} "
                              f"(signed, channel-summed, 99.5th-pct symmetric scale)")

    # --- class-conditional: one image, explanation toward each Imagenette class
    cls = [c for _, c, _ in IMAGENETTE_WNIDS]
    img = x[:1]
    cc = {}
    for name in ("SoftPullback", "DistributionPullback", "MarginPullback", "Gradient"):
        kw = MAP_METHODS.get(name, {})
        e = ALL_METHODS[name](b, **kw)
        cc[name] = np.concatenate([
            e.attribute(img, torch.tensor([c], device=DEV)).cpu().numpy() for c in cls])
    cc["__labels__"] = [IMAGENETTE_NAMES[c] for c in cls]
    class_conditional_grid(x_np[0], cc,
                           path=FIG / f"fig2_class_conditional_{model_name}.png",
                           suptitle=f"Target specificity -- {model_name}: same input, "
                                    f"explanation pulled back from each of the 10 classes")

    # --- counterfactual perturbations (paper Fig. 4 / Sec. H)
    pert = {}
    for name, kw in ASCENT_METHODS.items():
        pert[name] = _attr_chunked(ALL_METHODS[name](b, **kw), x[:4], y[:4])
    counterfactual_grid(x_np[:4], pert,
                        path=FIG / f"fig3_counterfactuals_{model_name}.png",
                        suptitle=f"Counterfactual perturbations -- {model_name} "
                                 f"(alpha=20, K=5)")
    del b
    torch.cuda.empty_cache()


def metric_figures(model_name, n, budget):
    csv = ROOT / "results" / f"summary_n{n}_{budget}.csv"
    if not csv.exists():
        print(f"  (no {csv.name}; skipping metric figures)")
        return
    df = pd.read_csv(csv)
    df = df[df.model == model_name]
    if df.empty:
        return
    metrics = [m for m in METRIC_DIRECTION if m in set(df.metric)]
    metric_bars(df, metrics, METRIC_DIRECTION, highlight=OURS,
                path=FIG / f"fig4_metrics_{model_name}.png",
                suptitle=f"Quantus explanation metrics -- {model_name}, "
                         f"n={n} Imagenette val (orange = this repo's extensions)")
    rank_heatmap(df, metrics, METRIC_DIRECTION,
                 path=FIG / f"fig5_rank_heatmap_{model_name}.png",
                 suptitle=f"Per-metric rank -- {model_name} (1 = best)")
    if {"MaxSensitivity", "FaithfulnessCorrelation"} <= set(metrics):
        tradeoff_scatter(df, "MaxSensitivity", "FaithfulnessCorrelation", METRIC_DIRECTION,
                         highlight=OURS, path=FIG / f"fig6_tradeoff_{model_name}.png",
                         suptitle=f"Robustness vs. faithfulness -- {model_name}")


def ablation_figures(model_name, n_abl):
    d = ROOT / "results" / "ablations"
    tag = f"{model_name}_n{n_abl}"
    cols = ["Infidelity", "MaxSensitivity", "FaithfulnessCorrelation",
            "delta_p", "stability_cos"]

    for key, fname, title, logx in [
        ("k", f"margin_k_{tag}.csv", "Margin Ascent: rival-set size k", False),
        ("tau_m", f"margin_tau_{tag}.csv",
         "Margin Ascent: rival temperature tau_m  (inf = uniform top-k, 1 = proportional)", True),
        ("p_floor", f"dist_floor_{tag}.csv",
         "Distribution Pullback: probability floor", True),
    ]:
        f = d / fname
        if not f.exists():
            continue
        t = pd.read_csv(f)
        xv = t[key].fillna(t[key].max() * 4 if key == "tau_m" else 0.0)
        if key == "p_floor":
            xv = xv.replace(0.0, 1e-7)
        n_eval = int(re.search(r"_n(\d+)\.csv$", fname).group(1))
        sweeps = {c: {"x": xv.values, "mean": t[c].values,
                      "std": t.get(c + "_std", pd.Series(np.zeros(len(t)))).values,
                      "n": n_eval}
                  for c in cols if c in t.columns}
        sweep_plot(sweeps, key, path=FIG / f"fig7_sweep_{key}_{model_name}.png",
                   suptitle=title + f"   (band = ±1 s.e.m., n={n_eval})", logx=logx)

    # --- selector-consistent infidelity
    f = d / f"selector_infidelity_{tag}.csv"
    if f.exists():
        t = pd.read_csv(f).sort_values("infid_target_logit")
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(t) + 1.4))
        pos = np.arange(len(t))
        ax.barh(pos - 0.19, t.infid_target_logit, height=0.36,
                color=PALETTE[7], label="vs. target logit  (what Quantus measures)")
        ax.barh(pos + 0.19, t.infid_explained_score, height=0.36,
                color=PALETTE[1], label="vs. the score the method ascends")
        ax.set_yticks(pos), ax.set_yticklabels(t.method, fontsize=7)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("Infidelity (lower better)")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_title("Infidelity depends on which score you measure against", fontsize=9)
        ax.grid(axis="x", lw=0.4, alpha=0.3), ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(FIG / f"fig8_selector_infidelity_{model_name}.png")
        plt.close(fig)

    # --- active-set dynamics
    f = d / f"active_set_{tag}.json"
    if f.exists():
        r = json.loads(f.read_text())
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.2))
        ax = axes[0]
        ax.plot(range(1, len(r["jaccard_consecutive"]) + 1), r["jaccard_consecutive"],
                "o-", color=PALETTE[0], label="consecutive steps")
        ax.plot(range(len(r["jaccard_vs_step0"])), r["jaccard_vs_step0"],
                "s--", color=PALETTE[1], label="vs. the initial set")
        ax.set_xlabel("ascent step"), ax.set_ylabel("Jaccard overlap of rival sets")
        ax.set_ylim(0, 1.02), ax.legend(fontsize=7)
        ax.set_title("The active set keeps moving ...", fontsize=9)
        ax.grid(lw=0.4, alpha=0.3), ax.set_axisbelow(True)

        # Which ingredient actually matters: the margin selector, re-deriving
        # the active set, or the soft adjoint?
        names = [k for k in r if isinstance(r[k], dict)]
        order = ["GradientAscent", "MarginGradientAscent", "PullbackAscent",
                 "MarginAscentFrozen", "MarginAscent"]
        names = [n for n in order if n in names] + [n for n in names if n not in order]
        soft = ["Gradient" not in n for n in names]  # uses the soft adjoint?
        colors = ["#D55E00" if s else "#9BB7C9" for s in soft]
        pos = np.arange(len(names))

        for ax, key, title, xlabel in [
            (axes[1], "p_end", "Margin selector: a small, real gain",
             "target probability after K=5 steps"),
            (axes[2], "stability_cos", "Soft adjoint: the dominant effect",
             "explanation cosine under sigma=0.03 noise"),
        ]:
            ax.barh(pos, [r[n][key] for n in names], color=colors, height=0.7)
            ax.set_yticks(pos), ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel(xlabel, fontsize=7.5)
            ax.set_title(title, fontsize=9)
            ax.grid(axis="x", lw=0.4, alpha=0.3), ax.set_axisbelow(True)
        axes[1].axvline(r[names[0]]["p_start"], color="#333", ls=":", lw=1.0,
                        label=f"start p = {r[names[0]]['p_start']:.3f}")
        axes[1].legend(fontsize=7, loc="lower right")
        axes[2].plot([], [], "s", color="#D55E00", label="soft adjoint")
        axes[2].plot([], [], "s", color="#9BB7C9", label="plain gradient")
        axes[2].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG / f"fig9_active_set_{model_name}.png")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-ablation", type=int, default=64)
    ap.add_argument("--budget", default="paper")
    ap.add_argument("--skip-maps", action="store_true")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    models = ["resnet50", "vgg11_bn", "pvt_v2_b1"] if args.models == ["all"] else args.models
    for m in models:
        print(f"[figures] {m}")
        if not args.skip_maps:
            maps_and_counterfactuals(m)
        metric_figures(m, args.n, args.budget)
        ablation_figures(m, args.n_ablation)
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()

