#!/usr/bin/env python
"""Sharpness of the attribution maps, and the frontier against faithfulness.

    python scripts/11_sharpness.py --model resnet50 --n 200

Attribution only -- no Quantus, no perturbation sampling -- so this is cheap
even on CPU: one attribution pass per method, the same cost as the
``attribute`` line of `02_run_benchmark.py` and none of the rest.

Writes `results/sharpness/`:

    sharpness__<tag>.csv    per-method means of every measure
    per_sample__<tag>.csv   per-method, per-image values
    frontier__<tag>.csv     Pareto frontier of sharpness vs each faithfulness
                            metric, from `results/summary_n*.csv`

The frontier is the point.  `docs/DESIGN_NOTES.md` §7.4 already caught a
sharpness-like measure ranking the raw `Gradient` first of 19 while its
FaithfulnessCorrelation was a fifth of Soft Pullback's, and `Saliency` --
literally `|Gradient|` -- keeps all of the concentration while losing the
direction.  A sharpness table sorted by sharpness therefore rewards the failure
mode; only "sharper *without* giving anything up" is a claim worth making.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from spx import build_explainers, load_bundle, load_imagenette
from spx.metrics import METRIC_DIRECTION
from spx.sharpness import SHARPNESS_DIRECTION, sharpness_measures, sharpness_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sharpness"

sys.path.insert(0, str(ROOT / "scripts"))


def _methods():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench", ROOT / "scripts" / "02_run_benchmark.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.METHODS), set(mod.ENTROPY_OPT_IN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reduce", default="abs_sum", choices=["abs_sum", "sum_abs"])
    ap.add_argument("--summary", default=None,
                    help="results/summary_n*.csv for the frontier; default matches --n")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{a.model}__n{a.n}__{a.reduce}"

    all_methods, opt_in = _methods()
    names = a.methods or [m for m in all_methods if m not in opt_in]
    spec = {m: all_methods.get(m, {}) for m in names}

    b = load_bundle(a.model, device)
    x, y, _ = load_imagenette(n=a.n, res=b.res, seed=a.seed)
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    explainers = build_explainers(b, spec)

    rows, recs = [], []
    for name, e in explainers.items():
        vals: dict[str, list[torch.Tensor]] = {}
        for i in range(0, len(x), a.batch_size):
            m = sharpness_measures(
                e.attribute(x[i:i + a.batch_size], y[i:i + a.batch_size]).detach(),
                reduce=a.reduce,
            )
            for k, v in m.items():
                vals.setdefault(k, []).append(v.cpu())
        rec = {"method": name}
        for k, v in vals.items():
            arr = torch.cat(v).numpy()
            rec[k] = float(np.nanmean(arr))
            rec[f"{k}__std"] = float(np.nanstd(arr))
            rows.extend(
                {"method": name, "sample": j, "measure": k, "value": float(s)}
                for j, s in enumerate(arr)
            )
        recs.append(rec)
        print(f"  {name:28s} gini {rec['gini']:.3f}  "
              f"eff.support {rec['effective_support']:.4f}  "
              f"top1% {rec['top1pct_mass']:.3f}")

    sharp = pd.DataFrame(recs).set_index("method")
    sharp.to_csv(OUT / f"sharpness__{tag}.csv")
    pd.DataFrame(rows).to_csv(OUT / f"per_sample__{tag}.csv", index=False)

    # -- the frontier ----------------------------------------------------
    summ = Path(a.summary) if a.summary else ROOT / "results" / f"summary_n{a.n}_paper.csv"
    if not summ.exists():
        print(f"\n[frontier skipped] {summ} not found -- run 02_run_benchmark.py first")
        return
    s = pd.read_csv(summ)
    s = s[s.model == a.model]
    faith = s.pivot_table(index="method", columns="metric", values="mean")

    fr = []
    for sm in [k for k, v in SHARPNESS_DIRECTION.items() if v != 0]:
        for fm in [m for m in METRIC_DIRECTION if m in faith.columns]:
            if fm == "Infidelity":
                continue  # fifteen decades; the rank-based view is in unify.py
            t = sharpness_report(sharp, faith, faith_metric=fm, sharp_metric=sm)
            t = t.reset_index().rename(columns={"index": "method", sm: "sharpness",
                                                fm: "faithfulness"})
            t.insert(0, "faith_metric", fm)
            t.insert(0, "sharp_metric", sm)
            fr.append(t)
    frontier = pd.concat(fr, ignore_index=True)
    frontier.to_csv(OUT / f"frontier__{tag}.csv", index=False)

    print("\n=== Pareto frontier, gini vs FaithfulnessCorrelation ===")
    sub = frontier[(frontier.sharp_metric == "gini")
                   & (frontier.faith_metric == "FaithfulnessCorrelation")]
    print(sub[["method", "sharpness", "faithfulness", "on_frontier"]]
          .round(4).to_string(index=False))
    print(f"\nwrote {OUT}")
    print("\nA method not on the frontier is beaten on BOTH axes by something "
          "else, and its sharpness is not a selling point.")


if __name__ == "__main__":
    main()
