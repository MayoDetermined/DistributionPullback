"""Main evaluation: the paper's Quantus suite (Tables 2-4) over the paper's
methods plus the two extensions.

Resumable: one JSON per (model, method) under ``results/raw/``.  Re-running
skips finished cells, so the sweep can be interrupted.

    python scripts/02_run_benchmark.py --models resnet50 --n 200
    python scripts/02_run_benchmark.py --models all --n 200 --budget paper
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

import numpy as np
import pandas as pd
import torch

from spx import load_bundle, load_imagenette
from spx.explainers import build_explainers
from spx.metrics import METRIC_DIRECTION, build_metrics, run_metric

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

warnings.filterwarnings("ignore")

#: Method -> constructor kwargs.  Paper defaults: alpha=2 for Double Pullback
#: (Sec. 3.3), alpha=20 / K=5 for Pullback Ascent (Sec. 3.4).
METHODS = {
    # --- the paper -------------------------------------------------------
    "SoftPullback": {},
    "DoublePullback": {"alpha": 2.0},
    "PullbackAscent": {"alpha": 20.0, "steps": 5},
    # --- extension 1: Distribution Pullback ------------------------------
    "DistributionPullback": {"p_floor": 1e-3, "eps": 1e-3},
    "DistributionDoublePullback": {"p_floor": 1e-3, "eps": 1e-3, "alpha": 2.0},
    "DistributionAscent": {"p_floor": 1e-3, "eps": 1e-3, "alpha": 20.0, "steps": 5},
    # --- extension 2: Margin Ascent --------------------------------------
    "MarginPullback": {"k": 8, "tau_m": float("inf")},
    "MarginAscent": {"k": 8, "tau_m": float("inf"), "alpha": 20.0, "steps": 5},
    # --- extension 3: Entropy-Flow ---------------------------------------
    # Not in the default sweep: `--methods EntropyFlow ClassEntropyFlow` opts
    # in.  `EntropyFlow` is target-independent by construction, so its
    # `RandomLogit` is ~1.0 and putting it in the default table next to the
    # class-conditional methods would read as a failure rather than as the
    # definition it is; `docs/DESIGN_NOTES.md` §8.3 has the argument.
    "EntropyFlow": {"layer_mode": "uniform"},
    "ClassEntropyFlow": {"layer_mode": "uniform"},
    "EntropyMaskedPullback": {"mode": "undecided"},
    "EntropyAscent": {"alpha": 20.0, "steps": 5},
    # --- ablations -------------------------------------------------------
    "GradientAscent": {"alpha": 20.0, "steps": 5},
    "MarginAscentFrozen": {"k": 8, "tau_m": float("inf"), "alpha": 20.0, "steps": 5},
    "MarginGradientAscent": {"k": 8, "tau_m": float("inf"), "alpha": 20.0, "steps": 5},
    # --- Captum baselines (Sec. 4.1) -------------------------------------
    "Gradient": {},
    "Saliency": {},
    "InputXGradient": {},
    "IntegratedGradients": {},
    "GradientShap": {},
    "DeepLift": {},
    "Deconvolution": {},
    "GuidedGradCam": {},
}

#: Excluded from the default sweep, reachable with an explicit ``--methods``.
ENTROPY_OPT_IN = frozenset({
    "EntropyFlow", "ClassEntropyFlow", "EntropyMaskedPullback", "EntropyAscent",
})


def attribute_all(explainer, x, y, device, batch_size):
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i:i + batch_size], device=device)
        yb = torch.as_tensor(y[i:i + batch_size], device=device)
        out.append(explainer.attribute(xb, yb).cpu().numpy())
        del xb, yb
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50"],
                    help="resnet50 vgg11_bn pvt_v2_b1, or 'all'")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--metrics", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--budget", default="paper", choices=["paper", "fast"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--metric-batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = ["resnet50", "vgg11_bn", "pvt_v2_b1"] if args.models == ["all"] else args.models
    # The Entropy-Flow family is opt-in; see the comment on METHODS.
    method_names = args.methods or [m for m in METHODS if m not in ENTROPY_OPT_IN]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RAW.mkdir(parents=True, exist_ok=True)

    # Cell tag.  seed 0 keeps the historical `n{N}__{budget}` naming so existing
    # raw JSONs stay resumable; any other seed gets its own namespace, otherwise
    # a replication on a fresh sample would silently overwrite the original run.
    tag = f"n{args.n}__{args.budget}" + ("" if args.seed == 0 else f"__s{args.seed}")

    for model_name in models:
        print(f"\n{'=' * 70}\n{model_name}   n={args.n}  budget={args.budget}\n{'=' * 70}")
        bundle = load_bundle(model_name, device)
        x, y, paths = load_imagenette(n=args.n, res=bundle.res, seed=args.seed)
        with torch.no_grad():
            acc = np.mean([
                (bundle.hard(torch.as_tensor(x[i:i + 32], device=device)).argmax(-1)
                 == torch.as_tensor(y[i:i + 32], device=device)).float().mean().item()
                for i in range(0, len(x), 32)
            ])
        print(f"top-1 accuracy on the evaluation sample: {acc:.3f}")

        spec = {m: METHODS[m] for m in method_names if m in METHODS}
        explainers = build_explainers(bundle, spec)
        metrics = build_metrics(args.budget)
        wanted = args.metrics or list(metrics)

        for name, expl in explainers.items():
            out_path = RAW / f"{model_name}__{name}__{tag}.json"
            if out_path.exists() and not args.force:
                print(f"  [skip] {name}")
                continue

            rec = {"model": model_name, "method": name, "n": args.n,
                   "budget": args.budget, "seed": args.seed,
                   "kwargs": {k: (None if v == float('inf') else v)
                              for k, v in METHODS[name].items()},
                   "top1": float(acc), "scores": {}, "timing": {}}
            try:
                t0 = time.time()
                a = attribute_all(expl, x, y, device, args.batch_size)
                rec["timing"]["attribute"] = time.time() - t0
                rec["attr_l2"] = float(np.linalg.norm(a.reshape(len(a), -1), axis=1).mean())
                print(f"  {name:28s} attr {rec['timing']['attribute']:6.1f}s", end="", flush=True)
            except Exception:
                rec["error"] = traceback.format_exc(limit=3)
                out_path.write_text(json.dumps(rec, indent=1))
                print(f"  {name:28s} ATTRIBUTION FAILED")
                continue

            qfn = expl.as_quantus_fn()
            for mname in wanted:
                t0 = time.time()
                try:
                    s = run_metric(metrics[mname], bundle.hard, x, y, a,
                                   str(device), args.metric_batch_size, qfn)
                    rec["scores"][mname] = [None if not np.isfinite(v) else float(v) for v in s]
                except Exception:
                    rec["scores"][mname] = None
                    rec.setdefault("metric_errors", {})[mname] = traceback.format_exc(limit=2)
                rec["timing"][mname] = time.time() - t0
                v = rec["scores"].get(mname)
                mu = np.nanmean([q for q in v if q is not None]) if v else float("nan")
                print(f" | {mname[:9]} {mu:9.3g} ({rec['timing'][mname]:.0f}s)", end="", flush=True)
            print()
            out_path.write_text(json.dumps(rec, indent=1))
            del a
            torch.cuda.empty_cache()

        del bundle, explainers
        torch.cuda.empty_cache()

    collect(tag)


def collect(tag: str):
    """Aggregate the raw per-cell JSONs of one ``tag`` into a tidy CSV."""
    rows = []
    for f in sorted(RAW.glob(f"*__{tag}.json")):
        rec = json.loads(f.read_text())
        for metric, vals in rec.get("scores", {}).items():
            if not vals:
                rows.append(dict(model=rec["model"], method=rec["method"],
                                 metric=metric, mean=np.nan, std=np.nan, n=0))
                continue
            v = np.array([q for q in vals if q is not None], dtype=float)
            v = v[np.isfinite(v)]
            rows.append(dict(model=rec["model"], method=rec["method"], metric=metric,
                             mean=v.mean() if len(v) else np.nan,
                             std=v.std() if len(v) else np.nan,
                             median=np.median(v) if len(v) else np.nan, n=len(v)))
    if not rows:
        print("no results to collect")
        return
    df = pd.DataFrame(rows)
    out = ROOT / "results" / f"summary_{tag.replace('__', '_')}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    for model in df.model.unique():
        sub = df[df.model == model]
        piv_m = sub.pivot_table(index="method", columns="metric", values="mean")
        piv_s = sub.pivot_table(index="method", columns="metric", values="std")
        cols = [c for c in METRIC_DIRECTION if c in piv_m.columns]
        print(f"\n--- {model} ---")
        table = pd.DataFrame(index=piv_m.index)
        for c in cols:
            table[c] = [f"{m:.3g} ± {s:.3g}" if np.isfinite(m) else "-"
                        for m, s in zip(piv_m[c], piv_s[c])]
        print(table.to_string())


if __name__ == "__main__":
    main()

