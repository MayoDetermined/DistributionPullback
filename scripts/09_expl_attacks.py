#!/usr/bin/env python
"""Adversarial control: attack the explanation, leave the prediction alone.

    python scripts/09_expl_attacks.py --model resnet50 --n 32 --eps 0.05
    python scripts/09_expl_attacks.py --model resnet50 --n 32 --methods SoftPullback \\
        DistributionPullback MarginPullback ClassEntropyFlow

For every method: a PGD attack on the map (gradient-based where the method is
twice differentiable, SPSA where it is not) **and** a random perturbation of
the same ``l_inf`` budget.  The claim in the papers is the ratio between them;
the raw post-attack similarity on its own is not interpretable, because maps
move under any perturbation -- that is what ``MaxSensitivity`` measures.

Samples whose prediction flips are dropped: an attack that changed the answer
has not attacked the explanation.  ``n_used`` reports how many survived, and a
row with a small ``n_used`` should not be quoted.

Cost.  The gradient path is a double backward through the backbone per step;
budget roughly ``steps`` x 6 x the cost of one attribution.  At the default
``--steps 40`` on CPU that is minutes per method per image, so start with
``--n 8 --steps 20`` and scale up on a GPU.  The SPSA path costs
``steps * 2 * spsa_samples`` attributions and is the slower of the two.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from spx import build_explainers, load_bundle, load_imagenette
from spx.expl_attack import DIFFERENTIABLE_METHODS, attack_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "attacks"

#: Default panel: one representative of each family plus the three extensions.
#: Deliberately not all 19 -- the SPSA path is expensive and the interesting
#: comparison is between the pullback family and the gradient baselines.
DEFAULT_METHODS = {
    "SoftPullback": {},
    "DoublePullback": {"alpha": 2.0},
    "DistributionPullback": {"p_floor": 1e-3, "eps": 1e-3},
    "MarginPullback": {"k": 8, "tau_m": float("inf")},
    "ClassEntropyFlow": {"layer_mode": "uniform"},
    "EntropyFlow": {"layer_mode": "uniform"},
    "Gradient": {},
    "IntegratedGradients": {},
    "GuidedGradCam": {},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--eps", type=float, default=0.05,
                    help="l_inf budget in normalised model-input units")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--objective", default="topk", choices=["topk", "cosine"])
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--spsa-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{a.model}__n{a.n}__eps{a.eps}__{a.objective}"

    spec = ({m: DEFAULT_METHODS.get(m, {}) for m in a.methods}
            if a.methods else dict(DEFAULT_METHODS))
    b = load_bundle(a.model, device)
    x, y, _ = load_imagenette(n=a.n, res=b.res, seed=a.seed)
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    explainers = build_explainers(b, spec)

    print(f"{a.model}  n={a.n}  eps={a.eps}  objective={a.objective}  device={device}")
    print(f"gradient path: {sorted(set(spec) & DIFFERENTIABLE_METHODS)}")
    print(f"SPSA path:     {sorted(set(spec) - DIFFERENTIABLE_METHODS)}\n")

    tables = []
    for name, e in explainers.items():
        out_path = OUT / f"{a.model}__{name}__{tag}.csv"
        if out_path.exists() and not a.force:
            print(f"  [skip] {name}")
            tables.append(pd.read_csv(out_path))
            continue
        t0 = time.time()
        chunks = []
        for i in range(0, len(x), a.batch_size):
            _, _, tab = attack_report(
                e, b, x[i:i + a.batch_size], y[i:i + a.batch_size],
                eps=a.eps, steps=a.steps, objective=a.objective,
                k=a.topk, spsa_samples=a.spsa_samples, seed=a.seed + i,
            )
            chunks.append(tab)
        tab = pd.concat(chunks)
        # weight the per-chunk means by how many samples each chunk kept
        w = tab["n_used"].clip(lower=0)
        agg = {"method": name, "mode": tab["mode"].iloc[0], "eps": a.eps,
               "n": int(tab["n"].sum()), "n_used": int(w.sum())}
        for c in tab.columns:
            if c.endswith(("__attacked", "__random", "__ratio")):
                agg[c] = float((tab[c] * w).sum() / w.sum()) if w.sum() else float("nan")
        row = pd.DataFrame([agg])
        row["seconds"] = time.time() - t0
        row.to_csv(out_path, index=False)
        tables.append(row)
        print(f"  {name:24s} {agg['mode']:5s} "
              f"topk {agg.get('topk_intersection__attacked', float('nan')):.3f} "
              f"(random {agg.get('topk_intersection__random', float('nan')):.3f}, "
              f"ratio {agg.get('topk_intersection__ratio', float('nan')):.2f})  "
              f"n_used {agg['n_used']}/{agg['n']}  {row['seconds'].iloc[0]:.0f}s")

    df = pd.concat(tables, ignore_index=True)
    summary = OUT / f"summary__{tag}.csv"
    df.to_csv(summary, index=False)

    cols = ["method", "mode", "n_used",
            "topk_intersection__attacked", "topk_intersection__random",
            "topk_intersection__ratio", "cosine__attacked", "cosine__random",
            "cosine__ratio"]
    cols = [c for c in cols if c in df.columns]
    print(f"\n{df[cols].round(3).to_string(index=False)}")
    print(f"\nwrote {summary}")
    print("\nRatio > 1 means an adversary beats noise at the same budget; the "
          "larger it is, the more fragile the explanation.")


if __name__ == "__main__":
    main()
