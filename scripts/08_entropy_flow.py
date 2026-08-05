#!/usr/bin/env python
"""Entropy-Flow diagnostics: the checks that are specific to the method.

The Quantus table for the Entropy-Flow family comes from the ordinary
benchmark, which reaches them by name::

    python scripts/02_run_benchmark.py --models resnet50 --n 200 \\
        --methods EntropyFlow ClassEntropyFlow EntropyMaskedPullback

This script covers what that table cannot say, because none of the six metrics
is defined against an *internal* scalar:

``--which gates``
    Per-layer gate entropy of the network on a real sample.  The precondition
    for everything else: if the gates were all decided, ``S_tau`` would be flat
    and its pullback zero.  Also the descriptive figure for the paper.
``--which behaviour``
    **The falsification test.**  ``K`` steps along ``+nu_H`` must raise the
    network's own gate entropy and ``-nu_H`` must lower it, and the output
    entropy ``H(p)`` should follow.  If it does not, the direction is not what
    it claims to be and nothing downstream matters.
``--which channel``
    Is the uncertainty channel a *different* channel?  Cosine of Entropy-Flow
    against the class-conditional maps of the whole pullback family, plus the
    same comparison after taking absolute values (a signed cosine of ~0 with a
    magnitude cosine of ~0.3 means "same regions, different sign", which §6.2
    already found for cross-target maps and which would make the
    orthogonality much less interesting).
``--which layers``
    Which layers carry the map: ``layer_mode`` in {uniform, depth, units} and
    the gate-mass profile of ``ClassEntropyFlow``.

    python scripts/08_entropy_flow.py --model resnet50 --n 32 --which all

Cost note.  ``EntropyFlow`` is ~3x ``SoftPullback`` per image (one extra
elementwise pass over every gate plus the retained graph); ``ClassEntropyFlow``
is ~3.5x.  On CPU that is a few seconds per image, so ``--n 32`` is minutes,
not hours -- unlike the Quantus suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from spx import load_bundle, load_imagenette
from spx.entropy_flow import (
    ClassEntropyFlow,
    EntropyFlow,
    EntropyMaskedPullback,
    GateEntropyProbe,
    gate_entropy_report,
    pullback_gate_mass,
)
from spx.explainers import (
    DistributionPullback,
    DoublePullback,
    MarginPullback,
    SoftPullback,
    _l2norm,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "entropy_flow"


def _batched(fn, x, y, bs):
    return torch.cat([fn(x[i:i + bs], y[i:i + bs]) for i in range(0, len(x), bs)])


def _cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


# --------------------------------------------------------------------------


def run_gates(b, x, tag):
    names, means, units = gate_entropy_report(b.soft, x)
    df = pd.DataFrame({
        "gate": names,
        "units": units,
        "mean_entropy_bits": means.mean(0).numpy(),
        "std_over_images": means.std(0).numpy(),
    })
    df["depth_rank"] = np.arange(len(df))
    df.to_csv(OUT / f"gate_entropy__{tag}.csv", index=False)
    print(f"\n{len(df)} gates, {df['units'].sum():,} units")
    print(f"gate entropy (bits): min {df.mean_entropy_bits.min():.3f}  "
          f"median {df.mean_entropy_bits.median():.3f}  "
          f"max {df.mean_entropy_bits.max():.3f}")
    print("\nmost undecided gates:")
    print(df.nlargest(5, "mean_entropy_bits").to_string(index=False))
    print("\nmost decided gates:")
    print(df.nsmallest(5, "mean_entropy_bits").to_string(index=False))
    return df


def run_behaviour(b, x, y, tag, alpha=20.0, steps=5, bs=4):
    """The falsification test."""
    ef = EntropyFlow(b)

    def _S_and_H(inp):
        with GateEntropyProbe(b.soft) as probe:
            with torch.no_grad():
                z = b.soft(inp)
            S = probe.functional()
        p = z.softmax(-1)
        H = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
        return S, H, p

    rows = []
    for sign, label in ((+1.0, "ascend"), (-1.0, "descend")):
        xt = x.clone()
        S0, H0, p0 = _S_and_H(xt)
        idx = torch.arange(len(x))
        for k in range(steps):
            nu = _batched(ef.attribute, xt, y, bs)
            xt = b.project(xt + sign * alpha * _l2norm(nu))
            S, H, p = _S_and_H(xt)
            rows.append({
                "direction": label, "step": k + 1,
                "dS": float((S - S0).mean()),
                "dH_out": float((H - H0).mean()),
                "dp_true": float((p[idx, y] - p0[idx, y]).mean()),
                "frac_S_moved_right": float(
                    ((S - S0) * sign > 0).float().mean()
                ),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"behaviour__{tag}.csv", index=False)
    print("\n=== falsification test: does nu_H move gate entropy the right way? ===")
    print(df.to_string(index=False))
    up = df[(df.direction == "ascend") & (df.step == steps)]
    dn = df[(df.direction == "descend") & (df.step == steps)]
    ok = float(up.dS.iloc[0]) > 0 and float(dn.dS.iloc[0]) < 0
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} "
          f"(dS ascend {float(up.dS.iloc[0]):+.4f}, descend {float(dn.dS.iloc[0]):+.4f})")
    return df


def run_channel(b, x, y, tag, bs=4):
    """Is the uncertainty channel a different channel?"""
    fam = {
        "SoftPullback": SoftPullback(b),
        "DoublePullback": DoublePullback(b, alpha=2.0),
        "DistributionPullback": DistributionPullback(b, p_floor=1e-3, eps=1e-3),
        "MarginPullback": MarginPullback(b, k=8),
        "ClassEntropyFlow": ClassEntropyFlow(b),
        "EntropyMaskedPullback": EntropyMaskedPullback(b, mode="undecided"),
    }
    a_ef = _batched(EntropyFlow(b).attribute, x, y, bs)
    rows = []
    for name, e in fam.items():
        a = _batched(e.attribute, x, y, bs)
        c_signed = _cos(a_ef, a)
        # magnitude maps, mean-centred: "same regions, opposite sign" would show
        # up here and not in the signed cosine -- the distinction §6.2 draws.
        m1 = a_ef.abs().flatten(1)
        m2 = a.abs().flatten(1)
        c_mag = F.cosine_similarity(m1 - m1.mean(1, keepdim=True),
                                    m2 - m2.mean(1, keepdim=True), dim=1)
        rows.append({
            "vs": name,
            "cos_signed_mean": float(c_signed.mean()),
            "cos_signed_std": float(c_signed.std()),
            "cos_signed_absmax": float(c_signed.abs().max()),
            "cos_magnitude_mean": float(c_mag.mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"channel__{tag}.csv", index=False)
    print("\n=== is Entropy-Flow a different channel? ===")
    print(df.round(4).to_string(index=False))
    print("\nRead: signed cosine ~ 0 with magnitude cosine also low = a genuinely")
    print("different region of the image.  Signed ~ 0 with magnitude high = the")
    print("same regions with disagreeing sign, which is much weaker evidence.")
    return df


def run_layers(b, x, y, tag, bs=4):
    modes = ["uniform", "depth", "units"]
    maps = {}
    for m in modes:
        maps[m] = _batched(EntropyFlow(b, layer_mode=m).attribute, x, y, bs)
    rows = []
    for i, a in enumerate(modes):
        for c in modes[i + 1:]:
            rows.append({"a": a, "b": c, "cos": float(_cos(maps[a], maps[c]).mean())})

    # where does the class-c pullback mass actually sit
    xr = x[:1].detach().requires_grad_(True)
    with GateEntropyProbe(b.soft) as probe:
        z = b.soft(xr)
        score = z[0, y[0]]
        w = pullback_gate_mass(probe, score, retain_graph=False)
    mass = pd.DataFrame({
        "gate": list(w),
        "entropy_bits": [float(probe.entropy[k].mean()) for k in w],
        "mass_weighted_entropy": [
            float((w[k] * probe.entropy[k]).sum()) for k in w
        ],
    })
    mass["depth_rank"] = np.arange(len(mass))
    mass.to_csv(OUT / f"gate_mass__{tag}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"layer_modes__{tag}.csv", index=False)
    print("\n=== layer weighting ===")
    print(df.round(4).to_string(index=False))
    print("\nmass-weighted gate entropy, top 8 gates for the true class:")
    print(mass.nlargest(8, "mass_weighted_entropy").round(4).to_string(index=False))
    return df, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--which", nargs="+",
                    default=["all"],
                    choices=["all", "gates", "behaviour", "channel", "layers"])
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    which = {"gates", "behaviour", "channel", "layers"} if "all" in a.which else set(a.which)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{a.model}__n{a.n}"

    b = load_bundle(a.model, device)
    x, y, _ = load_imagenette(n=a.n, res=b.res, seed=a.seed)
    x = torch.as_tensor(x, device=device)
    y = torch.as_tensor(y, device=device)
    print(f"{a.model}: {len(x)} images, device {device}")

    meta = {"model": a.model, "n": a.n, "seed": a.seed,
            "alpha": a.alpha, "steps": a.steps}
    if "gates" in which:
        run_gates(b, x[: min(8, len(x))], tag)
    if "behaviour" in which:
        run_behaviour(b, x, y, tag, alpha=a.alpha, steps=a.steps, bs=a.batch_size)
    if "channel" in which:
        run_channel(b, x, y, tag, bs=a.batch_size)
    if "layers" in which:
        run_layers(b, x, y, tag, bs=a.batch_size)
    (OUT / f"meta__{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
