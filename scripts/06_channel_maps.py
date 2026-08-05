"""Channel-resolved Distribution Pullback maps, target fluctuation, and
Margin Ascent counterfactuals.

Everything here keeps the three input channels apart instead of summing them,
which is what every other figure in this repo (and the paper) does.  The
reasoning and the numbers are in ``docs/DESIGN_NOTES.md`` §6; the narrated
version is ``notebooks/Distribution_Pullback.ipynb``.

    python scripts/06_channel_maps.py --model resnet50 --n-images 4

Writes ``figures/fig10..fig13_*_<model>.png`` and
``results/channel_stats_<model>.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from spx import load_bundle, load_imagenette
from spx.data import IMAGENETTE_NAMES, IMAGENETTE_WNIDS
from spx.explainers import ALL_METHODS, ascent_path
from spx.selectors import margin_score
from spx.viz import (
    ascent_trajectory_plot, channel_class_conditional_grid, channel_grid,
    channel_stats, counterfactual_target_grid, target_fluctuation_panel,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"

#: the ten Imagenette classes, as ImageNet-1k indices
CLASSES = [c for _, c, _ in IMAGENETTE_WNIDS]

MAP_METHODS = {"DistributionPullback": {}, "SoftPullback": {}}
ASCENT_KW = {"alpha": 20.0, "steps": 5}


def _attr(expl, x, y, chunk=2):
    out = []
    for i in range(0, len(x), chunk):
        out.append(expl.attribute(x[i:i + chunk], y[i:i + chunk]).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
# 1 + 2: channel-resolved maps
# --------------------------------------------------------------------------


def channel_figures(b, x, y, x_np, labels, model_name):
    attrs = {name: _attr(ALL_METHODS[name](b, **kw), x, y)
             for name, kw in MAP_METHODS.items()}
    channel_grid(
        x_np, attrs, titles=labels,
        path=FIG / f"fig10_channel_maps_{model_name}.png",
        suptitle=f"Distribution Pullback without the channel sum -- {model_name}\n"
                 f"(signed; R/G/B of one image share a 99.5th-pct scale, the sum has its own; "
                 f"'cancel' = 1 - |sum_c a_c| / sum_c |a_c|, the share the summed map destroys)",
    )

    rows = []
    for name, a in attrs.items():
        for i in range(len(a)):
            rows.append({"model": model_name, "method": name, "image": i,
                         "label": labels[i], **channel_stats(a[i])})
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / f"channel_stats_{model_name}.csv", index=False)
    print(df.groupby("method")[["cancellation", "share_R", "share_G", "share_B",
                                "cos_RG", "cos_RB", "cos_GB"]].mean().round(3))
    return attrs


# --------------------------------------------------------------------------
# 3: the same maps for every other target -- how much does the map fluctuate?
# --------------------------------------------------------------------------


def _class_conditional(b, img):
    """method -> ``(10, C, H, W)``: one attribution per Imagenette target."""
    out = {}
    for name, kw in MAP_METHODS.items():
        e = ALL_METHODS[name](b, **kw)
        out[name] = np.concatenate([
            e.attribute(img, torch.tensor([c], device=b.device)).cpu().numpy()
            for c in CLASSES])
    return out


def fluctuation_table(b, x, y, labels, model_name):
    """How far the map moves when only the *target* moves, over every image.

    ``cos_offdiag``  mean cosine between the maps of two different targets --
                     a target-invariant explainer sits near 1.
    ``energy_ratio`` ``||nu_true|| / mean_{j != true} ||nu_j||`` -- how much
                     more the explanation says for the class the image
                     actually is than for the nine it is not.
    ``share_sd``     largest s.d. of a channel's ``l1`` share across targets:
                     does the *channel balance* fluctuate, or only the layout.
    """
    rows = []
    for i in range(len(x)):
        cc = _class_conditional(b, x[i:i + 1])
        t = CLASSES.index(int(y[i]))
        for name, a in cc.items():
            flat = a.reshape(len(CLASSES), -1).astype(np.float64)
            nrm = np.linalg.norm(flat, axis=1)
            cos = (flat / nrm[:, None]) @ (flat / nrm[:, None]).T
            off = cos[~np.eye(len(CLASSES), dtype=bool)]
            share = np.abs(a).sum(axis=(2, 3))
            share = share / share.sum(1, keepdims=True)
            rows.append({
                "model": model_name, "method": name, "image": i, "label": labels[i],
                "cos_offdiag": float(off.mean()),
                "cos_true_vs_rivals": float(np.delete(cos[t], t).mean()),
                "energy_ratio": float(nrm[t] / np.delete(nrm, t).mean()),
                "share_sd": float(share.std(0).max()),
            })
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES / f"target_fluctuation_{model_name}.csv", index=False)
    print(df.groupby("method")[["cos_offdiag", "cos_true_vs_rivals",
                                "energy_ratio", "share_sd"]].mean().round(4))
    return df


def fluctuation_figures(b, x, y, x_np, model_name, img_idx=0):
    img = x[img_idx:img_idx + 1]
    cc = _class_conditional(b, img)
    labels = [IMAGENETTE_NAMES[c] for c in CLASSES]
    true_name = IMAGENETTE_NAMES[int(y[img_idx])]

    channel_class_conditional_grid(
        x_np[img_idx], cc, labels, true_label=true_name,
        path=FIG / f"fig11_channel_class_conditional_{model_name}.png",
        suptitle=f"Same image, one target per column, channels kept apart -- {model_name}\n"
                 f"(* = the true class; one shared scale per method across all targets, "
                 f"so a weaker target really looks weaker)",
    )
    target_fluctuation_panel(
        cc, labels, true_idx=CLASSES.index(int(y[img_idx])),
        path=FIG / f"fig12_target_fluctuation_{model_name}.png",
        suptitle=f"How the explanation of one image fluctuates with the target -- {model_name}",
    )
    return cc


# --------------------------------------------------------------------------
# 4: Margin Ascent as a counterfactual tool
# --------------------------------------------------------------------------


def _probe(b, xs, tgt, true_c, k=8):
    """``(p_target, p_true, margin)`` at every iterate of an ascent path."""
    p_t, p_o, mrg = [], [], []
    with torch.no_grad():
        for xt in xs:
            z = b.hard(xt)
            p = torch.softmax(z, -1)
            p_t.append(float(p[0, tgt]))
            p_o.append(float(p[0, true_c]))
            mrg.append(float(margin_score(z, torch.tensor([tgt], device=b.device), k=k)))
    return p_t, p_o, mrg


def counterfactual_figures(b, x, y, x_np, model_name, img_idx=0, n_targets=3):
    img = x[img_idx:img_idx + 1]
    true_c = int(y[img_idx])
    with torch.no_grad():
        p0 = torch.softmax(b.hard(img), -1)[0]

    # the rivals the network itself is entertaining, among the Imagenette ten
    rivals = sorted((c for c in CLASSES if c != true_c),
                    key=lambda c: -float(p0[c]))[:n_targets]
    targets = [true_c] + rivals

    rows, traj_rows = [], {}
    for c in targets:
        tgt = torch.tensor([c], device=b.device)
        e = ALL_METHODS["MarginAscent"](b, k=8, **ASCENT_KW)
        xs = ascent_path(e, img, tgt)
        delta = (xs[-1] - xs[0])[0].cpu().numpy()
        with torch.no_grad():
            p1 = torch.softmax(b.hard(xs[-1]), -1)[0]
        cap = (f"p({IMAGENETTE_NAMES[c][:12]}) {float(p0[c]):.3f} -> {float(p1[c]):.3f}")
        if c != true_c:
            cap += (f"\np({IMAGENETTE_NAMES[true_c][:12]}) {float(p0[true_c]):.3f} -> "
                    f"{float(p1[true_c]):.3f}")
        rows.append({
            "label": f"{IMAGENETTE_NAMES[c]}\n"
                     f"({'true class' if c == true_c else 'counterfactual'})",
            "delta": delta,
            "caption": cap,
        })

    counterfactual_target_grid(
        x_np[img_idx], rows,
        path=FIG / f"fig13_margin_counterfactuals_{model_name}.png",
        suptitle=f"Margin Ascent as a counterfactual generator -- {model_name} "
                 f"(alpha=20, K=5, k=8)\n"
                 f"'what would have to change for this {IMAGENETTE_NAMES[true_c]} "
                 f"to be read as each target'",
    )

    # one counterfactual target, three ascent selectors, step by step
    c = rivals[0]
    tgt = torch.tensor([c], device=b.device)
    for name, kw in [("MarginAscent", dict(k=8, **ASCENT_KW)),
                     ("PullbackAscent", dict(ASCENT_KW)),
                     ("DistributionAscent", dict(ASCENT_KW))]:
        xs = ascent_path(ALL_METHODS[name](b, **kw), img, tgt)
        p_t, p_o, mrg = _probe(b, xs, c, true_c)
        traj_rows[name] = {"p_target": p_t, "p_true": p_o, "margin": mrg}

    ascent_trajectory_plot(
        traj_rows,
        path=FIG / f"fig14_counterfactual_trajectory_{model_name}.png",
        suptitle=f"{model_name}: driving one {IMAGENETTE_NAMES[true_c]} to "
                 f"'{IMAGENETTE_NAMES[c]}' -- one image; the cross-image trade-off "
                 f"(margin suppresses harder, arrives less often) is in design notes §6.3",
    )
    for name, v in traj_rows.items():
        print(f"  {name:<20s} p_target {v['p_target'][0]:.4f} -> {v['p_target'][-1]:.4f}   "
              f"p_true {v['p_true'][0]:.4f} -> {v['p_true'][-1]:.4f}")
    return traj_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--image", type=int, default=0, help="index used for figs 11-14")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    b = load_bundle(args.model, dev)
    x_np, y_np, _ = load_imagenette(n=args.n_images, res=b.res, seed=args.seed,
                                    correct_only=True, model=b.hard, device=dev)
    x = torch.as_tensor(x_np, device=dev)
    y = torch.as_tensor(y_np, device=dev)
    labels = [IMAGENETTE_NAMES[int(v)] for v in y_np]

    print(f"[channel maps] {args.model}, {args.n_images} images")
    channel_figures(b, x, y, x_np, labels, args.model)
    print("[target fluctuation]")
    fluctuation_figures(b, x, y, x_np, args.model, img_idx=args.image)
    fluctuation_table(b, x, y, labels, args.model)
    fluctuation_table(b, x, y, labels, args.model)
    print("[margin counterfactuals]")
    counterfactual_figures(b, x, y, x_np, args.model, img_idx=args.image)
    print(f"wrote figures to {FIG}")


if __name__ == "__main__":
    main()
