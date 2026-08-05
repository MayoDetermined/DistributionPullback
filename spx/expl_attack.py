"""Attack the explanation, leave the prediction alone.

    find delta, ||delta||_inf <= eps, argmax f(x+delta) = argmax f(x),
    maximising the dissimilarity between e(x+delta) and e(x)

The Ghorbani / Dombrowski control (AAAI 2019 / NeurIPS 2019). Sharpest
faithfulness test available without ground-truth masks: a method that survives
can't be reporting an artefact of the local geometry, because there's a nearby
input the network treats identically and the map is asked whether it agrees.

Worth noting why the gradient version is even possible here. Dombrowski et al.
can't attack a ReLU net's gradient explanation directly -- ReLU'' = 0 almost
everywhere, so there's no usable second derivative, and they substitute a
softplus surrogate with a smoothing parameter beta, i.e. attack a model that
isn't the model. The soft adjoint kills that problem for free: its backward
multiplier Phi(z/tau) has derivative phi(z/tau)/tau, smooth and nonzero on a
band around the boundary, so nu(x) = W~(x;tau)^T u is differentiable in x as
written. tau plays beta's role while already being part of the method. That's
the paper's property, not ours.

Two backends:
  mode="grad"  PGD through the explanation. Needs it twice differentiable --
               the single-step pullback family qualifies, the ascent family and
               the Captum baselines don't.
  mode="spsa"  two-point gradient estimate, treats e(.) as a black box. Works
               for everything, which is what makes the control comparable
               across all 19. Weaker per step, so give it more.

attack_report always runs a random perturbation of matched norm alongside.
Maps move under any perturbation -- that's what MaxSensitivity measures -- so
the raw post-attack similarity means very little on its own. Quote the ratio
damage(attack)/damage(random). Nothing here returns an attack figure without
its control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .sharpness import spatial_map

__all__ = [
    "topk_intersection",
    "rank_correlation",
    "map_cosine",
    "mass_centre_shift",
    "dissimilarity",
    "AttackResult",
    "attack_explanation",
    "random_control",
    "attack_report",
    "DIFFERENTIABLE_METHODS",
]

#: Methods whose attribution is twice differentiable in ``x``, hence attackable
#: with ``mode="grad"``.  The single-step pullback family: one backward pass
#: composed of smooth operators.  The ascent family is excluded because its
#: output is the endpoint of a projected loop (the projection is non-smooth and
#: the loop multiplies memory by ``K``); the Captum baselines are excluded
#: because they run on the *hard* model, where the ReLU second derivative is
#: zero -- the problem Dombrowski's beta-smoothing exists to work around.
DIFFERENTIABLE_METHODS = frozenset({
    "SoftPullback",
    "DoublePullback",
    "DistributionPullback",
    "DistributionDoublePullback",
    "MarginPullback",
    "EntropyFlow",
    "ClassEntropyFlow",
    "EntropyMaskedPullback",
})


# --------------------------------------------------------------------------
# how different are two maps
# --------------------------------------------------------------------------


def topk_intersection(a0: torch.Tensor, a1: torch.Tensor, k: int = 1000) -> torch.Tensor:
    """Ghorbani's measure: the share of the original top-``k`` pixels still in
    the perturbed map's top-``k``.  1 = untouched, 0 = disjoint.

    ``k = 1000`` is ~2 % of a 224x224 image, the value the original paper uses.
    """
    s0, s1 = spatial_map(a0).flatten(1), spatial_map(a1).flatten(1)
    i0 = s0.topk(k, dim=1).indices
    i1 = s1.topk(k, dim=1).indices
    out = torch.zeros(len(s0), device=s0.device, dtype=s0.dtype)
    for b in range(len(s0)):
        out[b] = len(np.intersect1d(i0[b].cpu().numpy(), i1[b].cpu().numpy())) / k
    return out


def rank_correlation(a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    """Spearman correlation of the two maps' pixel rankings, per sample."""
    from scipy.stats import spearmanr

    s0, s1 = spatial_map(a0).flatten(1).cpu().numpy(), spatial_map(a1).flatten(1).cpu().numpy()
    return torch.tensor(
        [spearmanr(s0[b], s1[b]).statistic for b in range(len(s0))], dtype=torch.float32
    )


def map_cosine(a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    """Cosine of the signed maps.

    Catches sign flips, which the rank and top-k measures are blind to. §7.4:
    destroying Gradient's sign structure (= Saliency) is what kills it, so a
    sign-blind attack measure misses the most damaging failure.
    """
    return F.cosine_similarity(a0.flatten(1), a1.flatten(1), dim=1)


def mass_centre_shift(a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    """Euclidean shift of the attribution's centre of mass, in pixels."""
    def _c(a):
        s = spatial_map(a)
        H, W = s.shape[-2:]
        ii = torch.arange(H, device=s.device, dtype=s.dtype).view(1, H, 1)
        jj = torch.arange(W, device=s.device, dtype=s.dtype).view(1, 1, W)
        return torch.stack([(s * ii).flatten(1).sum(1), (s * jj).flatten(1).sum(1)], -1)

    return (_c(a0) - _c(a1)).norm(dim=-1)


def dissimilarity(a0: torch.Tensor, a1: torch.Tensor, k: int = 1000) -> Dict[str, torch.Tensor]:
    """All four, per sample."""
    return {
        "topk_intersection": topk_intersection(a0, a1, k),
        "rank_correlation": rank_correlation(a0, a1),
        "cosine": map_cosine(a0, a1),
        "centre_shift": mass_centre_shift(a0, a1),
    }


# --------------------------------------------------------------------------
# differentiable attribution (mode="grad")
# --------------------------------------------------------------------------


def differentiable_attribution(explainer, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Like explainers._pullback but with create_graph=True, so the map can be
    differentiated again.

    Raises for anything not in DIFFERENTIABLE_METHODS rather than quietly
    handing back a detached tensor, which would turn the attack into a no-op.
    """
    from .entropy_flow import ClassEntropyFlow, EntropyFlow, GateEntropyProbe
    from .explainers import (
        DistributionDoublePullback,
        DistributionPullback,
        DoublePullback,
        MarginPullback,
        SoftPullback,
    )

    name = getattr(explainer, "name", type(explainer).__name__)
    if name not in DIFFERENTIABLE_METHODS:
        raise NotImplementedError(
            f"{name} is not twice differentiable in x -- use mode='spsa'. "
            f"Differentiable: {sorted(DIFFERENTIABLE_METHODS)}"
        )

    b = explainer.b

    if isinstance(explainer, (EntropyFlow, ClassEntropyFlow)):
        cfg = explainer._config()
        with GateEntropyProbe(b.soft, include=cfg["include"]) as probe:
            z = b.soft(x)
            w = None
            if isinstance(explainer, ClassEntropyFlow):
                from .selectors import onehot_selector

                u = onehot_selector(z, y).detach()
                from .entropy_flow import pullback_gate_mass

                w = pullback_gate_mass(probe, (z * u).sum(-1), retain_graph=True)
            S = probe.functional(unit_weights=w, layer_mode=cfg["layer_mode"])
            (nu,) = torch.autograd.grad(S.sum(), x, create_graph=True)
        return nu

    model = b.soft
    if name == "EntropyMaskedPullback":
        model = explainer._model()

    def _sel(z):
        if isinstance(explainer, (DistributionPullback, MarginPullback)):
            return explainer._sel(z, y).detach()
        from .selectors import onehot_selector

        return onehot_selector(z, y).detach()

    def _one(xin):
        z = model(xin)
        (nu,) = torch.autograd.grad(z, xin, grad_outputs=_sel(z), create_graph=True)
        return nu

    nu = _one(x)
    if isinstance(explainer, (DoublePullback, DistributionDoublePullback)):
        alpha = explainer.kw.get("alpha", 2.0)
        x2 = x + alpha * nu
        nu = _one(x2)
    return nu


# --------------------------------------------------------------------------
# the attacks
# --------------------------------------------------------------------------


@dataclass
class AttackResult:
    """Per-sample outcome of one attack (or of its random control)."""

    delta: torch.Tensor
    a0: torch.Tensor
    a1: torch.Tensor
    pred_preserved: torch.Tensor          #: (B,) bool
    dp_target: torch.Tensor               #: (B,) change in p(target)
    linf: torch.Tensor                    #: (B,) achieved ||delta||_inf
    l2: torch.Tensor                      #: (B,) achieved ||delta||_2
    scores: Dict[str, torch.Tensor] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        out = {
            "pred_preserved": float(self.pred_preserved.float().mean()),
            "dp_target": float(self.dp_target.mean()),
            "linf": float(self.linf.mean()),
            "l2": float(self.l2.mean()),
        }
        out.update({k: float(v.mean()) for k, v in self.scores.items()})
        return out


def _finish(explainer, bundle, x, y, delta, a0, k) -> AttackResult:
    xa = bundle.project(x + delta)
    delta = (xa - x).detach()
    with torch.no_grad():
        p0 = bundle.hard(x).softmax(-1)
        p1 = bundle.hard(xa).softmax(-1)
    idx = torch.arange(len(x), device=x.device)
    a1 = explainer.attribute(xa, y).detach()
    return AttackResult(
        delta=delta,
        a0=a0,
        a1=a1,
        pred_preserved=(p0.argmax(-1) == p1.argmax(-1)),
        dp_target=(p1[idx, y] - p0[idx, y]).detach(),
        linf=delta.flatten(1).abs().max(1).values,
        l2=delta.flatten(1).norm(dim=1),
        scores=dissimilarity(a0, a1, k=k),
    )


def attack_explanation(
    explainer,
    bundle,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mode: str = "grad",
    objective: str = "topk",
    eps: float = 0.05,
    steps: int = 40,
    lr: Optional[float] = None,
    k: int = 1000,
    keep_margin: float = 0.0,
    lam_pred: float = 10.0,
    spsa_samples: int = 8,
    spsa_delta: float = 0.01,
    seed: int = 0,
) -> AttackResult:
    """Run a prediction-preserving attack on ``explainer``'s map.

    ``eps`` is an ``l_inf`` budget **in normalised (model input) units**; the
    ImageNet stds are ~0.225, so ``eps = 0.05`` is roughly 1.1/255 in raw pixel
    terms -- deliberately small, since the point is that the network cannot
    tell the difference.

    ``objective``
        ``"topk"``    push mass off the original top-``k`` pixels (Ghorbani).
        ``"cosine"``  minimise the signed cosine to the original map, which is
                      the harsher target because it can be won by a sign flip
                      alone.

    ``keep_margin`` / ``lam_pred`` hold the prediction: a hinge penalty on
    ``z_runner_up - z_target + keep_margin``.  Samples where the prediction
    flips anyway are reported in ``pred_preserved`` and must be **dropped**
    before quoting a number -- an attack that changed the answer has not
    attacked the explanation.
    """
    # Start from an admissible point.  ``_finish`` reports the *achieved*
    # ``delta = Pi_C(x + delta) - x``, so an inadmissible ``x`` would have its
    # own projection charged to the attack budget and the l_inf bound would be
    # violated by something the attack never did.
    x = bundle.project(x.detach())
    a0 = explainer.attribute(x, y).detach()
    lr = lr if lr is not None else 2.5 * eps / steps
    g = torch.Generator(device="cpu").manual_seed(seed)

    s0 = spatial_map(a0).flatten(1)
    top_idx = s0.topk(k, dim=1).indices
    idx = torch.arange(len(x), device=x.device)

    def _pred_penalty(z):
        zt = z[idx, y]
        zo = z.clone()
        zo[idx, y] = -float("inf")
        return F.relu(zo.max(-1).values - zt + keep_margin)

    def _loss_from_map(a, z):
        s = spatial_map(a).flatten(1)
        if objective == "topk":
            obj = s.gather(1, top_idx).sum(1)
        elif objective == "cosine":
            obj = map_cosine(a, a0)
        else:
            raise ValueError(objective)
        return obj + lam_pred * _pred_penalty(z)

    delta = torch.zeros_like(x)

    if mode == "grad":
        for _ in range(steps):
            d = delta.detach().requires_grad_(True)
            xa = x + d
            a = differentiable_attribution(explainer, xa, y)
            z = bundle.soft(xa)
            loss = _loss_from_map(a, z).sum()
            (gr,) = torch.autograd.grad(loss, d)
            delta = (delta - lr * gr.sign()).clamp(-eps, eps).detach()

    elif mode == "spsa":
        for _ in range(steps):
            est = torch.zeros_like(delta)
            for _ in range(spsa_samples):
                v = torch.randint(0, 2, x.shape, generator=g).to(x) * 2 - 1
                out = []
                for sgn in (+1, -1):
                    xa = bundle.project(x + delta + sgn * spsa_delta * v)
                    with torch.no_grad():
                        z = bundle.hard(xa)
                    a = explainer.attribute(xa, y).detach()
                    out.append(_loss_from_map(a, z))
                diff = (out[0] - out[1]) / (2 * spsa_delta)
                est = est + diff.view(-1, *([1] * (x.dim() - 1))) * v
            est = est / spsa_samples
            delta = (delta - lr * est.sign()).clamp(-eps, eps).detach()

    else:
        raise ValueError(mode)

    return _finish(explainer, bundle, x, y, delta, a0, k)


def random_control(
    explainer,
    bundle,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 0.05,
    k: int = 1000,
    seed: int = 0,
) -> AttackResult:
    """Uniform l_inf-eps perturbation: same budget, no search.

    Maps move under any perturbation. The question is whether an adversary
    beats noise, and this is the denominator for that.
    """
    g = torch.Generator(device="cpu").manual_seed(seed + 9999)
    x = bundle.project(x.detach())
    delta = (torch.rand(x.shape, generator=g).to(x) * 2 - 1) * eps
    a0 = explainer.attribute(x, y).detach()
    return _finish(explainer, bundle, x, y, delta, a0, k)


def attack_report(
    explainer,
    bundle,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    correct_only: bool = True,
    **kw,
):
    """``(attack, control, table)`` -- never the attack alone.

    ``table`` is a one-row ``pandas.DataFrame`` holding, for each dissimilarity
    measure, the attacked value, the random-control value at matched budget,
    and the ratio that is the actual claim.  ``n_used`` records how many
    samples survived the prediction-preservation filter, because a small
    ``n_used`` invalidates the row.
    """
    import pandas as pd

    mode = kw.pop("mode", None)
    if mode is None:
        name = getattr(explainer, "name", type(explainer).__name__)
        mode = "grad" if name in DIFFERENTIABLE_METHODS else "spsa"

    eps = kw.get("eps", 0.05)
    k = kw.get("k", 1000)
    seed = kw.get("seed", 0)
    atk = attack_explanation(explainer, bundle, x, y, mode=mode, **kw)
    ctl = random_control(explainer, bundle, x, y, eps=eps, k=k, seed=seed)

    keep = atk.pred_preserved & ctl.pred_preserved if correct_only else torch.ones_like(
        atk.pred_preserved
    )
    row = {"method": getattr(explainer, "name", "?"), "mode": mode, "eps": eps,
           "n": len(x), "n_used": int(keep.sum())}
    for m in atk.scores:
        a = float(atk.scores[m][keep].mean()) if keep.any() else float("nan")
        c = float(ctl.scores[m][keep].mean()) if keep.any() else float("nan")
        row[f"{m}__attacked"] = a
        row[f"{m}__random"] = c
        # for similarity measures the damage is 1 - similarity; centre_shift is
        # already a damage measure.
        da = a if m == "centre_shift" else 1.0 - a
        dc = c if m == "centre_shift" else 1.0 - c
        row[f"{m}__ratio"] = da / dc if abs(dc) > 1e-9 else float("nan")
    return atk, ctl, pd.DataFrame([row])
