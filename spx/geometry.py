"""The Fisher-Rao metric read *forward*, as a metric on input space.

``spx.selectors.fisher_rao_selector`` uses ``G(z)^+`` -- the pseudo-*inverse* of
the output Fisher metric -- to build a selector.  Pulling ``G`` all the way back
to input space and inverting *that*, ``G_x = J^T G J`` with a Riemannian step
``G_x^{-1} nu``, is a rejected variant: it collapses the target probability at
every step size tried (``docs/DESIGN_NOTES.md`` §2).

Using ``G_x`` in the forward direction -- to *measure* lengths and distances --
is a different operation, and a much better behaved one:

* it needs no inverse, hence no damping ``eps`` and no probability floor: the
  two stabilisations the selector cannot do without are simply absent here;
* ``G_x`` is PSD by construction;
* the length of one direction costs one forward-mode JVP, not a CG solve.

It is a *pseudo*-metric on images, not a metric: ``ker J`` is almost everything
(150 528 input dimensions against at most ``C - 1`` output directions), so
``d(x, y) = 0`` for any two images the network maps to the same distribution.
For explanation work that is arguably the point.

Two derived quantities are used as candidate evaluation metrics in
``scripts/07_geometric_metrics.py``:

``distributional_efficiency``
    ``sqrt(v^T G_x v)`` for the unit-``l2`` explanation direction: how much of
    the output distribution a method's map moves per unit of pixel change.
``max_sensitivity`` with ``mode="fr"``
    the paper's own robustness metric with the perturbation ball measured in
    Fisher-Rao arc length instead of pixel ``l2``, so all methods are compared
    at equal *distributional* displacement.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "fisher_rao_distance",
    "logit_fisher_length",
    "fisher_length",
    "distributional_efficiency",
    "calibrate_fisher_arc",
    "max_sensitivity",
]


# --------------------------------------------------------------------------
# distance on the simplex
# --------------------------------------------------------------------------


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact Fisher-Rao geodesic distance between categorical distributions.

    The simplex under the Fisher metric is a radius-2 sphere via ``a = 2 sqrt(p)``,
    so the geodesic distance is the arc between those points and the maximum
    (disjoint support) is ``pi``.

    Written as ``4 arcsin(||sqrt(p) - sqrt(q)||_2 / 2)`` rather than the usual
    ``2 arccos(sum sqrt(p q))``: the two are algebraically identical, but the
    Bhattacharyya form subtracts a number from 1 before taking ``arccos``, and in
    float32 that loses *all* precision below an arc of ~1e-2 -- it returns
    exactly 0.0 for a perturbation that moves the distribution measurably.  The
    ``arcsin`` form is accurate down to 1e-6 in float32 (checked in
    ``tests/test_geometry.py``).
    """
    d = (p.sqrt() - q.sqrt()).norm(dim=-1).clamp(max=2.0)
    return 4.0 * torch.arcsin(d / 2.0)


def logit_fisher_length(dz: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``sqrt(dz^T G(z) dz)`` with ``G = diag(p) - p p^T``: the Fisher length of a
    displacement expressed in *logit* coordinates.

    Equals ``sqrt(2 KL(p || p_{z+dz}))`` to second order, and agrees with
    :func:`fisher_rao_distance` in the small-displacement limit.  Shift
    invariant, as it must be -- adding a constant to every logit costs zero.
    """
    var = (p * dz.pow(2)).sum(-1) - (p * dz).sum(-1).pow(2)
    return var.clamp(min=0.0).sqrt()


# --------------------------------------------------------------------------
# length of an input-space direction
# --------------------------------------------------------------------------


def fisher_length(
    model: torch.nn.Module,
    x: torch.Tensor,
    v: torch.Tensor,
    p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``sqrt(v^T J^T G J v)`` -- the Fisher-Rao length of an input direction.

    One forward-mode JVP gives ``J v``; the quadratic form is then ``O(C)``.
    No transpose, no solve, no ``eps``.

    ``model`` is the *unmodified* network: the question "how far does this
    direction move the output distribution" is about the real Jacobian, not
    about the soft adjoint (which is a modified *backward* operator and has no
    forward-mode counterpart in the paper).
    """
    x = x.detach()
    v = v.detach()
    _, jv = torch.func.jvp(lambda t: model(t), (x,), (v,))
    if p is None:
        with torch.no_grad():
            p = torch.softmax(model(x), dim=-1)
    return logit_fisher_length(jv, p)


def distributional_efficiency(
    model: torch.nn.Module,
    x: torch.Tensor,
    a: torch.Tensor,
    p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """**Candidate metric 2.** Fisher-Rao length of the unit-``l2`` explanation
    direction: distributional movement per unit of pixel movement.

    Scale free (the map is normalised first), so it is defined for the ascent
    family -- whose "explanation" is a perturbation of norm ``~alpha K`` rather
    than a differential -- on the same footing as for the single-step methods.
    Needs no reference class, no baseline and no perturbation sampling: one JVP
    per sample.  Higher = the direction the method points along moves the output
    distribution more.
    """
    flat = a.flatten(1).norm(dim=1).clamp_min(1e-12)
    v = a / flat.view(-1, *([1] * (a.dim() - 1)))
    return fisher_length(model, x, v, p)


# --------------------------------------------------------------------------
# perturbations of a prescribed size
# --------------------------------------------------------------------------


@torch.no_grad()
def calibrate_fisher_arc(
    model: torch.nn.Module,
    x: torch.Tensor,
    d: torch.Tensor,
    arc: float,
    lo: float = 1e-3,
    hi: float = 1e3,
    iters: int = 24,
) -> torch.Tensor:
    """Per-sample scale ``s`` such that ``d_FR(p(x), p(x + s d)) == arc``.

    Bisection, batched, ``iters`` forward passes for the whole batch.  The arc is
    not guaranteed monotone in ``s`` for a deep network, but it is monotone on
    average and bisection on a bracketing interval is what the calibration
    needs: the result is *a* scale reaching the requested displacement, which is
    all the metric asks for.  Returns ``(B,)``.
    """
    B = x.shape[0]
    dev, dt = x.device, x.dtype
    p0 = torch.softmax(model(x), dim=-1)
    s_lo = torch.full((B,), lo, device=dev, dtype=dt)
    s_hi = torch.full((B,), hi, device=dev, dtype=dt)

    def _arc(s):
        xs = x + s.view(B, *([1] * (x.dim() - 1))) * d
        return fisher_rao_distance(p0, torch.softmax(model(xs), dim=-1))

    for _ in range(iters):
        mid = (s_lo + s_hi) / 2
        too_small = _arc(mid) < arc
        s_lo = torch.where(too_small, mid, s_lo)
        s_hi = torch.where(too_small, s_hi, mid)
    return (s_lo + s_hi) / 2


@torch.no_grad()
def sample_directions(
    x: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Unit-``l2`` isotropic random directions, one per sample."""
    d = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return d / d.flatten(1).norm(dim=1).clamp_min(1e-12).view(
        x.shape[0], *([1] * (x.dim() - 1))
    )


@torch.no_grad()
def build_perturbations(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int,
    mode: str,
    size: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(perturbed_inputs, achieved_arcs)`` of shapes ``(S, B, ...)``, ``(S, B)``.

    ``mode="l2"``  every perturbation has pixel norm ``||delta||_2 = size``;
                   the induced distributional displacement is whatever it is.
    ``mode="fr"``  every perturbation is rescaled so the Fisher-Rao arc equals
                   ``size``; the pixel norm is whatever it needs to be.

    The perturbations depend only on the model and the inputs, never on the
    explanation method, so one set is reused across all methods -- which also
    makes the comparison exactly paired.
    """
    with torch.no_grad():
        p0 = torch.softmax(model(x), dim=-1)
    xs, arcs = [], []
    for _ in range(n_samples):
        d = sample_directions(x, generator)
        if mode == "l2":
            s = torch.full((x.shape[0],), float(size), device=x.device, dtype=x.dtype)
        elif mode == "fr":
            s = calibrate_fisher_arc(model, x, d, float(size))
        else:
            raise ValueError(mode)
        xp = x + s.view(x.shape[0], *([1] * (x.dim() - 1))) * d
        with torch.no_grad():
            arcs.append(fisher_rao_distance(p0, torch.softmax(model(xp), dim=-1)))
        xs.append(xp)
    return torch.stack(xs), torch.stack(arcs)


def max_sensitivity(
    explainer,
    x: torch.Tensor,
    y: torch.Tensor,
    perturbed: torch.Tensor,
    a0: Optional[torch.Tensor] = None,
    batch_size: int = 8,
) -> torch.Tensor:
    """**Candidate metric 1** (with ``mode="fr"`` perturbations).

    ``max_s || e(x + delta_s) - e(x) ||_F / || e(x) ||_F`` over the supplied
    perturbations -- the paper's own MaxSensitivity, with *which* perturbations
    count left to the caller.  Feed it :func:`build_perturbations` output with
    ``mode="l2"`` to reproduce the usual pixel-ball version, or ``mode="fr"`` to
    compare methods at equal distributional displacement.

    Lower is better, as in the paper.
    """
    def _attr(xb, yb):
        out = []
        for i in range(0, len(xb), batch_size):
            out.append(explainer.attribute(xb[i:i + batch_size], yb[i:i + batch_size]))
        return torch.cat(out)

    if a0 is None:
        a0 = _attr(x, y)
    n0 = a0.flatten(1).norm(dim=1).clamp_min(1e-12)
    worst = torch.zeros(len(x), device=x.device, dtype=x.dtype)
    for xp in perturbed:
        ap = _attr(xp, y)
        rel = (ap - a0).flatten(1).norm(dim=1) / n0
        worst = torch.maximum(worst, rel)
    return worst
