"""Output-side selectors u, fed into the soft-adjoint backward pass.

A Semantic Pullback is nu_u(x) = W~(x; tau)^T u. The paper always uses the
one-hot u = e_c. Both extensions here keep the backward pass byte-for-byte and
change only u:

  fisher_rao_selector -- re-weight e_c by the Fisher-Rao metric of the output
                         distribution p = softmax(z)
  margin_selector     -- replace e_c by a contrastive selector that subtracts
                         the classes currently competing with the target

Both batched, O(C).

The selector is always detached, which is worth explaining since it looks like
a hack. Both depend on x through p = softmax(f(x)), and differentiating through
that would put us straight back in gradient-land. But Sec. 2.3 of the paper
already says a pullback propagates the neuron action through the effective
linear component and does *not* differentiate through the routing that selected
it -- and choosing which classes compete is routing. So detaching u applies the
paper's own rule one level up, the same way it treats a MaxPool switch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "onehot_selector",
    "fisher_rao_selector",
    "margin_selector",
    "margin_score",
    "fisher_rao_matrix",
]


def _as_batched(z: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if z.dim() == 1:
        return z.unsqueeze(0), True
    return z, False


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------


def onehot_selector(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``u = e_c`` -- the paper's own selector (Eq. 1 with ``u = e_c``)."""
    z, squeeze = _as_batched(logits)
    target = torch.as_tensor(target, device=z.device).reshape(-1)
    u = torch.zeros_like(z)
    u[torch.arange(z.shape[0], device=z.device), target] = 1.0
    return u.squeeze(0) if squeeze else u


# --------------------------------------------------------------------------
# Distribution Pullback (Fisher-Rao)
# --------------------------------------------------------------------------


def fisher_rao_matrix(p: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """``G(z) = diag(p) - p p^T``, the Fisher information of ``softmax(z)`` in
    natural (logit) coordinates.  Equals ``Hessian_z logsumexp(z)``; dense, for
    tests only."""
    C = p.shape[-1]
    return torch.diag(p) - torch.outer(p, p) + eps * torch.eye(
        C, device=p.device, dtype=p.dtype
    )


def fisher_rao_selector(
    logits: torch.Tensor,
    target: torch.Tensor,
    p_floor: float = 1e-3,
    eps: float = 1e-3,
    project_gauge: bool = True,
    normalise: bool = True,
) -> torch.Tensor:
    """**Distribution Pullback** selector: ``u = G_eps^+ (e_c - mean(e_c))``.

    Solves ``(diag(p_eff) - p_eff p_eff^T + eps I) u = e_c_centred`` in ``O(C)``
    by damped Sherman-Morrison -- no dense ``C x C`` solve.

    Two stabilisations, both load-bearing (each was found by an explicit
    failure, see ``docs/DESIGN_NOTES.md``):

    ``project_gauge``
        ``G`` is singular with null space ``span{1}`` (softmax is shift
        invariant).  Damping alone assigns that direction eigenvalue ``eps``,
        so the *inverse* amplifies it by ``1 / eps`` -- the exact opposite of
        pseudo-inverse behaviour.  Centring ``u`` first projects it onto the
        row space, and the damped solve then converges to the Moore-Penrose
        solution as ``eps -> 0`` (verified in ``tests/test_selectors.py``).

    ``p_floor``
        On a real 1000-way softmax most classes sit at ``p_i << eps``.  Their
        Fisher curvature is ~0, so the correction amplifies the entire
        background tail.  Flooring ``p`` from below collapses the tail into one
        bounded-curvature block.  Unlike a hard top-k restriction it is
        *continuous* in ``x``: a class drifting past the floor changes the
        selector smoothly instead of flipping set membership.

        The floored vector is **renormalised**.  Clamping alone leaves
        ``sum(p_eff) = 1 + n_floored * p_floor``, which is off the simplex, so
        ``diag(p_eff) - p_eff p_eff^T`` stops being a Fisher matrix and stops
        being PSD.  Concretely the Sherman-Morrison denominator
        ``1 - sum(p^2 / (p + eps))`` turns negative -- at ``C = 1000`` and
        ``p_floor = 1e-3`` it lands around ``-0.3`` -- which silently flips the
        sign of the whole correction and amplifies the background instead of
        suppressing it.  After renormalisation ``sum(p_eff) = 1``, hence
        ``sum(p^2/(p+eps)) < sum(p) = 1`` and the denominator is strictly
        positive by construction.  ``p_eff`` is then the Fisher matrix of a
        genuine floor-smoothed neighbour of ``p``.

    ``normalise`` rescales ``u`` to unit ``l2`` so the pullback magnitude stays
    comparable to the one-hot baseline (the preconditioner otherwise inflates
    it by ``~1 / p_c``); direction is unaffected.
    """
    z, squeeze = _as_batched(logits)
    p = F.softmax(z, dim=-1).detach()
    u = onehot_selector(z, target)

    if project_gauge:
        u = u - u.mean(-1, keepdim=True)

    p_eff = p.clamp(min=p_floor)
    p_eff = p_eff / p_eff.sum(-1, keepdim=True)  # stay on the simplex -- see docstring
    a_inv = 1.0 / (p_eff + eps)
    a_inv_u = a_inv * u
    a_inv_p = a_inv * p_eff
    denom = 1.0 - (p_eff * a_inv_p).sum(-1, keepdim=True)
    num = (p_eff * a_inv_u).sum(-1, keepdim=True)
    w = a_inv_u + a_inv_p * num / denom

    if normalise:
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-12)
    return w.squeeze(0) if squeeze else w


# --------------------------------------------------------------------------
# Margin Ascent
# --------------------------------------------------------------------------


def margin_selector(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 8,
    tau_m: float = float("inf"),
) -> torch.Tensor:
    """Contrastive selector ``u = e_c - softmax_{R_k}(z / tau_m)``.

    ``R_k(x)`` is the set of the ``k - 1`` highest-logit non-target classes.
    The rival weights are the softmax over that support at temperature
    ``tau_m``, so the family interpolates:

    ==============  ==========================================================
    ``tau_m -> 0``  all mass on the single strongest rival (hard ``argmax``)
    ``tau_m = 1``   mass proportional to the rivals' own probabilities
    ``tau_m = inf`` uniform ``-1 / (k - 1)`` over the top-``k-1`` rivals
    ==============  ==========================================================

    The two endpoints are exactly the two hand-designed variants that were
    compared head-to-head in the original notebook ("proportional" and
    "hard uniform"); here they are one operator with a temperature, which is
    the same softening knob the paper applies to ReLU gates and MaxPool
    switches (Sec. A, "Design principle").  Selecting *which* classes compete
    is routing, so ``tau_m`` softens the backward routing rather than adding a
    new heuristic.

    ``u`` sums to zero by construction, so the selector -- unlike ``e_c`` -- is
    invariant to a global shift of the logits, matching the invariance of the
    softmax it contrasts against.
    """
    z, squeeze = _as_batched(logits)
    B, C = z.shape
    k = max(2, min(int(k), C))
    idx = torch.arange(B, device=z.device)
    tgt = torch.as_tensor(target, device=z.device).reshape(-1).expand(B)

    z_masked = z.detach().clone()
    z_masked[idx, tgt] = -float("inf")
    rival_z, rival_idx = torch.topk(z_masked, k - 1, dim=-1)

    if tau_m == float("inf"):
        rival_w = torch.full_like(rival_z, 1.0 / (k - 1))
    else:
        rival_w = F.softmax(rival_z / max(tau_m, 1e-6), dim=-1)

    u = torch.zeros_like(z)
    u[idx, tgt] = 1.0
    u.scatter_add_(-1, rival_idx, -rival_w)
    return u.squeeze(0) if squeeze else u


def margin_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 8,
    tau_m: float = float("inf"),
    rival_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The scalar score whose selector :func:`margin_selector` is.

    ``m_c(x) = z_c - tau_m * LSE_{j in R_k}(z_j / tau_m)``, i.e. a smooth
    top-``k`` logit margin; ``tau_m -> inf`` degenerates to the mean rival
    logit and ``tau_m -> 0`` to ``max_{j in R_k} z_j``.  ``d m / d z`` is
    exactly the selector above, which is what makes Margin Ascent an ascent on
    a *fixed functional* rather than on an ad-hoc direction.

    Pass ``rival_idx`` to freeze the active set (needed when the score has to
    stay a fixed functional across a whole perturbation sweep, e.g. Infidelity).
    """
    z, squeeze = _as_batched(logits)
    B, C = z.shape
    k = max(2, min(int(k), C))
    idx = torch.arange(B, device=z.device)
    tgt = torch.as_tensor(target, device=z.device).reshape(-1).expand(B)

    if rival_idx is None:
        z_masked = z.detach().clone()
        z_masked[idx, tgt] = -float("inf")
        rival_idx = torch.topk(z_masked, k - 1, dim=-1).indices
    rival_z = z.gather(-1, rival_idx)

    if tau_m == float("inf"):
        agg = rival_z.mean(-1)
    else:
        t = max(tau_m, 1e-6)
        agg = t * torch.logsumexp(rival_z / t, dim=-1)
    s = z[idx, tgt] - agg
    return s.squeeze(0) if squeeze else s

