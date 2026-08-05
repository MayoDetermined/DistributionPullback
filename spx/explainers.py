"""Explanation methods: the paper's Semantic Pullbacks, the two extensions,
and the Captum baselines -- all behind one interface.

Every method here returns a signed attribution map with the same shape as the
input.  No post-processing is applied to the pullback family (no smoothing, no
positive-part clipping), matching Sec. 3.4 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .selectors import fisher_rao_selector, margin_selector, onehot_selector

__all__ = ["ModelBundle", "Explainer", "build_explainers", "ascent_path", "PULLBACK_METHODS"]


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """A pretrained model plus its soft-adjoint twin (forward-identical)."""

    name: str
    hard: nn.Module
    soft: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    gradcam_layer: Optional[nn.Module] = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    res: int = 224
    #: copy of ``hard`` with one module instance per activation -- Captum's
    #: DeepLift/Deconvolution hooks break on torchvision's shared ``self.relu``.
    captum_safe: Optional[nn.Module] = None

    def project(self, x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        """``Pi_C``: clamp to the admissible pixel range in raw image space."""
        raw = (x * self.std + self.mean).clamp(lo, hi)
        return (raw - self.mean) / self.std


def _l2norm(v: torch.Tensor) -> torch.Tensor:
    """Unit-``l2`` per sample (``Norm(.)`` in Eq. 8)."""
    flat = v.flatten(1).norm(dim=1).clamp_min(1e-12)
    return v / flat.view(-1, *([1] * (v.dim() - 1)))


def _pullback(model: nn.Module, x: torch.Tensor, u: Callable) -> torch.Tensor:
    """``nu_u(x) = W~(x; tau)^T u`` -- one backward pass through the soft adjoints.

    ``u`` is a callable ``logits -> selector`` so that selectors depending on
    the current logits stay a single forward/backward pair.  The selector is
    always detached (see ``selectors`` module docstring).
    """
    xt = x.detach().requires_grad_(True)
    z = model(xt)
    sel = u(z).detach()
    (nu,) = torch.autograd.grad(z, xt, grad_outputs=sel)
    return nu.detach()


# --------------------------------------------------------------------------
# the pullback family
# --------------------------------------------------------------------------


class Explainer:
    """Base: subclasses implement :meth:`attribute`."""

    name = "explainer"
    needs_soft = True

    def __init__(self, bundle: ModelBundle, **kw):
        self.b = bundle
        self.kw = kw

    def attribute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # quantus calls explain_func(model, inputs, targets, **kwargs) -> np.ndarray
    def as_quantus_fn(self) -> Callable:
        def _fn(model, inputs, targets, **kwargs):
            x = torch.as_tensor(inputs, dtype=torch.float32, device=self.b.device)
            y = torch.as_tensor(np.asarray(targets).reshape(-1), device=self.b.device).long()
            return self.attribute(x, y).cpu().numpy()

        _fn.__name__ = self.name
        return _fn


class SoftPullback(Explainer):
    """Eq. (6): ``nu~_{e_c}(x)``.  The paper's base method."""

    name = "SoftPullback"

    def attribute(self, x, y):
        return _pullback(self.b.soft, x, lambda z: onehot_selector(z, y))


class DoublePullback(Explainer):
    """Eq. (7): ``nu~(x + alpha * nu~(x))``, ``alpha = 2``."""

    name = "DoublePullback"

    def attribute(self, x, y):
        alpha = self.kw.get("alpha", 2.0)
        nu = _pullback(self.b.soft, x, lambda z: onehot_selector(z, y))
        return _pullback(self.b.soft, x + alpha * nu, lambda z: onehot_selector(z, y))


class _AscentBase(Explainer):
    """Algorithm 1 / Eq. (8)-(9): ``K`` projected, l2-normalised pullback steps;
    the explanation is the accumulated perturbation ``x^(K) - x``."""

    selector: Callable = staticmethod(onehot_selector)

    def _selector(self, z, y):
        return onehot_selector(z, y)

    def attribute(self, x, y):
        alpha = self.kw.get("alpha", 20.0)
        steps = self.kw.get("steps", 5)
        model = self.b.soft if self.needs_soft else self.b.hard
        x0 = x.detach()
        xt = x0.clone()
        for _ in range(steps):
            nu = _pullback(model, xt, lambda z: self._selector(z, y))
            xt = self.b.project(xt + alpha * _l2norm(nu))
        return (xt - x0).detach()


def ascent_path(explainer: "_AscentBase", x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Every iterate of the ascent loop, ``(K + 1, B, C, H, W)`` with ``[0] = x``.

    Same loop as :meth:`_AscentBase.attribute` -- ``attribute`` returns only
    ``x^(K) - x``, which is all a metric needs but hides the path.  Used by the
    counterfactual figures, where what happens *between* the endpoints (the
    original class falling as the target rises) is the thing being shown.
    """
    if not isinstance(explainer, _AscentBase):
        raise TypeError(f"{type(explainer).__name__} is not an ascent method")
    alpha = explainer.kw.get("alpha", 20.0)
    steps = explainer.kw.get("steps", 5)
    model = explainer.b.soft if explainer.needs_soft else explainer.b.hard
    xt = x.detach().clone()
    path = [xt.clone()]
    for _ in range(steps):
        nu = _pullback(model, xt, lambda z: explainer._selector(z, y))
        xt = explainer.b.project(xt + alpha * _l2norm(nu))
        path.append(xt.clone())
    return torch.stack(path)


def ascent_path(explainer: "_AscentBase", x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Every iterate of the ascent loop, ``(K + 1, B, C, H, W)`` with ``[0] = x``.

    Same loop as :meth:`_AscentBase.attribute` -- ``attribute`` returns only
    ``x^(K) - x``, which is all a metric needs but hides the path.  Used by the
    counterfactual figures, where what happens *between* the endpoints (the
    original class falling as the target rises) is the thing being shown.
    """
    if not isinstance(explainer, _AscentBase):
        raise TypeError(f"{type(explainer).__name__} is not an ascent method")
    alpha = explainer.kw.get("alpha", 20.0)
    steps = explainer.kw.get("steps", 5)
    model = explainer.b.soft if explainer.needs_soft else explainer.b.hard
    xt = x.detach().clone()
    path = [xt.clone()]
    for _ in range(steps):
        nu = _pullback(model, xt, lambda z: explainer._selector(z, y))
        xt = explainer.b.project(xt + alpha * _l2norm(nu))
        path.append(xt.clone())
    return torch.stack(path)


class PullbackAscent(_AscentBase):
    """Paper's PA (Sec. 3.4), ``alpha = 20``, ``K = 5``."""

    name = "PullbackAscent"


class GradientAscent(_AscentBase):
    """Same loop on the *unmodified* model: the paper's own naive baseline."""

    name = "GradientAscent"
    needs_soft = False


# --- extension 1: Distribution Pullback -----------------------------------


class DistributionPullback(Explainer):
    """**Distribution Pullback** (a.k.a. Natural / Fisher-Rao Pullback).

    Identical to :class:`SoftPullback` except that the selector is
    ``G(z)^+ e_c`` instead of ``e_c``: the target direction is re-expressed in
    the information geometry of the output distribution before being pulled
    back.  The backward pass itself is untouched.
    """

    name = "DistributionPullback"

    def _sel(self, z, y):
        return fisher_rao_selector(
            z, y,
            p_floor=self.kw.get("p_floor", 1e-3),
            eps=self.kw.get("eps", 1e-3),
        )

    def attribute(self, x, y):
        return _pullback(self.b.soft, x, lambda z: self._sel(z, y))


class DistributionDoublePullback(DistributionPullback):
    """Distribution Pullback with the paper's Double Pullback refinement."""

    name = "DistributionDoublePullback"

    def attribute(self, x, y):
        alpha = self.kw.get("alpha", 2.0)
        nu = _pullback(self.b.soft, x, lambda z: self._sel(z, y))
        return _pullback(self.b.soft, x + alpha * nu, lambda z: self._sel(z, y))


class DistributionAscent(_AscentBase, DistributionPullback):
    """Algorithm 1 driven by the Fisher-Rao selector."""

    name = "DistributionAscent"

    def _selector(self, z, y):
        return self._sel(z, y)


# --- extension 2: Margin Ascent -------------------------------------------


class MarginPullback(Explainer):
    """**Semantic Margin Pullback** -- single step of the Margin family.

    ``u = e_c - softmax_{R_k}(z / tau_m)``.  See :func:`spx.selectors.margin_selector`
    for why the rival temperature is the same kind of object as the paper's
    ReLU/MaxPool temperatures.
    """

    name = "MarginPullback"

    def _sel(self, z, y):
        return margin_selector(
            z, y, k=self.kw.get("k", 8), tau_m=self.kw.get("tau_m", float("inf"))
        )

    def attribute(self, x, y):
        return _pullback(self.b.soft, x, lambda z: self._sel(z, y))


class MarginAscent(_AscentBase, MarginPullback):
    """**Margin Ascent** -- Algorithm 1 driven by the margin selector,
    re-derived from the current iterate at every step (active-set dynamics)."""

    name = "MarginAscent"

    def _selector(self, z, y):
        return self._sel(z, y)


class MarginAscentFrozen(MarginAscent):
    """Ablation: the rival set is computed once at ``x`` and never re-derived."""

    name = "MarginAscentFrozen"

    def attribute(self, x, y):
        with torch.no_grad():
            z0 = self.b.hard(x)
        u0 = self._sel(z0, y).detach()
        alpha, steps = self.kw.get("alpha", 20.0), self.kw.get("steps", 5)
        x0 = x.detach()
        xt = x0.clone()
        for _ in range(steps):
            nu = _pullback(self.b.soft, xt, lambda z: u0)
            xt = self.b.project(xt + alpha * _l2norm(nu))
        return (xt - x0).detach()


class MarginGradientAscent(MarginAscent):
    """Ablation: margin selector, *hard* model (plain gradient backward)."""

    name = "MarginGradientAscent"
    needs_soft = False


PULLBACK_METHODS = {
    c.name: c
    for c in [
        SoftPullback, DoublePullback, PullbackAscent, GradientAscent,
        DistributionPullback, DistributionDoublePullback, DistributionAscent,
        MarginPullback, MarginAscent, MarginAscentFrozen, MarginGradientAscent,
    ]
}


# --------------------------------------------------------------------------
# Captum baselines (Sec. 4.1 "Methods")
# --------------------------------------------------------------------------


class _CaptumExplainer(Explainer):
    needs_soft = False

    #: DeepLift/Deconvolution need unshared activation modules; the rest are
    #: happy with the original network.
    use_captum_safe = False

    @property
    def _model(self):
        if self.use_captum_safe and self.b.captum_safe is not None:
            return self.b.captum_safe
        return self.b.hard

    def attribute(self, x, y):
        import captum.attr as ca

        x = x.detach().requires_grad_(True)
        algo = self._make(ca, self._model)
        with torch.enable_grad():
            a = algo.attribute(x, target=y, **self._call_kwargs(x))
        return a.detach()

    def _make(self, ca, model):
        raise NotImplementedError

    def _call_kwargs(self, x):
        return {}


def _captum(name, factory, call_kwargs=None, safe=False):
    class _C(_CaptumExplainer):
        pass

    _C.name = name
    _C.use_captum_safe = safe
    _C._make = lambda self, ca, model: factory(ca, model, self.b)
    _C._call_kwargs = lambda self, x: (call_kwargs or {})
    _C.__name__ = name
    return _C


Gradient = _captum("Gradient", lambda ca, m, b: ca.Saliency(m), {"abs": False})
Saliency = _captum("Saliency", lambda ca, m, b: ca.Saliency(m), {"abs": True})
InputXGradient = _captum("InputXGradient", lambda ca, m, b: ca.InputXGradient(m))
IntegratedGradients = _captum(
    "IntegratedGradients", lambda ca, m, b: ca.IntegratedGradients(m),
    {"n_steps": 32, "internal_batch_size": 16},
)
DeepLift = _captum("DeepLift", lambda ca, m, b: ca.DeepLift(m), safe=True)
Deconvolution = _captum("Deconvolution", lambda ca, m, b: ca.Deconvolution(m), safe=True)


class GradientShapExplainer(_CaptumExplainer):
    name = "GradientShap"

    def attribute(self, x, y):
        import captum.attr as ca

        x = x.detach().requires_grad_(True)
        baselines = torch.cat([torch.zeros_like(x), torch.ones_like(x)], dim=0)
        with torch.enable_grad():
            a = ca.GradientShap(self.b.hard).attribute(
                x, baselines=baselines, target=y, n_samples=8, stdevs=0.1
            )
        return a.detach()


class GuidedGradCamExplainer(_CaptumExplainer):
    name = "GuidedGradCam"

    def attribute(self, x, y):
        import captum.attr as ca

        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            a = ca.GuidedGradCam(self.b.hard, self.b.gradcam_layer).attribute(x, target=y)
        return a.detach()


CAPTUM_METHODS = {
    c.name: c
    for c in [
        Gradient, Saliency, InputXGradient, IntegratedGradients,
        GradientShapExplainer, DeepLift, Deconvolution, GuidedGradCamExplainer,
    ]
}


ALL_METHODS = {**PULLBACK_METHODS, **CAPTUM_METHODS}


def build_explainers(bundle: ModelBundle, spec: Dict[str, dict]) -> Dict[str, Explainer]:
    """``spec`` maps method name -> constructor kwargs."""
    out = {}
    for name, kw in spec.items():
        if name not in ALL_METHODS:
            raise KeyError(f"unknown method {name!r}; known: {sorted(ALL_METHODS)}")
        if name == "GuidedGradCam" and bundle.gradcam_layer is None:
            continue  # not defined for this backbone (paper omits it for PVT)
        out[name] = ALL_METHODS[name](bundle, **kw)
    return out

