"""Entropy-Flow -- pull back the entropy of the soft adjoint's own gates.

The other two extensions swap the output selector u. This one leaves the
selector alone and changes the *scalar* being pulled back: instead of
s_u(x) = <u, f(x)> we take S_tau(x), the total indecision of the network's
internal routing.

The soft adjoint already builds a distribution at every switch -- Phi(z/tau)
for ReLU/GELU, sigmoid for SiLU, softmax(window/tau) for MaxPool, the
attention map for PVT -- and then only ever uses its mean. The spread is free
information about how close each decision was, and nothing in the method reads
it. h2(g) = 0 means the gate is decided and hard/soft agree; h2(g) = 1 bit
means z ~ 0 and the hard adjoint is coin-flipping. Same story for the pooling
window, which is also the one place tau -> 0 provably doesn't recover the
gradient (see DESIGN_NOTES §1).

    S_tau(x) = sum_l lambda_l <w_l, H[gate_l(x)]>

w_l uniform            -> EntropyFlow, class-agnostic: which pixels make the
                          network undecided.
w_l ~ |dz_c / da_l|    -> ClassEntropyFlow: where the evidence for class c
                          passes through gates that haven't made up their mind.

dS/dx goes through the unmodified soft-adjoint backward pass -- the entropies
are elementwise functions of the pre-activations, so they differentiate
directly and everything below them propagates with the paper's operators.
Forward pass untouched, so assert_forward_unchanged still holds.

Not the Fisher-Rao selector, despite both sounding like "uncertainty":
G^+ ~ diag(1/p) *amplifies* written-off classes rather than measuring
anything (DESIGN_NOTES §6.2). Two inputs with identical output distributions
can have completely different gate entropies.

Cost: EntropyFlow is one fwd + one bwd like SoftPullback, plus an elementwise
pass over the gates. ClassEntropyFlow is one fwd and two bwd on the same graph.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .explainers import Explainer, _l2norm
from .soft_adjoint import (
    SoftGELU,
    SoftMaxPool2d,
    SoftReLU,
    SoftSiLU,
    _SoftActivation,
    normal_cdf,
)

__all__ = [
    "binary_gate_entropy",
    "categorical_gate_entropy",
    "GateEntropyProbe",
    "gate_entropy_report",
    "EntropyFlow",
    "ClassEntropyFlow",
    "EntropyMaskedPullback",
    "make_entropy_gated_model",
    "ENTROPY_METHODS",
]

LOG2 = math.log(2.0)

# Phi(z/tau) saturates to exactly 0.0/1.0 in float32 past |z/tau| ~ 5.7, and
# 0*log0 is a NaN that poisons the whole backward pass. Clamping sends the
# saturated gates to zero entropy and zero gradient, which is the right limit.
GATE_EPS = 1e-6

# MaxPool pads with -inf -> softmax gives exact zeros -> same NaN. Swap in a
# big finite negative instead; the weights are still 0 to float32.
NEG_FILL = -1e4


# --------------------------------------------------------------------------
# entropy of a single gate
# --------------------------------------------------------------------------


def binary_gate_entropy(g: torch.Tensor, normalise: bool = True) -> torch.Tensor:
    """Entropy of the Bernoulli gate g, in bits (or nats if normalise=False).

    g is the soft adjoint's backward multiplier. Peaks at g = 1/2, i.e. z = 0,
    where the unit sits on its switching boundary.
    """
    g = g.clamp(GATE_EPS, 1.0 - GATE_EPS)
    h = -(g * g.log() + (1.0 - g) * (1.0 - g).log())
    return h / LOG2 if normalise else h


def categorical_gate_entropy(
    logits: torch.Tensor, dim: int, normalise: bool = True
) -> torch.Tensor:
    """Entropy of softmax(logits, dim), reduced over dim -- the routing gates.

    normalise divides by log(fan-in) so a 9-way pooling gate and a 2-way ReLU
    gate land on the same scale. Without it the layer weights mean nothing.
    """
    logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, NEG_FILL))
    logw = F.log_softmax(logits, dim=dim)
    w = logw.exp()
    h = -(w * logw).sum(dim)
    if normalise:
        h = h / math.log(logits.shape[dim])
    return h


# --------------------------------------------------------------------------
# reading every gate of a soft-adjoint model
# --------------------------------------------------------------------------


class GateEntropyProbe:
    """Forward hooks exposing the entropy of every soft-adjoint gate.

        with GateEntropyProbe(bundle.soft) as probe:
            z = bundle.soft(x)
        S = probe.functional()                      # (B,)
        nu, = torch.autograd.grad(S.sum(), x)

    Captured tensors stay in the autograd graph -- that's the point, S_tau has
    to be differentiable back to the input.

    One forward per clear(). Gates are keyed "<module>#<call index>", NOT by
    module: torchvision's Bottleneck holds one self.relu and calls it three
    times, so keying by name keeps only the last call and drops two thirds of
    ResNet-50's gates (49 -> 17). models.py::_unshare_relus is the same problem
    on the Captum side.

    PVT self-attention is not covered -- timm's patched Attention.forward
    doesn't expose the attention map as a module output. Its GELU gates are.
    n_gates reports what was actually found.
    """

    def __init__(
        self,
        model: nn.Module,
        include: Sequence[str] = ("activation", "maxpool"),
        normalise: bool = True,
    ):
        self.model = model
        self.include = tuple(include)
        self.normalise = normalise
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        #: ``key -> (B, ...)`` per-unit normalised entropy, in forward order.
        self.entropy: Dict[str, torch.Tensor] = {}
        #: ``key -> (B, ...)`` the gate's own input, kept for diagnostics.
        self.gate_input: Dict[str, torch.Tensor] = {}
        #: ``key -> (B, ...)`` the tensor whose gradient measures the pullback
        #: mass arriving at this gate.  Always the gate's **output**, and
        #: always the same shape as :attr:`entropy` -- a ``MaxPool`` gate has
        #: one categorical distribution per output position, not per input
        #: pixel, so reading the mass at the input would neither line up nor
        #: mean the right thing.
        self.mass_site: Dict[str, torch.Tensor] = {}
        self._order: List[str] = []
        self._calls: Dict[str, int] = {}

    def _key(self, name: str) -> str:
        i = self._calls.get(name, 0)
        self._calls[name] = i + 1
        return f"{name}#{i}"

    def _store(self, name, h: torch.Tensor, gin: torch.Tensor, mass: torch.Tensor):
        if h.shape != mass.shape:
            raise RuntimeError(
                f"gate {name}: entropy {tuple(h.shape)} does not match mass site "
                f"{tuple(mass.shape)}"
            )
        key = self._key(name)
        self.entropy[key] = h
        self.gate_input[key] = gin
        self.mass_site[key] = mass
        self._order.append(key)

    # -- hooks ------------------------------------------------------------

    def _hook_activation(self, name: str):
        def _fn(mod, args, out):
            z = args[0]
            zt = z / mod.tau
            g = torch.sigmoid(zt) if mod.kind == "silu" else normal_cdf(zt)
            self._store(name, binary_gate_entropy(g, self.normalise), z, out)

        return _fn

    def _hook_maxpool(self, name: str):
        def _fn(mod, args, out):
            x = args[0]
            k = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
            s = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
            p = mod.padding if isinstance(mod.padding, int) else mod.padding[0]
            B, C = x.shape[0], x.shape[1]
            # exactly the unfold of _SoftMaxPool2dFn.backward, including the
            # -inf padding, so the entropy is the entropy of the weights the
            # backward pass actually uses.
            xp = F.pad(x, (p, p, p, p), value=float("-inf")) if p else x
            patches = F.unfold(xp, kernel_size=k, stride=s).view(B, C, k * k, -1)
            h = categorical_gate_entropy(patches / mod.tau, dim=2, normalise=self.normalise)
            self._store(name, h.view_as(out), x, out)

        return _fn

    def __enter__(self) -> "GateEntropyProbe":
        self.clear()
        for name, mod in self.model.named_modules():
            if "activation" in self.include and isinstance(
                mod, (SoftReLU, SoftGELU, SoftSiLU)
            ):
                self._handles.append(mod.register_forward_hook(self._hook_activation(name)))
            elif "maxpool" in self.include and isinstance(mod, SoftMaxPool2d):
                self._handles.append(mod.register_forward_hook(self._hook_maxpool(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def clear(self):
        self.entropy.clear()
        self.gate_input.clear()
        self.mass_site.clear()
        self._order.clear()
        self._calls.clear()

    # -- the functional ---------------------------------------------------

    @property
    def names(self) -> List[str]:
        """Gate call-site keys in forward order."""
        return list(self._order)

    @property
    def n_gates(self) -> int:
        return len(self.entropy)

    def layer_weights(self, mode: str = "uniform") -> Dict[str, float]:
        """lambda_l, summing to 1.

        uniform: every gate counts the same. Default, and the only scale-free
        option -- a raw sum over units is dominated by the stem (conv1's ReLU
        has 802816 units against layer4's 100352), which would turn this into a
        report on the first two layers.

        depth:   linear in depth, for asking whether the late gates are the
                 undecided ones.
        units:   proportional to unit count. Kept so the notebook can show the
                 stem domination instead of just asserting it.
        """
        names = self.names
        n = len(names)
        if mode == "uniform":
            return {k: 1.0 / n for k in names}
        if mode == "depth":
            raw = [i + 1 for i in range(n)]
            tot = float(sum(raw))
            return {k: r / tot for k, r in zip(names, raw)}
        if mode == "units":
            raw = [float(self.entropy[k][0].numel()) for k in names]
            tot = float(sum(raw))
            return {k: r / tot for k, r in zip(names, raw)}
        raise ValueError(mode)

    def functional(
        self,
        unit_weights: Optional[Dict[str, torch.Tensor]] = None,
        layer_mode: str = "uniform",
    ) -> torch.Tensor:
        """S_tau(x), one scalar per sample, differentiable back to x.

        unit_weights[key] is detached, non-negative and normalised per sample.
        None = uniform = EntropyFlow; pullback_gate_mass builds the
        class-conditional version.
        """
        lam = self.layer_weights(layer_mode)
        total = None
        for name in self.names:
            h = self.entropy[name]
            if unit_weights is not None and name in unit_weights:
                w = unit_weights[name]
                term = (w * h).flatten(1).sum(1)
            else:
                term = h.flatten(1).mean(1)
            term = lam[name] * term
            total = term if total is None else total + term
        if total is None:
            raise RuntimeError(
                "no gates were captured -- the model has no soft-adjoint gates, "
                "or the forward pass did not run inside the probe's context"
            )
        return total


@torch.no_grad()
def _normalise_mass(m: torch.Tensor) -> torch.Tensor:
    a = m.abs().flatten(1)
    return (a / a.sum(1, keepdim=True).clamp_min(1e-12)).view_as(m)


def pullback_gate_mass(
    probe: GateEntropyProbe,
    score: torch.Tensor,
    retain_graph: bool = True,
) -> Dict[str, torch.Tensor]:
    """Per-gate share of the pullback mass of score (usually z_c).

    One backward through the soft adjoints, read at each gate's output,
    normalised to 1 per sample. Detached on purpose: it's a routing weight, and
    Sec. 2.3 says don't differentiate through routing. Same rule selectors.py
    applies to u, one level down.
    """
    names = probe.names
    inputs = [probe.mass_site[n] for n in names]
    grads = torch.autograd.grad(
        score.sum(), inputs, retain_graph=retain_graph, allow_unused=True
    )
    out = {}
    for n, g in zip(names, grads):
        if g is not None:
            out[n] = _normalise_mass(g.detach())
    return out


def gate_entropy_report(
    model: nn.Module, x: torch.Tensor, include=("activation", "maxpool")
):
    """(names, mean_entropy, n_units) per gate. Diagnosis only, no gradients.

    Worth running before anything else: if every gate were decided, S_tau would
    be flat and the whole method returns zero.
    """
    with torch.no_grad():
        with GateEntropyProbe(model, include=include) as probe:
            model(x)
        names = probe.names
        means = [probe.entropy[n].flatten(1).mean(1) for n in names]
        units = [probe.entropy[n][0].numel() for n in names]
    return names, torch.stack(means, dim=1) if means else torch.empty(len(x), 0), units


# --------------------------------------------------------------------------
# the explainers
# --------------------------------------------------------------------------


class EntropyFlow(Explainer):
    """Class-agnostic Entropy-Flow: nu_H(x) = dS_tau/dx.

    Signed -- positive where raising the pixel makes the network less decided.

    Target-independent by construction, which is the whole claim: this is meant
    to be a channel the class-conditional maps don't contain. So RandomLogit
    will come out ~1.0 and that is not a failure. Use ClassEntropyFlow if you
    want something the metric suite can score.
    """

    name = "EntropyFlow"

    def _config(self):
        return dict(
            include=self.kw.get("include", ("activation", "maxpool")),
            layer_mode=self.kw.get("layer_mode", "uniform"),
        )

    def attribute(self, x, y):
        cfg = self._config()
        xt = x.detach().requires_grad_(True)
        with GateEntropyProbe(self.b.soft, include=cfg["include"]) as probe:
            self.b.soft(xt)
            S = probe.functional(layer_mode=cfg["layer_mode"])
            (nu,) = torch.autograd.grad(S.sum(), xt)
        return nu.detach()


class ClassEntropyFlow(EntropyFlow):
    """Class-conditional Entropy-Flow the uncertainty channel of a prediction.

    Gate entropies weighted by how much of class c's pullback mass goes through
    each gate. One forward, two backwards on the same graph: first pass gets
    the routing weights (detached), second differentiates the weighted entropy.
    """

    name = "ClassEntropyFlow"

    def attribute(self, x, y):
        from .selectors import onehot_selector

        cfg = self._config()
        xt = x.detach().requires_grad_(True)
        with GateEntropyProbe(self.b.soft, include=cfg["include"]) as probe:
            z = self.b.soft(xt)
            u = onehot_selector(z, y).detach()
            score = (z * u).sum(-1)
            w = pullback_gate_mass(probe, score, retain_graph=True)
            S = probe.functional(unit_weights=w, layer_mode=cfg["layer_mode"])
            (nu,) = torch.autograd.grad(S.sum(), xt)
        return nu.detach()


# --------------------------------------------------------------------------
# the probe variant: an entropy-reweighted backward pass
# --------------------------------------------------------------------------


class _EntropyGatedActivation(torch.autograd.Function):
    """Forward = the real activation; backward = v * g * m(h2(g)).

    mode="decided"   -> m = 1 - h2, keep mass routed through decided gates
    mode="undecided" -> m = h2,     keep mass routed through undecided ones

    rescale divides m by its per-sample mean. Not optional on a real network,
    see make_entropy_gated_model.
    """

    @staticmethod
    def forward(ctx, z, tau, kind, mode, rescale):
        ctx.save_for_backward(z)
        ctx.tau, ctx.kind, ctx.mode, ctx.rescale = tau, kind, mode, rescale
        if kind == "relu":
            return torch.relu(z)
        if kind == "gelu":
            return F.gelu(z)
        if kind == "silu":
            return F.silu(z)
        raise ValueError(kind)

    @staticmethod
    def backward(ctx, grad_out):
        (z,) = ctx.saved_tensors
        zt = z / ctx.tau
        gate = torch.sigmoid(zt) if ctx.kind == "silu" else normal_cdf(zt)
        h = binary_gate_entropy(gate, normalise=True)
        m = h if ctx.mode == "undecided" else (1.0 - h)
        if ctx.rescale:
            mean = m.flatten(1).mean(1).clamp_min(1e-12)
            m = m / mean.view(-1, *([1] * (m.dim() - 1)))
        return grad_out * gate * m, None, None, None, None


class _EntropyGatedAct(nn.Module):
    def __init__(self, src: _SoftActivation, mode: str, rescale: bool = True):
        super().__init__()
        self.tau, self.kind, self.mode = src.tau, src.kind, mode
        self.rescale = bool(rescale)

    def forward(self, z):
        return _EntropyGatedActivation.apply(
            z, self.tau, self.kind, self.mode, self.rescale
        )

    def extra_repr(self):
        return f"tau={self.tau}, mode={self.mode}, rescale={self.rescale}"


def make_entropy_gated_model(
    soft: nn.Module, mode: str = "undecided", rescale: bool = True
) -> nn.Module:
    """Copy of a soft-adjoint model with the activation gates entropy-masked.

    Builds a separate model rather than patching soft_adjoint.py, which is the
    verbatim Appendix A and shouldn't be perturbable from here.

    Why rescale defaults to True: the backward multipliers compose as a PRODUCT
    down the network, so a mask with mean m < 1 at every gate attenuates by
    m**n_gates. ResNet-50 has 50 gates at mean entropy ~0.27, i.e. 0.27**50 ~
    1e-29 -- exactly zero in float32. The first version of this returned an
    all-zero map on ResNet-50 while passing on the two-gate test net. Rescaling
    each mask to unit mean makes it a redistribution between units instead of a
    global attenuation, and the product stays O(1) at any depth.

    Neither setting gives an exact decomposition. rescale=False:
    nu = nu_decided + nu_undecided holds for one gate in series, fails for two.
    rescale=True: fails immediately, the two modes get different constants.
    It's a probe, not an identity.

    The attenuation figure is arguably the more interesting output here -- it
    says the pullback on ResNet-50 is a 50-term product, so any per-gate
    reweighting gets amplified exponentially by depth.
    """
    import copy

    if mode not in ("decided", "undecided"):
        raise ValueError(mode)
    out = copy.deepcopy(soft)

    def _walk(m):
        for name, child in m.named_children():
            if isinstance(child, _SoftActivation):
                setattr(m, name, _EntropyGatedAct(child, mode, rescale))
            else:
                _walk(child)

    _walk(out)
    return out


class EntropyMaskedPullback(Explainer):
    """The ordinary pullback with activation gates reweighted by h2 or 1 - h2.

    Narrower question than ClassEntropyFlow: not "what makes the network
    undecided" but "which part of *this class's* explanation went through
    undecided gates". See make_entropy_gated_model for the rescaling and for
    why the two modes don't sum back to the original.
    """

    name = "EntropyMaskedPullback"

    def __init__(self, bundle, **kw):
        super().__init__(bundle, **kw)
        self._masked = None

    def _model(self):
        if self._masked is None:
            self._masked = make_entropy_gated_model(
                self.b.soft,
                mode=self.kw.get("mode", "undecided"),
                rescale=self.kw.get("rescale", True),
            )
        return self._masked

    def attribute(self, x, y):
        from .explainers import _pullback
        from .selectors import onehot_selector

        return _pullback(self._model(), x, lambda z: onehot_selector(z, y))


class EntropyAscent(EntropyFlow):
    """Algorithm 1 driven by nu_H -- ascent on uncertainty.

    Mostly here because it's the one member of the family with an unambiguous
    behavioural check: K steps along +nu_H must raise S_tau and -nu_H must
    lower it. 08_entropy_flow.py --which behaviour runs it; first thing that
    would falsify the method.
    """

    name = "EntropyAscent"

    def attribute(self, x, y):
        alpha = self.kw.get("alpha", 20.0)
        steps = self.kw.get("steps", 5)
        cfg = self._config()
        x0 = x.detach()
        xt = x0.clone()
        for _ in range(steps):
            xr = xt.detach().requires_grad_(True)
            with GateEntropyProbe(self.b.soft, include=cfg["include"]) as probe:
                self.b.soft(xr)
                S = probe.functional(layer_mode=cfg["layer_mode"])
                (nu,) = torch.autograd.grad(S.sum(), xr)
            xt = self.b.project(xt + alpha * _l2norm(nu.detach()))
        return (xt - x0).detach()


ENTROPY_METHODS = {
    c.name: c
    for c in [EntropyFlow, ClassEntropyFlow, EntropyMaskedPullback, EntropyAscent]
}
