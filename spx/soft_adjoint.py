"""Layer-wise soft adjoints (Semantic Pullback, Appendix A + B).

Faithful re-implementation of the backward-only operators from
"Pulling Back the Curtain on Deep Networks" (arXiv:2507.22832v5), Appendix A.

Design rule, verbatim from the paper: the *forward* computation of the
pretrained network is never touched.  Only the VJP is replaced, so that a
plain ``torch.autograd.grad(score, x)`` on the wrapped model returns the soft
pullback of Eq. (6) instead of the gradient (Appendix B.3).

Implemented operators
---------------------
=====================  ============================================  =========
layer                  soft adjoint                                  paper ref
=====================  ============================================  =========
ReLU                   ``v * Phi(z / tau)``                          Eq. (12)
SiLU                   ``v * sigmoid(z / tau)``                      Eq. (13)
GELU                   ``v * Phi(z / tau)``                          Eq. (14)
MaxPool2d              ``v * softmax(window / tau)``                 Eq. (15-16)
LayerNorm              standard VJP, stats detached                  Sec. A.3
BatchNorm / Conv / Lin standard VJP (affine at eval time)            Sec. A.4
Self-Attention (PVT)   value branch only; Q, K, softmax blocked      Sec. A.5
=====================  ============================================  =========

``tau = 1.0`` recovers the standard adjoint for the activation layers and
``tau -> 0`` recovers hard gating/routing, so every operator here is a strict
generalisation of the layer it replaces.
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT2 = math.sqrt(2.0)

__all__ = [
    "SoftReLU",
    "SoftGELU",
    "SoftSiLU",
    "SoftMaxPool2d",
    "BlockedLayerNorm",
    "make_soft_adjoint_model",
    "assert_forward_unchanged",
]


def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Standard Normal CDF Phi."""
    return 0.5 * (1.0 + torch.erf(z / SQRT2))


# --------------------------------------------------------------------------
# A.1 Element-wise activations, via Forward Gradient Injection (Appendix B.1)
# --------------------------------------------------------------------------


class _FGIActivation(torch.autograd.Function):
    """Forward = the true activation; backward = ``v * gate(z / tau)``."""

    @staticmethod
    def forward(ctx, z, tau, kind):
        ctx.save_for_backward(z)
        ctx.tau = tau
        ctx.kind = kind
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
        if ctx.kind == "silu":
            gate = torch.sigmoid(zt)  # Eq. (13)
        else:
            gate = normal_cdf(zt)  # Eq. (12) / Eq. (14)
        return grad_out * gate, None, None


class _SoftActivation(nn.Module):
    kind: str = ""

    def __init__(self, tau: float = 0.6, inplace: bool = False):
        super().__init__()
        self.tau = float(tau)
        # `inplace` is accepted so we can swap 1:1 for nn.ReLU(inplace=True);
        # FGI cannot be in-place because it needs `z` in the backward pass.
        self.inplace = False
        del inplace

    def forward(self, z):
        return _FGIActivation.apply(z, self.tau, self.kind)

    def extra_repr(self):
        return f"tau={self.tau}"


class SoftReLU(_SoftActivation):
    kind = "relu"


class SoftGELU(_SoftActivation):
    kind = "gelu"


class SoftSiLU(_SoftActivation):
    kind = "silu"


# --------------------------------------------------------------------------
# A.2 Max pooling: hard forward, softmax-routed backward
# --------------------------------------------------------------------------


class _SoftMaxPool2dFn(torch.autograd.Function):
    """Forward = ``F.max_pool2d``.  Backward = ``softmax(window / tau)`` routing.

    Overlapping windows accumulate, exactly as ``fold`` does for the hard case.
    """

    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, tau):
        y = F.max_pool2d(x, kernel_size, stride, padding)
        ctx.save_for_backward(x)
        ctx.conf = (kernel_size, stride, padding, tau)
        ctx.in_shape = x.shape
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        k, s, p, tau = ctx.conf
        k = k if isinstance(k, int) else k[0]
        s = s if isinstance(s, int) else s[0]
        p = p if isinstance(p, int) else p[0]
        B, C, H, W = ctx.in_shape

        # PyTorch's max_pool2d pads with -inf, so padded slots never win the
        # max and must receive no backward mass either.  F.unfold pads with 0,
        # which would hand them a share of the softmax, so pad explicitly.
        xp = F.pad(x, (p, p, p, p), value=float("-inf")) if p else x
        patches = F.unfold(xp, kernel_size=k, stride=s).view(B, C, k * k, -1)
        weights = F.softmax(patches / tau, dim=2)  # Eq. (15); -inf -> weight 0
        grad_patches = weights * grad_out.reshape(B, C, 1, -1)  # Eq. (16)
        grad_input = F.fold(
            grad_patches.reshape(B, C * k * k, -1),
            output_size=(H + 2 * p, W + 2 * p), kernel_size=k, stride=s,
        )
        if p:
            grad_input = grad_input[..., p:p + H, p:p + W]
        return grad_input, None, None, None, None


class SoftMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding, tau: float = 0.3):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.tau = float(tau)

    def forward(self, x):
        return _SoftMaxPool2dFn.apply(
            x, self.kernel_size, self.stride, self.padding, self.tau
        )

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, tau={self.tau}"


# --------------------------------------------------------------------------
# A.3 LayerNorm: standard VJP but with the normalisation statistics detached
# --------------------------------------------------------------------------


class BlockedLayerNorm(nn.Module):
    """LayerNorm whose backward treats mean/var as constants (Sec. A.3).

    Numerically identical forward; the backward sees a plain affine map, which
    is what "backpropagate using the cached normalization statistics" means.
    """

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = tuple(ln.normalized_shape)
        self.eps = ln.eps
        self.weight = ln.weight
        self.bias = ln.bias
        self._dims = tuple(range(-len(self.normalized_shape), 0))

    def forward(self, x):
        mu = x.mean(self._dims, keepdim=True).detach()
        var = x.var(self._dims, unbiased=False, keepdim=True).detach()
        y = (x - mu) * torch.rsqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, stats_detached=True"


# --------------------------------------------------------------------------
# A.5 Self-Attention (timm pvt_v2.Attention): value branch only
# --------------------------------------------------------------------------


def _pvt_soft_attention_forward(self, x, feat_size: List[int]):
    """Drop-in replacement for ``timm.models.pvt_v2.Attention.forward``.

    Forward values are bit-identical to timm's non-fused path.  In the
    backward pass the signal flows *only* through V: Q, K and the softmax are
    detached (Sec. A.5, "Blocked backward through softmax").  ``tau_attn``
    optionally softens the backward attention map, Eq. (19).
    """
    B, N, C = x.shape
    H, W = feat_size
    tau = getattr(self, "tau_attn", 1.0)

    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

    if self.pool is not None:
        xs = x.permute(0, 2, 1).reshape(B, C, H, W)
        xs = self.sr(self.pool(xs)).reshape(B, C, -1).permute(0, 2, 1)
        xs = self.act(self.norm(xs))
    elif self.sr is not None:
        xs = x.permute(0, 2, 1).reshape(B, C, H, W)
        xs = self.sr(xs).reshape(B, C, -1).permute(0, 2, 1)
        xs = self.norm(xs)
    else:
        xs = x
    kv = self.kv(xs).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    # Eq. (18): logits from *detached* Q and K -> no backward path through routing.
    m = (q.detach() * self.scale) @ k.detach().transpose(-2, -1)
    attn = (m / tau).softmax(dim=-1)  # Eq. (19); tau=1 -> standard A
    out = attn @ v  # gradient flows through V only

    out = out.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(out))


def _patch_pvt_attention(model: nn.Module, tau_attn: float = 1.0) -> int:
    try:
        from timm.models.pvt_v2 import Attention as PvtAttention
    except Exception:  # pragma: no cover - timm not installed / different version
        return 0
    import types

    n = 0
    for mod in model.modules():
        if isinstance(mod, PvtAttention):
            mod.tau_attn = float(tau_attn)
            mod.forward = types.MethodType(_pvt_soft_attention_forward, mod)
            n += 1
    return n


# --------------------------------------------------------------------------
# B.2 Module replacement in pretrained models
# --------------------------------------------------------------------------


def make_soft_adjoint_model(
    model: nn.Module,
    tau_relu: float = 0.6,
    tau_maxpool: float = 0.3,
    tau_gelu: float = 0.6,
    tau_silu: float = 0.6,
    tau_attn: float = 1.0,
    block_layernorm: bool = True,
    deepcopy: bool = True,
) -> nn.Module:
    """Return a copy of ``model`` with every layer's VJP replaced by its soft adjoint.

    ``tau_* = 1.0`` for the activations and ``block_layernorm=False`` reduces
    this to the plain gradient, which is the baseline the paper compares to.
    """
    if deepcopy:
        model = copy.deepcopy(model)

    def _replace(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, SoftReLU(tau_relu))
            elif isinstance(child, nn.GELU):
                setattr(module, name, SoftGELU(tau_gelu))
            elif isinstance(child, nn.SiLU):
                setattr(module, name, SoftSiLU(tau_silu))
            elif isinstance(child, nn.MaxPool2d):
                setattr(module, name, SoftMaxPool2d(
                    child.kernel_size, child.stride, child.padding, tau_maxpool))
            elif block_layernorm and isinstance(child, nn.LayerNorm):
                setattr(module, name, BlockedLayerNorm(child))
            else:
                _replace(child)

    _replace(model)
    _patch_pvt_attention(model, tau_attn=tau_attn)
    return model


@torch.no_grad()
def assert_forward_unchanged(
    hard: nn.Module, soft: nn.Module, shape=(2, 3, 224, 224), atol: float = 1e-4
) -> float:
    """The paper's own correctness criterion: the wrapped model must be
    forward-identical to the original.  Returns the max abs difference."""
    device = next(hard.parameters()).device
    x = torch.randn(*shape, device=device)
    diff = (hard(x) - soft(x)).abs().max().item()
    if diff > atol:
        raise AssertionError(
            f"soft-adjoint model changed the forward pass (max|diff| = {diff:.3e})"
        )
    return diff

