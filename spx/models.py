"""Backbones used in the paper's test regime (Sec. 4.1): ResNet-50, VGG11-bn, PVT-v2-b1."""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .explainers import ModelBundle
from .soft_adjoint import assert_forward_unchanged, make_soft_adjoint_model

__all__ = ["MODEL_SPECS", "load_bundle"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# tau defaults.  The paper does not print exact values; Sec. B.4 reports the
# method is stable for tau in [0.3, 1.0] for ReLU-family gates and much wider
# for attention.  These are the values used in the original extension
# notebooks, kept so the reproduction is comparable to them.
MODEL_SPECS = {
    "resnet50": dict(tau_relu=0.6, tau_maxpool=0.3, res=224),
    "vgg11_bn": dict(tau_relu=0.6, tau_maxpool=0.3, res=224),
    "pvt_v2_b1": dict(tau_gelu=0.6, tau_attn=1.0, res=224),
}


def _basicblock_forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    return self.relu2(out + identity)


def _bottleneck_forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.relu2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    return self.relu3(out + identity)


def _unshare_relus(model: nn.Module) -> nn.Module:
    """Give every ReLU *call site* its own module instance, and disable in-place.

    Captum's DeepLift/Deconvolution attach one hook per module and raise if a
    module is invoked more than once per forward.  torchvision's ``BasicBlock``
    and ``Bottleneck`` both call a single ``self.relu`` two or three times, so
    swapping the module in place is not enough -- the ``forward`` has to be
    rewritten to use distinct instances.  Forward values are unchanged.
    """
    import types

    from torchvision.models.resnet import BasicBlock, Bottleneck

    model = copy.deepcopy(model)

    def _walk(m):
        for name, child in m.named_children():
            if isinstance(child, nn.ReLU):
                setattr(m, name, nn.ReLU(inplace=False))
            else:
                _walk(child)

    _walk(model)

    for mod in model.modules():
        if isinstance(mod, Bottleneck):
            for i in (1, 2, 3):
                setattr(mod, f"relu{i}", nn.ReLU(inplace=False))
            mod.forward = types.MethodType(_bottleneck_forward, mod)
        elif isinstance(mod, BasicBlock):
            for i in (1, 2):
                setattr(mod, f"relu{i}", nn.ReLU(inplace=False))
            mod.forward = types.MethodType(_basicblock_forward, mod)
    return model


def load_bundle(name: str, device: torch.device, verify: bool = True, **tau_overrides) -> ModelBundle:
    """Load a pretrained backbone and build its soft-adjoint twin."""
    spec = dict(MODEL_SPECS[name])
    res = spec.pop("res")
    spec.update(tau_overrides)

    gradcam_layer = None
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        hard = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        gradcam_layer_name = "layer4"
    elif name == "vgg11_bn":
        from torchvision.models import VGG11_BN_Weights, vgg11_bn

        hard = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        gradcam_layer_name = "features.27"
    elif name == "pvt_v2_b1":
        import timm

        hard = timm.create_model("pvt_v2_b1", pretrained=True)
        gradcam_layer_name = None  # GuidedGradCam is not defined for PVT (paper Tab. 3)
    else:
        raise KeyError(name)

    hard = hard.to(device).eval()
    for p in hard.parameters():
        p.requires_grad_(False)

    soft = make_soft_adjoint_model(hard, **spec).to(device).eval()
    for p in soft.parameters():
        p.requires_grad_(False)

    if verify:
        assert_forward_unchanged(hard, soft, shape=(2, 3, res, res))

    if gradcam_layer_name is not None:
        gradcam_layer = hard
        for part in gradcam_layer_name.split("."):
            gradcam_layer = gradcam_layer[int(part)] if part.isdigit() else getattr(gradcam_layer, part)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    bundle = ModelBundle(
        name=name, hard=hard, soft=soft, mean=mean, std=std,
        gradcam_layer=gradcam_layer, device=device,
    )
    bundle.res = res
    # Captum's DeepLift / Deconvolution need one module instance per call site.
    safe = _unshare_relus(hard).to(device).eval()
    assert_forward_unchanged(hard, safe, shape=(2, 3, res, res))
    for p in safe.parameters():
        p.requires_grad_(False)
    bundle.captum_safe = safe
    return bundle

