"""spx -- Semantic Pullback eXtensions.

Reproduction of "Pulling Back the Curtain on Deep Networks" (arXiv:2507.22832v5)
plus three extensions, evaluated in the paper's own test regime.

  Fisher-Rao Selector   u = G(z)^+ e_c, the inverse output Fisher metric used
                        as a preconditioner. selectors.fisher_rao_selector.
                        The explainer class is still called
                        DistributionPullback so older result files load.
  Margin Ascent         u = e_c - softmax_{R_k}(z/tau_m), which turns out to be
                        the differential of a smooth top-k logit margin.
  Entropy-Flow          not the selector -- the scalar being pulled back.
                        S_tau(x), the entropy of the soft adjoint's own gates.

Separately there's the Fisher-Rao *gauge* in geometry.py: the same metric read
forward, G_x = J^T G J, used to measure lengths and calibrate perturbation
balls. Not an explanation method. Keeping it apart from the selector matters --
they behave oppositely (the selector needs damping and a probability floor, the
gauge needs neither) and lumping them together under "distribution /
distributional" is what made the naming confusing in the first place.

All three reuse the paper's soft-adjoint backward pass verbatim and never touch
the forward pass.
"""

from .data import IMAGENETTE_WNIDS, denormalise, load_imagenette
from .explainers import (
    ALL_METHODS,
    Explainer,
    ModelBundle,
    ascent_path,
    build_explainers,
)
from .geometry import (
    distributional_efficiency,
    fisher_length,
    fisher_rao_distance,
    logit_fisher_length,
)
from .metrics import METRIC_DIRECTION, build_metrics, selector_infidelity
from .models import load_bundle
from .selectors import (
    fisher_rao_selector,
    margin_score,
    margin_selector,
    onehot_selector,
)
from .soft_adjoint import assert_forward_unchanged, make_soft_adjoint_model

# Imported for the side effect of registering the Entropy-Flow explainers in
# ``ALL_METHODS``, so ``build_explainers`` and ``02_run_benchmark.py`` reach
# them by name like any other method.  Done here rather than in
# ``explainers.py`` because ``entropy_flow`` imports *from* ``explainers``.
from .entropy_flow import ENTROPY_METHODS  # noqa: E402

ALL_METHODS.update(ENTROPY_METHODS)

__version__ = "0.2.0"

__all__ = [
    "ALL_METHODS", "ENTROPY_METHODS", "Explainer", "ModelBundle", "ascent_path",
    "build_explainers",
    "IMAGENETTE_WNIDS", "denormalise", "load_imagenette",
    "METRIC_DIRECTION", "build_metrics", "selector_infidelity",
    "fisher_rao_distance", "logit_fisher_length", "fisher_length",
    "distributional_efficiency",
    "load_bundle", "make_soft_adjoint_model", "assert_forward_unchanged",
    "onehot_selector", "fisher_rao_selector", "margin_selector", "margin_score",
]

