"""Explanation-quality metrics.

Two layers:

1. :func:`build_metrics` -- the paper's Quantus suite (Sec. 4.2 + F), with the
   deviations from Quantus defaults that Sec. F.1 records, and an explicit
   compute budget so the sweep is tractable.
2. :func:`selector_infidelity` -- a small native implementation of Infidelity
   (Eq. 30) that takes the *score the explanation actually claims to
   represent*.  Needed because the Margin and Distribution selectors do not
   explain the raw target logit; see ``docs/DESIGN_NOTES.md``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch

__all__ = ["METRIC_DIRECTION", "build_metrics", "selector_infidelity"]

#: +1 = higher is better, -1 = lower is better.
METRIC_DIRECTION = {
    "Infidelity": -1,
    "FaithfulnessCorrelation": +1,
    "MonotonicityCorrelation": +1,
    "FaithfulnessEstimate": +1,
    "MaxSensitivity": -1,
    "RandomLogit": -1,
}

#: Compute budgets.  ``paper`` follows Sec. F.1 as closely as the text allows;
#: ``fast`` shrinks the per-sample sample counts only (never the definitions),
#: for iterating on a laptop GPU.
BUDGETS = {
    "paper": dict(
        # one patch size rather than a ladder: Infidelity is ~50% of the whole
        # suite's wall-clock and the extra scales did not change the ordering.
        infid_samples=10, infid_patches=(56,),
        faithcorr_runs=50, faithcorr_subset=3136,
        mono_samples=10, mono_step=6272,
        faithest_step=6272,
        maxsens_samples=10,
    ),
    "fast": dict(
        infid_samples=4, infid_patches=(56,),
        faithcorr_runs=20, faithcorr_subset=3136,
        mono_samples=5, mono_step=12544,
        faithest_step=12544,
        maxsens_samples=5,
    ),
}


def build_metrics(budget: str = "paper", num_classes: int = 1000, seed: int = 42) -> Dict:
    """Instantiate the six metrics of Tables 2-4.

    Deviations from Quantus defaults, all of them stated in Sec. F.1 of the paper:

    * ``Infidelity``: ``perturb_baseline="uniform"`` (Quantus default is black),
      "to better reflect the semantics of the metric".
    * ``MaxSensitivity``: ``lower_bound=0.02`` instead of ``0.2``.
    * ``RandomLogit``: ``similarity_func = correlation_pearson`` instead of SSIM,
      "as it aligns better with our directional interpretation of pullbacks".
    * Faithfulness metrics: enlarged subset / step size "for faster evaluation".
      Quantus' own defaults (``features_in_step=1``) would need 150 528 forward
      passes per image; the exact enlarged values are not printed in the paper,
      so ours are recorded here and in the results manifest.

    ``return_aggregate=False`` everywhere: the paper reports mean +/- std over
    per-sample scores, not a single pooled number.
    """
    import quantus
    from quantus.functions.similarity_func import correlation_pearson

    b = BUDGETS[budget]
    return {
        "Infidelity": quantus.Infidelity(
            loss_func="mse",
            n_perturb_samples=b["infid_samples"],
            perturb_patch_sizes=list(b["infid_patches"]),
            perturb_baseline="uniform",
            normalise=False, abs=False,
            return_aggregate=False, disable_warnings=True,
        ),
        "FaithfulnessCorrelation": quantus.FaithfulnessCorrelation(
            nr_runs=b["faithcorr_runs"],
            subset_size=b["faithcorr_subset"],
            perturb_baseline="black",
            return_aggregate=False, disable_warnings=True,
        ),
        "MonotonicityCorrelation": quantus.MonotonicityCorrelation(
            nr_samples=b["mono_samples"],
            features_in_step=b["mono_step"],
            perturb_baseline="uniform",
            return_aggregate=False, disable_warnings=True,
        ),
        "FaithfulnessEstimate": quantus.FaithfulnessEstimate(
            features_in_step=b["faithest_step"],
            perturb_baseline="black",
            return_aggregate=False, disable_warnings=True,
        ),
        "MaxSensitivity": quantus.MaxSensitivity(
            nr_samples=b["maxsens_samples"],
            lower_bound=0.02,
            return_aggregate=False, disable_warnings=True,
        ),
        "RandomLogit": quantus.RandomLogit(
            similarity_func=correlation_pearson,
            num_classes=num_classes,
            seed=seed,
            return_aggregate=False, disable_warnings=True,
        ),
    }


def run_metric(metric, model, x, y, a, device: str, batch_size: int,
               explain_func: Optional[Callable] = None) -> np.ndarray:
    """Call a Quantus metric and always return a float array of length ``len(x)``."""
    out = metric(
        model=model, x_batch=x, y_batch=y, a_batch=a,
        channel_first=True, softmax=False, device=device,
        batch_size=batch_size, explain_func=explain_func,
    )
    return np.asarray(out, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------
# selector-consistent Infidelity
# --------------------------------------------------------------------------


@torch.no_grad()
def selector_infidelity(
    model: torch.nn.Module,
    x: torch.Tensor,
    attributions: torch.Tensor,
    selector: torch.Tensor,
    n_samples: int = 16,
    patch: int = 56,
    n_patches: int = 4,
    baseline: str = "uniform",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Infidelity (Eq. 30) against ``s_u(x) = <u, f(x)>`` for an arbitrary,
    *frozen* selector ``u``.

    ``Infid(x) = E_I[ (<e(x), I> - (s_u(x) - s_u(x - I)))^2 ]`` with ``I`` the
    baseline-replacement of ``n_patches`` random square patches, exactly the
    perturbation family Quantus uses.

    With ``u = e_c`` this *is* ``quantus.Infidelity`` on the target logit, so
    the number is directly comparable across selectors; the point of the
    function is that a margin explanation can be scored against the margin it
    ascends rather than against a logit it never claimed to represent.

    Returns one score per sample.
    """
    B, C, H, W = x.shape
    dev = x.device
    s0 = (model(x) * selector).sum(-1)
    errs = x.new_zeros(B, n_samples)

    for s in range(n_samples):
        mask = torch.zeros(B, 1, H, W, device=dev)
        for _ in range(n_patches):
            top = torch.randint(0, H - patch + 1, (B,), device=dev, generator=generator)
            left = torch.randint(0, W - patch + 1, (B,), device=dev, generator=generator)
            rows = torch.arange(H, device=dev).view(1, H, 1)
            cols = torch.arange(W, device=dev).view(1, 1, W)
            m = ((rows >= top.view(B, 1, 1)) & (rows < (top + patch).view(B, 1, 1)) &
                 (cols >= left.view(B, 1, 1)) & (cols < (left + patch).view(B, 1, 1)))
            mask = torch.maximum(mask, m.unsqueeze(1).float())

        if baseline == "uniform":
            base = torch.rand_like(x, generator=generator) * 2 - 1
        elif baseline == "black":
            base = torch.zeros_like(x)
        else:
            raise ValueError(baseline)

        # I = x - x_perturbed, i.e. the removed signal (Eq. 29 / 30).
        pert = (1 - mask) * x + mask * base
        i_vec = x - pert
        dot = (attributions * i_vec).flatten(1).sum(1)
        ds = s0 - (model(pert) * selector).sum(-1)
        errs[:, s] = (dot - ds) ** 2

    return errs.mean(1)

