# Semantic Pullback: three extensions

Three short notebooks on top of a reproduction of
[Pulling Back the Curtain on Deep Networks](https://arxiv.org/abs/2507.22832)
(Satkiewicz, Corizzo, Pietroń, arXiv:2507.22832v5).

One idea per notebook: the construction, what breaks if you do it naively, one
figure, the headline number, the catch. +- 16 cells, couple of minutes on CPU.
They're saved with outputs, so you can just read them.

* [`Fisher_Rao_Selector`](notebooks/Fisher_Rao_Selector.ipynb):
  the logits parameterise a distribution, so re-express `e_c` in the Fisher-Rao
  metric before pulling it back
* [`Margin_Ascent`](notebooks/Margin_Ascent.ipynb): explain the
  target *against its current rivals*, with a selector that turns out to be the
  gradient of a smooth top-k margin
* [`Entropy_Flow`](notebooks/Entropy_Flow.ipynb): the soft
  adjoint's gates are distributions; pull back their **entropy** instead of a
  class score

## What actually changes

The paper explains class `c` by pulling `u = e_c` back through a softened
backward pass. The three differ in *what* they change:

| | changes | leaves alone |
|---|---|---|
| Fisher-Rao Selector | the selector, `u = G(z)^+ e_c` | backward pass, forward pass |
| Margin Ascent | the selector, `u = e_c - softmax_{R_k}(z/τ_m)` | backward pass, forward pass |
| Entropy-Flow | the **scalar being pulled back**, `S_τ(x)` instead of `<u, f(x)>` | selector, backward pass, forward pass |

The first two are variants of one move. The third asks a different question and
doesn't touch the selector at all.

None of them modifies the forward pass. `assert_forward_unchanged` enforces
that at load time, which is the paper's own correctness criterion.

### On the naming

Two Fisher-Rao objects had been sharing a name. They're not variants of each
other; they behave oppositely:

| | Fisher-Rao **Selector** | Fisher-Rao **Gauge** |
|---|---|---|
| direction | backward | forward |
| object | `G(z)^+`, the pseudo-inverse | `G_x = J^TGJ`, the metric itself |
| does what | builds a selector `u` | measures lengths, calibrates perturbation balls |
| damping + probability floor | needs both | needs neither |
| explanation method? | yes | no, it's an evaluation instrument |

Inverting a metric needs stabilising; measuring with it doesn't. That's the
whole difference, and it's why they were easy to confuse.

The class in code is still `DistributionPullback` so older result files stay
loadable.

## Running it

```bash
pip install -r requirements.txt
python setup_data.py          # Imagenette-320, ok 342 MB, into ./data
jupyter lab notebooks/
```

ResNet-50 weights come from torchvision on first use.

The notebooks are all CPU-friendly. The expensive parts of the project (the
Quantus suite, the explanation attacks) aren't in them; their numbers are read
from `results/`.

## Layout

```
notebooks/     the three notebooks
spx/
  soft_adjoint.py   layer-wise soft adjoints (paper, Appendix A)
  selectors.py      one-hot, Fisher-Rao, margin
  entropy_flow.py   gate-entropy probe, S_tau, the Entropy-Flow family
  explainers.py     pullback family, ascent loop, Captum baselines
  geometry.py       the Fisher-Rao gauge (metric read forward)
  unify.py          what the six evaluation metrics span
  sharpness.py      concentration measures + Pareto frontier
  expl_attack.py    prediction-preserving attacks on the explanation
  metrics.py        Quantus suite with the paper's Sec. F.1 deviations
  models.py, data.py, viz.py, dossier.py
scripts/       08 Entropy-Flow diagnostics, 09 attacks, 10 metric unification,
               11 sharpness, plus 02/04/06/07
results/       n = 200 tables, ablation sweeps, unification, sharpness
```

## Where things stand

Fisher-Rao Selector and Margin Ascent are benchmarked at n = 200 on ResNet-50
in the paper's metric regime. Entropy-Flow has method, code and tests but !!! **no
Quantus numbers yet**. The explanation attacks are
implemented and unit-tested, not yet run at a sample size worth quoting.

## About this copy

!!! This folder was put together after the drive holding the main repo became
unavailable (i lost usb cable during backpacking), so a few things are worth flagging before you lean on it:

* `results/summary_n200_paper.csv` and `results/ablations/*.csv` were rebuilt. Means and standard
  deviations are the original numbers; the `median` column comes back as `NaN`. Nothing here uses it.
* `results/raw/*.json` (the per-sample scores) aren't here. That's the input to
  `scripts/10_metric_unification.py`, so that script can't be re-run from this
  folder. Its output is included as
  `results/unification/summary__resnet50__n200__paper.md`.
* `tests/` runs 57. The main repo has 76; the missing ones are older tests for
  `soft_adjoint`/`selectors`/`geometry`. Everything covering the three
  extensions is here and passes.
* `scripts/00,01,03,05` aren't here either. `00` is replaced by
  `setup_data.py`; the rest only regenerate `results/`, which is included.
* The full per-method dossiers (P1-P3) stayed in the main repo. These three
  notebooks are the short version.

## Citation

```bibtex
@article{satkiewicz2026pullingback,
  title   = {Pulling Back the Curtain on Deep Networks},
  author  = {Satkiewicz, Maciej and Corizzo, Roberto and Pietro\'{n}, Marcin},
  journal = {arXiv preprint arXiv:2507.22832},
  year    = {2026}
}
```
