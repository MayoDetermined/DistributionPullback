## The table

n = 200, ResNet-50, Imagenette validation, seed 0. Backbone top-1 on that
sample 0.848.

Column names are shortened; the full Quantus classes are `Infidelity`,
`FaithfulnessCorrelation`, `MonotonicityCorrelation`, `FaithfulnessEstimate`,
`MaxSensitivity`, `RandomLogit`. Lower is better for Infid, MaxSens and
RandLogit, higher for the other three. Bold marks the best value in a column.

| Method | Infid | FaithCorr | MonoCorr | FaithEst | MaxSens | RandLogit |
|---|---|---|---|---|---|---|
| **Paper** | | | | | | |
| SoftPullback | **8.86** | 0.072 | 0.208 | 0.515 | 0.401 | -0.018 |
| DoublePullback | 9.36 | 0.053 | 0.207 | 0.362 | 0.616 | 0.006 |
| PullbackAscent | 8.6e5 | 0.045 | 0.234 | 0.629 | 1.054 | 0.011 |
| **Fisher-Rao Selector** | | | | | | |
| DistributionPullback | 9.56 | **0.073** | 0.208 | 0.281 | 0.411 | **-0.059** |
| DistributionDoublePullback | 10.66 | 0.031 | 0.207 | 0.339 | 0.640 | 0.008 |
| DistributionAscent | 9.0e5 | 0.053 | 0.224 | 0.445 | 1.070 | 0.006 |
| **Margin Ascent** | | | | | | |
| MarginPullback | 8.90 | 0.058 | 0.226 | 0.557 | 0.410 | 0.084 |
| MarginAscent | 8.1e5 | 0.038 | 0.228 | **0.686** | 1.052 | 0.022 |
| **Ablations** | | | | | | |
| MarginAscentFrozen | 8.6e5 | 0.033 | 0.220 | 0.645 | 1.038 | 0.029 |
| MarginGradientAscent | 1.7e6 | 0.004 | 0.175 | 0.585 | 1.273 | 0.014 |
| GradientAscent | 1.7e6 | 0.040 | 0.221 | 0.546 | 1.282 | 0.011 |
| **Captum baselines** | | | | | | |
| Gradient | 10.09 | 0.014 | 0.157 | 0.384 | 1.030 | -0.026 |
| Saliency | 3.4e5 | 0.010 | 0.157 | 0.071 | 0.811 | 0.338 |
| InputXGradient | 14.38 | -0.034 | 0.264 | -0.295 | 1.092 | -0.020 |
| IntegratedGradients | 75.33 | -0.019 | 0.324 | -0.355 | 0.741 | -0.003 |
| GradientShap | 790.7 | -0.037 | 0.304 | -0.460 | 2.935 | -0.005 |
| DeepLift | 33.42 | -0.006 | 0.305 | -0.152 | 0.692 | 0.013 |
| Deconvolution | 9.2e15 | -0.004 | 0.216 | 0.173 | **0.181** | 1.000 |
| GuidedGradCam | 4.7e3 | 0.009 | **0.402** | 0.072 | 0.189 | 0.364 |

The class names are what the code and the result files use; the two group
headings give the names used in the text. `DistributionPullback` is the
Fisher-Rao Selector, kept under the old name so older result files stay
loadable. `Double` prefixes are the paper's Double Pullback refinement applied
to that selector, `Ascent` is Algorithm 1 driven by it.

Entropy-Flow has no row here. It is target-independent by construction, so
RandomLogit would come out at about 1.0 by definition rather than as a result,
and the rest of the suite scores a map against a change in a class score.
`ClassEntropyFlow` does depend on the class and is the variant to benchmark
first.

Standard deviations are in `results/summary_resnet50_n200.csv`. They are wide:
about 0.15 for FaithCorr, 0.35 for MonoCorr, 0.3 for FaithEst.
