# Metric unification resnet50, n = 200

Every aggregate is rank-based: `Infidelity` spans fifteen orders of magnitude across methods, so means and Pearson correlations over it are reports on `Deconvolution`'s tail.


## 1. How much of each metric is about the method at all

Two-way variance decomposition of the rank-transformed method x image panel. `var_method` is the share of variance attributable to which method produced the map.

| metric                  |   var_method |   var_image |   var_resid |   n_images |   n_methods |
|:------------------------|-------------:|------------:|------------:|-----------:|------------:|
| Infidelity              |        0.826 |       0.049 |       0.125 |        200 |          19 |
| FaithfulnessCorrelation |        0.046 |       0.125 |       0.829 |        200 |          19 |
| MonotonicityCorrelation |        0.027 |       0.478 |       0.495 |        200 |          19 |
| FaithfulnessEstimate    |        0.516 |       0.103 |       0.381 |        200 |          19 |
| MaxSensitivity          |        0.798 |       0.112 |       0.09  |        200 |          19 |
| RandomLogit             |        0.369 |       0.156 |       0.475 |        200 |          19 |

**FaithfulnessCorrelation, MonotonicityCorrelation** put under 10 % of their variance on the method.


## 2. Do the metrics agree about methods?

| metric                  |   Infidelity |   FaithfulnessCorrelation |   MonotonicityCorrelation |   FaithfulnessEstimate |   MaxSensitivity |   RandomLogit |
|:------------------------|-------------:|--------------------------:|--------------------------:|-----------------------:|-----------------:|--------------:|
| Infidelity              |         1    |                      0.37 |                      0.11 |                  -0.11 |             0.4  |          0.65 |
| FaithfulnessCorrelation |         0.37 |                      1    |                     -0.35 |                   0.62 |             0.28 |          0.05 |
| MonotonicityCorrelation |         0.11 |                     -0.35 |                      1    |                  -0.41 |             0.1  |         -0.03 |
| FaithfulnessEstimate    |        -0.11 |                      0.62 |                     -0.41 |                   1    |            -0.2  |         -0.3  |
| MaxSensitivity          |         0.4  |                      0.28 |                      0.1  |                  -0.2  |             1    |         -0.1  |
| RandomLogit             |         0.65 |                      0.05 |                     -0.03 |                  -0.3  |            -0.1  |          1    |


### Pairwise, with thresholds

| a                       | b                       |    rho |   abs_rho |     p | sig_05   | sig_bonf   |   n |
|:------------------------|:------------------------|-------:|----------:|------:|:---------|:-----------|----:|
| Infidelity              | RandomLogit             |  0.649 |     0.649 | 0.003 | True     | True       |  19 |
| FaithfulnessCorrelation | FaithfulnessEstimate    |  0.619 |     0.619 | 0.005 | True     | False      |  19 |
| MonotonicityCorrelation | FaithfulnessEstimate    | -0.407 |     0.407 | 0.084 | False    | False      |  19 |
| Infidelity              | MaxSensitivity          |  0.4   |     0.4   | 0.09  | False    | False      |  19 |
| Infidelity              | FaithfulnessCorrelation |  0.374 |     0.374 | 0.115 | False    | False      |  19 |
| FaithfulnessCorrelation | MonotonicityCorrelation | -0.346 |     0.346 | 0.147 | False    | False      |  19 |
| FaithfulnessEstimate    | RandomLogit             | -0.3   |     0.3   | 0.212 | False    | False      |  19 |
| FaithfulnessCorrelation | MaxSensitivity          |  0.281 |     0.281 | 0.244 | False    | False      |  19 |
| FaithfulnessEstimate    | MaxSensitivity          | -0.204 |     0.204 | 0.403 | False    | False      |  19 |
| Infidelity              | FaithfulnessEstimate    | -0.112 |     0.112 | 0.647 | False    | False      |  19 |
| Infidelity              | MonotonicityCorrelation |  0.111 |     0.111 | 0.652 | False    | False      |  19 |
| MaxSensitivity          | RandomLogit             | -0.096 |     0.096 | 0.694 | False    | False      |  19 |
| MonotonicityCorrelation | MaxSensitivity          |  0.096 |     0.096 | 0.697 | False    | False      |  19 |
| FaithfulnessCorrelation | RandomLogit             |  0.053 |     0.053 | 0.831 | False    | False      |  19 |
| MonotonicityCorrelation | RandomLogit             | -0.027 |     0.027 | 0.912 | False    | False      |  19 |


## 3. Do the metrics agree about images?

Spearman over the images *inside* each method, pooled by Fisher-z. This is the powered version of the question n = images, not n = 19 methods.

|                         |   Infidelity |   FaithfulnessCorrelation |   MonotonicityCorrelation |   FaithfulnessEstimate |   MaxSensitivity |   RandomLogit |
|:------------------------|-------------:|--------------------------:|--------------------------:|-----------------------:|-----------------:|--------------:|
| Infidelity              |         1    |                     -0.03 |                     -0.02 |                   0.03 |            -0.1  |          0.08 |
| FaithfulnessCorrelation |        -0.03 |                      1    |                     -0.02 |                   0.13 |             0.1  |         -0.03 |
| MonotonicityCorrelation |        -0.02 |                     -0.02 |                      1    |                  -0.09 |             0.03 |         -0    |
| FaithfulnessEstimate    |         0.03 |                      0.13 |                     -0.09 |                   1    |             0.18 |         -0.07 |
| MaxSensitivity          |        -0.1  |                      0.1  |                      0.03 |                   0.18 |             1    |         -0.06 |
| RandomLogit             |         0.08 |                     -0.03 |                     -0    |                  -0.07 |            -0.06 |          1    |


| a                       | b                       |    rho |   abs_rho |     p | sig_05   | sig_bonf   |   n |
|:------------------------|:------------------------|-------:|----------:|------:|:---------|:-----------|----:|
| FaithfulnessEstimate    | MaxSensitivity          |  0.181 |     0.181 | 0.01  | True     | False      | 200 |
| FaithfulnessCorrelation | FaithfulnessEstimate    |  0.127 |     0.127 | 0.072 | False    | False      | 200 |
| Infidelity              | MaxSensitivity          | -0.105 |     0.105 | 0.141 | False    | False      | 200 |
| FaithfulnessCorrelation | MaxSensitivity          |  0.098 |     0.098 | 0.167 | False    | False      | 200 |
| MonotonicityCorrelation | FaithfulnessEstimate    | -0.087 |     0.087 | 0.219 | False    | False      | 200 |
| Infidelity              | RandomLogit             |  0.081 |     0.081 | 0.257 | False    | False      | 200 |
| FaithfulnessEstimate    | RandomLogit             | -0.068 |     0.068 | 0.34  | False    | False      | 200 |
| MaxSensitivity          | RandomLogit             | -0.059 |     0.059 | 0.405 | False    | False      | 200 |
| Infidelity              | FaithfulnessEstimate    |  0.034 |     0.034 | 0.637 | False    | False      | 200 |
| Infidelity              | FaithfulnessCorrelation | -0.03  |     0.03  | 0.671 | False    | False      | 200 |
| FaithfulnessCorrelation | RandomLogit             | -0.029 |     0.029 | 0.684 | False    | False      | 200 |
| MonotonicityCorrelation | MaxSensitivity          |  0.029 |     0.029 | 0.685 | False    | False      | 200 |
| Infidelity              | MonotonicityCorrelation | -0.019 |     0.019 | 0.791 | False    | False      | 200 |
| FaithfulnessCorrelation | MonotonicityCorrelation | -0.017 |     0.017 | 0.809 | False    | False      | 200 |
| MonotonicityCorrelation | RandomLogit             | -0.004 |     0.004 | 0.957 | False    | False      | 200 |


Surviving Bonferroni: **none**. Largest |rho| at image level is 0.181 against 0.649 at method level.


## 4. How many axes do the six span?

|     |   eigenvalue |   explained |   cumulative |
|:----|-------------:|------------:|-------------:|
| PC1 |        1.949 |       0.325 |        0.325 |
| PC2 |        1.91  |       0.318 |        0.643 |
| PC3 |        1.156 |       0.193 |        0.836 |
| PC4 |        0.681 |       0.113 |        0.949 |
| PC5 |        0.195 |       0.033 |        0.982 |
| PC6 |        0.11  |       0.018 |        1     |


Kaiser criterion: **3** components. 80 % of variance needs **3**, 90 % needs **4**.


### Loadings

| metric                  |   PC1 |   PC2 |   PC3 |   PC4 |   PC5 |   PC6 |
|:------------------------|------:|------:|------:|------:|------:|------:|
| Infidelity              | -0.08 |  0.68 | -0.05 |  0.17 | -0.54 | -0.45 |
| FaithfulnessCorrelation |  0.55 |  0.38 |  0.14 |  0.2  |  0.66 | -0.25 |
| MonotonicityCorrelation | -0.48 | -0.01 |  0.31 |  0.79 |  0.15 |  0.14 |
| FaithfulnessEstimate    |  0.65 | -0.07 | -0.04 |  0.41 | -0.44 |  0.47 |
| MaxSensitivity          | -0.04 |  0.38 |  0.72 | -0.37 | -0.06 |  0.43 |
| RandomLogit             | -0.21 |  0.49 | -0.6  | -0.03 |  0.23 |  0.55 |


### Method positions

| method                     |   PC1 |   PC2 |   PC3 |   PC4 |   PC5 |   PC6 |
|:---------------------------|------:|------:|------:|------:|------:|------:|
| GradientAscent             |  1.36 | -1.26 | -1.04 | -0.16 |  0.54 | -0.37 |
| MarginAscentFrozen         |  1.34 | -1.09 |  0.01 |  0.44 | -0.56 |  0.13 |
| PullbackAscent             |  1.3  | -0.41 | -0.39 |  1.11 |  0.28 |  0.06 |
| MarginAscent               |  1.19 | -1.14 |  0.02 |  1.14 | -0.26 |  0.37 |
| MarginGradientAscent       |  1.06 | -1.98 | -1.07 | -0.45 | -0.61 |  0.27 |
| MarginPullback             |  1.06 |  1.38 |  1.63 |  0.94 | -0.66 | -0.67 |
| SoftPullback               |  1.04 |  2.53 |  0.33 |  0.21 |  0.11 |  0.39 |
| DoublePullback             |  0.83 |  1.54 |  0.07 | -0.41 |  0.11 | -0.08 |
| DistributionPullback       |  0.51 |  2.54 | -0.45 | -0.74 |  0.6  |  0.06 |
| DistributionAscent         |  0.38 | -0.77 | -0.42 |  1.1  |  0.97 |  0.03 |
| Saliency                   |  0.35 | -1.72 |  0.3  | -1.79 |  0.22 | -0.46 |
| Gradient                   |  0.26 |  1.3  | -1.53 | -1.03 | -0.59 |  0.05 |
| DistributionDoublePullback |  0.24 |  0.8  |  0.14 | -0.54 | -0.05 | -0.16 |
| Deconvolution              | -0.33 | -1.5  |  2.12 | -1.17 |  0.17 |  0.46 |
| GuidedGradCam              | -1.28 | -0.37 |  2.37 |  0.32 |  0.08 | -0.19 |
| DeepLift                   | -1.9  |  0.21 |  0.26 |  0.11 | -0.21 |  0.11 |
| InputXGradient             | -2.31 |  0.49 | -0.9  |  0.12 | -0.34 |  0.11 |
| IntegratedGradients        | -2.48 |  0.11 | -0.07 |  0.2  |  0.21 |  0.42 |
| GradientShap               | -2.61 | -0.65 | -1.36 |  0.61 | -0    | -0.55 |

### Within the pullback family only

[0.539, 0.282, 0.098, 0.044, 0.033, 0.004] explained; Kaiser 2, 80 % at 2.


| metric                  |   PC1 |   PC2 |
|:------------------------|------:|------:|
| Infidelity              |  0.5  |  0.28 |
| FaithfulnessCorrelation |  0.51 |  0.08 |
| MonotonicityCorrelation |  0.07 |  0.62 |
| FaithfulnessEstimate    | -0.35 |  0.5  |
| MaxSensitivity          |  0.47 |  0.27 |
| RandomLogit             |  0.38 | -0.46 |



### Within the captum family only

[0.486, 0.342, 0.116, 0.036, 0.017, 0.003] explained; Kaiser 2, 80 % at 2.


| metric                  |   PC1 |   PC2 |
|:------------------------|------:|------:|
| Infidelity              | -0.42 | -0.37 |
| FaithfulnessCorrelation |  0.41 | -0.39 |
| MonotonicityCorrelation | -0.12 |  0.56 |
| FaithfulnessEstimate    |  0.42 | -0.48 |
| MaxSensitivity          |  0.5  |  0.2  |
| RandomLogit             | -0.46 | -0.36 |



## 5. Where the extensions sit

| method               |   PC1 |   PC2 |   PC3 |
|:---------------------|------:|------:|------:|
| SoftPullback         |  1.04 |  2.53 |  0.33 |
| DistributionPullback |  0.51 |  2.54 | -0.45 |
| MarginPullback       |  1.06 |  1.38 |  1.63 |
| MarginAscent         |  1.19 | -1.14 |  0.02 |
