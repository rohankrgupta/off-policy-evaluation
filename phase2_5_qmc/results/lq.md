# QMC vs ROS vs PROPS with continuous actions (linear-quadratic control)

## Paired MSE ratios at n = 64 / 512 / 4096

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| configuration | comparison | n = 64 | n = 512 | n = 4096 |
|---|---|---|---|---|
| d = 1, H = 10, σ_w = 0 | Sobol RQMC vs i.i.d. | 0.0517 ✓ | 0.00746 ✓ | 0.00084 ✓ |
| d = 1, H = 10, σ_w = 0 | Array-RQMC vs i.i.d. | 0.0233 ✓ | 0.00423 ✓ | 0.00167 ✓ |
| d = 1, H = 10, σ_w = 0 | Sobol RQMC vs ROS (α tuned at each n) | 0.165 ✓ | 0.0287 ✓ | 0.00541 ✓ |
| d = 1, H = 10, σ_w = 0 | Array-RQMC vs ROS (α tuned at each n) | 0.074 ✓ | 0.0163 ✓ | 0.0108 ✓ |
| d = 1, H = 10, σ_w = 0 | Sobol RQMC vs PROPS (lr tuned at each n) | 0.113 ✓ | 0.0143 ✓ | 0.00149 ✓ |
| d = 1, H = 10, σ_w = 0 | Array-RQMC vs PROPS (lr tuned at each n) | 0.0508 ✓ | 0.00811 ✓ | 0.00297 ✓ |
| d = 1, H = 10, σ_w = 0 | ROS (α tuned at each n) vs PROPS (lr tuned at each n) | 0.686 ✓ | 0.499 ✓ | 0.275 ✓ |
| d = 1, H = 10, σ_w = 0.1 | Sobol RQMC vs i.i.d. | 0.208 ✓ | 0.175 ✓ | 0.162 ✓ |
| d = 1, H = 10, σ_w = 0.1 | Array-RQMC vs i.i.d. | 0.18 ✓ | 0.179 ✓ | 0.158 ✓ |
| d = 1, H = 10, σ_w = 0.1 | Sobol RQMC vs ROS (α tuned at each n) | 0.506 ✓ | 0.469 ✓ | 0.572 ✓ |
| d = 1, H = 10, σ_w = 0.1 | Array-RQMC vs ROS (α tuned at each n) | 0.439 ✓ | 0.481 ✓ | 0.557 ✓ |
| d = 1, H = 10, σ_w = 0.1 | Sobol RQMC vs PROPS (lr tuned at each n) | 0.394 ✓ | 0.291 ✓ | 0.263 ✓ |
| d = 1, H = 10, σ_w = 0.1 | Array-RQMC vs PROPS (lr tuned at each n) | 0.342 ✓ | 0.298 ✓ | 0.256 ✓ |
| d = 1, H = 10, σ_w = 0.1 | ROS (α tuned at each n) vs PROPS (lr tuned at each n) | 0.777 ✓ | 0.62 ✓ | 0.459 ✓ |
| d = 1, H = 10, σ_w = 0.3 | Sobol RQMC vs i.i.d. | 0.736 ✓ | 0.725 ✓ | 0.705 ✓ |
| d = 1, H = 10, σ_w = 0.3 | Array-RQMC vs i.i.d. | 0.779 ✓ | 0.736 ✓ | 0.741 ✓ |
| d = 1, H = 10, σ_w = 0.3 | Sobol RQMC vs ROS (α tuned at each n) | 0.974 ~ | 0.774 ✓ | 0.937 ~ |
| d = 1, H = 10, σ_w = 0.3 | Array-RQMC vs ROS (α tuned at each n) | 1.03 ~ | 0.786 ✓ | 0.986 ~ |
| d = 1, H = 10, σ_w = 0.3 | Sobol RQMC vs PROPS (lr tuned at each n) | 0.895 ~ | 0.804 ✓ | 0.817 ✓ |
| d = 1, H = 10, σ_w = 0.3 | Array-RQMC vs PROPS (lr tuned at each n) | 0.948 ~ | 0.816 ✓ | 0.859 ✓ |
| d = 1, H = 10, σ_w = 0.3 | ROS (α tuned at each n) vs PROPS (lr tuned at each n) | 0.919 ✓ | 1.04 ~ | 0.871 ✓ |
| d = 1, H = 50, σ_w = 0.1 | Sobol RQMC vs i.i.d. | 0.202 ✓ | 0.155 ✓ | 0.153 ✓ |
| d = 1, H = 50, σ_w = 0.1 | Array-RQMC vs i.i.d. | 0.18 ✓ | 0.159 ✓ | 0.147 ✓ |
| d = 1, H = 50, σ_w = 0.1 | Sobol RQMC vs ROS (α tuned at each n) | 0.588 ✓ | 0.793 ✓ | 0.929 ~ |
| d = 1, H = 50, σ_w = 0.1 | Array-RQMC vs ROS (α tuned at each n) | 0.522 ✓ | 0.812 ✓ | 0.891 ~ |
| d = 1, H = 50, σ_w = 0.1 | Sobol RQMC vs PROPS (lr tuned at each n) | 0.353 ✓ | 0.252 ✓ | 0.208 ✓ |
| d = 1, H = 50, σ_w = 0.1 | Array-RQMC vs PROPS (lr tuned at each n) | 0.313 ✓ | 0.258 ✓ | 0.2 ✓ |
| d = 1, H = 50, σ_w = 0.1 | ROS (α tuned at each n) vs PROPS (lr tuned at each n) | 0.6 ✓ | 0.318 ✓ | 0.224 ✓ |
| d = 4, H = 50, σ_w = 0.1 | Sobol RQMC vs i.i.d. | 0.944 ✓ | 0.98 ~ | 0.932 ✓ |
| d = 4, H = 50, σ_w = 0.1 | Array-RQMC vs i.i.d. | 0.959 ~ | 1.02 ~ | 0.928 ✓ |
| d = 4, H = 50, σ_w = 0.1 | Sobol RQMC vs ROS (α tuned at each n) | 0.969 ~ | 0.996 ~ | 0.958 ~ |
| d = 4, H = 50, σ_w = 0.1 | Array-RQMC vs ROS (α tuned at each n) | 0.984 ~ | 1.03 ~ | 0.954 ~ |
| d = 4, H = 50, σ_w = 0.1 | Sobol RQMC vs PROPS (lr tuned at each n) | 0.951 ✓ | 0.977 ~ | 0.939 ✓ |
| d = 4, H = 50, σ_w = 0.1 | Array-RQMC vs PROPS (lr tuned at each n) | 0.965 ~ | 1.01 ~ | 0.934 ✓ |
| d = 4, H = 50, σ_w = 0.1 | ROS (α tuned at each n) vs PROPS (lr tuned at each n) | 0.981 ✓ | 0.982 ~ | 0.979 ✓ |

## Bias and error bars

bias/SE over 1000 seeds; coverage of 95% t-intervals from groups of 10 independent runs, and their mean width relative to i.i.d. sampling's.

| configuration | method | bias/SE, n = 64 | coverage | width | bias/SE, n = 4096 | coverage | width |
|---|---|---|---|---|---|---|---|
| d = 1, H = 10, σ_w = 0 | i.i.d. | -1.4 | 94% | 0.969 | +1.5 | 97% | 0.975 |
| d = 1, H = 10, σ_w = 0 | Sobol RQMC | +2.0 | 94% | 0.219 | +3.1 | 94% | 0.0281 |
| d = 1, H = 10, σ_w = 0 | antithetic | -0.6 | 96% | 1.35 | +1.4 | 95% | 1.35 |
| d = 1, H = 10, σ_w = 0 | Array-RQMC | -0.5 | 94% | 0.146 | -1.0 | 94% | 0.0397 |
| d = 1, H = 10, σ_w = 0 | ROS (α tuned at each n) | -12.7 | 77% | 0.504 | -12.6 | 82% | 0.357 |
| d = 1, H = 10, σ_w = 0 | PROPS (lr tuned at each n) | -13.6 | 76% | 0.604 | -7.1 | 91% | 0.715 |
| d = 1, H = 10, σ_w = 0.1 | i.i.d. | -1.1 | 95% | 0.969 | +1.9 | 96% | 0.978 |
| d = 1, H = 10, σ_w = 0.1 | Sobol RQMC | +0.8 | 93% | 0.442 | +0.8 | 94% | 0.388 |
| d = 1, H = 10, σ_w = 0.1 | antithetic | -0.9 | 95% | 1.31 | +1.9 | 93% | 1.3 |
| d = 1, H = 10, σ_w = 0.1 | Array-RQMC | -0.4 | 95% | 0.413 | +0.8 | 94% | 0.383 |
| d = 1, H = 10, σ_w = 0.1 | ROS (α tuned at each n) | -9.9 | 82% | 0.592 | -6.8 | 90% | 0.507 |
| d = 1, H = 10, σ_w = 0.1 | PROPS (lr tuned at each n) | -11.1 | 85% | 0.669 | -5.5 | 92% | 0.759 |
| d = 1, H = 10, σ_w = 0.3 | i.i.d. | -0.6 | 94% | 0.973 | +2.3 | 95% | 0.975 |
| d = 1, H = 10, σ_w = 0.3 | Sobol RQMC | -0.2 | 96% | 0.828 | +1.2 | 89% | 0.802 |
| d = 1, H = 10, σ_w = 0.3 | antithetic | -1.4 | 95% | 1.11 | +2.6 | 95% | 1.09 |
| d = 1, H = 10, σ_w = 0.3 | Array-RQMC | +0.2 | 96% | 0.859 | +1.1 | 94% | 0.834 |
| d = 1, H = 10, σ_w = 0.3 | ROS (α tuned at each n) | -4.4 | 93% | 0.841 | -2.5 | 97% | 0.849 |
| d = 1, H = 10, σ_w = 0.3 | PROPS (lr tuned at each n) | -5.3 | 95% | 0.877 | -1.5 | 96% | 0.905 |
| d = 1, H = 50, σ_w = 0.1 | i.i.d. | -1.2 | 98% | 0.974 | +0.2 | 92% | 0.958 |
| d = 1, H = 50, σ_w = 0.1 | Sobol RQMC | +1.4 | 94% | 0.437 | -0.3 | 94% | 0.378 |
| d = 1, H = 50, σ_w = 0.1 | antithetic | -0.8 | 90% | 1.24 | -0.2 | 90% | 1.25 |
| d = 1, H = 50, σ_w = 0.1 | Array-RQMC | -0.6 | 95% | 0.415 | +0.5 | 99% | 0.373 |
| d = 1, H = 50, σ_w = 0.1 | ROS (α tuned at each n) | -2.7 | 95% | 0.575 | +0.3 | 93% | 0.392 |
| d = 1, H = 50, σ_w = 0.1 | PROPS (lr tuned at each n) | -16.3 | 72% | 0.661 | -14.8 | 76% | 0.746 |
| d = 4, H = 50, σ_w = 0.1 | i.i.d. | -0.7 | 94% | 0.979 | +1.1 | 96% | 0.975 |
| d = 4, H = 50, σ_w = 0.1 | Sobol RQMC | -0.5 | 97% | 0.956 | +0.7 | 97% | 0.951 |
| d = 4, H = 50, σ_w = 0.1 | antithetic | -0.6 | 94% | 0.987 | +0.7 | 95% | 0.972 |
| d = 4, H = 50, σ_w = 0.1 | Array-RQMC | -0.3 | 94% | 0.958 | +1.0 | 96% | 0.945 |
| d = 4, H = 50, σ_w = 0.1 | ROS (α tuned at each n) | -1.4 | 94% | 0.967 | +1.1 | 97% | 0.964 |
| d = 4, H = 50, σ_w = 0.1 | PROPS (lr tuned at each n) | -1.4 | 93% | 0.976 | +0.3 | 96% | 0.973 |

## n·MSE/Var(g)

i.i.d. sampling = 1; lower is better. Slope of log MSE against log n between n = 512 and 4096 (−1 is the Monte Carlo rate).

### d = 1, H = 10, σ_w = 0

| method | n = 64 | n = 512 | n = 4096 | slope (512 → 4096) |
|---|---|---|---|---|
| i.i.d. | 1.02 | 0.982 | 1.02 | -0.98 |
| Sobol RQMC | 0.0527 | 0.00733 | 0.000853 | -2.03 |
| antithetic | 1.98 | 2.05 | 1.99 | -1.01 |
| Array-RQMC | 0.0237 | 0.00415 | 0.0017 | -1.43 |
| ROS (α tuned at each n) | 0.32 | 0.256 | 0.158 | -1.23 |
| PROPS (lr tuned at each n) | 0.466 | 0.512 | 0.573 | -0.95 |

### d = 1, H = 10, σ_w = 0.1

| method | n = 64 | n = 512 | n = 4096 | slope (512 → 4096) |
|---|---|---|---|---|
| i.i.d. | 1 | 0.96 | 0.998 | -0.98 |
| Sobol RQMC | 0.209 | 0.168 | 0.162 | -1.02 |
| antithetic | 1.83 | 1.78 | 1.81 | -0.99 |
| Array-RQMC | 0.181 | 0.172 | 0.158 | -1.04 |
| ROS (α tuned at each n) | 0.412 | 0.358 | 0.283 | -1.11 |
| PROPS (lr tuned at each n) | 0.53 | 0.577 | 0.618 | -0.97 |

### d = 1, H = 10, σ_w = 0.3

| method | n = 64 | n = 512 | n = 4096 | slope (512 → 4096) |
|---|---|---|---|---|
| i.i.d. | 0.985 | 0.988 | 1 | -0.99 |
| Sobol RQMC | 0.725 | 0.716 | 0.707 | -1.01 |
| antithetic | 1.28 | 1.14 | 1.26 | -0.95 |
| Array-RQMC | 0.768 | 0.727 | 0.743 | -0.99 |
| ROS (α tuned at each n) | 0.744 | 0.924 | 0.754 | -1.10 |
| PROPS (lr tuned at each n) | 0.81 | 0.891 | 0.865 | -1.01 |

### d = 1, H = 50, σ_w = 0.1

| method | n = 64 | n = 512 | n = 4096 | slope (512 → 4096) |
|---|---|---|---|---|
| i.i.d. | 1.05 | 1.02 | 1.07 | -0.98 |
| Sobol RQMC | 0.212 | 0.158 | 0.164 | -0.98 |
| antithetic | 1.8 | 1.89 | 1.87 | -1.00 |
| Array-RQMC | 0.188 | 0.162 | 0.158 | -1.01 |
| ROS (α tuned at each n) | 0.36 | 0.199 | 0.177 | -1.06 |
| PROPS (lr tuned at each n) | 0.6 | 0.627 | 0.79 | -0.89 |

### d = 4, H = 50, σ_w = 0.1

| method | n = 64 | n = 512 | n = 4096 | slope (512 → 4096) |
|---|---|---|---|---|
| i.i.d. | 1.09 | 1.02 | 0.985 | -1.02 |
| Sobol RQMC | 1.03 | 0.998 | 0.918 | -1.04 |
| antithetic | 1.11 | 1.03 | 0.973 | -1.03 |
| Array-RQMC | 1.05 | 1.04 | 0.914 | -1.06 |
| ROS (α tuned at each n) | 1.07 | 1 | 0.958 | -1.02 |
| PROPS (lr tuned at each n) | 1.09 | 1.02 | 0.978 | -1.02 |
