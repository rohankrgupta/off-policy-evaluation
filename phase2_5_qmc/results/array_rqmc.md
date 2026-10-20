# Array-RQMC on the GridWorld

## Paired MSE ratios at n = 64 / 512 / 8192

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI). BPS = exact BPS target.

| domain | comparison | n = 64 | n = 512 | n = 8192 |
|---|---|---|---|---|
| πe-A | Array-RQMC, MC vs ROS-MC (α tuned at each n) | 2.41 ✗ | 3 ✗ | 10.5 ✗ |
| πe-A | Array-RQMC, MC vs Sobol RQMC, MC | 0.484 ✓ | 0.293 ✓ | 0.146 ✓ |
| πe-A | BPS + Array-RQMC, B vs BPS + Sobol RQMC, B | 0.452 ✓ | 0.29 ✓ | 0.255 ✓ |
| πe-A | BPS + Array-RQMC, B vs BPS+ROS (α tuned at each n) | 13.1 ✗ | 1.41 ✗ | 0.712 ✓ |
| πe-A | BPS + Array-RQMC, B vs ROS-MC (α tuned at each n) | 2.36 ✗ | 3.25 ✗ | 17.4 ✗ |
| πe-A | ROS-MC (α tuned at n = 8192) vs ROS-MC (α tuned at each n) | 1.51 ✗ | 2.74 ✗ | 1 ~ |
| πe-B | Array-RQMC, MC vs ROS-MC (α tuned at each n) | 27.3 ✗ | 68 ✗ | 147 ✗ |
| πe-B | Array-RQMC, MC vs Sobol RQMC, MC | 0.485 ✓ | 0.169 ✓ | 0.0282 ✓ |
| πe-B | BPS + Array-RQMC, B vs BPS + Sobol RQMC, B | 0.758 ✓ | 0.479 ✓ | 0.607 ✓ |
| πe-B | BPS + Array-RQMC, B vs BPS+ROS (α tuned at each n) | 0.634 ✓ | 0.384 ✓ | 0.386 ✓ |
| πe-B | BPS + Array-RQMC, B vs ROS-MC (α tuned at each n) | 13 ✗ | 63 ✗ | 927 ✗ |
| πe-B | ROS-MC (α tuned at n = 8192) vs ROS-MC (α tuned at each n) | 1.32 ✗ | 1.35 ✗ | 1 ~ |
| πe-B, slippery | Array-RQMC, MC vs ROS-MC (α tuned at each n) | 2.41 ✗ | 1.33 ✗ | 1.07 ~ |
| πe-B, slippery | Array-RQMC, MC vs Sobol RQMC, MC | 0.656 ✓ | 0.382 ✓ | 0.317 ✓ |
| πe-B, slippery | BPS + Array-RQMC, B vs BPS + Sobol RQMC, B | 0.876 ✓ | 0.676 ✓ | 0.815 ✓ |
| πe-B, slippery | BPS + Array-RQMC, B vs BPS+ROS (α tuned at each n) | 0.749 ✓ | 0.705 ✓ | 0.697 ✓ |
| πe-B, slippery | BPS + Array-RQMC, B vs ROS-MC (α tuned at each n) | 1.51 ✗ | 1.18 ✗ | 1.24 ✗ |
| πe-B, slippery | ROS-MC (α tuned at n = 8192) vs ROS-MC (α tuned at each n) | 1 ✗ | 0.951 ~ | 1 ~ |

## Bias and error bars (on-policy MC)

bias/SE over 1000 seeds (|value| > 2 suggests bias). Coverage of 95% t-intervals built from groups of 10 independent runs (100 groups), and their mean width relative to OS-MC's.

| domain | method | bias/SE, n = 64 | coverage | width | bias/SE, n = 8192 | coverage | width |
|---|---|---|---|---|---|---|---|
| πe-A | OS-MC | -0.6 | 95% | 0.964 | -1.8 | 93% | 0.967 |
| πe-A | ROS-MC (α tuned at each n) | -14.1 | 78% | 0.232 | -20.5 | 55% | 0.0285 |
| πe-A | per-state QMC, MC | +4.7 | 92% | 0.239 | +7.6 | 89% | 0.0402 |
| πe-A | Sobol RQMC, MC | +0.2 | 97% | 0.554 | -1.6 | 94% | 0.29 |
| πe-A | Array-RQMC, MC | -1.9 | 94% | 0.384 | -0.1 | 99% | 0.113 |
| πe-B | OS-MC | -0.1 | 97% | 0.963 | +0.5 | 95% | 0.965 |
| πe-B | ROS-MC (α tuned at each n) | -19.3 | 59% | 0.0954 | -11.9 | 76% | 0.00915 |
| πe-B | per-state QMC, MC | +11.4 | 80% | 0.203 | +8.9 | 88% | 0.0224 |
| πe-B | Sobol RQMC, MC | +0.2 | 93% | 0.845 | -0.6 | 97% | 0.702 |
| πe-B | Array-RQMC, MC | +0.2 | 95% | 0.585 | -0.3 | 92% | 0.119 |
| πe-B, slippery | OS-MC | +0.8 | 97% | 0.975 | +0.8 | 93% | 0.965 |
| πe-B, slippery | ROS-MC (α tuned at each n) | -2.8 | 94% | 0.451 | -2.2 | 96% | 0.442 |
| πe-B, slippery | per-state QMC, MC | +4.1 | 93% | 0.528 | -1.2 | 95% | 0.429 |
| πe-B, slippery | Sobol RQMC, MC | +1.0 | 97% | 0.89 | +0.4 | 98% | 0.806 |
| πe-B, slippery | Array-RQMC, MC | +0.5 | 88% | 0.698 | -0.6 | 95% | 0.445 |

## n·MSE/Var(g)

On-policy Monte Carlo = 1; lower is better. Slope of log MSE against log n between n = 512 and 8192 (−1 is the Monte Carlo rate).

### πe-A

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) |
|---|---|---|---|---|
| OS-MC | 1 | 0.942 | 0.987 | -0.98 |
| ROS-MC (α tuned at each n) | 0.0659 | 0.019 | 0.00124 | -1.98 |
| ROS-MC (α tuned at n = 8192) | 0.0996 | 0.0521 | 0.00124 | -2.35 |
| per-state QMC, MC | 0.0625 | 0.0184 | 0.00173 | -1.85 |
| Sobol RQMC, MC | 0.327 | 0.194 | 0.0893 | -1.28 |
| Array-RQMC, MC | 0.159 | 0.0569 | 0.013 | -1.53 |
| BPS-OIS | 0.92 | 0.812 | 0.901 | -0.96 |
| BPS+ROS (α tuned at each n) | 0.0118 | 0.0438 | 0.0303 | -1.13 |
| BPS + Sobol RQMC, B | 0.344 | 0.213 | 0.0844 | -1.33 |
| BPS + Array-RQMC, B | 0.155 | 0.0617 | 0.0215 | -1.38 |

### πe-B

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) |
|---|---|---|---|---|
| OS-MC | 0.962 | 0.987 | 1.09 | -0.96 |
| ROS-MC (α tuned at each n) | 0.0128 | 0.0016 | 0.000109 | -1.97 |
| ROS-MC (α tuned at n = 8192) | 0.0168 | 0.00216 | 0.000109 | -2.08 |
| per-state QMC, MC | 0.0475 | 0.0083 | 0.000623 | -1.93 |
| Sobol RQMC, MC | 0.718 | 0.641 | 0.57 | -1.04 |
| Array-RQMC, MC | 0.348 | 0.109 | 0.0161 | -1.69 |
| BPS-OIS | 0.265 | 0.275 | 0.289 | -0.98 |
| BPS+ROS (α tuned at each n) | 0.262 | 0.262 | 0.262 | -1.00 |
| BPS + Sobol RQMC, B | 0.219 | 0.21 | 0.167 | -1.08 |
| BPS + Array-RQMC, B | 0.166 | 0.1 | 0.101 | -1.00 |

### πe-B, slippery

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) |
|---|---|---|---|---|
| OS-MC | 0.916 | 1.01 | 1.08 | -0.97 |
| ROS-MC (α tuned at each n) | 0.204 | 0.22 | 0.218 | -1.00 |
| ROS-MC (α tuned at n = 8192) | 0.205 | 0.209 | 0.218 | -0.98 |
| per-state QMC, MC | 0.271 | 0.232 | 0.21 | -1.04 |
| Sobol RQMC, MC | 0.751 | 0.763 | 0.737 | -1.01 |
| Array-RQMC, MC | 0.493 | 0.292 | 0.234 | -1.08 |
| BPS-OIS | 0.408 | 0.388 | 0.388 | -1.00 |
| BPS+ROS (α tuned at each n) | 0.413 | 0.368 | 0.39 | -0.98 |
| BPS + Sobol RQMC, B | 0.353 | 0.384 | 0.333 | -1.05 |
| BPS + Array-RQMC, B | 0.31 | 0.259 | 0.272 | -0.98 |
