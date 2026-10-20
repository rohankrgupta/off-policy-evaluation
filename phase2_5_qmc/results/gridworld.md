# QMC-type action sampling on the GridWorld

## Paired MSE ratios at n = 8192 [95% CI]

✓ the first method is better, ✗ worse, ~ inconclusive. BPS = exact BPS target.

| domain | comparison | MSE ratio |
|---|---|---|
| πe-A | Sobol RQMC, MC vs OS-MC | 0.0905 [0.0798, 0.103] ✓ |
| πe-A | Latin hypercube, MC vs OS-MC | 0.421 [0.371, 0.477] ✓ |
| πe-A | antithetic, MC vs OS-MC | 0.744 [0.656, 0.841] ✓ |
| πe-A | ROS + Sobol RQMC, MC vs ROS-MC (α=100000) | 1.1 [0.973, 1.25] ~ |
| πe-A | BPS + Sobol RQMC, B vs BPS-OIS | 0.0936 [0.0825, 0.106] ✓ |
| πe-A | BPS + Latin hypercube, B vs BPS-OIS | 0.445 [0.39, 0.508] ✓ |
| πe-A | BPS + antithetic, B vs BPS-OIS | 0.871 [0.769, 0.991] ✓ |
| πe-A | BPS + Sobol RQMC, B vs BPS+ROS-R (α=1000) | 2.79 [2.46, 3.16] ✗ |
| πe-A | BPS + Sobol RQMC, B vs ROS-MC (α=100000) | 68.2 [60.2, 77.3] ✗ |
| πe-B | Sobol RQMC, MC vs OS-MC | 0.521 [0.459, 0.594] ✓ |
| πe-B | Latin hypercube, MC vs OS-MC | 0.704 [0.618, 0.802] ✓ |
| πe-B | antithetic, MC vs OS-MC | 0.72 [0.633, 0.817] ✓ |
| πe-B | ROS + Sobol RQMC, MC vs ROS-MC (α=1e+06) | 0.977 [0.885, 1.07] ~ |
| πe-B | BPS + Sobol RQMC, B vs BPS-OIS | 0.578 [0.51, 0.656] ✓ |
| πe-B | BPS + Latin hypercube, B vs BPS-OIS | 0.855 [0.753, 0.971] ✓ |
| πe-B | BPS + antithetic, B vs BPS-OIS | 0.972 [0.858, 1.1] ~ |
| πe-B | BPS + Sobol RQMC, B vs BPS+ROS-R (α=100) | 0.637 [0.564, 0.718] ✓ |
| πe-B | BPS + Sobol RQMC, B vs ROS-MC (α=1e+06) | 1.53e+03 [1.36e+03, 1.71e+03] ✗ |
| πe-B, slippery | Sobol RQMC, MC vs OS-MC | 0.68 [0.6, 0.769] ✓ |
| πe-B, slippery | Latin hypercube, MC vs OS-MC | 0.757 [0.669, 0.857] ✓ |
| πe-B, slippery | antithetic, MC vs OS-MC | 0.763 [0.67, 0.859] ✓ |
| πe-B, slippery | ROS + Sobol RQMC, MC vs ROS-MC (α=1e+06) | 1.14 [1.01, 1.28] ✗ |
| πe-B, slippery | BPS + Sobol RQMC, B vs BPS-OIS | 0.859 [0.759, 0.974] ✓ |
| πe-B, slippery | BPS + Latin hypercube, B vs BPS-OIS | 1.08 [0.946, 1.22] ~ |
| πe-B, slippery | BPS + antithetic, B vs BPS-OIS | 0.942 [0.832, 1.07] ~ |
| πe-B, slippery | BPS + Sobol RQMC, B vs BPS+ROS-R (α=100) | 0.855 [0.757, 0.965] ✓ |
| πe-B, slippery | BPS + Sobol RQMC, B vs ROS-MC (α=1e+06) | 1.53 [1.35, 1.72] ✗ |

## Per-state QMC vs ROS with α tuned at each n (MSE ratio, QMC / ROS)

| domain | n = 16 | n = 64 | n = 256 | n = 1024 | n = 8192 |
|---|---|---|---|---|---|
| πe-A | 0.62 [0.53, 0.72] ✓ (α=100) | 0.95 [0.84, 1.08] ~ (α=1000) | 0.87 [0.77, 0.98] ✓ (α=10000) | 1.53 [1.38, 1.68] ✗ (α=1e+06) | 1.39 [1.24, 1.57] ✗ (α=100000) |
| πe-B | 4.47 [4.01, 4.97] ✗ (α=100000) | 3.72 [3.31, 4.17] ✗ (α=10000) | 7.42 [6.53, 8.50] ✗ (α=1e+06) | 6.49 [5.76, 7.32] ✗ (α=100000) | 5.70 [5.10, 6.38] ✗ (α=1e+06) |
| πe-B, slippery | 1.23 [1.08, 1.39] ✗ (α=1000) | 1.33 [1.17, 1.50] ✗ (α=100000) | 1.18 [1.05, 1.34] ✗ (α=100000) | 1.05 [0.92, 1.19] ~ (α=1e+06) | 0.96 [0.85, 1.09] ~ (α=1e+06) |

## n·MSE/Var(g) by method

On-policy Monte Carlo = 1; lower is better. 1000 seeds. Slope: of log MSE against log n over n = 1024 … 8192 (−1 is the Monte Carlo rate). Latin hypercubes are run only at the reported n.

### πe-A

| method | n = 64 | n = 512 | n = 8192 | slope |
|---|---|---|---|---|
| OS-MC | 1 | 0.942 | 0.987 | -0.98 |
| ROS-MC (α=100000) | 0.0996 | 0.0521 | 0.00124 | -2.02 |
| per-state QMC, MC | 0.0625 | 0.0184 | 0.00173 | -1.86 |
| Sobol RQMC, MC | 0.327 | 0.194 | 0.0893 | -1.29 |
| Latin hypercube, MC | 0.426 | 0.411 | 0.416 | – |
| antithetic, MC | 0.75 | 0.723 | 0.734 | -1.01 |
| ROS + Sobol RQMC, MC | 0.1 | 0.0522 | 0.00136 | -2.00 |
| BPS-OIS | 0.92 | 0.812 | 0.901 | -0.96 |
| BPS+ROS-R (α=1000) | 0.0716 | 0.062 | 0.0303 | -1.21 |
| BPS + Sobol RQMC, B | 0.344 | 0.213 | 0.0844 | -1.32 |
| BPS + Sobol RQMC, R | 0.184 | 0.0907 | 0.0193 | -1.54 |
| BPS + Latin hypercube, B | 0.402 | 0.371 | 0.402 | – |
| BPS + antithetic, B | 0.698 | 0.691 | 0.786 | -0.91 |
| BPS + per-state QMC, B | 0.0774 | 0.134 | 1.63 | -0.06 |

### πe-B

| method | n = 64 | n = 512 | n = 8192 | slope |
|---|---|---|---|---|
| OS-MC | 0.962 | 0.987 | 1.09 | -0.98 |
| ROS-MC (α=1e+06) | 0.0168 | 0.00216 | 0.000109 | -1.88 |
| per-state QMC, MC | 0.0475 | 0.0083 | 0.000623 | -1.88 |
| Sobol RQMC, MC | 0.718 | 0.641 | 0.57 | -1.03 |
| Latin hypercube, MC | 0.835 | 0.786 | 0.77 | – |
| antithetic, MC | 0.891 | 0.888 | 0.788 | -1.06 |
| ROS + Sobol RQMC, MC | 0.0162 | 0.00206 | 0.000107 | -1.87 |
| BPS-OIS | 0.265 | 0.275 | 0.289 | -1.02 |
| BPS+ROS-R (α=100) | 0.374 | 0.265 | 0.262 | -1.02 |
| BPS + Sobol RQMC, B | 0.219 | 0.21 | 0.167 | -1.06 |
| BPS + Sobol RQMC, R | 0.585 | 0.233 | 0.18 | -1.07 |
| BPS + Latin hypercube, B | 0.262 | 0.278 | 0.247 | – |
| BPS + antithetic, B | 0.28 | 0.272 | 0.281 | -1.01 |
| BPS + per-state QMC, B | 1.07 | 7.24 | 114 | -0.00 |

### πe-B, slippery

| method | n = 64 | n = 512 | n = 8192 | slope |
|---|---|---|---|---|
| OS-MC | 0.916 | 1.01 | 1.08 | -0.95 |
| ROS-MC (α=1e+06) | 0.205 | 0.209 | 0.218 | -0.99 |
| per-state QMC, MC | 0.271 | 0.232 | 0.21 | -1.03 |
| Sobol RQMC, MC | 0.751 | 0.763 | 0.737 | -1.03 |
| Latin hypercube, MC | 0.815 | 0.861 | 0.821 | – |
| antithetic, MC | 0.902 | 0.953 | 0.827 | -1.07 |
| ROS + Sobol RQMC, MC | 0.243 | 0.209 | 0.249 | -0.92 |
| BPS-OIS | 0.408 | 0.388 | 0.388 | -1.04 |
| BPS+ROS-R (α=100) | 0.535 | 0.374 | 0.39 | -1.00 |
| BPS + Sobol RQMC, B | 0.353 | 0.384 | 0.333 | -1.06 |
| BPS + Sobol RQMC, R | 2.05 | 0.413 | 0.335 | -1.07 |
| BPS + Latin hypercube, B | 0.373 | 0.383 | 0.417 | – |
| BPS + antithetic, B | 0.403 | 0.391 | 0.365 | -1.05 |
| BPS + per-state QMC, B | 0.923 | 5.2 | 78.2 | -0.02 |
