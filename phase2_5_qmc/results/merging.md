# Prefix-weight merging and Q-ordered actions (Array-RQMC, BPS target, PDIS)

## Paired MSE ratios at n = 64 / 512 / 8192

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| domain | comparison | n = 64 | n = 512 | n = 8192 |
|---|---|---|---|---|
| πe-A | Array-RQMC, merged vs Array-RQMC, weight-sorted | 0.947 ✓ | 0.924 ✓ | 0.737 ✓ |
| πe-A | Array-RQMC, merged, Q-ordered vs Array-RQMC, merged | 0.761 ✓ | 0.83 ✓ | 0.697 ✓ |
| πe-A | Array-RQMC, merged vs BPS-OIS (i.i.d.) | 0.296 ✓ | 0.118 ✓ | 0.0143 ✓ |
| πe-A | Array-RQMC, merged vs BPS+ROS (Phase 2) | 23 ✗ | 2.18 ✗ | 0.427 ✓ |
| πe-A | Array-RQMC, merged vs ROS-MC (α tuned at each n) | 4.14 ✗ | 5.04 ✗ | 10.4 ✗ |
| πe-B | Array-RQMC, merged vs Array-RQMC, weight-sorted | 0.813 ✓ | 0.683 ✓ | 0.575 ✓ |
| πe-B | Array-RQMC, merged, Q-ordered vs Array-RQMC, merged | 0.476 ✓ | 0.377 ✓ | 0.448 ✓ |
| πe-B | Array-RQMC, merged vs BPS-OIS (i.i.d.) | 0.859 ✓ | 0.188 ✓ | 0.0171 ✓ |
| πe-B | Array-RQMC, merged vs BPS+ROS (Phase 2) | 0.87 ✓ | 0.188 ✓ | 0.0186 ✓ |
| πe-B | Array-RQMC, merged vs ROS-MC (α tuned at each n) | 17.9 ✗ | 32.4 ✗ | 45.2 ✗ |
| slip 0.1 | Array-RQMC, merged vs Array-RQMC, weight-sorted | 0.898 ~ | 0.804 ✓ | 0.924 ~ |
| slip 0.1 | Array-RQMC, merged, Q-ordered vs Array-RQMC, merged | 0.784 ✓ | 0.949 ~ | 0.934 ~ |
| slip 0.1 | Array-RQMC, merged vs BPS-OIS (i.i.d.) | 0.872 ✓ | 0.552 ✓ | 0.527 ✓ |
| slip 0.1 | Array-RQMC, merged vs BPS+ROS (Phase 2) | 0.861 ✓ | 0.583 ✓ | 0.525 ✓ |
| slip 0.1 | Array-RQMC, merged vs ROS-MC (α tuned at each n) | 1.74 ✗ | 0.975 ~ | 0.936 ~ |
| slip 0.2 | Array-RQMC, merged vs Array-RQMC, weight-sorted | 0.872 ✓ | 0.946 ~ | 0.92 ~ |
| slip 0.2 | Array-RQMC, merged, Q-ordered vs Array-RQMC, merged | 0.908 ~ | 0.921 ~ | 1.06 ~ |
| slip 0.2 | Array-RQMC, merged vs BPS-OIS (i.i.d.) | 0.87 ✓ | 0.754 ✓ | 0.587 ✓ |
| slip 0.2 | Array-RQMC, merged vs BPS+ROS (Phase 2) | 0.862 ✓ | 0.8 ✓ | 0.608 ✓ |
| slip 0.2 | Array-RQMC, merged vs ROS-MC (α tuned at each n) | 1.19 ✗ | 0.97 ~ | 0.886 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, merged vs Array-RQMC, weight-sorted | 0.924 ~ | 0.896 ~ | 1.04 ~ |
| slip 0.1 + noisy trap | Array-RQMC, merged, Q-ordered vs Array-RQMC, merged | 0.831 ✓ | 0.919 ~ | 0.987 ~ |
| slip 0.1 + noisy trap | Array-RQMC, merged vs BPS-OIS (i.i.d.) | 0.851 ✓ | 0.71 ✓ | 0.675 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, merged vs BPS+ROS (Phase 2) | 0.789 ✓ | 0.728 ✓ | 0.687 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, merged vs ROS-MC (α tuned at each n) | 1.08 ~ | 0.872 ✓ | 0.772 ✓ |

## n·MSE/Var(g)

On-policy Monte Carlo = 1; lower is better. 1000 seeds.

### πe-A

Exact first-order n·Var/Var(g) — J (coordination, path weights): 0; MB with the same data: 0.

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | 0.0659 | 0.019 | 0.00124 | -1.98 | -20.5 |
| BPS-OIS (i.i.d.) | 0.92 | 0.812 | 0.901 | -0.96 | +0.5 |
| BPS+ROS (Phase 2) | 0.0118 | 0.0438 | 0.0303 | -1.13 | +6.8 |
| Array-RQMC, weight-sorted | 0.288 | 0.104 | 0.0175 | -1.64 | +0.3 |
| Array-RQMC, weight-sorted, Q-ordered | 0.227 | 0.0853 | 0.0146 | -1.64 | +1.2 |
| Array-RQMC, merged | 0.272 | 0.0958 | 0.0129 | -1.72 | +0.3 |
| Array-RQMC, merged, Q-ordered | 0.207 | 0.0795 | 0.009 | -1.79 | +0.3 |

### πe-B

Exact first-order n·Var/Var(g) — J (coordination, path weights): 0; MB with the same data: 0.

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | 0.0128 | 0.0016 | 0.000109 | -1.97 | -11.9 |
| BPS-OIS (i.i.d.) | 0.265 | 0.275 | 0.289 | -0.98 | +0.9 |
| BPS+ROS (Phase 2) | 0.262 | 0.274 | 0.266 | -1.01 | -0.4 |
| Array-RQMC, weight-sorted | 0.28 | 0.0756 | 0.0086 | -1.78 | +1.8 |
| Array-RQMC, weight-sorted, Q-ordered | 0.155 | 0.0412 | 0.00556 | -1.72 | -0.1 |
| Array-RQMC, merged | 0.228 | 0.0517 | 0.00494 | -1.85 | +1.4 |
| Array-RQMC, merged, Q-ordered | 0.108 | 0.0195 | 0.00222 | -1.78 | +0.1 |

### slip 0.1

Exact first-order n·Var/Var(g) — J (coordination, path weights): 0.2; MB with the same data: 0.163.

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | 0.204 | 0.22 | 0.218 | -1.00 | -2.2 |
| BPS-OIS (i.i.d.) | 0.408 | 0.388 | 0.388 | -1.00 | -0.1 |
| BPS+ROS (Phase 2) | 0.413 | 0.368 | 0.39 | -0.98 | -1.1 |
| Array-RQMC, weight-sorted | 0.396 | 0.267 | 0.221 | -1.07 | -0.7 |
| Array-RQMC, weight-sorted, Q-ordered | 0.339 | 0.232 | 0.205 | -1.04 | +0.3 |
| Array-RQMC, merged | 0.356 | 0.214 | 0.204 | -1.02 | -0.5 |
| Array-RQMC, merged, Q-ordered | 0.279 | 0.204 | 0.191 | -1.02 | -0.5 |

### slip 0.2

Exact first-order n·Var/Var(g) — J (coordination, path weights): 0.372; MB with the same data: 0.326.

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | 0.41 | 0.405 | 0.379 | -1.02 | -1.3 |
| BPS-OIS (i.i.d.) | 0.562 | 0.52 | 0.571 | -0.97 | -0.6 |
| BPS+ROS (Phase 2) | 0.567 | 0.491 | 0.552 | -0.96 | -8.9 |
| Array-RQMC, weight-sorted | 0.561 | 0.415 | 0.364 | -1.05 | -0.6 |
| Array-RQMC, weight-sorted, Q-ordered | 0.443 | 0.381 | 0.4 | -0.98 | -0.2 |
| Array-RQMC, merged | 0.489 | 0.392 | 0.335 | -1.06 | -0.1 |
| Array-RQMC, merged, Q-ordered | 0.444 | 0.361 | 0.357 | -1.00 | +0.5 |

### slip 0.1 + noisy trap

Exact first-order n·Var/Var(g) — J (coordination, path weights): 0.344; MB with the same data: 0.292.

| method | n = 64 | n = 512 | n = 8192 | slope (512 → 8192) | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | 0.427 | 0.416 | 0.471 | -0.96 | -2.2 |
| BPS-OIS (i.i.d.) | 0.541 | 0.511 | 0.539 | -0.98 | -0.1 |
| BPS+ROS (Phase 2) | 0.584 | 0.499 | 0.53 | -0.98 | -1.5 |
| Array-RQMC, weight-sorted | 0.498 | 0.405 | 0.348 | -1.05 | -0.5 |
| Array-RQMC, weight-sorted, Q-ordered | 0.418 | 0.359 | 0.331 | -1.03 | +0.1 |
| Array-RQMC, merged | 0.46 | 0.363 | 0.364 | -1.00 | -0.4 |
| Array-RQMC, merged, Q-ordered | 0.383 | 0.333 | 0.359 | -0.97 | +0.1 |
