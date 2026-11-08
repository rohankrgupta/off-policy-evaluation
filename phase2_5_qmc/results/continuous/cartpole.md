# Coordinated BPS in a continuous state space: CartPole

n·MSE/Var(g): on-policy Monte Carlo = 1, lower is better; 1000 seeds (stored repo rows: 200). 'Equal steps' rescales by the method's mean episode length, i.e. compares at equal environment steps. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = 4096 on 10 cores.

## Force noise 0 N

Truth v = 36.0234 ± 0.0024, Var(g) = 268.5, mean length 48.4. BPS target (Part A): Var_q(OIS)/Var(g) = 0.247, mean length 59.4; learned offline in 46 s from 3.2e6 episodes (6e5 for the gradient steps, 2.6e6 for validation; not counted in Part A).

### Part A: BPS target given

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.965 | 1.09 | 1 | 1 | +0.4 | 97% | 1.00 | 5.2 | – |
| ROS-MC | 0.383 | 0.311 | 0.265 | 0.265 | -3.7 | 96% | 0.52 | 20.3 | 10, 30, 30 |
| BPS + OIS (i.i.d.) | 0.254 | 0.252 | 0.251 | 0.308 | -0.2 | 96% | 0.50 | 12.0 | – |
| BPS + PDIS (i.i.d.) | 0.404 | 0.428 | 0.413 | 0.507 | +1.1 | 98% | 0.64 | 12.0 | – |
| Array-RQMC, on-policy (MC) | 0.803 | 0.695 | 0.741 | 0.741 | -1.2 | 94% | 0.86 | 5.8 | – |
| BPS + Array-RQMC, OIS | 0.167 | 0.158 | 0.158 | 0.193 | -0.2 | 94% | 0.39 | 12.7 | – |
| BPS + Array-RQMC, PDIS | 0.313 | 0.312 | 0.294 | 0.361 | +0.2 | 96% | 0.54 | 12.7 | – |
| BPS + Array-RQMC, PDIS, merged pairs | 0.424 | 0.411 | 0.911 | 1.12 | +43.7 | 3% | 0.56 | 13.3 | – |

### Part B: every episode counted

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.965 | 1.09 | 1 | 1 | +0.4 | 97% | 1.00 | 5.2 | – |
| ROS-MC | 0.383 | 0.311 | 0.265 | 0.265 | -3.7 | 96% | 0.52 | 20.3 | 10, 30, 30 |
| BPS online, OIS [1] | 1.03 | 0.836 | 0.472 | 0.536 | +0.2 | 96% | 0.69 | 18.2 | 0.0008, 0.0002, 0.0002 |
| BPS online, then Array-RQMC, OIS | 0.901 | 0.676 | 0.433 | 0.489 | -0.1 | 92% | 0.65 | 14.5 | 5e-05, 0.0004, 0.0002 |
| OS-MC (ROS repo, stored) | 1.01 | 1.05 | 1.02 | 1.02 | +1.9 | 100% | 1.04 | – | – |
| ROS-MC, lr 10 (ROS repo, stored) | 0.4 | 0.347 | 0.342 | 0.342 | -2.0 | 95% | 0.58 | – | – |
| BPS online, OIS (ROS repo's BPG, stored) | 1.06 | 0.984 | 0.643 | 0.643 | +2.5 | 90% | 0.79 | – | – |

## Force noise 3 N

Truth v = 34.8234 ± 0.0024, Var(g) = 262.1, mean length 46.4. BPS target (Part A): Var_q(OIS)/Var(g) = 0.322, mean length 56.2; learned offline in 42 s from 3.2e6 episodes (6e5 for the gradient steps, 2.6e6 for validation; not counted in Part A).

### Part A: BPS target given

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.897 | 1.09 | 1.07 | 1.07 | +1.0 | 98% | 1.00 | 5.0 | – |
| ROS-MC | 0.42 | 0.372 | 0.341 | 0.341 | -2.1 | 96% | 0.56 | 19.4 | 10, 30, 30 |
| BPS + OIS (i.i.d.) | 0.325 | 0.317 | 0.328 | 0.397 | +1.5 | 97% | 0.56 | 11.8 | – |
| BPS + PDIS (i.i.d.) | 0.483 | 0.481 | 0.462 | 0.56 | +0.7 | 98% | 0.66 | 11.8 | – |
| Array-RQMC, on-policy (MC) | 0.811 | 0.791 | 0.787 | 0.787 | -1.2 | 94% | 0.85 | 5.6 | – |
| BPS + Array-RQMC, OIS | 0.267 | 0.256 | 0.242 | 0.293 | -0.3 | 97% | 0.47 | 12.4 | – |
| BPS + Array-RQMC, PDIS | 0.392 | 0.35 | 0.374 | 0.453 | -0.8 | 98% | 0.59 | 12.4 | – |
| BPS + Array-RQMC, PDIS, merged pairs | 0.42 | 0.39 | 0.673 | 0.816 | +29.0 | 30% | 0.58 | 12.8 | – |

### Part B: every episode counted

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.897 | 1.09 | 1.07 | 1.07 | +1.0 | 98% | 1.00 | 5.0 | – |
| ROS-MC | 0.42 | 0.372 | 0.341 | 0.341 | -2.1 | 96% | 0.56 | 19.4 | 10, 30, 30 |
| BPS online, OIS [1] | 0.883 | 0.801 | 0.54 | 0.609 | +0.5 | 100% | 0.72 | 18.6 | 0.0008, 0.0004, 0.0002 |
| BPS online, then Array-RQMC, OIS | 0.944 | 0.718 | 0.459 | 0.515 | -0.1 | 95% | 0.65 | 14.4 | 0.0008, 0.0004, 0.0002 |

## Paired MSE ratios per episode

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| force noise | comparison | n = 64 | n = 512 | n = 4096 |
|---|---|---|---|---|
| 0 N | BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 0.659 ✓ | 0.629 ✓ | 0.628 ✓ |
| 0 N | BPS + Array-RQMC, OIS vs ROS-MC | 0.436 ✓ | 0.508 ✓ | 0.595 ✓ |
| 0 N | BPS + Array-RQMC, PDIS vs BPS + PDIS (i.i.d.) | 0.775 ✓ | 0.73 ✓ | 0.712 ✓ |
| 0 N | BPS + Array-RQMC, PDIS, merged pairs vs BPS + Array-RQMC, PDIS | 1.36 ✗ | 1.32 ✗ | 3.1 ✗ |
| 0 N | Array-RQMC, on-policy (MC) vs OS-MC | 0.832 ✓ | 0.639 ✓ | 0.739 ✓ |
| 0 N | BPS online, then Array-RQMC, OIS vs BPS online, OIS [1] | 0.876 ~ | 0.809 ✓ | 0.918 ~ |
| 0 N | BPS online, then Array-RQMC, OIS vs ROS-MC | 2.35 ✗ | 2.17 ✗ | 1.63 ✗ |
| 0 N | BPS online, OIS [1] vs ROS-MC | 2.68 ✗ | 2.68 ✗ | 1.78 ✗ |
| 3 N | BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 0.822 ✓ | 0.808 ✓ | 0.738 ✓ |
| 3 N | BPS + Array-RQMC, OIS vs ROS-MC | 0.637 ✓ | 0.688 ✓ | 0.708 ✓ |
| 3 N | BPS + Array-RQMC, PDIS vs BPS + PDIS (i.i.d.) | 0.811 ✓ | 0.727 ✓ | 0.808 ✓ |
| 3 N | BPS + Array-RQMC, PDIS, merged pairs vs BPS + Array-RQMC, PDIS | 1.07 ✗ | 1.11 ✗ | 1.8 ✗ |
| 3 N | Array-RQMC, on-policy (MC) vs OS-MC | 0.904 ~ | 0.724 ✓ | 0.732 ✓ |
| 3 N | BPS online, then Array-RQMC, OIS vs BPS online, OIS [1] | 1.07 ~ | 0.897 ~ | 0.851 ✓ |
| 3 N | BPS online, then Array-RQMC, OIS vs ROS-MC | 2.25 ✗ | 1.93 ✗ | 1.35 ✗ |
| 3 N | BPS online, OIS [1] vs ROS-MC | 2.1 ✗ | 2.15 ✗ | 1.58 ✗ |
