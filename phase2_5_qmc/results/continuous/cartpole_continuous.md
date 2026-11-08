# Coordinated BPS in a continuous state space: CartPole (continuous actions)

n·MSE/Var(g): on-policy Monte Carlo = 1, lower is better; 1000 seeds (stored repo rows: 200). 'Equal steps' rescales by the method's mean episode length, i.e. compares at equal environment steps. Coverage: 95% t-intervals from groups of 10 runs. Time: µs per episode at n = 4096 on 10 cores.

## Force noise 0 N

Truth v = 35.4117 ± 0.0037, Var(g) = 386.2, mean length 49.7. BPS target (Part A): Var_q(OIS)/Var(g) = 0.414, mean length 61.7; learned offline in 47 s from 3.2e6 episodes (6e5 for the gradient steps, 2.6e6 for validation; not counted in Part A).

### Part A: BPS target given

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.969 | 1.1 | 1.02 | 1.02 | -0.3 | 96% | 1.00 | 5.6 | – |
| ROS-MC | 0.885 | 0.842 | 0.638 | 0.638 | -0.4 | 95% | 0.79 | 22.0 | 0.03, 0.1, 0.3 |
| BPS + OIS (i.i.d.) | 0.42 | 0.433 | 0.409 | 0.509 | -1.2 | 98% | 0.64 | 12.7 | – |
| BPS + PDIS (i.i.d.) | 0.545 | 0.598 | 0.564 | 0.701 | -1.4 | 98% | 0.76 | 12.7 | – |
| Array-RQMC, on-policy (MC) | 0.885 | 0.803 | 0.915 | 0.915 | -1.2 | 92% | 0.94 | 6.3 | – |
| BPS + Array-RQMC, OIS | 0.365 | 0.373 | 0.341 | 0.424 | -1.2 | 95% | 0.58 | 14.3 | – |
| BPS + Array-RQMC, PDIS | 0.465 | 0.517 | 0.453 | 0.564 | -1.2 | 96% | 0.67 | 14.3 | – |
| BPS + Array-RQMC, PDIS, merged pairs | 0.541 | 0.544 | 0.729 | 0.906 | +24.9 | 40% | 0.66 | 14.1 | – |

### Part B: every episode counted

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.969 | 1.1 | 1.02 | 1.02 | -0.3 | 96% | 1.00 | 5.6 | – |
| ROS-MC | 0.885 | 0.842 | 0.638 | 0.638 | -0.4 | 95% | 0.79 | 22.0 | 0.03, 0.1, 0.3 |
| BPS online, OIS [1] | 0.963 | 1.09 | 0.935 | 0.957 | -0.7 | 97% | 0.96 | 29.4 | 2e-06, 1e-06, 1e-06 |
| BPS online, then Array-RQMC, OIS | 0.926 | 0.948 | 9.6 | 9.91 | -1.6 | 96% | 1.31 | 23.6 | 1.25e-07, 2e-06, 2e-06 |
| OS-MC (ROS repo, stored) | 0.921 | 0.865 | 0.978 | 0.978 | -1.9 | 100% | 1.00 | – | – |
| ROS-MC, lr 0.1 (ROS repo, stored) | 0.796 | 0.714 | 0.73 | 0.73 | -0.4 | 95% | 0.87 | – | – |
| BPS online, OIS (ROS repo's BPG, stored) | 0.872 | 0.918 | 0.871 | 0.871 | -1.6 | 90% | 0.93 | – | – |

## Force noise 3 N

Truth v = 34.5711 ± 0.0034, Var(g) = 373.3, mean length 48.0. BPS target (Part A): Var_q(OIS)/Var(g) = 0.439, mean length 59.2; learned offline in 63 s from 3.2e6 episodes (6e5 for the gradient steps, 2.6e6 for validation; not counted in Part A).

### Part A: BPS target given

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.938 | 1.02 | 1.05 | 1.05 | +0.3 | 97% | 1.00 | 7.4 | – |
| ROS-MC | 0.853 | 0.816 | 0.674 | 0.674 | +0.4 | 98% | 0.80 | 27.8 | 0.03, 0.1, 0.3 |
| BPS + OIS (i.i.d.) | 0.443 | 0.447 | 0.42 | 0.519 | -0.5 | 99% | 0.63 | 17.2 | – |
| BPS + PDIS (i.i.d.) | 0.546 | 0.582 | 0.596 | 0.735 | +0.1 | 98% | 0.75 | 17.2 | – |
| Array-RQMC, on-policy (MC) | 0.972 | 0.827 | 0.809 | 0.809 | -0.2 | 95% | 0.87 | 8.0 | – |
| BPS + Array-RQMC, OIS | 0.371 | 0.371 | 0.392 | 0.484 | +0.1 | 93% | 0.59 | 19.2 | – |
| BPS + Array-RQMC, PDIS | 0.516 | 0.51 | 0.508 | 0.626 | -0.1 | 94% | 0.68 | 19.2 | – |
| BPS + Array-RQMC, PDIS, merged pairs | 0.597 | 0.543 | 0.743 | 0.917 | +21.6 | 46% | 0.68 | 18.1 | – |

### Part B: every episode counted

| method | n = 64 | n = 512 | n = 4096 | equal steps, n = 4096 | bias/SE | coverage | width | µs/episode | tuned (per n) |
|---|---|---|---|---|---|---|---|---|---|
| OS-MC | 0.938 | 1.02 | 1.05 | 1.05 | +0.3 | 97% | 1.00 | 7.4 | – |
| ROS-MC | 0.853 | 0.816 | 0.674 | 0.674 | +0.4 | 98% | 0.80 | 27.8 | 0.03, 0.1, 0.3 |
| BPS online, OIS [1] | 0.94 | 1 | 0.95 | 0.971 | +0.3 | 97% | 0.95 | 22.4 | 2e-06, 1e-06, 1e-06 |
| BPS online, then Array-RQMC, OIS | 0.957 | 0.988 | 0.882 | 0.898 | +1.0 | 93% | 0.90 | 18.6 | 2e-06, 2.5e-07, 1e-06 |

## Paired MSE ratios per episode

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| force noise | comparison | n = 64 | n = 512 | n = 4096 |
|---|---|---|---|---|
| 0 N | BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 0.869 ✓ | 0.861 ✓ | 0.834 ✓ |
| 0 N | BPS + Array-RQMC, OIS vs ROS-MC | 0.413 ✓ | 0.443 ✓ | 0.535 ✓ |
| 0 N | BPS + Array-RQMC, PDIS vs BPS + PDIS (i.i.d.) | 0.853 ✓ | 0.865 ✓ | 0.804 ✓ |
| 0 N | BPS + Array-RQMC, PDIS, merged pairs vs BPS + Array-RQMC, PDIS | 1.16 ✗ | 1.05 ✗ | 1.61 ✗ |
| 0 N | Array-RQMC, on-policy (MC) vs OS-MC | 0.914 ~ | 0.729 ✓ | 0.898 ~ |
| 0 N | Array-RQMC, on-policy (MC) vs ROS-MC | 1 ~ | 0.953 ~ | 1.43 ✗ |
| 0 N | BPS online, then Array-RQMC, OIS vs BPS online, OIS [1] | 0.961 ~ | 0.872 ✓ | 10.3 ~ |
| 0 N | BPS online, then Array-RQMC, OIS vs ROS-MC | 1.05 ~ | 1.12 ✗ | 15 ✗ |
| 0 N | BPS online, OIS [1] vs ROS-MC | 1.09 ✗ | 1.29 ✗ | 1.47 ✗ |
| 3 N | BPS + Array-RQMC, OIS vs BPS + OIS (i.i.d.) | 0.837 ✓ | 0.83 ✓ | 0.932 ~ |
| 3 N | BPS + Array-RQMC, OIS vs ROS-MC | 0.435 ✓ | 0.455 ✓ | 0.582 ✓ |
| 3 N | BPS + Array-RQMC, PDIS vs BPS + PDIS (i.i.d.) | 0.945 ~ | 0.876 ✓ | 0.852 ✓ |
| 3 N | BPS + Array-RQMC, PDIS, merged pairs vs BPS + Array-RQMC, PDIS | 1.16 ✗ | 1.07 ✗ | 1.46 ✗ |
| 3 N | Array-RQMC, on-policy (MC) vs OS-MC | 1.04 ~ | 0.81 ✓ | 0.77 ✓ |
| 3 N | Array-RQMC, on-policy (MC) vs ROS-MC | 1.14 ✗ | 1.01 ~ | 1.2 ✗ |
| 3 N | BPS online, then Array-RQMC, OIS vs BPS online, OIS [1] | 1.02 ~ | 0.988 ~ | 0.929 ~ |
| 3 N | BPS online, then Array-RQMC, OIS vs ROS-MC | 1.12 ~ | 1.21 ✗ | 1.31 ✗ |
| 3 N | BPS online, OIS [1] vs ROS-MC | 1.1 ✗ | 1.23 ✗ | 1.41 ✗ |
