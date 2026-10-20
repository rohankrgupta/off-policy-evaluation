# A behavior policy for coordinated sampling (J target), GridWorld

## Paired MSE ratios at n = 64 / 512 / 8192

✓ the first method is better, ✗ worse, ~ inconclusive (95% CI).

| domain | comparison | n = 64 | n = 512 | n = 8192 |
|---|---|---|---|---|
| slip 0.1 | Array-RQMC, J target, PDIS vs BPS-OIS (i.i.d.) | 0.981 ~ | 0.635 ✓ | 0.51 ✓ |
| slip 0.1 | Array-RQMC, J target, PDIS vs BPS+ROS (Phase 2) | 0.968 ~ | 0.67 ✓ | 0.508 ✓ |
| slip 0.1 | Array-RQMC, BPS target, PDIS vs BPS-OIS (i.i.d.) | 0.97 ~ | 0.687 ✓ | 0.571 ✓ |
| slip 0.1 | Array-RQMC, BPS target, PDIS vs BPS+ROS (Phase 2) | 0.958 ~ | 0.725 ✓ | 0.568 ✓ |
| slip 0.1 | Array-RQMC, J target, PDIS vs Array-RQMC, BPS target, PDIS | 1.01 ~ | 0.924 ~ | 0.894 ~ |
| slip 0.1 | Array-RQMC, J target, PDIS vs Array-RQMC, J target, OIS | 0.998 ~ | 0.825 ✓ | 0.702 ✓ |
| slip 0.1 | Array-RQMC, J target, PDIS vs Array-RQMC, on-policy, MC | 0.812 ✓ | 0.845 ✓ | 0.846 ✓ |
| slip 0.1 | Array-RQMC, J target, PDIS vs ROS-MC (α tuned at each n) | 1.96 ✗ | 1.12 ~ | 0.906 ~ |
| slip 0.1 | Array-RQMC, BPS target, PDIS vs ROS-MC (α tuned at each n) | 1.94 ✗ | 1.21 ✗ | 1.01 ~ |
| slip 0.2 | Array-RQMC, J target, PDIS vs BPS-OIS (i.i.d.) | 1.01 ~ | 0.811 ✓ | 0.627 ✓ |
| slip 0.2 | Array-RQMC, J target, PDIS vs BPS+ROS (Phase 2) | 1 ~ | 0.86 ✓ | 0.649 ✓ |
| slip 0.2 | Array-RQMC, BPS target, PDIS vs BPS-OIS (i.i.d.) | 0.998 ~ | 0.797 ✓ | 0.638 ✓ |
| slip 0.2 | Array-RQMC, BPS target, PDIS vs BPS+ROS (Phase 2) | 0.989 ~ | 0.845 ✓ | 0.66 ✓ |
| slip 0.2 | Array-RQMC, J target, PDIS vs Array-RQMC, BPS target, PDIS | 1.02 ~ | 1.02 ~ | 0.983 ~ |
| slip 0.2 | Array-RQMC, J target, PDIS vs Array-RQMC, J target, OIS | 1.02 ~ | 0.887 ✓ | 0.854 ✓ |
| slip 0.2 | Array-RQMC, J target, PDIS vs Array-RQMC, on-policy, MC | 0.868 ✓ | 0.882 ✓ | 0.866 ✓ |
| slip 0.2 | Array-RQMC, J target, PDIS vs ROS-MC (α tuned at each n) | 1.39 ✗ | 1.04 ~ | 0.946 ~ |
| slip 0.2 | Array-RQMC, BPS target, PDIS vs ROS-MC (α tuned at each n) | 1.37 ✗ | 1.03 ~ | 0.962 ~ |
| slip 0.1 + noisy trap | Array-RQMC, J target, PDIS vs BPS-OIS (i.i.d.) | 0.927 ~ | 0.727 ✓ | 0.631 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, J target, PDIS vs BPS+ROS (Phase 2) | 0.859 ✓ | 0.746 ✓ | 0.642 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, BPS target, PDIS vs BPS-OIS (i.i.d.) | 0.921 ~ | 0.793 ✓ | 0.646 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, BPS target, PDIS vs BPS+ROS (Phase 2) | 0.854 ✓ | 0.813 ✓ | 0.658 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, J target, PDIS vs Array-RQMC, BPS target, PDIS | 1.01 ~ | 0.918 ~ | 0.977 ~ |
| slip 0.1 + noisy trap | Array-RQMC, J target, PDIS vs Array-RQMC, J target, OIS | 0.817 ✓ | 0.649 ✓ | 0.609 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, J target, PDIS vs Array-RQMC, on-policy, MC | 0.81 ✓ | 0.754 ✓ | 0.755 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, J target, PDIS vs ROS-MC (α tuned at each n) | 1.17 ✗ | 0.893 ~ | 0.722 ✓ |
| slip 0.1 + noisy trap | Array-RQMC, BPS target, PDIS vs ROS-MC (α tuned at each n) | 1.17 ✗ | 0.973 ~ | 0.739 ✓ |

## n·MSE/Var(g)

On-policy Monte Carlo = 1; lower is better. 1000 seeds. Predicted: J(q)/Var(g), the first-order value under perfect coordination.

### slip 0.1

Mean episode length: πe 19.3, BPS target 26.2, J target 27.9.

| method | predicted | n = 64 | n = 512 | n = 8192 | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | – | 0.204 | 0.22 | 0.218 | -2.2 |
| BPS-OIS (i.i.d.) | – | 0.408 | 0.388 | 0.388 | -0.1 |
| BPS+ROS (Phase 2) | – | 0.413 | 0.368 | 0.39 | -1.1 |
| Array-RQMC, on-policy, MC | 0.219 | 0.493 | 0.292 | 0.234 | -0.6 |
| Array-RQMC, BPS target, OIS | 0.200 | 0.318 | 0.257 | 0.248 | -0.4 |
| Array-RQMC, BPS target, PDIS | 0.200 | 0.396 | 0.267 | 0.221 | -0.7 |
| Array-RQMC, J target, OIS | 0.188 | 0.401 | 0.299 | 0.282 | -0.5 |
| Array-RQMC, J target, PDIS | 0.188 | 0.4 | 0.247 | 0.198 | -1.3 |

### slip 0.2

Mean episode length: πe 21.4, BPS target 26.6, J target 27.8.

| method | predicted | n = 64 | n = 512 | n = 8192 | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | – | 0.41 | 0.405 | 0.379 | -1.3 |
| BPS-OIS (i.i.d.) | – | 0.562 | 0.52 | 0.571 | -0.6 |
| BPS+ROS (Phase 2) | – | 0.567 | 0.491 | 0.552 | -8.9 |
| Array-RQMC, on-policy, MC | 0.403 | 0.656 | 0.478 | 0.414 | -0.9 |
| Array-RQMC, BPS target, OIS | 0.372 | 0.512 | 0.411 | 0.386 | -0.1 |
| Array-RQMC, BPS target, PDIS | 0.372 | 0.561 | 0.415 | 0.364 | -0.6 |
| Array-RQMC, J target, OIS | 0.359 | 0.56 | 0.476 | 0.419 | +0.3 |
| Array-RQMC, J target, PDIS | 0.359 | 0.57 | 0.422 | 0.358 | +0.6 |

### slip 0.1 + noisy trap

Mean episode length: πe 19.3, BPS target 25.3, J target 33.1.

| method | predicted | n = 64 | n = 512 | n = 8192 | bias/SE (n = 8192) |
|---|---|---|---|---|---|
| ROS-MC (α tuned at each n) | – | 0.427 | 0.416 | 0.471 | -2.2 |
| BPS-OIS (i.i.d.) | – | 0.541 | 0.511 | 0.539 | -0.1 |
| BPS+ROS (Phase 2) | – | 0.584 | 0.499 | 0.53 | -1.5 |
| Array-RQMC, on-policy, MC | 0.435 | 0.619 | 0.493 | 0.45 | -0.5 |
| Array-RQMC, BPS target, OIS | 0.344 | 0.459 | 0.446 | 0.394 | -1.2 |
| Array-RQMC, BPS target, PDIS | 0.344 | 0.498 | 0.405 | 0.348 | -0.5 |
| Array-RQMC, J target, OIS | 0.328 | 0.614 | 0.573 | 0.558 | -0.6 |
| Array-RQMC, J target, PDIS | 0.328 | 0.501 | 0.372 | 0.34 | -0.7 |
