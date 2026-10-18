# Model-based estimation with an optimized allocation (GridWorld, equal environment steps)

## Paired MSE ratios at the full budget T [95% CI]

✓ the first method is better, ✗ worse, ~ inconclusive. * = oracle allocation, ^ = learned from stage-1 data (counted in the budget).

| domain | comparison | MSE ratio |
|---|---|---|
| slip 0.1 | ours* vs ReVar* | 0.859 [0.766, 0.960] ✓ |
| slip 0.1 | ours^ vs ReVar^ | 0.979 [0.873, 1.099] ~ |
| slip 0.1 | ours* + ROS vs ours* | 1.044 [0.928, 1.174] ~ |
| slip 0.1 | ours^ + ROS vs ours^ | 0.933 [0.836, 1.037] ~ |
| slip 0.1 | OS-MB vs ROS-MC | 1.078 [0.952, 1.220] ~ |
| slip 0.1 | ours^ vs ROS-MC | 1.038 [0.917, 1.177] ~ |
| slip 0.1 | ours^ + ROS vs ROS-MC | 0.968 [0.858, 1.093] ~ |
| slip 0.2 | ours* vs ReVar* | 0.836 [0.745, 0.937] ✓ |
| slip 0.2 | ours^ vs ReVar^ | 0.931 [0.830, 1.045] ~ |
| slip 0.2 | ours* + ROS vs ours* | 1.028 [0.926, 1.140] ~ |
| slip 0.2 | ours^ + ROS vs ours^ | 0.964 [0.889, 1.044] ~ |
| slip 0.2 | OS-MB vs ROS-MC | 0.968 [0.873, 1.077] ~ |
| slip 0.2 | ours^ vs ROS-MC | 0.866 [0.773, 0.971] ✓ |
| slip 0.2 | ours^ + ROS vs ROS-MC | 0.835 [0.747, 0.933] ✓ |
| slip 0.1 + noisy trap | ours* vs ReVar* | 0.755 [0.677, 0.845] ✓ |
| slip 0.1 + noisy trap | ours^ vs ReVar^ | 0.813 [0.728, 0.910] ✓ |
| slip 0.1 + noisy trap | ours* + ROS vs ours* | 1.006 [0.904, 1.124] ~ |
| slip 0.1 + noisy trap | ours^ + ROS vs ours^ | 0.954 [0.862, 1.057] ~ |
| slip 0.1 + noisy trap | OS-MB vs ROS-MC | 0.950 [0.861, 1.047] ~ |
| slip 0.1 + noisy trap | ours^ vs ROS-MC | 0.634 [0.563, 0.713] ✓ |
| slip 0.1 + noisy trap | ours^ + ROS vs ROS-MC | 0.605 [0.537, 0.682] ✓ |

## MSE relative to on-policy Monte Carlo at the same number of environment steps

On-policy MC = 1; lower is better. 1000 seeds. Predicted: exact first-order value for oracle allocations.

### slip 0.1

Per-step gain over on-policy MB: at most 1.21× for any behavior (1 + CV²); the oracle allocation achieves 1.14× (ReVar's: 1.01×).

| method | sampling | α | predicted | T/4 | T/2 | T |
|---|---|---|---|---|---|---|
| ROS-MC | ROS | 100000 | – | 0.215 | 0.206 | 0.210 |
| OS-MB | i.i.d. | – | 0.219 | 0.226 | 0.235 | 0.227 |
| ReVar* | i.i.d. | – | 0.216 | 0.212 | 0.237 | 0.233 |
| ours* | i.i.d. | – | 0.192 | 0.193 | 0.203 | 0.200 |
| ours* + ROS | ROS | 100000 | 0.192 | 0.191 | 0.200 | 0.209 |
| ReVar^ | i.i.d. | – | – | 0.207 | 0.206 | 0.223 |
| ours^ | i.i.d. | – | – | 0.200 | 0.210 | 0.219 |
| ours^ + ROS | ROS | 10000 | – | 0.211 | 0.206 | 0.204 |

### slip 0.2

Per-step gain over on-policy MB: at most 1.22× for any behavior (1 + CV²); the oracle allocation achieves 1.15× (ReVar's: 1.01×).

| method | sampling | α | predicted | T/4 | T/2 | T |
|---|---|---|---|---|---|---|
| ROS-MC | ROS | 10000 | – | 0.447 | 0.417 | 0.435 |
| OS-MB | i.i.d. | – | 0.403 | 0.408 | 0.392 | 0.422 |
| ReVar* | i.i.d. | – | 0.399 | 0.398 | 0.399 | 0.413 |
| ours* | i.i.d. | – | 0.351 | 0.348 | 0.350 | 0.345 |
| ours* + ROS | ROS | 10000 | 0.351 | 0.347 | 0.353 | 0.355 |
| ReVar^ | i.i.d. | – | – | 0.383 | 0.394 | 0.405 |
| ours^ | i.i.d. | – | – | 0.346 | 0.349 | 0.377 |
| ours^ + ROS | ROS | 1000 | – | 0.349 | 0.357 | 0.363 |

### slip 0.1 + noisy trap

Per-step gain over on-policy MB: at most 1.87× for any behavior (1 + CV²); the oracle allocation achieves 1.55× (ReVar's: 1.23×).

| method | sampling | α | predicted | T/4 | T/2 | T |
|---|---|---|---|---|---|---|
| ROS-MC | ROS | 1000 | – | 0.459 | 0.452 | 0.482 |
| OS-MB | i.i.d. | – | 0.435 | 0.470 | 0.448 | 0.458 |
| ReVar* | i.i.d. | – | 0.353 | 0.343 | 0.380 | 0.373 |
| ours* | i.i.d. | – | 0.280 | 0.296 | 0.292 | 0.282 |
| ours* + ROS | ROS | 10000 | 0.280 | 0.290 | 0.287 | 0.284 |
| ReVar^ | i.i.d. | – | – | 0.363 | 0.378 | 0.376 |
| ours^ | i.i.d. | – | – | 0.325 | 0.304 | 0.305 |
| ours^ + ROS | ROS | 10000 | – | 0.324 | 0.312 | 0.292 |
