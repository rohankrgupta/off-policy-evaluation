"""Bootstrap confidence intervals and convergence rates."""
import numpy as np


def bootstrap_ci(per_seed, level=0.95, n_boot=10000, seed=0):
    """Percentile bootstrap CI of the mean over seeds (axis 0), for every column."""
    x = np.asarray(per_seed, float)
    idx = np.random.default_rng(seed).integers(0, x.shape[0], (n_boot, x.shape[0]))
    means = x[idx].mean(1)
    return tuple(np.quantile(means, [(1 - level) / 2, (1 + level) / 2], axis=0))


def slope(ns, mse, last=4):
    """Least-squares slope of log MSE against log n over the last `last` checkpoints
    (-1 is the usual Monte Carlo rate)."""
    return float(np.polyfit(np.log(np.asarray(ns, float)[-last:]), np.log(np.asarray(mse, float)[-last:]), 1)[0])
