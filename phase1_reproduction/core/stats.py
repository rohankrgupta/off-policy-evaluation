"""Bootstrap confidence intervals, paired MSE ratios and convergence rates."""
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


def paired_mse_ratio(est_x, est_y, truth, level=0.95, n_boot=10000, seed=0):
    """MSE(x) / MSE(y) with a paired bootstrap CI over seeds (x and y share seeds and random numbers)."""
    sx, sy = (np.asarray(est_x, float) - truth) ** 2, (np.asarray(est_y, float) - truth) ** 2
    idx = np.random.default_rng(seed).integers(0, len(sx), (n_boot, len(sx)))
    lo, hi = np.quantile(sx[idx].mean(1) / sy[idx].mean(1), [(1 - level) / 2, (1 + level) / 2])
    return sx.mean() / sy.mean(), lo, hi
