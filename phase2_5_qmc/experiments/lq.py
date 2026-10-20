"""QMC vs ROS vs PROPS with continuous actions: linear-quadratic control with a Gaussian policy (core/lq.py).

Configurations (state dimension d, horizon H, environment noise sigma_w): (1, 10, 0), (1, 10, 0.1), (1, 10, 0.3),
(1, 50, 0.1), (4, 50, 0.1). With sigma_w = 0 all randomness is the agent's; sigma_w > 0 adds an uncontrollable floor.

Methods, all scored by Monte Carlo (on-policy): i.i.d., Sobol RQMC, antithetic, Array-RQMC, ROS and PROPS.
ROS's alpha and PROPS's learning rate are tuned at each n on 500 other seeds. n up to 4096; 1000 seeds.
Reported as in experiments/array_rqmc.py: n·MSE/Var(g), paired MSE ratios, bias/SE and the coverage of 95% intervals
from groups of 10 independent runs.

Run:    python -m phase2_5_qmc.experiments.lq     (about 1 hour)
Output: results/lq.md
"""
from pathlib import Path

import numpy as np

from phase1_reproduction.core import stats
from phase1_reproduction.core.rng import stream_key
from phase2_5_qmc.core import lq as L
from phase2_5_qmc.experiments.array_rqmc import ci_stats
from phase2_5_qmc.experiments.gridworld import md

OUT = Path(__file__).resolve().parents[1] / "results"
CONFIGS = [(1, 10, 0.0), (1, 10, 0.1), (1, 10, 0.3), (1, 50, 0.1), (4, 50, 0.1)]
NS = 2 ** np.arange(4, 13)                      # n = 16 ... 4096
SHOW = (64, 512, 4096)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)          # ROS
LRS = (0.001, 0.003, 0.01, 0.03)                # PROPS
N_EVAL, N_TUNE = 1000, 500
EVAL, TUNE = stream_key("lq"), stream_key("lq-tune")


def prefix_means(returns):
    """Estimates at each n in NS from per-trajectory returns (seeds, n_max)."""
    return (np.cumsum(returns, 1) / np.arange(1, returns.shape[1] + 1))[:, NS - 1]


def tuned_per_n(lq, v, method, params, kw):
    """Estimates (seeds, len(NS)) using, at each n, the parameter with the lowest MSE on the tuning seeds."""
    tune = {p: np.nan_to_num(((prefix_means(L.adaptive(lq, NS[-1], method, n_seeds=N_TUNE, key=TUNE, **{kw: p})) - v) ** 2)
                             .mean(0), nan=np.inf) for p in params}             # a diverged setting never wins
    best = [min(params, key=lambda p: tune[p][i]) for i in range(len(NS))]
    evals = {p: prefix_means(L.adaptive(lq, NS[-1], method, n_seeds=N_EVAL, key=EVAL, **{kw: p})) for p in set(best)}
    return np.stack([evals[p][:, i] for i, p in enumerate(best)], 1), best


def run_config(d, H, sigma_w):
    lq = L.make_lq(d, H, sigma_w)
    v = L.value(lq)
    iid_returns = L.adaptive(lq, NS[-1], "ros", n_seeds=N_EVAL, key=EVAL)
    var = iid_returns.var()
    sobol, anti = np.empty((N_EVAL, len(NS))), np.empty((N_EVAL, len(NS)))
    array = np.full((N_EVAL, len(NS)), np.nan)
    for s in range(N_EVAL):
        eps, w = L.noise(EVAL, s, NS[-1], H, d)
        sobol[s] = prefix_means(L.rollout(lq, L.sobol_eps(NS[-1], H, s), w)[None])[0]
        anti_eps = np.stack([eps[: NS[-1] // 2], -eps[: NS[-1] // 2]], 1).reshape(NS[-1], H)
        anti[s] = prefix_means(L.rollout(lq, anti_eps, w)[None])[0]
        for n in SHOW:                          # Array-RQMC: one lockstep design per n
            array[s, list(NS).index(n)] = L.array_rqmc(lq, w[:n], s, EVAL).mean()
    ros, ros_alpha = tuned_per_n(lq, v, "ros", ALPHAS, "alpha")
    props, props_lr = tuned_per_n(lq, v, "props", LRS, "lr")
    methods = {"i.i.d.": prefix_means(iid_returns), "Sobol RQMC": sobol, "antithetic": anti, "Array-RQMC": array,
               "ROS (α tuned at each n)": ros, "PROPS (lr tuned at each n)": props}
    name = f"d = {d}, H = {H}, σ_w = {sigma_w:g}"
    print(f"{name} done (ROS α {ros_alpha}, PROPS lr {props_lr})", flush=True)
    return {"name": name, "v": v, "var": var, "methods": methods}


def main():
    OUT.mkdir(exist_ok=True)
    results = [run_config(*c) for c in CONFIGS]
    cols = [list(NS).index(n) for n in SHOW]
    mark = lambda lo, hi: "✓" if hi < 1 else "✗" if lo > 1 else "~"
    acc, contrasts, practical = [], [], []
    for D in results:
        M, v = D["methods"], D["v"]
        rows = []
        for k, e in M.items():
            y = np.array(SHOW) * ((e[:, cols] - v) ** 2).mean(0) / D["var"]
            rows.append([k, *(f"{x:.3g}" for x in y), f"{np.log(y[2] / y[1]) / np.log(SHOW[2] / SHOW[1]) - 1:+.2f}"])
        acc.append(f"### {D['name']}\n\n" + md(["method", *(f"n = {n}" for n in SHOW), "slope (512 → 4096)"], rows))
        for x, y in (("Sobol RQMC", "i.i.d."), ("Array-RQMC", "i.i.d."), ("Sobol RQMC", "ROS (α tuned at each n)"),
                     ("Array-RQMC", "ROS (α tuned at each n)"), ("Sobol RQMC", "PROPS (lr tuned at each n)"),
                     ("Array-RQMC", "PROPS (lr tuned at each n)"), ("ROS (α tuned at each n)", "PROPS (lr tuned at each n)")):
            cells = []
            for c in cols:
                r, lo, hi = stats.paired_mse_ratio(M[x][:, c], M[y][:, c], v)
                cells.append(f"{r:.3g} {mark(lo, hi)}")
            contrasts.append([D["name"], f"{x} vs {y}", *cells])
        for k in M:
            row = [D["name"], k]
            for c in (cols[0], cols[2]):
                e = M[k][:, c]
                ref = 2.262 * M["i.i.d."][:, c].std(ddof=1) / np.sqrt(10)      # t(0.975, 9) half-width of i.i.d.
                cov, width = ci_stats(e, v, ref)
                row += [f"{(e - v).mean() / (e.std(ddof=1) / np.sqrt(len(e))):+.1f}", f"{cov:.0%}", f"{width:.3g}"]
            practical.append(row)
    text = ("# QMC vs ROS vs PROPS with continuous actions (linear-quadratic control)\n\n"
            "## Paired MSE ratios at n = 64 / 512 / 4096\n\n✓ the first method is better, ✗ worse, ~ inconclusive "
            "(95% CI).\n\n" + md(["configuration", "comparison", *(f"n = {n}" for n in SHOW)], contrasts)
            + "\n\n## Bias and error bars\n\nbias/SE over 1000 seeds; coverage of 95% t-intervals from groups of 10 "
            "independent runs, and their mean width relative to i.i.d. sampling's.\n\n"
            + md(["configuration", "method", "bias/SE, n = 64", "coverage", "width", "bias/SE, n = 4096", "coverage",
                  "width"], practical)
            + "\n\n## n·MSE/Var(g)\n\ni.i.d. sampling = 1; lower is better. Slope of log MSE against log n between "
            "n = 512 and 4096 (−1 is the Monte Carlo rate).\n\n" + "\n\n".join(acc) + "\n")
    (OUT / "lq.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
