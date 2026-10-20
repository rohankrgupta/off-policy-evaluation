"""Array-RQMC on the GridWorld, against ROS and the other QMC samplers, plus the practical selling points.

Domains, policies and the exact BPS target are as in experiments/gridworld.py. n in {64, 512, 8192}; 1000 seeds.
  * Accuracy: n·MSE/Var(g) for on-policy MC (OS, ROS, per-state QMC, Sobol RQMC, Array-RQMC) and for the BPS
    target with OIS (i.i.d., ROS, Sobol RQMC, Array-RQMC). ROS's alpha is tuned on 500 other seeds at each n.
  * No tuning: ROS with alpha tuned at n = 8192 but used at every n, against ROS tuned at that n.
  * Unbiasedness: bias / SE over 1000 seeds.
  * Error bars: the 1000 seeds form 100 groups of 10 independent runs; each group gives a 95% t-interval for v
    from its 10 estimates. Reported: coverage over the 100 groups, and mean interval width relative to OS-MC's.

Run:    python -m phase2_5_qmc.experiments.array_rqmc     (about 4 min)
Output: results/array_rqmc.md
"""
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from phase1_reproduction.core import exact, sampling, stats
from phase2_5_qmc.core import array_rqmc, qmc
from phase2_5_qmc.experiments.gridworld import ALPHAS, N_EVAL, N_TUNE, NS, domains, md
from phase2_bps_plus_ros.core import bps_ros

OUT = Path(__file__).resolve().parents[1] / "results"
SHOW = (64, 512, 8192)
GROUP = 10                                         # independent runs per confidence interval


def run_domain(name, mdp, th_e):
    pe = exact.softmax(th_e)
    v, var = exact.value(mdp, pe), exact.return_variance(mdp, pe)
    bps = bps_ros.oracle_bps_target(mdp, th_e)
    cols = [list(NS).index(n) for n in SHOW]
    mse = lambda est: ((est - v) ** 2).mean(0)

    # ROS toward pi_e and toward the BPS target, at every alpha (all n in one run), on evaluation and tuning seeds
    ros = {a: sampling.run(mdp, th_e, a, NS, N_EVAL, "e4")[:, cols] for a in ALPHAS}
    ros_tune = {a: mse(sampling.run(mdp, th_e, a, NS, N_TUNE, "tune-e4")[:, cols]) for a in ALPHAS}
    tracked = {a: bps_ros.run(mdp, th_e, bps, a, NS, N_EVAL, "e4") for a in ALPHAS}
    tracked_tune = {a: bps_ros.run(mdp, th_e, bps, a, NS, N_TUNE, "tune-e4") for a in ALPHAS}

    def tuned(evals, tunes, j):                    # estimates at column j with the alpha that tunes best there
        return evals[min(ALPHAS, key=lambda a: tunes[a][j])][:, j]

    columns = {key: [] for key in ("ROS-MC", "ROS-MC, α tuned at n = 8192", "BPS+ROS")}
    a_big = min(ALPHAS, key=lambda a: ros_tune[a][-1])
    for j in range(len(SHOW)):
        columns["ROS-MC"].append(tuned(ros, ros_tune, j))
        columns["ROS-MC, α tuned at n = 8192"].append(ros[a_big][:, j])
        best = {e: tuned({a: tracked[a][e][:, cols] for a in ALPHAS}, {a: mse(tracked_tune[a][e][:, cols]) for a in ALPHAS}, j)
                for e in ("B", "R")}
        columns["BPS+ROS"].append(min(best.values(), key=lambda x: ((x - v) ** 2).mean()))
    per_state = qmc.run(mdp, th_e, th_e, "per-state", 0.0, NS, N_EVAL, "e4")["MC"][:, cols]
    sobol = qmc.run(mdp, th_e, th_e, "sobol", 0.0, NS, N_EVAL, "e4")["MC"][:, cols]
    sobol_bps = qmc.run(mdp, th_e, bps, "sobol", 0.0, NS, N_EVAL, "e4")["B"][:, cols]
    arr = [array_rqmc.run(mdp, th_e, th_e, n, N_EVAL, "e4") for n in SHOW]
    arr_bps = [array_rqmc.run(mdp, th_e, bps, n, N_EVAL, "e4") for n in SHOW]
    methods = {   # label -> estimates (seeds, len(SHOW))
        "OS-MC": sampling.run(mdp, th_e, 0.0, NS, N_EVAL, "e4")[:, cols],
        "ROS-MC (α tuned at each n)": np.stack(columns["ROS-MC"], 1),
        "ROS-MC (α tuned at n = 8192)": np.stack(columns["ROS-MC, α tuned at n = 8192"], 1),
        "per-state QMC, MC": per_state,
        "Sobol RQMC, MC": sobol,
        "Array-RQMC, MC": np.stack([x["MC"] for x in arr], 1),
        "BPS-OIS": bps_ros.run(mdp, th_e, bps, 0.0, NS, N_EVAL, "e4")["B"][:, cols],
        "BPS+ROS (α tuned at each n)": np.stack(columns["BPS+ROS"], 1),
        "BPS + Sobol RQMC, B": sobol_bps,
        "BPS + Array-RQMC, B": np.stack([x["B"] for x in arr_bps], 1),
    }
    print(f"{name} done", flush=True)
    return {"name": name, "v": v, "var": var, "methods": methods}


def ci_stats(est, v, reference_width):
    """Coverage and relative width of 95% t-intervals from groups of GROUP independent runs."""
    groups = est[: len(est) // GROUP * GROUP].reshape(-1, GROUP)
    half = student_t.ppf(0.975, GROUP - 1) * groups.std(1, ddof=1) / np.sqrt(GROUP)
    covered = np.abs(groups.mean(1) - v) <= half
    return covered.mean(), half.mean() / reference_width


def main():
    OUT.mkdir(exist_ok=True)
    results = [run_domain(*d) for d in domains()]
    acc, contrasts, practical = [], [], []
    mark = lambda lo, hi: "✓" if hi < 1 else "✗" if lo > 1 else "~"
    for D in results:
        M, v = D["methods"], D["v"]
        rows = []
        for k, e in M.items():
            y = np.array(SHOW) * ((e - v) ** 2).mean(0) / D["var"]
            rows.append([k, *(f"{x:.3g}" for x in y), f"{np.log(y[2] / y[1]) / np.log(SHOW[2] / SHOW[1]) - 1:+.2f}"])
        acc.append(f"### {D['name']}\n\n" + md(["method", *(f"n = {n}" for n in SHOW), "slope (512 → 8192)"], rows))
        for x, y in (("Array-RQMC, MC", "ROS-MC (α tuned at each n)"), ("Array-RQMC, MC", "Sobol RQMC, MC"),
                     ("BPS + Array-RQMC, B", "BPS + Sobol RQMC, B"), ("BPS + Array-RQMC, B", "BPS+ROS (α tuned at each n)"),
                     ("BPS + Array-RQMC, B", "ROS-MC (α tuned at each n)"),
                     ("ROS-MC (α tuned at n = 8192)", "ROS-MC (α tuned at each n)")):
            cells = []
            for j in range(len(SHOW)):
                r, lo, hi = stats.paired_mse_ratio(M[x][:, j], M[y][:, j], v)
                cells.append(f"{r:.3g} {mark(lo, hi)}")
            contrasts.append([D["name"], f"{x} vs {y}", *cells])
        for k in ("OS-MC", "ROS-MC (α tuned at each n)", "per-state QMC, MC", "Sobol RQMC, MC", "Array-RQMC, MC"):
            row = [D["name"], k]
            for j in (0, 2):
                e = M[k][:, j]
                ref = student_t.ppf(0.975, GROUP - 1) * M["OS-MC"][:, j].std(ddof=1) / np.sqrt(GROUP)
                cov, width = ci_stats(e, v, ref)
                row += [f"{(e - v).mean() / (e.std(ddof=1) / np.sqrt(len(e))):+.1f}", f"{cov:.0%}", f"{width:.3g}"]
            practical.append(row)
    text = ("# Array-RQMC on the GridWorld\n\n"
            "## Paired MSE ratios at n = 64 / 512 / 8192\n\n✓ the first method is better, ✗ worse, ~ inconclusive "
            "(95% CI). BPS = exact BPS target.\n\n"
            + md(["domain", "comparison", *(f"n = {n}" for n in SHOW)], contrasts)
            + "\n\n## Bias and error bars (on-policy MC)\n\n"
            "bias/SE over 1000 seeds (|value| > 2 suggests bias). Coverage of 95% t-intervals built from groups of 10 "
            "independent runs (100 groups), and their mean width relative to OS-MC's.\n\n"
            + md(["domain", "method", "bias/SE, n = 64", "coverage", "width", "bias/SE, n = 8192", "coverage", "width"],
                 practical)
            + "\n\n## n·MSE/Var(g)\n\nOn-policy Monte Carlo = 1; lower is better. Slope of log MSE against log n "
            "between n = 512 and 8192 (−1 is the Monte Carlo rate).\n\n" + "\n\n".join(acc) + "\n")
    (OUT / "array_rqmc.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
