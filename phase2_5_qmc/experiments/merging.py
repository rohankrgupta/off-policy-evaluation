"""Prefix-weight merging and Q-ordered actions for Array-RQMC with the BPS target (core/array_rqmc.py).

Domains: deterministic GridWorld with pi_e-A and pi_e-B, and the noisy GridWorlds of Phase 3 (slip 0.1, slip 0.2,
slip 0.1 + noisy trap) with pi_e-B. Sampling policy: the exact BPS target. n = 64 / 512 / 8192, 1000 seeds.

Methods, all PDIS from Array-RQMC unless noted:
  weight-sorted (the previous best), merged (past weights averaged within each state at every step),
  each with actions in the default order or ordered by pi_e/q * Q(s, a) (exact Q, a best case);
  baselines: ROS-MC (alpha tuned at each n), BPS-OIS (i.i.d.) and BPS+ROS (Phase 2; alpha and B/R tuned at each n).
Exact first-order references (÷ Var(g)): J(BPS target), the ideal for coordination with path weights, and the
model-based (MB) variance with data from the BPS target.

Run:    python -m phase2_5_qmc.experiments.merging     (about 8 min)
Output: results/merging.md
"""
from pathlib import Path

import numpy as np

from phase1_reproduction.core import exact, sampling, stats
from phase2_5_qmc.core import array_rqmc, j_target
from phase2_5_qmc.experiments.gridworld import ALPHAS, N_EVAL, N_TUNE, domains as gridworld_domains, md
from phase2_bps_plus_ros.core import bps_ros
from phase3_mb_allocation.core import allocation
from phase3_mb_allocation.experiments.gridworld import domains as noisy_domains

OUT = Path(__file__).resolve().parents[1] / "results"
SHOW = (64, 512, 8192)


def q_order(mdp, pi_e, theta_q):
    """Action order per (t, s): increasing pi_e/q * Q_t(s, a), so the map from uniform to weighted value is monotone."""
    V = exact._backward(mdp, pi_e)[1]
    q = exact.softmax(theta_q)
    order = np.zeros((mdp.horizon, mdp.S, mdp.A), np.int64)
    for t in range(mdp.horizon):
        Q = (mdp.P * (mdp.R + mdp.gamma * V[t + 1][None, None, :])).sum(2)
        order[t] = np.argsort(pi_e / q * Q, axis=1)
    return order


def run_domain(name, mdp, th_e):
    pe = exact.softmax(th_e)
    v, var = exact.value(mdp, pe), exact.return_variance(mdp, pe)
    bps = exact.oracle_behavior(mdp, pe)[0]
    ns = np.array(SHOW)
    mse = lambda est: ((est - v) ** 2).mean(0)

    ros = {a: sampling.run(mdp, th_e, a, ns, N_EVAL, "e4") for a in ALPHAS}
    ros_tune = {a: mse(sampling.run(mdp, th_e, a, ns, N_TUNE, "tune-e4")) for a in ALPHAS}
    tracked = {a: bps_ros.run(mdp, th_e, bps, a, ns, N_EVAL, "e4") for a in ALPHAS}
    tracked_tune = {a: bps_ros.run(mdp, th_e, bps, a, ns, N_TUNE, "tune-e4") for a in ALPHAS}
    best = [min(((e, a) for e in ("B", "R") for a in ALPHAS), key=lambda ea: mse(tracked_tune[ea[1]][ea[0]])[j])
            for j in range(len(SHOW))]
    methods = {
        "ROS-MC (α tuned at each n)": np.stack([ros[min(ALPHAS, key=lambda a: ros_tune[a][j])][:, j]
                                                for j in range(len(SHOW))], 1),
        "BPS-OIS (i.i.d.)": bps_ros.run(mdp, th_e, bps, 0.0, ns, N_EVAL, "e4")["B"],
        "BPS+ROS (Phase 2)": np.stack([tracked[a][e][:, j] for j, (e, a) in enumerate(best)], 1),
    }
    order = q_order(mdp, pe, bps)
    for label, kw in (("weight-sorted", dict(by_weight=True)), ("weight-sorted, Q-ordered", dict(by_weight=True, action_order=order)),
                      ("merged", dict(merge=True)), ("merged, Q-ordered", dict(merge=True, action_order=order))):
        methods[f"Array-RQMC, {label}"] = np.stack(
            [array_rqmc.run(mdp, th_e, bps, n, N_EVAL, "e4", **kw)["PDIS"] for n in SHOW], 1)
    reference = {"J (coordination, path weights)": j_target.J(mdp, pe, bps) / var,
                 "MB with the same data": allocation.mb_variance(mdp, pe, exact.softmax(bps))[0] / var}
    print(f"{name} done", flush=True)
    return {"name": name, "v": v, "var": var, "methods": methods, "reference": reference}


def main():
    OUT.mkdir(exist_ok=True)
    domains = gridworld_domains()[:2] + noisy_domains()          # deterministic pi_e-A, pi_e-B; then the noisy ones
    results = [run_domain(*d) for d in domains]
    mark = lambda lo, hi: "✓" if hi < 1 else "✗" if lo > 1 else "~"
    acc, contrasts = [], []
    for D in results:
        M, v = D["methods"], D["v"]
        rows = []
        for k, e in M.items():
            y = np.array(SHOW) * ((e - v) ** 2).mean(0) / D["var"]
            bias = (e[:, -1] - v).mean() / (e[:, -1].std(ddof=1) / np.sqrt(len(e)))
            rows.append([k, *(f"{x:.3g}" for x in y), f"{np.log(y[2] / y[1]) / np.log(SHOW[2] / SHOW[1]) - 1:+.2f}",
                         f"{bias:+.1f}"])
        refs = "; ".join(f"{k}: {x:.3g}" for k, x in D["reference"].items())
        acc.append(f"### {D['name']}\n\nExact first-order n·Var/Var(g) — {refs}.\n\n"
                   + md(["method", *(f"n = {n}" for n in SHOW), "slope (512 → 8192)", "bias/SE (n = 8192)"], rows))
        for x, y in (("Array-RQMC, merged", "Array-RQMC, weight-sorted"),
                     ("Array-RQMC, merged, Q-ordered", "Array-RQMC, merged"),
                     ("Array-RQMC, merged", "BPS-OIS (i.i.d.)"), ("Array-RQMC, merged", "BPS+ROS (Phase 2)"),
                     ("Array-RQMC, merged", "ROS-MC (α tuned at each n)")):
            cells = []
            for j in range(len(SHOW)):
                r, lo, hi = stats.paired_mse_ratio(M[x][:, j], M[y][:, j], v)
                cells.append(f"{r:.3g} {mark(lo, hi)}")
            contrasts.append([D["name"], f"{x} vs {y}", *cells])
    text = ("# Prefix-weight merging and Q-ordered actions (Array-RQMC, BPS target, PDIS)\n\n"
            "## Paired MSE ratios at n = 64 / 512 / 8192\n\n✓ the first method is better, ✗ worse, ~ inconclusive "
            "(95% CI).\n\n" + md(["domain", "comparison", *(f"n = {n}" for n in SHOW)], contrasts)
            + "\n\n## n·MSE/Var(g)\n\nOn-policy Monte Carlo = 1; lower is better. 1000 seeds.\n\n" + "\n\n".join(acc) + "\n")
    (OUT / "merging.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
