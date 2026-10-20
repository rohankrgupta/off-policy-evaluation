"""A behavior policy designed for coordinated sampling (core/j_target.py), against BPS's target and ROS-MC.

Domains: Phase 3's noisy GridWorlds (slip 0.1, slip 0.2, slip 0.1 + noisy trap) with pi_e-B. In a deterministic
GridWorld J is identically 0, so it has nothing to optimize there.

Methods (n = 64 / 512 / 8192 episodes, 1000 seeds):
  ROS-MC (alpha tuned at each n on 500 other seeds); on-policy Array-RQMC with MC;
  the Phase 2 baselines from the exact BPS target: i.i.d. + OIS (BPS-OIS) and ROS + B or R (BPS+ROS, alpha and
  the weighting chosen at each n on the tuning seeds);
  Array-RQMC from the exact BPS target and from the exact J-optimal target, each with OIS and PDIS.
Array-RQMC orders episodes by state and, within a state, by current importance weight.
"predicted" is J(q) / Var(g): the n·MSE that perfect coordination would leave.

Run:    python -m phase2_5_qmc.experiments.j_target     (about 8 min)
Output: results/j_target.md
"""
from pathlib import Path

import numpy as np

from phase1_reproduction.core import exact, sampling, stats
from phase2_5_qmc.core import array_rqmc, j_target
from phase2_5_qmc.experiments.gridworld import ALPHAS, N_EVAL, N_TUNE, md
from phase2_bps_plus_ros.core import bps_ros
from phase3_mb_allocation.experiments.gridworld import domains

OUT = Path(__file__).resolve().parents[1] / "results"
SHOW = (64, 512, 8192)


def run_domain(name, mdp, th_e):
    pe = exact.softmax(th_e)
    v, var = exact.value(mdp, pe), exact.return_variance(mdp, pe)
    targets = {"BPS target": exact.oracle_behavior(mdp, pe)[0], "J target": j_target.optimal_target(mdp, pe)}

    # ROS-MC with alpha tuned at each n
    ns = np.array(SHOW)
    ros = {a: sampling.run(mdp, th_e, a, ns, N_EVAL, "e4") for a in ALPHAS}
    tune = {a: ((sampling.run(mdp, th_e, a, ns, N_TUNE, "tune-e4") - v) ** 2).mean(0) for a in ALPHAS}
    methods = {"ROS-MC (α tuned at each n)":
               np.stack([ros[min(ALPHAS, key=lambda a: tune[a][j])][:, j] for j in range(len(SHOW))], 1)}
    # Phase 2 baselines from the BPS target: i.i.d. + OIS, and ROS with (alpha, B or R) picked at each n on tuning seeds
    bps = targets["BPS target"]
    methods["BPS-OIS (i.i.d.)"] = bps_ros.run(mdp, th_e, bps, 0.0, ns, N_EVAL, "e4")["B"]
    tracked = {a: bps_ros.run(mdp, th_e, bps, a, ns, N_EVAL, "e4") for a in ALPHAS}
    tracked_tune = {a: bps_ros.run(mdp, th_e, bps, a, ns, N_TUNE, "tune-e4") for a in ALPHAS}
    best = [min(((e, a) for e in ("B", "R") for a in ALPHAS), key=lambda ea: ((tracked_tune[ea[1]][ea[0]][:, j] - v) ** 2).mean())
            for j in range(len(SHOW))]
    methods["BPS+ROS (Phase 2)"] = np.stack([tracked[a][e][:, j] for j, (e, a) in enumerate(best)], 1)
    on_policy = [array_rqmc.run(mdp, th_e, th_e, n, N_EVAL, "e4") for n in SHOW]
    methods["Array-RQMC, on-policy, MC"] = np.stack([o["MC"] for o in on_policy], 1)
    predicted = {"Array-RQMC, on-policy, MC": j_target.J(mdp, pe, th_e) / var}
    lengths = {"πe": exact.mean_length(mdp, pe)}
    for label, theta in targets.items():
        runs = [array_rqmc.run(mdp, th_e, theta, n, N_EVAL, "e4", by_weight=True) for n in SHOW]
        for est, est_label in (("B", "OIS"), ("PDIS", "PDIS")):
            methods[f"Array-RQMC, {label}, {est_label}"] = np.stack([o[est] for o in runs], 1)
            predicted[f"Array-RQMC, {label}, {est_label}"] = j_target.J(mdp, pe, theta) / var
        lengths[label] = exact.mean_length(mdp, exact.softmax(theta))
    print(f"{name} done", flush=True)
    return {"name": name, "v": v, "var": var, "methods": methods, "predicted": predicted, "lengths": lengths}


def main():
    OUT.mkdir(exist_ok=True)
    results = [run_domain(*d) for d in domains()]
    mark = lambda lo, hi: "✓" if hi < 1 else "✗" if lo > 1 else "~"
    acc, contrasts = [], []
    for D in results:
        M, v = D["methods"], D["v"]
        rows = []
        for k, e in M.items():
            y = np.array(SHOW) * ((e - v) ** 2).mean(0) / D["var"]
            bias = (e[:, -1] - v).mean() / (e[:, -1].std(ddof=1) / np.sqrt(len(e)))
            pred = D["predicted"].get(k)
            rows.append([k, "–" if pred is None else f"{pred:.3f}", *(f"{x:.3g}" for x in y), f"{bias:+.1f}"])
        lengths = ", ".join(f"{k} {L:.1f}" for k, L in D["lengths"].items())
        acc.append(f"### {D['name']}\n\nMean episode length: {lengths}.\n\n"
                   + md(["method", "predicted", *(f"n = {n}" for n in SHOW), "bias/SE (n = 8192)"], rows))
        for x, y in (("Array-RQMC, J target, PDIS", "BPS-OIS (i.i.d.)"), ("Array-RQMC, J target, PDIS", "BPS+ROS (Phase 2)"),
                     ("Array-RQMC, BPS target, PDIS", "BPS-OIS (i.i.d.)"), ("Array-RQMC, BPS target, PDIS", "BPS+ROS (Phase 2)"),
                     ("Array-RQMC, J target, PDIS", "Array-RQMC, BPS target, PDIS"),
                     ("Array-RQMC, J target, PDIS", "Array-RQMC, J target, OIS"),
                     ("Array-RQMC, J target, PDIS", "Array-RQMC, on-policy, MC"),
                     ("Array-RQMC, J target, PDIS", "ROS-MC (α tuned at each n)"),
                     ("Array-RQMC, BPS target, PDIS", "ROS-MC (α tuned at each n)")):
            cells = []
            for j in range(len(SHOW)):
                r, lo, hi = stats.paired_mse_ratio(M[x][:, j], M[y][:, j], v)
                cells.append(f"{r:.3g} {mark(lo, hi)}")
            contrasts.append([D["name"], f"{x} vs {y}", *cells])
    text = ("# A behavior policy for coordinated sampling (J target), GridWorld\n\n"
            "## Paired MSE ratios at n = 64 / 512 / 8192\n\n✓ the first method is better, ✗ worse, ~ inconclusive "
            "(95% CI).\n\n" + md(["domain", "comparison", *(f"n = {n}" for n in SHOW)], contrasts)
            + "\n\n## n·MSE/Var(g)\n\nOn-policy Monte Carlo = 1; lower is better. 1000 seeds. Predicted: J(q)/Var(g), "
            "the first-order value under perfect coordination.\n\n" + "\n\n".join(acc) + "\n")
    (OUT / "j_target.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
