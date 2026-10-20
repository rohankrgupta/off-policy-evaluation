"""QMC-type action sampling on the GridWorld: on its own, and combined with ROS and with BPS.

Domains (as in Phase 2): deterministic GridWorld with pi_e-A and pi_e-B, slippery GridWorld (p = 0.1) with pi_e-B.

Samplers of the action uniforms (core/qmc.py): per-state QMC, and three trajectory-level designs, namely
scrambled Sobol (RQMC), Latin hypercube and antithetic pairs. Each is used
  * on-policy, scored by Monte Carlo (MC), against i.i.d. sampling (OS-MC) and ROS (ROS-MC); and
  * with the exact BPS target, scored by OIS (B, and R for Sobol), against i.i.d. (BPS-OIS) and ROS (BPS+ROS).
The i.i.d. and ROS rows use Phase 1-2 code and Phase 2's random streams. ROS's alpha is tuned on 500 other seeds:
at n = 8192 for the main tables, and separately at each n for the per-state QMC vs ROS table.
Latin hypercubes are balanced only as whole designs, so they are run separately at each reported n.

Run:    python -m phase2_5_qmc.experiments.gridworld     (about 6 min)
Output: results/gridworld.md, results/gridworld.png
"""
from pathlib import Path

import matplotlib
import numpy as np

from phase1_reproduction.core import envs, exact, sampling, stats
from phase2_5_qmc.core import qmc
from phase2_bps_plus_ros.core import bps_ros

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
NS = 2 ** np.arange(4, 14)                      # n = 16 ... 8192
SHOW = (64, 512, 8192)
ALPHAS = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
N_EVAL, N_TUNE = 1000, 500


def domains():
    g = envs.gridworld()
    pi_a = envs.repo_policy_logits("GridWorld_5000", g)
    pi_b = envs.eps_greedy_logits(pi_a, 0.7, g.terminal)
    return [("πe-A", g, pi_a), ("πe-B", g, pi_b), ("πe-B, slippery", envs.gridworld(slip=0.1), pi_b)]


def run_domain(name, mdp, th_e):
    pe = exact.softmax(th_e)
    v, var = exact.value(mdp, pe), exact.return_variance(mdp, pe)
    bps = bps_ros.oracle_bps_target(mdp, th_e)
    sq = lambda est: ((est - v) ** 2).mean(0)                  # MSE per checkpoint

    # ROS toward pi_e at every alpha: alpha tuned at n = 8192 for the main tables, and at each n for table 3
    ros = {a: sampling.run(mdp, th_e, a, NS, N_EVAL, "e4") for a in ALPHAS}
    ros_tune = {a: sq(sampling.run(mdp, th_e, a, NS, N_TUNE, "tune-e4")) for a in ALPHAS}
    a_mc = min(ALPHAS, key=lambda a: ros_tune[a][-1])
    # ROS toward the BPS target, B and R weights each with its own tuned alpha; keep whichever tunes better
    tracked = {}
    for est in ("B", "R"):
        a = min(ALPHAS, key=lambda a: sq(bps_ros.run(mdp, th_e, bps, a, NS, N_TUNE, "tune-e4")[est])[-1])
        tracked[est] = (a, bps_ros.run(mdp, th_e, bps, a, NS, N_EVAL, "e4")[est])
    best = min(tracked, key=lambda e: sq(tracked[e][1])[-1])

    def lhs(target, est):                                      # one Latin hypercube per reported n
        out = np.full((N_EVAL, len(NS)), np.nan)
        for n in SHOW:
            out[:, list(NS).index(n)] = qmc.run(mdp, th_e, target, "lhs", 0.0, [n], N_EVAL, "e4")[est][:, 0]
        return out

    sobol_bps = qmc.run(mdp, th_e, bps, "sobol", 0.0, NS, N_EVAL, "e4")
    per_state = qmc.run(mdp, th_e, th_e, "per-state", 0.0, NS, N_EVAL, "e4")["MC"]
    methods = {   # label -> estimates (seeds, checkpoints)
        "OS-MC": sampling.run(mdp, th_e, 0.0, NS, N_EVAL, "e4"),
        f"ROS-MC (α={a_mc:g})": ros[a_mc],
        "per-state QMC, MC": per_state,
        "Sobol RQMC, MC": qmc.run(mdp, th_e, th_e, "sobol", 0.0, NS, N_EVAL, "e4")["MC"],
        "Latin hypercube, MC": lhs(th_e, "MC"),
        "antithetic, MC": qmc.run(mdp, th_e, th_e, "antithetic", 0.0, NS, N_EVAL, "e4")["MC"],
        "ROS + Sobol RQMC, MC": qmc.run(mdp, th_e, th_e, "sobol", a_mc, NS, N_EVAL, "e4")["MC"],
        "BPS-OIS": bps_ros.run(mdp, th_e, bps, 0.0, NS, N_EVAL, "e4")["B"],
        f"BPS+ROS-{best} (α={tracked[best][0]:g})": tracked[best][1],
        "BPS + Sobol RQMC, B": sobol_bps["B"],
        "BPS + Sobol RQMC, R": sobol_bps["R"],
        "BPS + Latin hypercube, B": lhs(bps, "B"),
        "BPS + antithetic, B": qmc.run(mdp, th_e, bps, "antithetic", 0.0, NS, N_EVAL, "e4")["B"],
        "BPS + per-state QMC, B": qmc.run(mdp, th_e, bps, "per-state", 0.0, NS, N_EVAL, "e4")["B"],
    }
    # Table 3: per-state QMC against ROS with alpha tuned at each n
    per_n = []
    for i, n in enumerate(NS):
        if n in (16, 64, 256, 1024, 8192):
            a = min(ALPHAS, key=lambda a: ros_tune[a][i])
            per_n.append((n, a, stats.paired_mse_ratio(per_state[:, i], ros[a][:, i], v)))
    print(f"{name} done", flush=True)
    return {"name": name, "v": v, "var": var, "methods": methods, "per_n": per_n}


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def mark(lo, hi):
    return "✓" if hi < 1 else "✗" if lo > 1 else "~"


def main():
    OUT.mkdir(exist_ok=True)
    results = [run_domain(*d) for d in domains()]
    sections, contrasts, per_n_rows = [], [], []
    for D in results:
        M = D["methods"]
        rows = []
        for k, e in M.items():
            y = NS * ((e - D["v"]) ** 2).mean(0) / D["var"]
            tail = y[-4:]
            rows.append([k, *(f"{y[list(NS).index(n)]:.3g}" for n in SHOW),
                         "–" if np.isnan(tail).any() else f"{stats.slope(NS, y / NS):+.2f}"])
        sections.append(f"### {D['name']}\n\n" + md(["method", "n = 64", "n = 512", "n = 8192", "slope"], rows))
        ros_mc = next(k for k in M if k.startswith("ROS-MC"))
        tracked = next(k for k in M if k.startswith("BPS+ROS"))
        for x, y in (("Sobol RQMC, MC", "OS-MC"), ("Latin hypercube, MC", "OS-MC"), ("antithetic, MC", "OS-MC"),
                     ("ROS + Sobol RQMC, MC", ros_mc), ("BPS + Sobol RQMC, B", "BPS-OIS"),
                     ("BPS + Latin hypercube, B", "BPS-OIS"), ("BPS + antithetic, B", "BPS-OIS"),
                     ("BPS + Sobol RQMC, B", tracked), ("BPS + Sobol RQMC, B", ros_mc)):
            r, lo, hi = stats.paired_mse_ratio(M[x][:, -1], M[y][:, -1], D["v"])
            contrasts.append([D["name"], f"{x} vs {y}", f"{r:.3g} [{lo:.3g}, {hi:.3g}] {mark(lo, hi)}"])
        per_n_rows.append([D["name"]] + [f"{r:.2f} [{lo:.2f}, {hi:.2f}] {mark(lo, hi)} (α={a:g})"
                                         for n, a, (r, lo, hi) in D["per_n"]])
    text = ("# QMC-type action sampling on the GridWorld\n\n"
            "## Paired MSE ratios at n = 8192 [95% CI]\n\n"
            "✓ the first method is better, ✗ worse, ~ inconclusive. BPS = exact BPS target.\n\n"
            + md(["domain", "comparison", "MSE ratio"], contrasts)
            + "\n\n## Per-state QMC vs ROS with α tuned at each n (MSE ratio, QMC / ROS)\n\n"
            + md(["domain"] + [f"n = {n}" for n, _, _ in results[0]["per_n"]], per_n_rows)
            + "\n\n## n·MSE/Var(g) by method\n\nOn-policy Monte Carlo = 1; lower is better. 1000 seeds. Slope: of log "
            "MSE against log n over n = 1024 … 8192 (−1 is the Monte Carlo rate). Latin hypercubes are run only at "
            "the reported n.\n\n" + "\n\n".join(sections) + "\n")
    (OUT / "gridworld.md").write_text(text)
    print(text)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    style = {"OS-MC": ("0.6", "-"), "ROS-MC": ("k", "-"), "per-state QMC, MC": ("tab:green", "-"),
             "Sobol RQMC, MC": ("tab:purple", "-"), "BPS-OIS": ("tab:blue", "--"), "BPS+ROS": ("tab:orange", "--"),
             "BPS + Sobol RQMC, B": ("tab:red", "--")}
    for ax, D in zip(axes, results):
        for prefix, (color, ls) in style.items():
            key = next(k for k in D["methods"] if k.startswith(prefix))
            y = NS * ((D["methods"][key] - D["v"]) ** 2).mean(0) / D["var"]
            ax.plot(NS, y, ls, color=color, marker="o", ms=3, label=key)
        ax.set(xscale="log", yscale="log", title=D["name"], xlabel="n (trajectories)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("n·MSE / Var(g)")
    axes[0].legend(fontsize=7)
    fig.suptitle("QMC action sampling (solid: on-policy MC; dashed: exact BPS target with OIS)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gridworld.png", dpi=150)


if __name__ == "__main__":
    main()
