"""BPS + ROS in a one-step problem: the ROS repository's 30-arm bandit.

Arm means are scaled by rho in {0.1, 0.3, 1, 3, 10} with the reward noise fixed. Small rho means most of
Var(g) is reward noise; large rho means most of it comes from which arm is pulled.

Methods (1000 seeds, n up to 4096 draws, alpha tuned per method on 500 other seeds):
  OS-MC, ROS-MC        on-policy Monte Carlo, i.i.d. or with ROS tracking pi_e (Phase 1 code);
  BPS-OIS, BPS-R       i.i.d. draws from the learned BPS target, weighted by OIS (B) or regression IS (R);
  BPS+ROS-A/B/R        ROS tracking the learned BPS target, with the three weightings.
The exact BPS target (q ∝ pi_e * sqrt(mu^2 + sigma^2)) is also run. Each method is shown next to its
closed-form limit of n * MSE as n grows.

Run:    python -m phase2_bps_plus_ros.experiments.bandit     (about 1 min)
Output: results/bandit.md, results/bandit.png
"""
from pathlib import Path

import matplotlib
import numpy as np

from phase1_reproduction.core import envs, exact, sampling, stats
from phase2_bps_plus_ros.core import bps_ros

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
RHOS = (0.1, 0.3, 1.0, 3.0, 10.0)
NS = 2 ** np.arange(4, 13)                      # n = 16 ... 4096
ALPHAS = (0.0, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
N_EVAL, N_TUNE = 1000, 500


def bandit(rho):
    base = envs.multibandit()
    return envs.bandit(base.R[0, :, 1] * rho, base.reward_std[0, :, 1], f"MultiBandit, rho = {rho:g}")


def limits(pi, mu, sd, q):
    """n * MSE as n grows, for target q (one row per seed): i.i.d. + OIS, and ROS or R (between-arm
    variance removed). With q = pi the second is ROS-MC's limit."""
    q = np.atleast_2d(q)
    v = pi @ mu
    ois = ((pi ** 2 * (mu ** 2 + sd ** 2) / q).sum(-1) - v ** 2).mean()
    tracked = (pi ** 2 * sd ** 2 / q).sum(-1).mean()
    return ois, tracked


def run_rho(rho):
    mdp = bandit(rho)
    th_e = envs.repo_policy_logits("MultiBandit_5000", mdp)
    pi, mu, sd = exact.softmax(th_e)[0], mdp.R[0, :, 1], mdp.reward_std[0, :, 1]
    v, var = pi @ mu, exact.return_variance(mdp, exact.softmax(th_e))
    oracle = np.zeros_like(th_e)
    oracle[0] = np.log(pi * np.sqrt(mu ** 2 + sd ** 2))
    targets = {"BPS": (bps_ros.learned_bps_target(mdp, th_e, N_EVAL, "stage1"),
                       bps_ros.learned_bps_target(mdp, th_e, N_TUNE, "stage1-tune")),
               "BPS*": (oracle, oracle)}
    # runs[(method, alpha)] = (eval estimates (seeds, n), tuning MSE at the largest n)
    runs = {}
    for alpha in ALPHAS:
        ev = sampling.run(mdp, th_e, alpha, NS, N_EVAL, "e3")
        tu = sampling.run(mdp, th_e, alpha, NS, N_TUNE, "tune-e3")
        runs[("MC", alpha)] = (ev, ((tu[:, -1] - v) ** 2).mean())
        for name, (t_eval, t_tune) in targets.items():
            ev = bps_ros.run(mdp, th_e, t_eval, alpha, NS, N_EVAL, "e3")
            tu = bps_ros.run(mdp, th_e, t_tune, alpha, NS, N_TUNE, "tune-e3")
            for est in ("A", "B", "R"):
                runs[(f"{name}-{est}", alpha)] = (ev[est], ((tu[est][:, -1] - v) ** 2).mean())

    def tuned(method, alphas=ALPHAS[1:]):
        return min(alphas, key=lambda a: runs[(method, a)][1])

    lim = {name: limits(pi, mu, sd, exact.softmax(t[0])[..., 0, :]) for name, t in targets.items()}
    rows = {"OS-MC": ("MC", 0.0, var), "ROS-MC": ("MC", tuned("MC"), limits(pi, mu, sd, pi)[1])}
    for name in targets:
        rows.update({f"{name}-OIS": (f"{name}-B", 0.0, lim[name][0]), f"{name}-R": (f"{name}-R", 0.0, lim[name][1]),
                     f"{name}+ROS-A": (f"{name}-A", tuned(f"{name}-A", (1.0, 10.0, 1e2)), None),
                     f"{name}+ROS-B": (f"{name}-B", tuned(f"{name}-B"), lim[name][1]),
                     f"{name}+ROS-R": (f"{name}-R", tuned(f"{name}-R"), lim[name][1])})
    return {"v": v, "var": var, "runs": runs, "rows": rows, "tuned": tuned}


def main():
    OUT.mkdir(exist_ok=True)
    results = {rho: run_rho(rho) for rho in RHOS}
    # Table 1: n·MSE/Var(g) at n = 4096 (closed-form limit in parentheses)
    labels = list(results[RHOS[0]]["rows"])
    table1 = []
    for label in labels:
        row = [label]
        for rho in RHOS:
            r = results[rho]
            method, alpha, limit = r["rows"][label]
            nmse = NS[-1] * ((r["runs"][(method, alpha)][0][:, -1] - r["v"]) ** 2).mean() / r["var"]
            row.append(f"{nmse:.3f}" + ("" if limit is None else f" ({limit / r['var']:.3f})"))
        table1.append(row)
    # Table 2: paired contrasts for BPS+ROS (B or R, whichever tuned better) at n = 4096
    table2 = []
    for name in ("BPS", "BPS*"):
        for rho in RHOS:
            r = results[rho]
            est = min(("B", "R"), key=lambda e: r["runs"][(f"{name}-{e}", r["tuned"](f"{name}-{e}"))][1])
            x = r["runs"][(f"{name}-{est}", r["tuned"](f"{name}-{est}"))][0][:, -1]
            row = [name, f"{rho:g}", f"{name}+ROS-{est}"]
            for comp in (("MC", r["tuned"]("MC")), (f"{name}-B", 0.0), (f"{name}-R", 0.0)):
                ratio, lo, hi = stats.paired_mse_ratio(x, r["runs"][comp][0][:, -1], r["v"])
                row.append(f"{ratio:.2f} [{lo:.2f}, {hi:.2f}] " + ("✓" if hi < 1 else "✗" if lo > 1 else "~"))
            table2.append(row)
    md = lambda header, rows: "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                                        + ["| " + " | ".join(r) + " |" for r in rows])
    text = ("# BPS + ROS on the 30-arm bandit\n\n"
            "## n·MSE/Var(g) at n = 4096 (closed-form limit as n grows in parentheses)\n\n"
            "On-policy Monte Carlo = 1; lower is better. 1000 seeds; α tuned on 500 other seeds. BPS = learned "
            "target (BPG-V on 1024 draws per seed); BPS* = exact target.\n\n"
            + md(["method"] + [f"ρ = {rho:g}" for rho in RHOS], table1)
            + "\n\n## Paired MSE ratios of BPS + ROS at n = 4096 [95% CI]\n\n"
            "✓ BPS + ROS is better, ✗ worse, ~ inconclusive. The tracked method uses B or R, whichever tuned better.\n\n"
            + md(["target", "ρ", "method", "vs ROS-MC", "vs BPS-OIS", "vs BPS-R (i.i.d. + R)"], table2) + "\n")
    (OUT / "bandit.md").write_text(text)
    print(text)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, color in (("ROS-MC", "k"), ("BPS-OIS", "tab:blue"), ("BPS-R", "tab:green"), ("BPS+ROS-R", "tab:red")):
        sim, lim = [], []
        for rho in RHOS:
            r = results[rho]
            method, alpha, limit = r["rows"][label]
            sim.append(NS[-1] * ((r["runs"][(method, alpha)][0][:, -1] - r["v"]) ** 2).mean() / r["var"])
            lim.append(limit / r["var"])
        ax.plot(RHOS, sim, "o-", color=color, label=label)
        ax.plot(RHOS, lim, ":", color=color, lw=1)
    ax.set(xscale="log", yscale="log", xlabel="ρ (scale of arm means relative to reward noise)",
           ylabel="n·MSE / Var(g) at n = 4096", title="Bandit: BPS + ROS never beats both parents")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "bandit.png", dpi=150)


if __name__ == "__main__":
    main()
